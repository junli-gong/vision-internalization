"""DEV-0021 第 2 条：新增 token 行的训练开关 adapter.trainable_token_rows（规则见 checkpoint.TRAINABLE_ROWS_RULE）。
CPU fp32 小模型，三个家族；Qwen 另测不共享 lm_head 的小模型（共享 / 不共享两种情况都覆盖）。

- 一步优化器更新（全词表的分块交叉熵，梯度流过嵌入查表与 lm_head）之后：打开的行每一行都变了，其余行逐位不变（嵌入、
  不共享时的 lm_head、Gemma 的 PLE），视觉塔与 LLM 的 base 权重逐位不变，LoRA 变了；冻结行的梯度恰为 0。
- 共享权重的家族（Qwen 小模型、Gemma）只返回、只处理一个张量。
- 保存再加载：token 表的行哈希相同、行逐位相同；合并 LoRA 之后这些行仍是训练后的值（不是均值初始化）。
- 对这些参数做 weight decay 会改冻结行：save_trained 的抽样核对报错。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import checkpoint, factory
from src.adapters.checkpoint import CheckpointError, LoraStateError
from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy

POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=False, max_patches=1),
}
INIT_STD = 0.1  # 同 test_adapter_tokens_lora
N_GRID = 8
TRAIN = ("<g1>", "<g4>", "<g6>")  # 打开训练的 grid token（名字）
# (注册键, 小模型的 tie_word_embeddings；None = 家族默认)：Qwen 默认共享、InternVL 默认不共享、Gemma 总是共享
CASES = [("qwen3.5-0.8b", None), ("qwen3.5-0.8b", False), ("gemma-4-e2b-it", None), ("internvl3.5-2b-hf", None)]
CASE_IDS = ["qwen-tied", "qwen-untied", "gemma-tied", "internvl-untied"]


def _make(key, tie=None, **kw):
    if tie is not None:
        kw["tie_word_embeddings"] = tie
    try:
        return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True,
                            init_std=INIT_STD, **kw)
    except OSError as e:
        pytest.fail(f"取不到 {key} 的小文件：{e!r}（需要联网下载一次）", pytrace=False)


def _img(seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (100, 150, 3), dtype=np.uint8)
    a[25:50, 40:80, 1] = 230
    return Image.fromarray(a)


def _setup(key, tie=None, *, lora_init=False):
    ad = _make(key, tie)
    grid = ad.add_grid_tokens([f"<g{i}>" for i in range(N_GRID)])
    ad.inject_lora(r=4, lora_alpha=8, init_lora_weights=lora_init)
    return ad, grid


def _batch(ad, grid):
    """两行输入：指令里写全部 grid token（嵌入查表经过每一行），目标也含全部 grid token（lm_head 经过每一行）。"""
    surf = {g["name"]: g["surface"] for g in ad.token_table()["grid_tokens"]}
    text = "Cells: " + "".join(surf[f"<g{i}>"] for i in range(N_GRID))
    inputs = [ad.build_inputs([_img(0)], text, "instruction_first"), ad.build_inputs([], text + " again", "image_first")]
    ids = [grid[f"<g{i}>"] for i in range(N_GRID)]
    return inputs, [ids + [1001, 1002], ids[::-1]]


def _step(ad, grid, params, *, token_wd=0.0, checkpointing=False):
    """一步训练（全词表 LM 交叉熵）+ AdamW：LoRA 的参数组 weight_decay = 0.01，token 行的参数组 weight_decay = token_wd。"""
    inputs, targets = _batch(ad, grid)
    ad.train_mode()
    if checkpointing:
        ad.set_gradient_checkpointing(True)
    out = ad.training_forward(inputs, targets, enable_thinking=False)
    loss = ad.chunked_lm_loss(out.target_hidden, out.target_ids, chunk_size=5)
    loss.backward()
    lora = [p for n, p in ad.model.named_parameters() if "lora_" in n]
    opt = torch.optim.AdamW([{"params": lora, "weight_decay": 0.01}, {"params": params, "weight_decay": token_wd}], lr=1e-2)
    grads = {id(p): p.grad.detach().clone() for p in params}
    opt.step()
    opt.zero_grad(set_to_none=True)
    ad.eval_mode()
    return grads


@pytest.mark.parametrize("key,tie", CASES, ids=CASE_IDS)
def test_one_step_changes_only_trainable_rows(key, tie):
    ad, grid = _setup(key, tie)
    train_ids = sorted(grid[n] for n in TRAIN)
    tables = dict(ad._row_tables())
    emb, head = tables["embed"], tables["lm_head"]
    # 名字与 id 混用都可以
    params = ad.trainable_token_rows([TRAIN[0], grid[TRAIN[1]], TRAIN[2]])
    assert ad.trainable_token_row_ids == tuple(train_ids)
    if ad.tied_word_embeddings:
        assert emb is head and len(params) == 1 and params[0] is emb  # 共享：同一个张量，只处理一次
    else:
        assert emb is not head and len(params) == 2 and params[0] is emb and params[1] is head
    assert all(p.requires_grad for p in params)
    snap = {n: p.detach().clone() for n, p in ad.model.named_parameters()}
    grads = _step(ad, grid, params, checkpointing=(tie is False))
    frozen = torch.ones(emb.shape[0], dtype=torch.bool)
    frozen[train_ids] = False
    for p in params:  # 冻结行的梯度恰为 0；打开的行每一行都有梯度
        g = grads[id(p)]
        assert torch.count_nonzero(g[frozen]) == 0
        assert all(torch.count_nonzero(g[i]) > 0 for i in train_ids)
    row_params = {id(p) for p in params}
    n_lora_changed = 0
    for n, p in ad.model.named_parameters():
        before = snap[n]
        if "lora_" in n:
            n_lora_changed += int(not torch.equal(p, before))
        elif id(p) in row_params:
            assert torch.equal(p[frozen], before[frozen]), n  # 其余行逐位不变
            for i in train_ids:
                assert not torch.equal(p[i], before[i]), (n, i)  # 打开的行每一行都变了
        else:
            assert torch.equal(p, before), n  # 视觉塔、投影层、LLM base 权重、Gemma 的 PLE
    assert n_lora_changed > 0


@pytest.mark.parametrize("key,tie", CASES, ids=CASE_IDS)
def test_save_load_and_merge_keep_trained_rows(key, tie, tmp_path):
    ad, grid = _setup(key, tie)
    init_rows = ad.token_rows()
    params = ad.trainable_token_rows(list(TRAIN))
    _step(ad, grid, params)
    table, rows = ad.token_table(), ad.token_rows()
    train_pos = [table["row_ids"].index(grid[n]) for n in TRAIN]
    other_pos = [k for k in range(len(table["row_ids"])) if k not in train_pos]
    for kind, r in rows.items():  # 训练后的行与初始化不同，其余新增行仍是初始化的值；Gemma 的 PLE 不训练
        if kind == "per_layer":
            assert torch.equal(r, init_rows[kind])
            continue
        assert all(not torch.equal(r[k], init_rows[kind][k]) for k in train_pos), kind
        assert torch.equal(r[other_pos], init_rows[kind][other_pos]), kind
    d = ad.save_trained(tmp_path / "ck")
    man = checkpoint.read_manifest(d)
    assert man["trainable_token_rows"] == sorted(grid[n] for n in TRAIN)

    b, _ = factory.load_trained(d, device="cpu", merge=False)
    assert b.token_table() == table  # 行哈希相同
    assert all(torch.equal(v, rows[k]) for k, v in b.token_rows().items()) and set(b.token_rows()) == set(rows)
    c, _ = factory.load_trained(d, device="cpu")  # merge=True
    assert c.lora_state == "merged" and c.token_table() == table
    assert all(torch.equal(v, rows[k]) for k, v in c.token_rows().items())
    ad.merge_lora()  # 原适配器合并之后，行仍是训练后的值
    assert ad.token_table() == table
    assert all(torch.equal(v, rows[k]) for k, v in ad.token_rows().items())


def test_weight_decay_on_token_rows_is_caught(tmp_path):
    """梯度掩码只让冻结行的梯度为 0；AdamW 的 weight decay 不看梯度，会改所有行 —— save_trained 的抽样核对报错。"""
    ad, grid = _setup("internvl3.5-2b-hf")
    params = ad.trainable_token_rows(list(TRAIN))
    _step(ad, grid, params, token_wd=0.1)
    with pytest.raises(CheckpointError, match="weight_decay"):
        ad.save_trained(tmp_path / "ck")


def test_trainable_rows_validation():
    ad = _make("qwen3.5-0.8b")
    grid = ad.add_grid_tokens([f"<g{i}>" for i in range(N_GRID)])
    with pytest.raises(LoraStateError, match="inject_lora"):  # 注入 LoRA 之前
        ad.trainable_token_rows(["<g0>"])
    ad.inject_lora(r=2, lora_alpha=4)
    with pytest.raises(ValueError, match="不是本适配器加过的"):
        ad.trainable_token_rows(["<nope>"])
    with pytest.raises(ValueError, match="grid token 的行"):
        ad.trainable_token_rows([15])  # 基础词表的行
    with pytest.raises(ValueError, match="重复"):
        ad.trainable_token_rows(["<g0>", grid["<g0>"]])
    with pytest.raises(ValueError, match="为空"):
        ad.trainable_token_rows([])
    with pytest.raises(TypeError):
        ad.trainable_token_rows("<g0>")
    assert ad.trainable_token_row_ids is None
    # 再调一次就换成新的行集合（旧钩子移除）
    (w,) = ad.trainable_token_rows(["<g0>"])
    (w2,) = ad.trainable_token_rows(["<g2>"])
    assert w is w2 and ad.trainable_token_row_ids == (grid["<g2>"],)
    inputs, _ = _batch(ad, grid)
    out = ad.training_forward(inputs[1:], [[grid["<g0>"], grid["<g2>"]]], enable_thinking=False)
    ad.chunked_lm_loss(out.target_hidden, out.target_ids, chunk_size=8).backward()
    nz = torch.count_nonzero(w.grad, dim=1).nonzero().flatten().tolist()
    assert nz == [grid["<g2>"]]
    ad.merge_lora()
    with pytest.raises(LoraStateError, match="合并之前"):
        ad.trainable_token_rows(["<g1>"])
