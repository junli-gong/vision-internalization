"""DEV-0020 第二部分：新增 token 的表与核对式恢复、查询占位符固定在预留区末行、LoRA 目标 / 注入 / 合并、检查点的保存与加载。
CPU fp32 小模型，三个家族。

- token 表：名字 → id、每行 bf16 字节的 sha256、查询登记、家族与 revision；restore_tokens 只核对不初始化（D21-2 的"静默重置为
  均值"在这里钉住：恢复后的行是训练过的行，不是均值）。
- 查询占位符（Qwen、InternVL）固定在预留区末行：先注入查询、先加 grid token、交错进行，grid token 的 id 都相同；300 的预算照旧成立。
- LoRA 目标只含 LLM 线性层（含 GDN 投影），覆盖 LLM 的全部线性层；真实尺寸（meta 设备）上的模块数与 rank 32 的参数量与
  env_report §7、§8 的实测相同。顺序约束：LoRA 之后再加 token / 注入查询 / 恢复 token 表报错。
- 合并一致性：随机非零的 LoRA，合并前后 generate_batch 的输出逐个相同，training_forward 的隐藏状态与 logits 在容差内相同。
- 检查点：只有 LoRA（peft 格式）、token 表与行、查询嵌入、extra 与 manifest，没有合并后的权重；factory.load_trained 重建 base、
  恢复、合并，输出与保存前相同；文件被改、base 不同、目录里多文件都报错。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import checkpoint, factory, registry
from src.adapters.checkpoint import CheckpointError, LoraStateError, TokenRestoreError
from src.adapters.factory import make_adapter
from src.adapters.image_policy import (
    INTERNVL_NATIVE_TILES,
    FixedSize,
    PixelBudget,
    SoftTokenBudget,
    TilePolicy,
    parse_policy,
)
from src.adapters.training import TokenOrderError
from src.constants import MAX_NEW_GRID_TOKENS

FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
RESERVED_KEYS = ("qwen3.5-0.8b", "internvl3.5-2b-hf")  # 有预留行的两家
POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=False, max_patches=1),
}
INIT_STD = 0.1
# 预留区：attach 时的 len(tokenizer) 与嵌入行数（model_notes §5.1、§7.1）；占位符取末行
RESERVED = {"qwen3.5-0.8b": (248077, 248320), "internvl3.5-2b-hf": (151679, 151936)}
# 真实尺寸、rank 32 的 LoRA（env_report §7 / §8 的实测：模块数, 可训练参数）
ENV_REPORT_LORA = {
    "qwen3.5-0.8b": (186, 21_645_312),
    "gemma-4-e2b-it": (276, 52_666_368),
    "internvl3.5-2b-hf": (196, 34_865_152),
}
TF_ATOL = 1e-4  # 合并前后 fp32 的隐藏状态 / logits：只差合并带来的舍入（W + BA 与 W x + B(Ax) 的求和顺序）


def _make(key, **kw):
    try:
        return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True,
                            init_std=INIT_STD, **kw)
    except OSError as e:
        pytest.fail(f"取不到 {key} 的小文件：{e!r}（需要联网下载一次）", pytrace=False)


def _img(rng, h, w):
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[h // 4: h // 2, w // 4: w // 2, 1] = 230
    return Image.fromarray(a)


def _inputs(ad, seed=0):
    rng = np.random.default_rng(seed)
    return [ad.build_inputs([_img(rng, 100, 150)], "Describe the image in one short sentence.", "instruction_first"),
            ad.build_inputs([], "Hi", "instruction_first"),
            ad.build_inputs([_img(rng, 150, 100)], "Which side is brighter?", "image_first")]


def _randomize_lora(ad, seed=0, std=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in ad.model.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn(p.shape, generator=g) * std)


def _train_rows(ad, ids, seed=1):
    """模拟训练：把新增行改成随机值（与均值初始化区分开）。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor(ids)
    with torch.no_grad():
        for kind, w in ad._row_tables():
            if kind == "lm_head" and ad.tied_word_embeddings:
                continue
            w[idx] = torch.randn(len(ids), w.shape[1], generator=g) * 0.1


# ---------------------------------------------------------------------------
# 查询占位符固定在预留区末行；grid token 的 id 与调用顺序无关；300 的预算
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", RESERVED_KEYS)
def test_query_row_is_last_reserved_row_and_order_independent(key):
    n_tok, n_rows = RESERVED[key]
    names = [f"<g{i}>" for i in range(MAX_NEW_GRID_TOKENS)]
    a = _make(key)
    a.inject_query_tokens(4, "seq1d")
    ids_a = a.add_grid_tokens(names)
    b = _make(key)
    ids_b = b.add_grid_tokens(names)
    b.inject_query_tokens(4, "seq1d")
    c = _make(key)
    ids_c = c.add_grid_tokens(names[:100])
    c.inject_query_tokens(4, "seq1d")
    ids_c.update(c.add_grid_tokens(names[100:]))
    assert ids_a == ids_b == ids_c
    for ad in (a, b, c):
        assert ad.query_token_id == n_rows - 1 and ad.query_token_id not in ad.grid_token_ids.values()
        assert ad.n_grid_tokens == MAX_NEW_GRID_TOKENS  # 300 的预算（查询占位符不计）
        assert ad.model.get_input_embeddings().weight.shape[0] == n_tok + MAX_NEW_GRID_TOKENS + 1
        assert ad.tokenizer.convert_tokens_to_ids("<|visint_query|>") == n_rows - 1  # 越过末行时登记进 tokenizer 占住
        with pytest.raises(ValueError, match="上限"):
            ad.add_grid_tokens(["<extra>"])
    want = list(range(n_tok, n_rows - 1)) + list(range(n_rows, n_rows + MAX_NEW_GRID_TOKENS - (n_rows - 1 - n_tok)))
    assert list(ids_a.values()) == want
    ta, tb, tc = a.token_table(), b.token_table(), c.token_table()
    assert ta["grid_tokens"] == tb["grid_tokens"] == tc["grid_tokens"] and ta["row_ids"] == tb["row_ids"]
    # 不越过末行时占位符不进 tokenizer，grid id 也不受影响
    d, e = _make(key), _make(key)
    d.inject_query_tokens(4, "seq1d")
    assert d.add_grid_tokens(names[:10]) == e.add_grid_tokens(names[:10])
    assert len(d.tokenizer) == n_tok + 10 and "<|visint_query|>" not in d.tokenizer.get_vocab()
    inp = d.build_inputs([], "Hi", "instruction_first")
    ids = inp.tensors["input_ids"][0].tolist()
    assert ids[inp.query_span.start: inp.query_span.end] == [n_rows - 1] * 4
    # 占位符行按 D9-3 / D13-1 初始化为基础行均值
    emb = d.model.get_input_embeddings().weight
    assert torch.equal(emb[n_rows - 1], emb[: d.n_base_vocab].double().mean(0).to(emb.dtype))


def test_gemma_query_row_order_independent():
    names = [f"<vg{i}>" for i in range(MAX_NEW_GRID_TOKENS)]
    a, b = _make("gemma-4-e2b-it"), _make("gemma-4-e2b-it")
    a.inject_query_tokens(4, "seq1d")
    ids_a = a.add_grid_tokens(names)
    ids_b = b.add_grid_tokens(names)
    b.inject_query_tokens(4, "seq1d")
    assert ids_a == ids_b and a.query_token_id == b.query_token_id == max(a.unused_token_ids)


# ---------------------------------------------------------------------------
# token 表与核对式恢复
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_token_table_contents(key):
    ad = _make(key)
    ids = ad.add_grid_tokens(["<vg_a>", "<vg_b>", "<vg_c>"])
    ad.inject_query_tokens(4, "seq1d")
    t = ad.token_table()
    entry = registry.get_model(key)
    assert (t["family"], t["model_key"], t["repo"], t["revision"], t["tiny"]) == (
        ad.family, f"tiny({key})", entry.repo, entry.revision, True)
    assert [(g["name"], g["id"]) for g in t["grid_tokens"]] == sorted(ids.items(), key=lambda kv: kv[1])
    assert t["query"]["id"] == ad.query_token_id and t["query"]["n"] == 4 and t["query"]["position_mode"] == "seq1d"
    assert t["row_ids"] == sorted(list(ids.values()) + [ad.query_token_id])
    kinds = {"embed", "lm_head"} | ({"per_layer"} if key.startswith("gemma") else set())
    assert set(t["row_tables"]) == kinds and t["tied_word_embeddings"] == ad.tied_word_embeddings
    idx = torch.tensor(t["row_ids"])
    for kind, w in ad._row_tables():
        assert t["row_tables"][kind]["sha256"] == checkpoint.row_sha256(w[idx])
    json.dumps(t)  # 可以原样写进 JSON
    rows = ad.token_rows()
    assert set(rows) == (kinds - {"lm_head"} if ad.tied_word_embeddings else kinds)
    # 哈希按 bf16 字节：fp32 的微小差别（低于 bf16 精度）不改哈希，超过就改
    r = rows["embed"][0].clone()
    assert checkpoint.row_sha256(r[None])[0] == t["row_tables"]["embed"]["sha256"][0]
    r2 = r.clone()
    r2[0] += 1.0
    assert checkpoint.row_sha256(r2[None])[0] != t["row_tables"]["embed"]["sha256"][0]


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_restore_tokens_verifies_and_does_not_reinit(key):
    """D21-2：训练后的模型重新挂适配器、再 add_grid_tokens 会把训练过的行重置为均值；restore_tokens 写回训练过的行。"""
    names = [f"<g{i}>" for i in range(250)] if key in RESERVED_KEYS else [f"<vg{i}>" for i in range(20)]
    ad = _make(key)
    ids = ad.add_grid_tokens(names)
    mode = "grid2d" if key.startswith("qwen") else "seq1d"
    ad.inject_query_tokens(9, mode)
    _train_rows(ad, ad.token_table()["row_ids"])
    table, rows = ad.token_table(), ad.token_rows()

    naive = _make(key)  # 旧做法：重新挂、再加 grid token —— 行是均值，不是训练过的值
    naive.add_grid_tokens(names)
    w_naive = naive.model.get_input_embeddings().weight
    idx = torch.tensor(table["row_ids"])
    assert naive.grid_token_ids == ids  # id 相同，但行被重置成了均值
    mean = w_naive[: naive.n_base_vocab].double().mean(0).to(w_naive.dtype) if key in RESERVED_KEYS else None
    if mean is not None:
        assert torch.equal(w_naive[list(ids.values())], mean.expand(len(ids), -1))
    assert not torch.equal(w_naive[idx[:5]], rows["embed"][:5])

    fresh = _make(key)
    fresh.restore_tokens(table, rows)
    assert fresh.grid_token_ids == ids and fresh.query_token_id == ad.query_token_id
    assert fresh.token_table() == table
    for kind, r in fresh.token_rows().items():
        assert torch.equal(r, rows[kind]), kind
    a = ad.build_inputs([], "Hi", "instruction_first")
    b = fresh.build_inputs([], "Hi", "instruction_first")
    assert torch.equal(a.tensors["input_ids"], b.tensors["input_ids"]) and a.query_span == b.query_span
    assert a.query_position_mode == b.query_position_mode == mode


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_restore_tokens_rejects_mismatch(key):
    ad = _make(key)
    ad.add_grid_tokens(["<vg_a>", "<vg_b>"])
    table, rows = ad.token_table(), ad.token_rows()
    used = _make(key)
    used.add_grid_tokens(["<vg_z>"])
    with pytest.raises(TokenRestoreError, match="刚挂好"):  # 不是刚挂好的适配器
        used.restore_tokens(table, rows)
    bad_rows = {k: v.clone() for k, v in rows.items()}
    bad_rows["embed"][1, 0] += 1.0
    with pytest.raises(TokenRestoreError, match="哈希"):
        _make(key).restore_tokens(table, bad_rows)
    bad_table = json.loads(json.dumps(table))
    bad_table["grid_tokens"][1]["id"] += 1
    with pytest.raises(TokenRestoreError, match="id"):
        _make(key).restore_tokens(bad_table, rows)
    bad_table = json.loads(json.dumps(table))
    bad_table["revision"] = "0" * 40
    with pytest.raises(TokenRestoreError, match="revision"):
        _make(key).restore_tokens(bad_table, rows)
    with pytest.raises(TokenRestoreError, match="行"):
        _make(key).restore_tokens(table, {k: v for k, v in rows.items() if k != "embed"})
    f = _make(key)
    f.inject_lora(r=2, lora_alpha=4)
    with pytest.raises(TokenOrderError):
        f.restore_tokens(table, rows)


# ---------------------------------------------------------------------------
# LoRA 目标模块
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_lora_targets_only_llm_linear_and_cover_all(key):
    ad = _make(key)
    targets = ad.lora_target_modules()
    linear = {n for n, m in ad.model.named_modules() if isinstance(m, torch.nn.Linear)}
    llm = {n for n in linear if n.startswith("model.language_model.")}
    assert set(targets) == llm and len(targets) == len(set(targets))  # 覆盖 LLM 的全部线性层
    assert not set(targets) & {n for n in linear if not n.startswith("model.language_model.")}  # 不含视觉塔、投影层、lm_head
    for n in targets:
        assert not any(s in n for s in ("visual", "vision_tower", "multi_modal_projector", "embed_vision", "lm_head"))
    suffixes = {n.rsplit(".", 1)[-1] for n in targets}
    assert suffixes == set(checkpoint.LORA_TARGET_SUFFIXES[ad.family])
    if key.startswith("qwen"):
        assert {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= suffixes  # GDN 的投影


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_lora_targets_real_size_match_env_report(key):
    """真实尺寸的模型建在 meta 设备上（不占内存、不下载权重）：模块数与 rank 32 的参数量与 env_report 的实测相同。"""
    from transformers import AutoConfig, AutoModelForImageTextToText

    entry = registry.get_model(key)
    cfg = AutoConfig.from_pretrained(entry.repo, revision=entry.revision)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg)
    targets = checkpoint.lora_targets_for(model, entry.family)
    mods = dict(model.named_modules())
    n_params = sum(32 * (mods[t].in_features + mods[t].out_features) for t in targets)
    assert (len(targets), n_params) == ENV_REPORT_LORA[key]


def test_lora_targets_reject_unknown_linear():
    ad = _make("internvl3.5-2b-hf")
    ad.model.model.language_model.layers[0].mlp.extra_proj = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="目标表"):
        ad.lora_target_modules()


# ---------------------------------------------------------------------------
# 注入、顺序约束、合并一致性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_lora_order_and_state(key):
    ad = _make(key)
    ad.add_grid_tokens(["<vg_a>"])
    ad.inject_query_tokens(4, "seq1d")
    model = ad.model
    ad.inject_lora(r=4, lora_alpha=8)
    assert ad.model is model and ad.lora_state == "injected"
    assert ad.lora_parameter_count() == sum(4 * (m.in_features + m.out_features)
                                            for n, m in dict(ad.model.named_modules()).items()
                                            if n in {t + ".base_layer" for t in ad.lora_target_modules()})
    assert all(not p.requires_grad for n, p in ad.model.named_parameters() if "lora_" not in n)
    with pytest.raises(TokenOrderError, match="inject_lora"):
        ad.add_grid_tokens(["<vg_b>"])
    ad.inject_query_tokens(9, "seq1d")  # 已分配过占位符的行：再调只改个数，LoRA 之后也可以
    ad.clear_query_tokens()
    late = _make(key)
    late.inject_lora(r=2, lora_alpha=4)
    with pytest.raises(TokenOrderError, match="inject_lora"):  # 首次注入要分配行：顺序反了报错
        late.inject_query_tokens(4, "seq1d")
    with pytest.raises(LoraStateError):
        ad.inject_lora(r=4, lora_alpha=8)
    ad.merge_lora()
    assert ad.lora_state == "merged" and ad.peft_model is None
    assert not [n for n, _ in ad.model.named_modules() if "lora_" in n]
    with pytest.raises(LoraStateError):
        ad.merge_lora()
    with pytest.raises(TokenOrderError):
        ad.add_grid_tokens(["<vg_b>"])
    for bad in (dict(r=0, lora_alpha=8), dict(r=4, lora_alpha=0), dict(r=4, lora_alpha=8, lora_dropout=1.0)):
        with pytest.raises(ValueError):
            _make(key).inject_lora(**bad)


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_merge_consistency(key):
    """随机非零的 LoRA：合并前后 generate_batch 逐个相同，training_forward 的隐藏状态与 logits 在 TF_ATOL 内相同。"""
    ad = _make(key)
    grid = ad.add_grid_tokens([f"<g{i}>" for i in range(8)])
    inputs = _inputs(ad)
    base = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=8)
    ad.inject_lora(r=4, lora_alpha=8, init_lora_weights=False)
    _randomize_lora(ad)
    rep = checkpoint.merge_consistency(ad, inputs, max_new_tokens=8, enable_thinking=False)
    assert ad.lora_state == "merged"
    assert rep["generate"]["exact_rows"] == 1.0 and rep["generate"]["token_agreement"] == 1.0
    assert rep["teacher_forcing"]["argmax_agreement"] == 1.0
    assert rep["teacher_forcing"]["max_abs_hidden"] <= TF_ATOL and rep["teacher_forcing"]["max_abs_logit"] <= TF_ATOL
    assert [r.token_ids for r in rep["before"]] != [b.token_ids for b in base]  # 前提：LoRA 确实改变了输出
    allowed = sorted(set(grid.values()) | set(ad._stop_ids()))
    ad2 = _make(key)  # 受限解码（grid token）同样一致
    ad2.add_grid_tokens([f"<g{i}>" for i in range(8)])
    ad2.inject_lora(r=4, lora_alpha=8, init_lora_weights=False)
    _randomize_lora(ad2)
    rep2 = checkpoint.merge_consistency(ad2, inputs, max_new_tokens=8, enable_thinking=True, allowed_ids=allowed)
    assert rep2["generate"]["exact_rows"] == 1.0 and rep2["teacher_forcing"]["argmax_agreement"] == 1.0
    assert rep2["teacher_forcing"]["max_abs_logit"] <= TF_ATOL


# ---------------------------------------------------------------------------
# 检查点
# ---------------------------------------------------------------------------


def _plain_inputs(ad):
    """不带查询 token 的生成输入（clear 之后建，再恢复原来的查询登记；不碰任何行）。"""
    n, mode = ad.token_table()["query"]["n"], ad.token_table()["query"]["position_mode"]
    ad.clear_query_tokens()
    out = _inputs(ad)
    if n is not None:
        ad.inject_query_tokens(n, mode)
    return out


def _trained(key, tmp_path, *, lora=True, query=True):
    ad = _make(key)
    ad.add_grid_tokens([f"<g{i}>" for i in range(12)])
    if query:
        ad.inject_query_tokens(4, "seq1d")
    _train_rows(ad, ad.token_table()["row_ids"])
    if lora:
        ad.inject_lora(r=4, lora_alpha=8, init_lora_weights=False)
        _randomize_lora(ad)
    qe = torch.randn(4, ad.hidden_size(), generator=torch.Generator().manual_seed(3)) * 0.05 if query else None
    extra = {"decoder.w": torch.randn(3, 5), "mix.logits": torch.arange(4, dtype=torch.float32)}
    d = ad.save_trained(tmp_path / "ck", extra=extra, query_embeds=qe)
    return ad, d, qe, extra


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_save_and_load_round_trip(key, tmp_path):
    ad, d, qe, extra = _trained(key, tmp_path)
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert files == sorted([checkpoint.MANIFEST, checkpoint.TOKENS_JSON, checkpoint.TOKEN_ROWS, checkpoint.LORA_CONFIG,
                            checkpoint.LORA_WEIGHTS, checkpoint.QUERY_EMBEDS, checkpoint.EXTRA])
    man = checkpoint.read_manifest(d)
    entry = registry.get_model(key)
    assert (man["repo"], man["revision"], man["family"], man["model_key"]) == (
        entry.repo, entry.revision, ad.family, f"tiny({key})")
    assert man["lora"]["r"] == 4 and man["lora"]["n_targets"] == len(ad.lora_target_modules())
    assert man["build_spec"]["key"] == key and man["build_spec"]["tiny"] is True
    lora_cfg = json.loads((d / checkpoint.LORA_CONFIG).read_text())
    assert lora_cfg["peft_type"] == "LORA" and lora_cfg["revision"] == entry.revision
    # 不存合并后的权重：检查点的体积只有 LoRA、新增行、查询嵌入与 extra
    total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    n_model = sum(p.numel() * p.element_size() for p in ad.model.parameters())
    assert total < n_model / 5

    inputs = _plain_inputs(ad)
    want_gen = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=6)
    qinp = ad.build_inputs([], "Segment.", "instruction_first")
    with torch.no_grad():
        want_tf = ad.training_forward([qinp], [[1001, 1002]], enable_thinking=False, query_embeds=qe).target_hidden

    b, ck = factory.load_trained(d, device="cpu", merge=False)
    assert b.lora_state == "injected" and b.token_table() == ad.token_table()
    assert torch.equal(ck.query_embeds, qe) and set(ck.extra) == set(extra)
    assert all(torch.equal(ck.extra[k], v) for k, v in extra.items())
    assert [g.token_ids for g in b.generate_batch(_plain_inputs(b), enable_thinking=False, max_new_tokens=6)] == [
        g.token_ids for g in want_gen]
    with torch.no_grad():
        got = b.training_forward([b.build_inputs([], "Segment.", "instruction_first")], [[1001, 1002]],
                                 enable_thinking=False, query_embeds=ck.query_embeds).target_hidden
    assert torch.equal(got, want_tf)

    c, ck2 = factory.load_trained(d, device="cpu")  # 默认 merge=True：在内存里合并
    assert c.lora_state == "merged"
    assert [g.token_ids for g in c.generate_batch(_plain_inputs(c), enable_thinking=False, max_new_tokens=6)] == [
        g.token_ids for g in want_gen]
    with torch.no_grad():
        got = c.training_forward([c.build_inputs([], "Segment.", "instruction_first")], [[1001, 1002]],
                                 enable_thinking=False, query_embeds=ck2.query_embeds).target_hidden
    torch.testing.assert_close(got, want_tf, rtol=0, atol=TF_ATOL)
    with pytest.raises(LoraStateError, match="合并"):
        c.save_trained(tmp_path / "again")


def test_save_without_lora_and_query(tmp_path):
    """C1（冻结 LLM）之类：没有 LoRA、没有查询，只有 token 行与 extra。"""
    ad, d, _, _ = _trained("qwen3.5-0.8b", tmp_path, lora=False, query=False)
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert files == sorted([checkpoint.MANIFEST, checkpoint.TOKENS_JSON, checkpoint.TOKEN_ROWS, checkpoint.EXTRA])
    b, ck = factory.load_trained(d, device="cpu")
    assert b.lora_state == "none" and ck.query_embeds is None and b.token_table() == ad.token_table()


def test_checkpoint_rejects_tampering_and_wrong_base(tmp_path):
    key = "internvl3.5-2b-hf"
    ad, d, qe, _ = _trained(key, tmp_path)
    # 文件被改
    p = d / checkpoint.EXTRA
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 1
    p.write_bytes(bytes(raw))
    with pytest.raises(CheckpointError, match="sha256"):
        factory.load_trained(d, device="cpu")
    raw[-1] ^= 1
    p.write_bytes(bytes(raw))
    checkpoint.read_manifest(d)
    # 多出文件（例如有人把合并后的权重放进来）；以 "." 开头的文件不算
    (d / "model.safetensors").write_bytes(b"x")
    with pytest.raises(CheckpointError, match="多出"):
        checkpoint.read_manifest(d)
    (d / "model.safetensors").unlink()
    (d / ".DS_Store").write_bytes(b"x")
    checkpoint.read_manifest(d)
    # base 不同（种子不同 → 指纹不同）
    other = _make(key, seed=1)
    with pytest.raises(CheckpointError, match="base_fingerprint"):
        other.load_trained(d)
    # 图像口径不同
    other = make_adapter(key, device="cpu", dtype=torch.float32, tiny=True, init_std=INIT_STD,
                         image_policy=INTERNVL_NATIVE_TILES)
    with pytest.raises(CheckpointError, match="image_policy"):
        other.load_trained(d)
    # 已经注入了 LoRA / 加过 token 的适配器不能再恢复
    used = _make(key)
    used.add_grid_tokens(["<vg_z>"])
    with pytest.raises(TokenRestoreError):
        used.load_trained(d)
    # 小模型的检查点不能靠"按模型键加载"恢复
    from src.adapters.internvl35 import InternVL35Adapter

    with pytest.raises(CheckpointError, match="小模型"):
        InternVL35Adapter(device="cpu", crop_to_patches=False, max_patches=1).load_trained(d)
    # 保存的入参
    with pytest.raises(CheckpointError, match="空目录"):
        ad.save_trained(d, query_embeds=qe)
    with pytest.raises(CheckpointError, match="query_embeds"):
        ad.save_trained(tmp_path / "x1")
    with pytest.raises(CheckpointError, match="形状"):
        ad.save_trained(tmp_path / "x2", query_embeds=qe[:2])
    with pytest.raises(CheckpointError, match="extra"):
        ad.save_trained(tmp_path / "x3", query_embeds=qe, extra={"a": 1})


@pytest.mark.parametrize("policy", [FixedSize(64, 96), PixelBudget(300), SoftTokenBudget(280), TilePolicy(False, 1),
                                    INTERNVL_NATIVE_TILES])
def test_policy_spec_round_trip(policy):
    assert parse_policy(checkpoint.policy_spec(policy)) == policy


def test_base_fingerprint_ignores_lora_and_expansion():
    ad = _make("qwen3.5-0.8b")
    fp = checkpoint.base_fingerprint(ad.model)
    assert fp == ad._base_fingerprint
    ad.add_grid_tokens([f"<g{i}>" for i in range(250)])  # 扩表
    ad.inject_lora(r=2, lora_alpha=4, init_lora_weights=False)
    assert checkpoint.base_fingerprint(ad.model) == fp
    assert checkpoint.base_fingerprint(_make("qwen3.5-0.8b", seed=1).model) != fp
