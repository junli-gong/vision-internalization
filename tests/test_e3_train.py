"""DEV-0025 §3、§4：E3b 的训练循环（src/e3/train.py）。CPU fp32 小模型，三个家族。

- 三家族 × {网格分割、深度、两种修正对、C3 的框 / 多边形 / 点对} 各跑几步：损失有限；梯度只落在 LoRA 与网格 token 行上（嵌入 / lm_head
  的其余行梯度为 0）；冻结的参数逐位不变（视觉塔、LLM 的 base 权重、嵌入的其余行），save_trained 的冻结行核对放行；
- 过拟合：4 条样本训练到教师强制逐格正确，generate_restricted 生成的序列与目标逐个相同，parse 解析回 GT 网格（训练与推理的链路通）；
- 保存、加载（factory.load_trained）、合并之后，受限生成与合并前逐位相同（CPU fp32）；
- token 行的两种模式（fp32 主副本 / 整个矩阵交给优化器）在 fp32 上逐位相同；
- 续跑：中途崩溃后从检查点接着跑，train_log.jsonl 与最终检查点的张量与一次跑完逐位相同（网格与 C3）。
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.adapters import factory, registry
from src.adapters.checkpoint import CheckpointError, read_manifest
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e3 import data as D
from src.e3.config import load_config, resolve_run
from src.e3.targets import grid_to_ids, parse_grid_ids, seq_len
from src.e3.train import RunSpec, Trainer, leaf_dir, lr_factor, run_training

FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()


def _tiny(key, **kw):
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]), **kw)


@pytest.fixture(scope="module")
def pools():
    paths = D.E3Paths.fixtures()
    return {t: D.load_pool(t, paths, CFG.task(t)).records for t in CFG.tasks}


def _plans(sched: D.Schedule, want: list[tuple[str, str]]) -> list[D.SlotPlan]:
    out = []
    for task, kind in want:
        for k in range(4000):
            p = sched.plan(k)
            if p.task == task and p.kind == kind and (p.text_kind == kind if task == "refseg" and kind in ("box", "polygon")
                                                      else True):
                out.append(p)
                break
        else:
            raise AssertionError((task, kind))
    return out


def _c3_plans(sched: D.Schedule) -> list[D.SlotPlan]:
    want = {"box": None, "polygon": None, "pairs": None}
    for k in range(4000):
        p = sched.plan(k)
        if want.get(p.text_kind, 1) is None:
            want[p.text_kind] = p
        if all(want.values()):
            break
    return list(want.values())


def _setup(key, supervision, pools, *, rc=None, seed=0):
    ad = _tiny(key)
    rc = rc or resolve_run(CFG, supervision=supervision, tiny=True)
    tr = Trainer(ad, CFG, rc, supervision=supervision, seed=seed)
    tr.setup_fresh()
    b = D.ExampleBuilder(CFG, pools, seed=seed, supervision=supervision, stop_id=tr.stop_id, tokens=tr.tokens,
                         encode=lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"]),
                         family=registry.get_model(key).family)
    s = D.Schedule(pools, CFG, seed=seed, supervision=supervision, batch_size=int(rc["batch_size"]))
    return ad, tr, b, s


# ---------------------------------------------------------------------------
# 三家族 × 全部变体：损失有限、梯度只落在 LoRA 与网格行上、冻结的参数逐位不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILIES)
@pytest.mark.parametrize("supervision", ["grid", "text"])
def test_few_steps_grads_only_on_lora_and_grid_rows(key, supervision, pools, tmp_path):
    ad, tr, b, sched = _setup(key, supervision, pools)
    if supervision == "grid":
        plans = _plans(sched, [("refseg", "grid"), ("depth", "grid"), ("refseg", "correction"), ("depth", "correction")])
    else:
        plans = _c3_plans(sched)
    examples = [b.build(p) for p in plans]
    assert {(e.plan.task, e.plan.kind) for e in examples} == {(p.task, p.kind) for p in plans}
    lora = {n for n, _ in ad.model.named_parameters() if "lora_" in n}
    row_params = {id(p) for p in (tr.rows.params if tr.rows else [])}
    grid_rows = sorted(tr.tokens.grid_ids) if tr.tokens else []

    # 一次手动的前向 + 反向：梯度只在 LoRA 与网格 token 行上
    ad.train_mode()
    out = ad.training_forward([tr.build_inputs(e) for e in examples], [e.target_ids for e in examples], enable_thinking=False)
    loss, per, _ = tr._loss(out, examples)
    assert torch.isfinite(loss) and set(per) == {"refseg", "depth"}
    loss.backward()
    for n, p in ad.model.named_parameters():
        if p.grad is None or not torch.any(p.grad != 0):
            continue
        if id(p) in row_params:
            other = torch.ones(p.shape[0], dtype=torch.bool)
            other[grid_rows] = False
            assert torch.count_nonzero(p.grad[other]) == 0, n
            assert torch.count_nonzero(p.grad[grid_rows]) > 0, n
        else:
            assert n in lora, f"{n} 不是 LoRA 也不是网格 token 行，却有梯度"
    assert any(ad.model.get_parameter(n).grad is not None for n in lora)
    tr.opt.zero_grad(set_to_none=True)
    tr.rows and tr.rows.zero_dense_grads()

    # 两步正式的优化：冻结的参数逐位不变，LoRA 与网格行变了
    snap = {n: p.detach().clone() for n, p in ad.model.named_parameters()}
    for _ in range(2):
        rec, times, comps = tr.optimize(examples)
        assert math.isfinite(rec["loss"]) and rec["n_samples"] == len(examples) and len(comps) == len(examples)
        assert rec["n_supervised_tokens"] == sum(e.n_supervised for e in examples)
    changed_lora = 0
    head = ad.model.get_output_embeddings().weight
    fed = {t for e in examples for t in e.target_ids[:-1]}  # 教师强制时作为输入喂进去的网格 token
    for n, p in ad.model.named_parameters():
        if n in lora:
            changed_lora += int(not torch.equal(p, snap[n]))
        elif id(p) in row_params:
            other = torch.ones(p.shape[0], dtype=torch.bool)
            other[grid_rows] = False
            assert torch.equal(p[other], snap[n][other]), n  # 冻结行逐位不变
            changed = {i for i in tr.rows.ids if not torch.equal(p[i], snap[n][i])}
            # lm_head（共享时就是嵌入）：受限交叉熵经过允许集合里的每一行，67 行都变；不共享时输入嵌入只有喂进去的那些行有梯度
            want = set(tr.rows.ids) if p is head else (fed & set(tr.rows.ids))
            assert changed == want, (n, sorted(set(tr.rows.ids) - changed)[:5])
        else:
            assert torch.equal(p, snap[n]), f"{n} 变了"
    assert changed_lora > 0
    ad.eval_mode()
    d = ad.save_trained(tmp_path / "ck")  # 冻结行的抽样核对放行
    man = read_manifest(d)
    assert man["lora"]["r"] == CFG.doc["lora"]["r"]
    assert man["trainable_token_rows"] == (sorted(tr.rows.ids) if tr.rows else None)


def test_save_trained_catches_changed_frozen_rows(pools, tmp_path):
    ad, tr, b, sched = _setup("qwen3.5-0.8b", "grid", pools)
    with torch.no_grad():
        ad.model.get_input_embeddings().weight[0] += 1.0  # 冻结行被改了（例如误加 weight decay）
    with pytest.raises(CheckpointError, match="weight_decay"):
        ad.save_trained(tmp_path / "ck")


def test_lr_schedule():
    f = [lr_factor(s, warmup=3, total=10, min_ratio=0.0) for s in range(10)]
    assert f[:3] == [1 / 3, 2 / 3, 1.0] and f[3] == 1.0 and all(a >= b for a, b in zip(f[3:], f[4:]))
    assert lr_factor(10, warmup=3, total=10, min_ratio=0.1) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 过拟合：训练与推理的链路
# ---------------------------------------------------------------------------

OVERFIT_N = 8  # 小网格：CPU 小模型在 32×32 上逐格过拟合要几百步（实测 900 步、5 分钟仍只有 99.5%），见完成总结


def _overfit(key, *, images: bool, max_steps: int):
    ad = _tiny(key)
    rc = resolve_run(CFG, supervision="grid", tiny=True)
    rc["optim"] = copy.deepcopy(rc["optim"])
    rc["optim"]["lora"]["lr"] = 3e-3
    rc["optim"]["token_rows"]["lr"] = 3e-2
    rc["lora"] = {**rc["lora"], "dropout": 0.0}
    rc["steps"], rc["warmup_steps"], rc["micro_batch_size"] = max_steps, 0, 4
    rc["optim"]["min_lr_ratio"] = 1.0  # 常数 lr
    tr = Trainer(ad, CFG, rc, supervision="grid", seed=0)
    tr.setup_fresh()
    tok = tr.tokens
    rng = np.random.default_rng(1)
    examples = []
    for k in range(4):
        g = np.zeros((OVERFIT_N, OVERFIT_N), dtype=np.int64)
        r0, c0 = rng.integers(0, OVERFIT_N // 2, 2)
        g[r0:r0 + rng.integers(2, OVERFIT_N // 2), c0:c0 + rng.integers(2, OVERFIT_N // 2)] = 1
        img = np.full((96, 128, 3), 40, dtype=np.uint8)
        img[r0 * 12:(r0 + 2) * 12, c0 * 16:(c0 + 2) * 16] = (200, 50, 50)
        ids = grid_to_ids(g, tok.value_ids("refseg"), tok.row_id, tok.stop_id, n=OVERFIT_N)
        plan = D.SlotPlan(k=k, step=0, task="refseg", j=k, index=k, epoch=0, sample_id=f"overfit/{k}", kind="grid")
        examples.append(D.Example(plan, [img] if images else [], f"Segment object number {k}.", "instruction_first", ids,
                                  [1] * len(ids), info={"grid": g}))
    allowed = tok.allowed_ids("refseg", "task")
    inputs = [tr.build_inputs(e) for e in examples]
    targets = [e.target_ids for e in examples]

    def tf_correct() -> bool:
        ad.eval_mode()
        with torch.no_grad():
            out = ad.training_forward(inputs, targets, enable_thinking=False)
            am = torch.tensor(allowed)[ad.restricted_logits(out.target_hidden, allowed).argmax(-1)]
        ad.train_mode()
        return bool(torch.equal(am, out.target_ids))

    ad.train_mode()
    ok, n = False, 0
    for n in range(1, max_steps + 1):
        tr.optimize(examples)
        if n % 10 == 0 and tf_correct():
            ok = True
            break
    ad.eval_mode()
    assert ok, f"{key}：{max_steps} 步内教师强制没有逐格正确"
    gen = ad.generate_restricted(inputs, allowed, max_new_tokens=seq_len(OVERFIT_N) + 5, stop_ids=[tok.stop_id],
                                 enable_thinking=False)
    for e, r in zip(examples, gen):
        assert list(r.token_ids) == e.target_ids and r.stopped
        p = parse_grid_ids(r.token_ids, tok.value_ids("refseg"), tok.row_id, tok.stop_id, n=OVERFIT_N)
        assert p.failure is None and np.array_equal(p.grid, e.info["grid"])
    return n


@pytest.mark.parametrize("key", FAMILIES)
def test_overfit_generate_and_parse(key):
    _overfit(key, images=True, max_steps=500)  # 实测 Qwen 130 步、Gemma 130 步、InternVL 70 步


# ---------------------------------------------------------------------------
# 保存 → 加载 → 合并：受限生成与合并前逐位相同
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILIES)
def test_save_load_merge_same_generation(key, pools, tmp_path):
    ad, tr, b, sched = _setup(key, "grid", pools)
    plans = _plans(sched, [("refseg", "grid"), ("depth", "correction")])
    examples = [b.build(p) for p in plans]
    ad.train_mode()
    for _ in range(3):
        tr.optimize(examples)
    ad.eval_mode()
    inputs = [tr.build_inputs(e) for e in examples]
    allowed = tr.tokens.allowed_ids("refseg", "all")
    kw = dict(max_new_tokens=40, stop_ids=[tr.stop_id], enable_thinking=False)
    before = [r.token_ids for r in ad.generate_restricted(inputs, allowed, **kw)]
    d = ad.save_trained(tmp_path / "ck")
    for merge in (True, False):
        ad2, ck = factory.load_trained(d, device="cpu", merge=merge)
        assert ad2.lora_state == ("merged" if merge else "injected")
        assert ad2.grid_token_ids == ad.grid_token_ids
        inputs2 = [ad2.build_inputs(e.images, e.instruction, e.order) for e in examples]
        after = [r.token_ids for r in ad2.generate_restricted(inputs2, allowed, **kw)]
        assert after == before, f"{key} merge={merge}"


# ---------------------------------------------------------------------------
# token 行的两种模式在 fp32 上逐位相同
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["qwen3.5-0.8b", "internvl3.5-2b-hf"])  # 共享 / 不共享 lm_head 各一
def test_master_rows_equal_dense_param_fp32(key, pools):
    """两种 token 行模式在 fp32 上逐位相同（本意不放松）。

    DEV-0029：这条测试在高负载下偶发失败（D60-1：两套全量同时跑时失败，单独重跑 5/5 通过）。两种模式依次跑、互相比较，逐位
    相同要求每次前向 / 反向的浮点求和顺序都相同；CPU 上的多线程算子按线程切块归约，负载变化时切块与合并的顺序可能变，fp32 末位
    就会不同——这与两种模式本身无关。所以这条测试里把 torch 的 intra-op 线程数固定为 1（测完恢复）：每个算子的求和顺序固定，
    比较的只剩两种模式本身。不改成带容差的比较：在固定的求和顺序下两种模式确实逐位相同（主副本就是 fp32 的那几行，梯度是同一份，
    AdamW 逐元素更新），容差只会放过真正的差异。"""
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        res = []
        for mode in ("fp32_master_rows", "dense_param"):
            rc = resolve_run(CFG, supervision="grid", tiny=True)
            rc["optim"] = copy.deepcopy(rc["optim"])
            rc["optim"]["token_rows"]["mode"] = mode
            rc["optim"]["max_grad_norm"] = None  # 只记范数不裁：两种模式的范数只差求和顺序
            ad, tr, b, sched = _setup(key, "grid", pools, rc=rc)
            examples = [b.build(p) for p in sched.step_plans(0)]
            ad.train_mode()
            for _ in range(3):
                tr.optimize(examples)
            res.append({n: p.detach().clone() for n, p in ad.model.named_parameters()})
    finally:
        torch.set_num_threads(n_threads)
    assert torch.get_num_threads() == n_threads
    a, b_ = res
    assert set(a) == set(b_)
    for n in a:
        assert torch.equal(a[n], b_[n]), n


@pytest.mark.parametrize("key", ["qwen3.5-0.8b", "internvl3.5-2b-hf"])
def test_bf16_base_keeps_fp32_lora_and_master_rows(key, pools):
    """bf16 的 base（CPU 上也能跑）：peft 把 LoRA 升到 fp32（入口断言）；token 行的主副本是 fp32，写回后权重的这几行 = 主副本取整到
    bf16，其余行逐位不变。"""
    fam = registry.get_model(key).family
    ad = make_adapter(key, device="cpu", dtype=torch.bfloat16, image_policy=parse_policy(CFG.doc["tiny"]["image_policies"][fam]),
                      tiny=True, init_std=float(CFG.doc["tiny"]["init_std"]))
    rc = resolve_run(CFG, supervision="grid", tiny=True)
    tr = Trainer(ad, CFG, rc, supervision="grid", seed=0)
    tr.setup_fresh()
    assert tr.audit["lora_dtypes"] == ["torch.float32"]
    assert all(m.dtype == torch.float32 for m in tr.rows.masters) and all(p.dtype == torch.bfloat16 for p in tr.rows.params)
    b = D.ExampleBuilder(CFG, pools, seed=0, supervision="grid", stop_id=tr.stop_id, tokens=tr.tokens)
    s = D.Schedule(pools, CFG, seed=0, supervision="grid", batch_size=int(rc["batch_size"]))
    snap = [p.detach().clone() for p in tr.rows.params]
    ad.train_mode()
    rec, _, _ = tr.optimize([b.build(p) for p in s.step_plans(0)])
    assert math.isfinite(rec["loss"])
    idx = torch.tensor(tr.rows.ids)
    for p, m, before in zip(tr.rows.params, tr.rows.masters, snap):
        assert torch.equal(p.detach().index_select(0, idx), m.detach().to(torch.bfloat16))
        other = torch.ones(p.shape[0], dtype=torch.bool)
        other[idx] = False
        assert torch.equal(p.detach()[other], before[other])


# ---------------------------------------------------------------------------
# 续跑：中途崩溃后接着跑，与一次跑完逐位相同
# ---------------------------------------------------------------------------


def _spec(key, supervision, ad):
    e = registry.get_model(key)
    from src.adapters.checkpoint import policy_spec

    return RunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                   image_policy=policy_spec(ad.image_policy), model_dtype="fp32", supervision=supervision, seed=0)


def _tensors(d: Path) -> dict:
    from safetensors.torch import load_file

    out = {}
    for f in sorted(d.rglob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            out[f"{f.relative_to(d)}:{k}"] = v
    return out


@pytest.mark.parametrize("key,supervision", [("qwen3.5-0.8b", "grid"), ("qwen3.5-0.8b", "text"), ("gemma-4-e2b-it", "grid")])
def test_resume_after_crash_is_bitwise_identical(key, supervision, tmp_path):
    rc = resolve_run(CFG, supervision=supervision, tiny=True)
    paths = D.E3Paths.fixtures()
    a, b = tmp_path / "A", tmp_path / "B"
    ad = _tiny(key)
    sa = run_training(ad, CFG, _spec(key, supervision, ad), rc, paths, a, log=lambda m: None)
    assert sa["complete"] and sa["steps"] == rc["steps"]
    ad = _tiny(key)
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        run_training(ad, CFG, _spec(key, supervision, ad), rc, paths, b, log=lambda m: None, _fail_after_step=3)
    lb = leaf_dir(b, rc["variant"], key, 0)
    assert (lb / "ckpt" / "latest" / "resume_state.pt").exists()
    ad = _tiny(key)
    with pytest.raises(FileExistsError):
        run_training(ad, CFG, _spec(key, supervision, ad), rc, paths, b, log=lambda m: None)
    ad = _tiny(key)
    sb = run_training(ad, CFG, _spec(key, supervision, ad), rc, paths, b, resume=True, log=lambda m: None)
    la = leaf_dir(a, rc["variant"], key, 0)
    for f in ("train_log.jsonl", "config.yaml", "README.md", "train_summary.json"):
        assert (la / f).read_bytes() == (lb / f).read_bytes(), f
    ta, tb = _tensors(la / "ckpt" / "final" / "model"), _tensors(lb / "ckpt" / "final" / "model")
    assert set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)
    assert not (la / "ckpt" / "latest").exists() and not (lb / "ckpt" / "latest").exists()  # 完成后删掉续跑状态
    assert sa == sb
    log = [json.loads(l) for l in (la / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == list(range(rc["steps"])) and all(math.isfinite(r["loss"]) for r in log)
    acc = json.loads((lb / "accounting.json").read_text())
    assert acc["n_runs"] == 2 and acc["n_steps_logged"] == rc["steps"] and acc["n_steps_timed"] == rc["steps"] + 1
    comp = json.loads((la / "compute.json").read_text())
    assert comp["n_samples"] == rc["steps"] * rc["batch_size"] and comp["generated_tokens"] == sum(r["n_target_tokens"] for r in log)
    # 已完成的叶目录再跑一次：直接返回
    ad = _tiny(key)
    assert run_training(ad, CFG, _spec(key, supervision, ad), rc, paths, b, resume=True, log=lambda m: None) == sb


def test_data_workers_do_not_change_results(tmp_path):
    """下一步的样本由线程池构造：线程数（0 = 主线程串行）不改变任何结果。"""
    paths = D.E3Paths.fixtures()
    logs, tens = [], []
    for w in (0, 3):
        rc = resolve_run(CFG, supervision="grid", tiny=True)
        rc["data_workers"], rc["steps"] = w, 3
        ad = _tiny("qwen3.5-0.8b")
        run_training(ad, CFG, _spec("qwen3.5-0.8b", "grid", ad), rc, paths, tmp_path / f"w{w}", log=lambda m: None)
        leaf = leaf_dir(tmp_path / f"w{w}", "e3b", "qwen3.5-0.8b", 0)
        logs.append((leaf / "train_log.jsonl").read_bytes())
        tens.append(_tensors(leaf / "ckpt" / "final" / "model"))
    assert logs[0] == logs[1]
    assert set(tens[0]) == set(tens[1]) and all(torch.equal(tens[0][k], tens[1][k]) for k in tens[0])
