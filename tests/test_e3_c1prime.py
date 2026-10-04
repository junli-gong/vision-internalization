"""DEV-0035 §1：C1′（冻结 LLM，只训练网格 token 的嵌入行与输出行；scripts/train_e3b.py --supervision grid --variant c1prime）。

- resolve_run：frozen_llm、lora = null、optim 只剩 token 行一组；c1prime 只配网格监督；token 行必须是 fp32 主副本；
- 三家族：训练前后 LLM 的参数逐位不变，只有网格 token 的行变（lm_head 共享时就是嵌入矩阵的这 67 行）；没有 LoRA；优化器里只有 fp32 的
  主副本；检查点只有 token 行（没有 lora/），载入（merge 两种）后受限生成与存之前逐位相同；冻结行被改了，存检查点之前报错；
- 过拟合（DEV-0025 的 8×8 流程，lr 放大）：InternVL（lm_head 不共享）学会，教师强制逐格正确、受限生成解析回 GT；Qwen、Gemma（嵌入与
  lm_head 共享）在小模型上学不全，实测 400 步停在教师强制 95%–97%（lr 0.1 也不过 95%），这里只钉"明显学到了"（≥ 0.9），如实报告；
- 续跑：中途崩溃后接着跑，与一次跑完逐位相同；
- 评测：eval_e3b 的合并核对跳过 merge、restricted 照做（含 across_merge：同一个检查点两次加载）；run_e1 --checkpoint 的合并核对
  一项不剩，只写说明跳过的 JSON。
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.adapters import factory, registry
from src.adapters.checkpoint import policy_spec, read_manifest
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e3 import data as D
from src.e3.config import ConfigError, load_config, resolve_run
from src.e3.merge_check import SKIPPED_FORMAT, run_merge_check
from src.e3.targets import grid_to_ids, parse_grid_ids, seq_len
from src.e3.train import RunSpec, Trainer, leaf_dir, run_training

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()


def _tiny(key):
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


@pytest.fixture(scope="module")
def pools():
    paths = D.E3Paths.fixtures()
    return {t: D.load_pool(t, paths, CFG.task(t)).records for t in CFG.tasks}


def test_resolve_run_c1prime():
    e3b = resolve_run(CFG, supervision="grid", tiny=False)
    rc = resolve_run(CFG, supervision="grid", tiny=False, variant="c1prime")
    assert rc["variant"] == "c1prime" and rc["frozen_llm"] is True and rc["lora"] is None and "lora" not in rc["optim"]
    assert rc["optim"]["token_rows"] == e3b["optim"]["token_rows"]  # token 行的学习率沿用配置的 token_rows
    same = {k: v for k, v in e3b.items() if k not in ("variant", "lora", "optim")}
    assert {k: rc[k] for k in same} == same  # 数据、步数、批、调度同 E3b
    assert set(rc) - set(e3b) == {"frozen_llm"} and "frozen_llm" not in e3b
    with pytest.raises(ConfigError, match="c1prime"):
        resolve_run(CFG, supervision="text", tiny=False, variant="c1prime")
    with pytest.raises(ConfigError):
        resolve_run(CFG, supervision="grid", tiny=False, variant="c3")
    cfg = copy.deepcopy(CFG)
    object.__setattr__(cfg, "doc", copy.deepcopy(dict(CFG.doc)))
    cfg.doc["optim"]["token_rows"]["mode"] = "dense_param"
    with pytest.raises(ConfigError, match="fp32_master_rows"):
        resolve_run(cfg, supervision="grid", tiny=True, variant="c1prime")
    assert "basis" in CFG.doc["c1prime"]


def _setup(key, pools, *, rc=None):
    ad = _tiny(key)
    rc = rc or resolve_run(CFG, supervision="grid", tiny=True, variant="c1prime")
    tr = Trainer(ad, CFG, rc, supervision="grid", seed=0)
    tr.setup_fresh()
    b = D.ExampleBuilder(CFG, pools, seed=0, supervision="grid", stop_id=tr.stop_id, tokens=tr.tokens)
    s = D.Schedule(pools, CFG, seed=0, supervision="grid", batch_size=int(rc["batch_size"]))
    return ad, tr, b, s


def _plans(sched, want):
    out = []
    for task, kind in want:
        for k in range(4000):
            p = sched.plan(k)
            if p.task == task and p.kind == kind:
                out.append(p)
                break
        else:
            raise AssertionError((task, kind))
    return out


@pytest.mark.parametrize("key", FAMILIES)
def test_only_grid_rows_change(key, pools, tmp_path):
    ad, tr, b, sched = _setup(key, pools)
    examples = [b.build(p) for p in _plans(sched, [("refseg", "grid"), ("depth", "grid"), ("refseg", "correction"),
                                                   ("depth", "correction")])]
    assert ad.lora_state == "none" and not [n for n, _ in ad.model.named_parameters() if "lora_" in n]
    emb, head = ad.model.get_input_embeddings().weight, ad.lm_head_weight()
    assert [id(p) for p in tr.rows.params] == ([id(emb)] if emb is head else [id(emb), id(head)])
    assert {n for n, p in ad.model.named_parameters() if p.requires_grad} == {
        n for n, p in ad.model.named_parameters() if any(p is q for q in tr.rows.params)}
    assert [g["name"] for g in tr.opt.param_groups] == ["token_rows"]
    assert all(m.dtype == torch.float32 for m in tr.rows.masters) and tr.rows.mode == "fp32_master_rows"
    assert tr.audit["frozen_llm"] is True and tr.audit["n_lora_params"] == 0
    grid_rows = sorted(tr.tokens.grid_ids)
    assert tr.rows.ids == grid_rows
    snap = {n: p.detach().clone() for n, p in ad.model.named_parameters()}
    ad.train_mode()
    for _ in range(2):
        rec, _, _ = tr.optimize(examples)
        assert math.isfinite(rec["loss"]) and set(rec["lr"]) == {"token_rows"}
    fed = {t for e in examples for t in e.target_ids[:-1]}
    for n, p in ad.model.named_parameters():
        if any(p is q for q in tr.rows.params):
            other = torch.ones(p.shape[0], dtype=torch.bool)
            other[grid_rows] = False
            assert torch.equal(p[other], snap[n][other]), n  # 网格 token 以外的行逐位不变
            changed = {i for i in grid_rows if not torch.equal(p[i], snap[n][i])}
            want = set(grid_rows) if p is head else (fed & set(grid_rows))
            assert changed == want, (n, sorted(set(want) - changed)[:5])
        else:
            assert torch.equal(p, snap[n]), f"{n} 变了（冻结的 LLM）"
    # 检查点：只有 token 行（没有 LoRA）；载入后受限生成与存之前逐位相同
    ad.eval_mode()
    inputs = [tr.build_inputs(e) for e in examples[:2]]
    allowed = tr.tokens.allowed_ids("refseg", "all")
    kw = dict(max_new_tokens=30, stop_ids=[tr.stop_id], enable_thinking=False)
    before = [r.token_ids for r in ad.generate_restricted(inputs, allowed, **kw)]
    tr.check_frozen()
    d = ad.save_trained(tmp_path / "ck")
    assert sorted(x.name for x in d.iterdir()) == ["manifest.json", "token_rows.safetensors", "tokens.json"]
    assert read_manifest(d)["lora"] is None
    for merge in (True, False):
        ad2, _ = factory.load_trained(d, device="cpu", merge=merge)
        assert ad2.lora_state == "none" and ad2.grid_token_ids == ad.grid_token_ids
        inputs2 = [ad2.build_inputs(e.images, e.instruction, e.order) for e in examples[:2]]
        assert [r.token_ids for r in ad2.generate_restricted(inputs2, allowed, **kw)] == before


def test_frozen_probe_catches_changed_rows(pools):
    ad, tr, _, _ = _setup("internvl3.5-2b-hf", pools)
    tr.check_frozen()
    with torch.no_grad():
        ad.lm_head_weight()[0] += 1.0  # 网格 token 以外的行被改了
    with pytest.raises(RuntimeError, match="网格 token 以外"):
        tr.check_frozen()


# ---------------------------------------------------------------------------
# 过拟合（DEV-0025 的 8×8 流程）
# ---------------------------------------------------------------------------

OVERFIT_N = 8


def _overfit(key, *, max_steps: int, lr: float = 3e-2):
    ad = _tiny(key)
    rc = resolve_run(CFG, supervision="grid", tiny=True, variant="c1prime")
    rc["optim"] = copy.deepcopy(rc["optim"])
    rc["optim"]["token_rows"]["lr"] = lr
    rc["optim"]["min_lr_ratio"] = 1.0
    rc["steps"], rc["warmup_steps"], rc["micro_batch_size"] = max_steps, 0, 4
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
        examples.append(D.Example(plan, [img], f"Segment object number {k}.", "instruction_first", ids, [1] * len(ids),
                                  info={"grid": g}))
    allowed = tok.allowed_ids("refseg", "task")
    inputs = [tr.build_inputs(e) for e in examples]
    targets = [e.target_ids for e in examples]

    def tf_acc() -> float:
        ad.eval_mode()
        with torch.no_grad():
            out = ad.training_forward(inputs, targets, enable_thinking=False)
            am = torch.tensor(allowed)[ad.restricted_logits(out.target_hidden, allowed).argmax(-1)]
        ad.train_mode()
        return float((am == out.target_ids).float().mean())

    acc0 = tf_acc()
    ad.train_mode()
    acc = acc0
    for n in range(1, max_steps + 1):
        tr.optimize(examples)
        if n % 25 == 0:
            acc = tf_acc()
            if acc == 1.0:
                break
    ad.eval_mode()
    return ad, tok, examples, inputs, allowed, acc0, acc


def test_overfit_untied_lm_head_learns():
    """InternVL（lm_head 不共享，嵌入行与输出行各 67 行）：实测 400 步教师强制逐格正确。"""
    ad, tok, examples, inputs, allowed, acc0, acc = _overfit("internvl3.5-2b-hf", max_steps=500)
    assert acc == 1.0, f"500 步内教师强制没有逐格正确（{acc0:.3f} → {acc:.3f}）"
    gen = ad.generate_restricted(inputs, allowed, max_new_tokens=seq_len(OVERFIT_N) + 5, stop_ids=[tok.stop_id],
                                 enable_thinking=False)
    for e, r in zip(examples, gen):
        assert list(r.token_ids) == e.target_ids and r.stopped
        p = parse_grid_ids(r.token_ids, tok.value_ids("refseg"), tok.row_id, tok.stop_id, n=OVERFIT_N)
        assert p.failure is None and np.array_equal(p.grid, e.info["grid"])


def test_overfit_tied_learns_partially():
    """Qwen（嵌入与 lm_head 共享）：冻结的小模型只靠 67 行学不全 8×8（实测 400 步停在 95%，lr 0.1 也一样；Gemma 96%–97%），
    这正是 C1′ 要量的"冻结的 LLM 本来能表达多少"。这里只钉"明显学到了"：150 步后教师强制 ≥ 0.9，且比开始时高。"""
    _, _, _, _, _, acc0, acc = _overfit("qwen3.5-0.8b", max_steps=150)
    assert acc >= 0.9 and acc > acc0 + 0.3, (acc0, acc)


# ---------------------------------------------------------------------------
# 续跑
# ---------------------------------------------------------------------------


def _spec(key, ad):
    e = registry.get_model(key)
    return RunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                   image_policy=policy_spec(ad.image_policy), model_dtype="fp32", supervision="grid", seed=0, frozen_llm=True)


def _tensors(d: Path) -> dict:
    from safetensors.torch import load_file

    return {f"{f.relative_to(d)}:{k}": v for f in sorted(d.rglob("*.safetensors")) for k, v in load_file(str(f)).items()}


def test_resume_after_crash_is_bitwise_identical(tmp_path):
    key = "qwen3.5-0.8b"
    rc = resolve_run(CFG, supervision="grid", tiny=True, variant="c1prime")
    paths = D.E3Paths.fixtures()
    ad = _tiny(key)
    sa = run_training(ad, CFG, _spec(key, ad), rc, paths, tmp_path / "A", log=lambda m: None)
    assert sa["complete"] and sa["variant"] == "c1prime" and sa["frozen_llm"] is True and sa["lora"] is None
    assert sa["trainable_token_rows"] == sorted(int(v) for v in ad.grid_token_ids.values())
    ad = _tiny(key)
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        run_training(ad, CFG, _spec(key, ad), rc, paths, tmp_path / "B", log=lambda m: None, _fail_after_step=3)
    ad = _tiny(key)
    sb = run_training(ad, CFG, _spec(key, ad), rc, paths, tmp_path / "B", resume=True, log=lambda m: None)
    la, lb = leaf_dir(tmp_path / "A", "c1prime", key, 0), leaf_dir(tmp_path / "B", "c1prime", key, 0)
    for f in ("train_log.jsonl", "config.yaml", "README.md", "train_summary.json"):
        assert (la / f).read_bytes() == (lb / f).read_bytes(), f
    ta, tb = _tensors(la / "ckpt" / "final" / "model"), _tensors(lb / "ckpt" / "final" / "model")
    assert set(ta) == set(tb) == {"token_rows.safetensors:embed"} and all(torch.equal(ta[k], tb[k]) for k in ta)
    assert sa == sb
    assert "C1′" in (la / "README.md").read_text() and "不注入 LoRA" in (la / "README.md").read_text()


# ---------------------------------------------------------------------------
# 命令行与评测
# ---------------------------------------------------------------------------


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True, cwd=REPO)


@pytest.fixture(scope="module")
def c1_ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("c1prime_train")
    p = run_script("train_e3b", "--tiny", "--model", "qwen3.5-0.8b", "internvl3.5-2b-hf", "--supervision", "grid", "--variant",
                   "c1prime", "--seed", "0", "--out", str(d))
    assert p.returncode == 0, p.stderr[-3000:]
    return d


def test_cli_outputs(c1_ckpt):
    for key in ("qwen3.5-0.8b", "internvl3.5-2b-hf"):
        leaf = c1_ckpt / "c1prime" / key / "seed0"
        files = sorted(x.name for x in (leaf / "ckpt" / "final" / "model").iterdir())
        assert files == ["manifest.json", "token_rows.safetensors", "tokens.json"]
        import yaml

        ident = yaml.safe_load((leaf / "config.yaml").read_text())
        assert ident["variant"] == "c1prime" and ident["training"]["frozen_llm"] is True and ident["training"]["lora"] is None
        assert ident["noise"] is not None  # 修正对照旧（与 E3b 相同的数据）
        run = json.loads(sorted((leaf / "logs").glob("run_*.json"))[0].read_text())
        assert run["audit"]["frozen_llm"] is True and run["audit"]["n_lora_params"] == 0


def test_eval_merge_check_skips_merge_and_restricted_runs(c1_ckpt, tmp_path):
    ck = c1_ckpt / "c1prime" / "qwen3.5-0.8b" / "seed0" / "ckpt" / "final" / "model"
    p = run_script("eval_e3b", "--tiny", "--checkpoint", str(ck), "--model", "qwen3.5-0.8b", "--variant", "c1prime", "--seed", "0",
                   "--task", "depth", "--limit", "1", "--out", str(tmp_path / "e"))
    assert p.returncode == 0, p.stderr[-3000:]
    mdir = tmp_path / "e" / "eval" / "c1prime" / "qwen3.5-0.8b" / "seed0"
    mc = json.loads((mdir / "merge_check.json").read_text())
    assert mc["pass"] is True and list(mc["checks"]) == ["restricted"] and mc["skipped_checks"] == ["merge"]
    r = mc["checks"]["restricted"]
    assert r["by_thinking"]["thinking_off"]["teacher_forcing"]["agreement"] == 1.0
    assert r["across_merge"]["by_thinking"]["thinking_off"]["teacher_forcing"]["argmax_agreement"] == 1.0  # 两次加载，逐个相同
    m = json.loads((mdir / "depth" / "metrics.json").read_text())
    assert m["complete"] is True and m["n"] == 1
    # T 文本评测：只有 merge 一项，跳过之后一项不剩，只写说明
    p = run_script("run_e1", "--tiny", "--model", "qwen3.5-0.8b", "--task", "grid16", "--limit", "1", "--checkpoint", str(ck),
                   "--checkpoint-label", "t_text/c1prime/seed0", "--out", str(tmp_path / "t"))
    assert p.returncode == 0, p.stderr[-3000:]
    base = tmp_path / "t" / "t_text" / "c1prime" / "seed0" / "qwen3.5-0.8b"
    doc = json.loads((base / "merge_check.json").read_text())
    assert doc["format"] == SKIPPED_FORMAT and doc["pass"] is True and doc["skipped_checks"] == ["merge"] and doc["checks"] == {}
    import yaml

    cfg = yaml.safe_load((base / "grid16" / "config.yaml").read_text())
    assert cfg["checkpoint"]["lora"] is None and cfg["checkpoint"]["merge_check"]["checks"] == []


def test_run_merge_check_skip_rules(c1_ckpt, tmp_path):
    ck = c1_ckpt / "c1prime" / "internvl3.5-2b-hf" / "seed0" / "ckpt" / "final" / "model"
    out = tmp_path / "mc.json"
    s = run_merge_check(ck, model_key="internvl3.5-2b-hf", tiny=True, n=2, checks=["merge"], out_json=out, device="cpu",
                        dtype="fp32", log=lambda m: None)
    assert s["pass"] is True and s["checks"] == [] and s["skipped_checks"] == ["merge"] and s["n_used"] == 0
    s = run_merge_check(ck, model_key="internvl3.5-2b-hf", tiny=True, n=2, checks=["merge", "restricted"], out_json=out,
                        device="cpu", dtype="fp32", log=lambda m: None)
    assert s["pass"] is True and s["checks"] == ["restricted"] and s["skipped_checks"] == ["merge"] and not s["reused"]
    doc = json.loads(out.read_text())
    assert doc["skipped_checks"] == ["merge"] and "merge" not in doc["checks"]
    s = run_merge_check(ck, model_key="internvl3.5-2b-hf", tiny=True, n=2, checks=["merge", "restricted"], out_json=out,
                        device="cpu", dtype="fp32", log=lambda m: None)
    assert s["reused"] is True and s["skipped_checks"] == ["merge"]
