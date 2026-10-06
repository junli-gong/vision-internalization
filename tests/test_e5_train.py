"""DEV-0037 §3–§5：E5 主比较的训练与训练后评测（CPU fp32 小模型，--tiny）。

- 四种起点（base_warmup、e3b、e3a、base）在三个家族上都能训练、评测跑通（起点用 scripts/train_e3b.py --tiny 的网格监督与网格格式预热、
  scripts/train_e3a.py --tiny 的小检查点）；评测前的合并核对通过；E5 检查点只存新 LoRA、网格 token 行与起点的引用；
- 起点 LoRA 合并后的一致性：叠加出来的模型与 factory.load_trained(起点, merge=True) 逐位相同（权重指纹也相同）；同一个种子下各起点的
  新 LoRA 初始值逐位相同；
- 坏起点（病态的 LoRA：合并前后分家）：起点的合并核对不通过，拒绝训练（不建叶目录）；坏的 E5 LoRA：评测前的合并核对不通过，拒绝评测；
- 过拟合：每类 1 条样本（8×8 网格），训练后在训练样本上受限解码，网格 IoU 满分、起伏误差为 0；
- 续跑：训练中断后 --resume，train_log 与最终检查点逐字节相同；评测中断在一批中间后 --resume，确定的输出逐字节相同。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.adapters import factory
from src.adapters.checkpoint import read_manifest
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e3.data import SlotPlan
from src.e3.tokens import add_vocab
from src.e5 import stack as STK
from src.e5 import trained_eval as TE
from src.e5.protocol import E5_TASKS, headline_scope_text, num_bins
from src.e5.token_grid import max_new_tokens, sample_images, task_format
from src.e5.train import E5Trainer
from src.e5.train_config import load_train_config, resolve_run
from src.e5.train_data import ExampleBuilder, TrainSet
from src.synth import generate

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_train_config()
STARTS = ("base_warmup", "e3b", "e3a", "base")
_SCRIPTS: dict[str, object] = {}


def _script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_e5", REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.path[:] = saved
        _SCRIPTS[name] = mod
    return _SCRIPTS[name]


def _tiny(key: str):
    from src.adapters import registry

    t = CFG.e3b.doc["tiny"]
    fam = registry.get_model(key).family
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _from_manifest(ck: Path):
    spec = read_manifest(ck)["build_spec"]
    return make_adapter(spec["key"], device="cpu", dtype=torch.float32, image_policy=parse_policy(spec["image_policy"]),
                        tiny=True, **dict(spec["family_kw"]))


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module", params=FAMILIES)
def starts(request, tmp_path_factory):
    """三种起点检查点（DEV-0025 的网格监督、DEV-0035 的预热、DEV-0034 的 E3a），各跑配置的 --tiny 规模。"""
    key = request.param
    root = tmp_path_factory.mktemp(f"e5starts_{key}")
    e3, e3a = _script("train_e3b"), _script("train_e3a")
    assert e3.main(["--tiny", "--model", key, "--supervision", "grid", "--seed", "0", "--out", str(root / "e3")]) == 0
    assert e3.main(["--tiny", "--model", key, "--supervision", "warmup", "--seed", "0", "--out", str(root / "e3")]) == 0
    assert e3a.main(["--tiny", "--model", key, "--variant", "full", "--seed", "0", "--out", str(root / "e3a")]) == 0
    fin = Path("seed0") / "ckpt" / "final" / "model"
    return {"key": key, "root": root, "base": None, "e3b": root / "e3" / "e3b" / key / fin,
            "base_warmup": root / "e3" / "warmup" / key / fin, "e3a": root / "e3a" / "e3a" / "full" / key / fin}


# ---------------------------------------------------------------------------
# 四种起点 × 三个家族：训练、评测跑通
# ---------------------------------------------------------------------------


def test_four_starts_train_and_eval(starts, tmp_path):
    key = starts["key"]
    tr, ev = _script("train_e5"), _script("eval_e5_trained")
    out = tmp_path / "o"
    for init in STARTS:
        ck = starts[init]
        args = ["--tiny", "--init", init, "--model", key, "--n-train", "20", "--seed", "0", "--out", str(out)]
        if ck is not None:
            args += ["--start-checkpoint", str(ck)]
        assert tr.main(args) == 0, init
        leaf = out / "train" / init / key / "n20" / "seed0"
        final = leaf / "ckpt" / "final"
        summary = json.loads((leaf / "train_summary.json").read_text())
        assert summary["complete"] and summary["steps"] == 5 and summary["epochs"] == 1  # 20 条 × 1 epoch ÷ 批 4（不设下限的档）
        assert summary["selection"] is None and summary["selected_step"] == 5  # 不做 val 选择，取最后一步（DEV-0038）
        pair = yaml.safe_load((leaf / "config.yaml").read_text())["seed_pairing"]
        assert pair["status"] == ("not_applicable" if ck is None else "match") and pair["start_seed"] == (None if ck is None else 0)
        assert sorted(p.name for p in final.iterdir()) == ["e5_stack.json", "model"]
        files = set(json.loads((final / "model" / "manifest.json").read_text())["files"])
        assert {"lora/adapter_model.safetensors", "token_rows.safetensors", "tokens.json"} <= files  # 不存合并后的整模型（D34）
        assert not any("model.safetensors" == f or f.endswith("pytorch_model.bin") for f in files)
        stack = json.loads((final / "e5_stack.json").read_text())
        assert stack["start"]["kind"] == init and stack["tag"] == "n20" and stack["seed"] == 0
        if ck is None:
            assert STK.read_marker(final / "model") is None and stack["start"]["manifest_sha256"] is None
        else:
            sha = _sha(Path(ck) / "manifest.json")
            assert STK.read_marker(final / "model") == sha == stack["start"]["manifest_sha256"]
            checks = out / "train" / init / key / "start_checks"
            mc = json.loads((checks / f"merge_check_{sha[:16]}.json").read_text())
            assert mc["pass"] is True
            assert sorted(mc["checks"]) == (["merge", "restricted"] if init != "e3a" else ["merge"])
            ref = json.loads((checks / f"reference_{sha[:16]}.json").read_text())
            assert ref["fingerprint"] == stack["fingerprint"]["value"]  # 叠加后的权重 = 检查点合并评测时的权重
            assert stack["start"]["leaf_variant"] == CFG.doc["starts"][init]["variant"]
        tokens = json.loads((final / "model" / "tokens.json").read_text())
        assert tokens["n_grid_tokens"] == 67 and tokens["query"]["n"] is None  # e3a 起点的查询 token 不带进 E5
        tasks = ("rotation3d", "imagery") if init == "e3b" else ("rotation3d",)  # 二值路径另由合并核对与续跑测试覆盖（省时间）
        assert ev.main(["--tiny", "--checkpoint", str(final), "--model", key, "--out", str(out), "--limit", "1",
                        "--task", *tasks]) == 0, init
        mdir = out / "trained" / init / key / "n20" / "seed0"
        mc = json.loads((mdir / "merge_check.json").read_text())
        assert mc["pass"] is True and mc["data"]["n"] == 5 and mc["data"]["n_runs"] == 2  # 每类 1 条，二值、起伏各一批
        for task in tasks:
            m = json.loads((mdir / task / "metrics.json").read_text())
            assert m["complete"] and m["n"] == 1 and m["system"] == "trained" and m["start"] == init
            assert "gate" in m and m["headline_scope"] == headline_scope_text(task)  # 与零样本同一个标题口径（protocol）
            rec = json.loads((mdir / task / "predictions.jsonl").read_text().splitlines()[0])
            assert rec["n_generated_tokens"] <= max_new_tokens(32) and "score" in rec
        cfgy = (mdir / "rotation3d" / "config.yaml").read_text()
        assert "generate_restricted" in cfgy and "1057" in cfgy


def _same_params(a, b, skip_rows) -> None:
    pa, pb = dict(a.model.named_parameters()), dict(b.model.named_parameters())
    assert set(pa) == set(pb)
    for k in pa:
        x, y = pa[k].detach(), pb[k].detach()
        if x.shape == y.shape and torch.equal(x, y):
            continue
        assert x.dim() == 2 and y.dim() == 2 and x.shape[1] == y.shape[1], k  # 只有嵌入类的表允许差在新增行上
        n = min(x.shape[0], y.shape[0])
        keep = torch.ones(n, dtype=torch.bool)
        keep[[r for r in skip_rows if r < n]] = False
        assert torch.equal(x[:n][keep], y[:n][keep]), k


def test_stacking_equals_merged_checkpoint(starts):
    """叠加（合并起点 LoRA + 网格 token）出来的模型与 factory.load_trained(起点, merge=True) 逐位相同：带网格 token 的起点连同网格行
    全部相同；e3a 只差新增的行（网格 token 行与查询占位符行）。权重指纹三者一致。"""
    for init in ("base_warmup", "e3b", "e3a"):
        ck = starts[init]
        start = STK.resolve_start(CFG, init, ck, model_key=starts["key"], tiny=True)
        ad = _from_manifest(ck)
        _, info = STK.setup_start(ad, start, CFG)
        ref, _ = factory.load_trained(ck, device="cpu", dtype=torch.float32, merge=True)
        assert ad.lora_state == "none" and ref.lora_state == "merged"
        skip = set(ad.token_table()["row_ids"]) | set(ref.token_table()["row_ids"])
        if init != "e3a":
            assert ad.token_table()["row_tables"] == ref.token_table()["row_tables"]
            skip = set()
        _same_params(ad, ref, skip)
        assert STK.stack_fingerprint(ad) == STK.stack_fingerprint(ref) == info["fingerprint"]
        base = _from_manifest(ck)
        assert STK.stack_fingerprint(base) != info["fingerprint"]  # 起点的 LoRA 确实合并进去了


def test_new_lora_same_init_across_starts(starts):
    if starts["key"] != FAMILIES[0]:
        pytest.skip("一个家族就够")
    rc = resolve_run(CFG, n_train=10, n_examples=10, tiny=True)
    lora = {}
    for init in ("base", "e3b", "e3a"):
        ck = starts[init]
        start = STK.resolve_start(CFG, init, ck, model_key=starts["key"], tiny=True)
        ad = _tiny(starts["key"]) if ck is None else _from_manifest(ck)
        tr = E5Trainer(ad, CFG, rc, seed=1, start=start)
        tr.setup_fresh()
        assert ad.lora_state == "injected" and len(tr.tokens.vocab.names) == 67
        lora[init] = {n: p.detach().clone() for n, p in ad.model.named_parameters() if "lora_" in n}
        assert tr.audit["token_row_mode"] == "fp32_master_rows"
    assert lora["base"].keys() == lora["e3b"].keys() == lora["e3a"].keys()
    assert all(torch.equal(lora["base"][k], lora["e3b"][k]) and torch.equal(lora["base"][k], lora["e3a"][k]) for k in lora["base"])


def test_start_rules(starts, tmp_path):
    key = starts["key"]
    with pytest.raises(STK.StartError):
        STK.resolve_start(CFG, "e3a", starts["e3b"], model_key=key, tiny=True)  # e3a 不该带网格 token
    with pytest.raises(STK.StartError):
        STK.resolve_start(CFG, "e3b", starts["e3a"], model_key=key, tiny=True)  # e3b 要带 67 个网格 token
    with pytest.raises(STK.StartError, match="变体"):
        STK.resolve_start(CFG, "e3b", starts["base_warmup"], model_key=key, tiny=True)  # 预热与 E3b 弄混
    with pytest.raises(STK.StartError):
        STK.resolve_start(CFG, "base", starts["e3b"], model_key=key, tiny=True)
    with pytest.raises(STK.StartError):
        STK.resolve_start(CFG, "e3b", None, model_key=key, tiny=True)
    other = [f for f in FAMILIES if f != key][0]
    with pytest.raises(STK.StartError, match="构造方式"):
        STK.resolve_start(CFG, "e3b", starts["e3b"], model_key=other, tiny=True)


# ---------------------------------------------------------------------------
# 坏起点 / 坏的 E5 LoRA：拒绝
# ---------------------------------------------------------------------------


def _pathological_lora(ad) -> None:
    """同 tests/test_e3_eval.bad_checkpoint：A 的两行几乎相同、B 的两列相反且放大 1e7；合并与不合并两条路径的中间量都很大、舍入各不
    相同，fp32 上也分家。"""
    mods = dict(ad.model.named_modules())
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name in ad.lora_target_modules():
            m = mods[name]
            a_w, b_w = m.lora_A["default"].weight, m.lora_B["default"].weight
            u = torch.randn(a_w.shape[1], generator=g)
            a_w.zero_()
            b_w.zero_()
            a_w[0] = u
            a_w[1] = u + 1e-6 * torch.randn(a_w.shape[1], generator=g)
            v = torch.randn(b_w.shape[0], generator=g)
            b_w[:, 0] = v * 1e7
            b_w[:, 1] = -v * 1e7


def test_bad_start_refused(tmp_path):
    """病态 LoRA 的起点（这里按 e3a 的结构：没有网格 token，合并核对只有 merge，CPU 上快）：起点的合并核对不过，拒绝训练。"""
    key = FAMILIES[0]
    ad = _tiny(key)
    ad.inject_lora(r=2, lora_alpha=2, init_lora_weights=False)
    _pathological_lora(ad)
    bad = ad.save_trained(tmp_path / "bad_ck")
    out = tmp_path / "o"
    # 不在训练叶目录下的检查点，配种子核对不了（DEV-0038）：这里只测合并核对，所以放开
    code = _script("train_e5").main(["--tiny", "--init", "e3a", "--start-checkpoint", str(bad), "--model", key, "--n-train", "10",
                                     "--seed", "0", "--out", str(out), "--allow-seed-mismatch"])
    assert code == 1
    assert not (out / "train" / "e3a" / key / "n10").exists()  # 不建叶目录
    (mc,) = (out / "train" / "e3a" / key / "start_checks").glob("merge_check_*.json")
    doc = json.loads(mc.read_text())
    assert doc["pass"] is False and doc["failures"]["merge"]
    # 带网格 token 的坏起点：检查项是 grid_eval（merge + restricted），同样不过（resolve_start 认它是 e3b 的结构）
    ad2 = _tiny(key)
    add_vocab(ad2, CFG.e3b.vocab)
    ad2.inject_lora(r=2, lora_alpha=2, init_lora_weights=False)
    _pathological_lora(ad2)
    bad2 = ad2.save_trained(tmp_path / "bad_ck2")
    assert STK.resolve_start(CFG, "e3b", bad2, model_key=key, tiny=True).grid_tokens == "from_checkpoint"


def test_bad_e5_lora_refused(tmp_path):
    key = FAMILIES[0]
    ad = _tiny(key)
    start = STK.resolve_start(CFG, "base", None, model_key=key, tiny=True)
    tr = E5Trainer(ad, CFG, resolve_run(CFG, n_train=10, n_examples=10, tiny=True), seed=0, start=start)
    tr.setup_fresh()
    _pathological_lora(ad)
    final = tmp_path / "final"
    ad.save_trained(final / "model")
    ts = TrainSet.build(CFG, 10, None, 0)
    STK.write_stack_file(final, {"model_dir": "model", "start": start.record(), "seed": 0, "tag": ts.tag, "data": ts.identity(),
                                 "fingerprint": {"value": tr.stack_info["fingerprint"], "device_key": STK.device_key(ad)},
                                 "identity_sha256": "test", "config": CFG.identity(), "tokens_e5": tr.e5.layout()})
    out = tmp_path / "o"
    code = _script("eval_e5_trained").main(["--tiny", "--checkpoint", str(final), "--model", key, "--out", str(out), "--limit", "1"])
    assert code == 1
    mdir = out / "trained" / "base" / key / "n10" / "seed0"
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json"]  # 没有任何任务目录
    doc = json.loads((mdir / "merge_check.json").read_text())
    assert doc["pass"] is False and (doc["failures"]["merge"] or doc["failures"]["restricted"])


# ---------------------------------------------------------------------------
# 过拟合
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILIES)
def test_overfit_near_perfect(key):
    """每类 1 条训练样本（8×8 网格；DEV-0025 的过拟合超参），训到教师强制逐格正确，再在这些样本上受限解码（与评测同一个解析与评分）：
    网格 IoU 满分，a、b 的起伏误差为 0。"""
    n = 8
    ad = _tiny(key)
    rc = resolve_run(CFG, n_train=20, n_examples=5, tiny=True)
    rc["optim"] = copy.deepcopy(rc["optim"])
    rc["optim"]["lora"]["lr"] = 3e-3
    rc["optim"]["token_rows"]["lr"] = 3e-2
    rc["optim"]["min_lr_ratio"] = 1.0
    rc["lora"] = {**rc["lora"], "dropout": 0.0}
    rc["steps"], rc["warmup_steps"], rc["micro_batch_size"] = 800, 0, 5
    start = STK.resolve_start(CFG, "base", None, model_key=key, tiny=True)
    tr = E5Trainer(ad, CFG, rc, seed=0, start=start)
    tr.setup_fresh()
    builder = ExampleBuilder(TrainSet.build(CFG, 10, None, 0), tokens=tr.e5, prompt=CFG.prompt_format(), n=n, tiny=True)
    samples, examples = [], []
    for k, t in enumerate(E5_TASKS):
        s = generate(t, "train", 7, bin=7 % num_bins(t), grid_size=n)
        samples.append(s)
        plan = SlotPlan(k=k, step=0, task=t, j=0, index=7, epoch=0, sample_id=f"{t}/train/7", kind="grid")
        examples.append(builder.example(plan, s))

    def tf_correct() -> bool:
        ad.eval_mode()
        with torch.no_grad():
            out = ad.training_forward([tr.build_inputs(e) for e in examples], [e.target_ids for e in examples],
                                      enable_thinking=False)
            ok = True
            for b, e in enumerate(examples):
                sel = (out.target_rows == b).nonzero().flatten()
                allowed = tr.e5.allowed_ids(e.plan.task)
                am = torch.tensor(allowed)[ad.restricted_logits(out.target_hidden.index_select(0, sel), allowed).argmax(-1)]
                ok = ok and bool(torch.equal(am, out.target_ids.index_select(0, sel)))
        ad.train_mode()
        return ok

    ad.train_mode()
    ok = False
    for step in range(1, 801):
        tr.optimize(examples)
        if step % 10 == 0 and tf_correct():
            ok = True
            break
    assert ok, f"{key}：800 步内教师强制没有逐格正确"
    ad.eval_mode()
    pf = CFG.prompt_format()
    for s in samples:
        text = pf.render(s, n)
        (res,) = ad.generate_restricted([ad.build_inputs(sample_images(s), text, pf.order)], tr.e5.allowed_ids(s.task),
                                        max_new_tokens=max_new_tokens(n), stop_ids=[tr.e5.stop_id], enable_thinking=False)
        rec = TE.record(s, text, res, tr.e5, n)
        assert rec["valid"] and rec["score"]["iou_grid"] == 1.0, (key, s.task, rec["failure"])
        if task_format(s.task) == "relief" and not rec["score"]["relief"]["target_flat"]:
            assert rec["score"]["relief"]["error"] == 0.0


# ---------------------------------------------------------------------------
# 续跑
# ---------------------------------------------------------------------------


def test_train_resume_bytewise(tmp_path):
    key = FAMILIES[0]
    tr = _script("train_e5")
    base = ["--tiny", "--init", "base", "--model", key, "--n-train", "20", "--seed", "1"]
    assert tr.main(base + ["--out", str(tmp_path / "a")]) == 0
    assert tr.main(base + ["--out", str(tmp_path / "b"), "--stop-after-steps", "3"]) == 0
    leaf_b = tmp_path / "b" / "train" / "base" / key / "n20" / "seed1"
    assert not (leaf_b / "train_summary.json").exists() and (leaf_b / "ckpt" / "latest").exists()
    assert tr.main(base + ["--out", str(tmp_path / "b")]) == 2  # 非空目录不加 --resume：拒绝
    assert tr.main(base + ["--out", str(tmp_path / "b"), "--resume"]) == 0
    leaf_a = tmp_path / "a" / "train" / "base" / key / "n20" / "seed1"
    assert json.loads((leaf_a / "train_summary.json").read_text())["steps"] == 5
    for rel in ("train_log.jsonl", "config.yaml", "tokens_init.json", "ckpt/final/model/lora/adapter_model.safetensors",
                "ckpt/final/model/token_rows.safetensors", "ckpt/final/model/tokens.json"):
        assert (leaf_a / rel).read_bytes() == (leaf_b / rel).read_bytes(), rel
    assert not (leaf_b / "ckpt" / "latest").exists()
    # 已完成的叶目录：运行身份不同（这里换了 epoch 数）就拒绝混写，不当作"已完成"跳过
    doc = json.loads((REPO / "configs" / "e5_train.json").read_text())
    doc["training"]["tiny"]["epochs"] = 2
    (tmp_path / "c.json").write_text(json.dumps(doc))
    assert tr.main(base + ["--out", str(tmp_path / "b"), "--resume", "--config", str(tmp_path / "c.json")]) == 2
    # 数据种子不等于种子时叶目录分开
    from src.e5.train import seed_dir_name

    assert seed_dir_name(1, 1) == "seed1" and seed_dir_name(1, 2) == "seed1_d2"


def test_eval_resume_bytewise(tmp_path):
    key = FAMILIES[0]
    out = tmp_path / "o"
    assert _script("train_e5").main(["--tiny", "--init", "base", "--model", key, "--n-train", "10", "--seed", "0",
                                     "--out", str(out)]) == 0
    final = out / "train" / "base" / key / "n10" / "seed0" / "ckpt" / "final"
    ev = _script("eval_e5_trained")
    args = ["--tiny", "--checkpoint", str(final), "--model", key, "--out", str(out), "--task", "c", "--limit", "3",
            "--batch-size", "2"]
    assert ev.main(args) == 0
    leaf = out / "trained" / "base" / key / "n10" / "seed0" / "occlusion"
    det = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")
    before = {f: (leaf / f).read_bytes() for f in det}
    assert ev.main(args) == 2  # 非空目录不加 --resume：拒绝
    lines = (leaf / "predictions.jsonl").read_bytes().splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_bytes(lines[0] + lines[1][:7])  # 中断在一批中间、末行不完整
    (leaf / "metrics.json").unlink()
    assert ev.main(args + ["--resume"]) == 0
    assert {f: (leaf / f).read_bytes() for f in det} == before
    mc = json.loads((leaf.parent / "merge_check.json").read_text())
    assert mc["pass"] is True
    # 换批大小：运行身份不同，拒绝混写
    assert ev.main(args[:-1] + ["3", "--resume"]) == 2
