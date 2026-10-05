"""DEV-0034 §3、§5：E3a 的评测（src/readout/evaluate.py；scripts/train_e3a.py、eval_e3a.py）。CPU fp32 小模型。

- 规则的单元测试：分批（最后一批不留单条）、批内打乱（换到不同图）、语义分割分块 argmax 与一次算全部通道相同、完美预测的逐样本指标、
  配置里钉住的评测设置；
- 过拟合检查（Qwen，完整方法）：每个任务几条样本训到教师强制下接近满分，评测（guided 与 bilinear）在训练样本上接近满分；泄漏检查在这个
  检查点上确实崩溃：批内打乱后四个任务都塌下去，去掉指令后指代分割塌下去（同一张图上三句指代句指三个不同的物体）；
- 续跑：中断在一批中间后 --resume，确定的输出（scores、metrics、README、arrays）与一次跑完逐字节相同；评分进程数不改变结果；
  配对比较（完整方法对 C2）按样本 id 对齐；
- 命令行：三家族 × full / c1 / c2 的 --tiny 训练与评测都跑通（full 先做合并核对，c1、c2 跳过）；C2 的去掉指令恒为零差；
  坏检查点（病态 LoRA）的合并核对不通过，评测拒绝运行。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.adapters import registry
from src.data.records import LoadResult
from src.readout import evaluate as EV
from src.readout.config import load_config
from src.readout.targets import to_original
from tests._e3a_overfit import overfit_examples, overfit_samples, overfit_trainer

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
VARIANTS = ("full", "c1", "c2")
CFG = load_config()
DETERMINISTIC = ("README.md", "config.yaml", "metrics.json", "scores.jsonl")
OVERFIT_STEPS = 130


def _load_script(name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def _spec(key: str, variant: str, *, limit=None) -> EV.EvalSpec:
    e = registry.get_model(key)
    return EV.EvalSpec(model_key=key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=key, tiny=True,
                       dtype="fp32", variant=variant, seed=0, checkpoint={"model_manifest_sha256": "test"}, limit=limit)


def _det(leaf: Path) -> dict:
    out = {f: (leaf / f).read_bytes() for f in DETERMINISTIC}
    for p in sorted((leaf / "arrays").glob("*.npz")):
        with np.load(p) as z:
            out[p.name] = {k: z[k].tobytes() for k in z.files}
    return out


# ---------------------------------------------------------------------------
# 规则的单元测试
# ---------------------------------------------------------------------------


def test_chunk_bounds_and_shuffle_perm():
    assert EV.chunk_bounds(7, 2) == [(0, 2), (2, 4), (4, 7)]  # 最后一批只有 1 条：并进前一批
    assert EV.chunk_bounds(6, 2) == [(0, 2), (2, 4), (4, 6)]
    assert EV.chunk_bounds(1, 4) == [(0, 1)] and EV.chunk_bounds(5, 32) == [(0, 5)]
    with pytest.raises(ValueError):
        EV.chunk_bounds(5, 1)
    perm, fb = EV.shuffle_perm(["a", "a", "b", "c"])
    assert perm == [2, 2, 3, 0] and fb == 0  # 换到循环顺序里下一条不同图的样本
    assert EV.shuffle_perm(["a", "a"]) == ([1, 0], 2)  # 全批同图：退回下一条并计数
    assert EV.shuffle_perm(["x"]) == ([0], 1)


def test_split_merge_roundtrip():
    rng = np.random.default_rng(0)
    inter = np.zeros(150, np.int64)
    union = np.zeros(150, np.int64)
    union[[3, 40, 149]] = [10, 7, 2]
    inter[[3, 149]] = [4, 2]
    st = {"normal": {"guided": {"inter": inter, "union": union, "correct": 6, "total": 19}, "bilinear": None}}
    scal, arrs = EV.split_arrays("semseg", st)
    assert arrs == {} and scal["normal"]["guided"]["union"] == {"3": 10, "40": 7, "149": 2}
    assert scal["normal"]["guided"]["inter"] == {"3": 4, "149": 2}
    back = EV.merge_arrays("semseg", json.loads(json.dumps(scal)), {}, 0)
    assert np.array_equal(back["normal"]["guided"]["inter"], inter) and np.array_equal(back["normal"]["guided"]["union"], union)
    assert back["normal"]["bilinear"] is None
    hist = rng.integers(0, 5, 1800)
    st = {"normal": {"guided": {"sum": 1.5, "n": 3, "within": [1, 2, 3], "hist": hist}}}
    scal, arrs = EV.split_arrays("normals", st)
    assert "hist" not in scal["normal"]["guided"] and np.array_equal(arrs["normal.guided.hist"], hist)
    back = EV.merge_arrays("normals", scal, {k: v[None] for k, v in arrs.items()}, 0)
    assert np.array_equal(back["normal"]["guided"]["hist"], hist)
    assert EV.merge_arrays("normals", scal, None, 0)["normal"]["guided"]["hist"] is None


def test_eval_config_pins():
    s = EV.resolve_eval(CFG, tiny=False)
    assert s["batch_size"] == 32 and s["upsample"]["primary"] == "guided" and s["upsample"]["methods"] == ["guided", "bilinear"]
    assert s["leak_checks"]["no_instruction"] == {"tasks": ["refseg"], "instruction": "", "upsample": "guided"}
    assert s["leak_checks"]["shuffle"]["upsample"] == "guided"
    assert s["merge_check"] == {"n": 200, "checks": ["merge"]} and s["score_workers"] == 16
    t = EV.resolve_eval(CFG, tiny=True, batch_size=3)
    assert t["batch_size"] == 3 and t["merge_check"]["n"] == 3 and t["score_workers"] == 0
    assert "score_workers" not in EV.identity_settings(s) and "merge_check" not in EV.identity_settings(s)
    from src.e3.merge_check import load_cti

    assert set(s["merge_check"]["checks"]) <= set(load_cti().CHECKS)
    with pytest.raises(ValueError):
        EV.resolve_eval(CFG, tiny=False, batch_size=1)


def test_semseg_argmax_chunking_is_exact():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((150, 128, 128)).astype(np.float32)
    logits[7, 10:20, 30:40] = 5.0
    logits[8, 10:20, 30:40] = 5.0  # 平票取靠前的类
    img = (rng.random((160, 200, 3)) * 255).astype(np.uint8)
    for m in ("guided", "bilinear"):
        full = np.argmax(to_original(logits.transpose(1, 2, 0), img, m), axis=2)
        for chunk in (1, 16, 150):
            assert np.array_equal(EV.semseg_argmax(logits, img, m, chunk), full)


def test_sample_stats_on_perfect_predictions():
    rng = np.random.default_rng(1)
    img = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
    gt = np.zeros((96, 128), bool)
    gt[24:72, 32:96] = True
    small = gt[np.ix_((np.arange(128) * 96) // 128, np.arange(128))]
    st = EV.sample_stat("refseg", np.where(small, 20.0, -20.0)[None], img, gt, "bilinear", 16)
    assert st["iou"] > 0.97 and st["inter"] <= st["union"]
    lab = np.zeros((96, 128), np.uint8)
    lab[:, :64] = 3
    lab[:, 64:] = 9
    lg = np.full((150, 128, 128), -5.0, np.float32)
    lg[2, :, :64] = 5.0
    lg[8, :, 64:] = 5.0
    st = EV.sample_stat("semseg", lg, img, lab, "bilinear", 16)
    assert st["correct"] == st["total"] == 96 * 128 and st["inter"][2] == st["union"][2] == 96 * 64
    st = EV.sample_stat("semseg", lg, img, lab, "guided", 16)  # 引导图是噪声：边界上差几个像素
    assert st["correct"] > 0.99 * st["total"]
    depth = np.full((480, 640), 2.5, np.float32)
    big = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    st = EV.sample_stat("depth", np.full((1, 128, 128), np.log(2.5), np.float32), big, depth, "guided", 16)
    assert st["abs_rel"] < 1e-6 and st["d1"] == 1.0
    n = np.zeros((480, 640, 3), np.float32)
    n[..., 1] = 1.0
    st = EV.sample_stat("normals", np.stack([np.zeros((128, 128)), np.full((128, 128), 3.0), np.zeros((128, 128))]),
                        big, (n, np.ones((480, 640), bool)), "bilinear", 16)
    assert st["n"] == 480 * 640 and st["sum"] < 1e-3 and st["within"][0] == st["n"]


# ---------------------------------------------------------------------------
# 过拟合检查点：接近满分；泄漏检查确实崩溃
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def overfit(tmp_path_factory):
    root = tmp_path_factory.mktemp("e3a_overfit")
    samples = overfit_samples(root / "data")
    tr = overfit_trainer("qwen3.5-0.8b", "full", steps=OVERFIT_STEPS)
    ex = overfit_examples(samples, tr.readout.spec.out_size)
    tr.ad.train_mode()
    for _ in range(OVERFIT_STEPS):
        rec, _, _ = tr.optimize(ex)
    tr.ad.eval_mode()
    ck = tr.save(root / "ck")
    return {"ck": ck, "samples": samples, "root": root, "last": rec}


def _eval(ov, out: Path, task: str, *, variant: str = "full", ck=None, settings=None, resume=False, model=None):
    model = model or EV.EvalModel.from_checkpoint(ck or ov["ck"], CFG, device="cpu")
    settings = settings or EV.resolve_eval(CFG, tiny=True)
    sel = LoadResult(ov["samples"][task], {"set": "overfit"})
    m = EV.run_eval_task(model, CFG, settings, _spec("qwen3.5-0.8b", variant), task, sel, out, resume=resume,
                         log=lambda _m: None)
    return m, EV.leaf_dir(out, _spec("qwen3.5-0.8b", variant), task)


def test_overfit_near_perfect_and_leak_checks_collapse(overfit, tmp_path):
    assert overfit["last"]["loss_by_task"]["semseg"] < 0.05 and overfit["last"]["loss_by_task"]["refseg"] < 0.1
    model = EV.EvalModel.from_checkpoint(overfit["ck"], CFG, device="cpu")
    out = tmp_path / "e"
    res = {t: _eval(overfit, out, t, model=model)[0] for t in ("refseg", "semseg", "depth", "normals")}
    for m in ("guided", "bilinear"):
        r = res["refseg"]["by_upsample"][m]
        assert r["cIoU"]["value"] > 0.95 and r["gIoU"]["value"] > 0.95
        assert res["semseg"]["by_upsample"][m]["miou"] > 0.95 and res["semseg"]["by_upsample"][m]["pixel_acc"] > 0.97
        assert res["depth"]["by_upsample"][m]["abs_rel"] < 0.1 and res["depth"]["by_upsample"][m]["d1"] > 0.95
        assert res["normals"]["by_upsample"][m]["mean"] < 12.0 and res["normals"]["by_upsample"][m]["within_11.25"] > 0.7
    assert res["refseg"]["headline"]["upsample"] == "guided" and res["refseg"]["by_upsample"]["guided"]["BIoU"]["value"] > 0.9
    # 批内打乱：四个任务都塌下去，配对差的 CI 下界 > 0（深度的 AbsRel 越小越好：配对差 < 0）
    sh = {t: res[t]["leak_checks"]["shuffle"] for t in res}
    assert sh["refseg"]["metrics"]["gIoU"]["value"] < 0.3
    assert sh["refseg"]["paired_normal_minus_check"]["gIoU"]["low"] > 0.5
    assert sh["semseg"]["metrics"]["miou"] < 0.5 and sh["semseg"]["paired_normal_minus_check"]["miou"]["low"] > 0.3
    assert sh["depth"]["metrics"]["d1"] < 0.5 and sh["depth"]["paired_normal_minus_check"]["abs_rel"]["high"] < 0
    assert sh["normals"]["metrics"]["mean"] > 45.0 and sh["normals"]["paired_normal_minus_check"]["mean"]["high"] < 0
    assert all(s["n_same_image_fallback"] == 0 for t, s in sh.items() if t != "refseg")
    # 去掉指令：指代分割塌下去（图 A 上三句指三个物体，读出不知道指的是哪个）
    ni = res["refseg"]["leak_checks"]["no_instruction"]
    assert ni["instruction"] == "" and ni["metrics"]["gIoU"]["value"] < 0.7
    assert ni["paired_normal_minus_check"]["gIoU"]["estimate"] > 0.3
    assert set(res["refseg"]["leak_checks"]) == {"shuffle", "no_instruction"} and set(res["depth"]["leak_checks"]) == {"shuffle"}
    assert res["refseg"]["groups"] == {"non_person": 2, "person": 2}
    assert res["refseg"]["compute"]["n_calls_no_instruction"] == 4


def test_eval_resume_and_workers_identical(overfit, tmp_path):
    settings = EV.resolve_eval(CFG, tiny=True, batch_size=2)
    for task in ("refseg", "semseg"):
        _, leaf_a = _eval(overfit, tmp_path / "a", task, settings=settings)
        before = _det(leaf_a)
        # 中断在一批中间：scores 截到 1 条（第一批有 2 条），后面的数组与汇总都删掉
        lines = (leaf_a / "scores.jsonl").read_text().splitlines(keepends=True)
        (leaf_a / "scores.jsonl").write_text(lines[0])
        for f in ("metrics.json", "README.md"):
            (leaf_a / f).unlink()
        for p in (leaf_a / "arrays").glob("*.npz"):
            p.unlink()
        with pytest.raises(FileExistsError):
            _eval(overfit, tmp_path / "a", task, settings=settings)
        _eval(overfit, tmp_path / "a", task, settings=settings, resume=True)
        assert _det(leaf_a) == before
        # 评分进程数不改变结果
        w = dict(settings, score_workers=2)
        _, leaf_b = _eval(overfit, tmp_path / "b", task, settings=w)
        assert _det(leaf_b) == before
        # 只读入库的文件（不读 arrays/）也能还原配对比较要的逐样本统计
        ra, _ = EV.load_rows(leaf_a, task)
        rb, _ = EV.load_rows(leaf_a, task, arrays=False)
        for x, y in zip(ra, rb):
            for k in ("inter", "union") if task == "semseg" else ("inter", "union", "iou"):
                assert np.array_equal(np.asarray(x["stats"]["normal"]["guided"][k]), np.asarray(y["stats"]["normal"]["guided"][k]))
        key = "miou" if task == "semseg" else "gIoU"
        assert EV.paired_compare(leaf_a, leaf_b)["a_minus_b"][key]["estimate"] == 0.0
    # 运行身份变了（批大小）拒绝续跑
    with pytest.raises(ValueError, match="运行身份"):
        _eval(overfit, tmp_path / "a", "refseg", settings=EV.resolve_eval(CFG, tiny=True, batch_size=3), resume=True)


def test_paired_compare(overfit, tmp_path):
    model = EV.EvalModel.from_checkpoint(overfit["ck"], CFG, device="cpu")
    _, leaf = _eval(overfit, tmp_path / "x", "refseg", model=model)
    _, leaf_c = _eval(overfit, tmp_path / "y", "refseg", model=model, variant="c1")
    pc = EV.paired_compare(leaf, leaf_c)
    assert pc["n"] == 4 and pc["upsample"] == "guided" and pc["a"]["variant"] == "full" and pc["b"]["variant"] == "c1"
    assert pc["a_minus_b"]["gIoU"]["estimate"] == 0.0  # 同一个模型：配对差恰为 0
    rows, _ = EV.load_rows(leaf, "refseg")
    assert [r["id"] for r in rows] == [s.id for s in overfit["samples"]["refseg"]]
    assert rows[0]["stats"]["normal"]["guided"]["iou"] > 0.9 and rows[0]["shuffle_source"] != rows[0]["id"]


# ---------------------------------------------------------------------------
# 命令行：三家族 × full / c1 / c2
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("e3a_cli")
    tr = _load_script("train_e3a")
    for v in VARIANTS:
        code = tr.main(["--tiny", "--model", *FAMILIES, "--variant", v, "--seed", "0", "--out", str(out / "train")])
        assert code == 0
    return out


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("key", FAMILIES)
def test_cli_tiny_train_and_eval(tiny_runs, key, variant):
    ev = _load_script("eval_e3a")
    ck = tiny_runs / "train" / "e3a" / variant / key / "seed0" / "ckpt" / "final"
    assert (ck / "model" / "manifest.json").is_file() and (ck / "readout" / "readout.json").is_file()
    out = tiny_runs / "eval" / variant / key
    args = ["--tiny", "--checkpoint", str(ck), "--model", key, "--variant", variant, "--seed", "0", "--out", str(out)]
    assert ev.main(args) == 0
    mdir = out / "eval" / "e3a" / variant / key / "seed0"
    assert (mdir / "merge_check.json").is_file() == (variant == "full")
    if variant == "full":
        mc = json.loads((mdir / "merge_check.json").read_text())
        assert mc["pass"] is True and list(mc["checks"]) == ["merge"]
    for task in ("refseg", "depth", "semseg", "normals"):
        m = json.loads((mdir / task / "metrics.json").read_text())
        assert m["complete"] and m["variant"] == variant and set(m["by_upsample"]) == {"guided", "bilinear"}
        assert "shuffle" in m["leak_checks"] and ("no_instruction" in m["leak_checks"]) == (task == "refseg")
        assert (mdir / task / "compute.json").is_file()
    if variant == "c2":  # C2 的解码器看不到指令：去掉指令恒为零差
        ni = json.loads((mdir / "refseg" / "metrics.json").read_text())["leak_checks"]["no_instruction"]
        assert ni["paired_normal_minus_check"]["gIoU"]["estimate"] == 0.0
        assert ni["paired_normal_minus_check"]["gIoU"]["low"] == ni["paired_normal_minus_check"]["gIoU"]["high"] == 0.0


def test_cli_resume_subprocess_and_rejections(tiny_runs, tmp_path):
    key = FAMILIES[0]
    ck = tiny_runs / "train" / "e3a" / "full" / key / "seed0" / "ckpt" / "final"
    ev = _load_script("eval_e3a")
    out = tmp_path / "r"
    args = ["--tiny", "--checkpoint", str(ck), "--model", key, "--variant", "full", "--seed", "0", "--out", str(out),
            "--task", "refseg", "semseg"]
    assert ev.main(args) == 0
    mdir = out / "eval" / "e3a" / "full" / key / "seed0"
    # 续跑：已完成的直接复用（合并核对也复用）；非空目录不加 --resume 拒绝；变体对不上拒绝
    before = {t: (mdir / t / "metrics.json").read_bytes() for t in ("refseg", "semseg")}
    assert ev.main(args + ["--resume"]) == 0
    assert {t: (mdir / t / "metrics.json").read_bytes() for t in before} == before
    assert ev.main(args) == 2
    assert ev.main(["--tiny", "--checkpoint", str(ck), "--model", key, "--variant", "c1", "--seed", "0",
                    "--out", str(out / "w")]) == 2
    assert ev.main(["--tiny", "--checkpoint", str(ck), "--model", FAMILIES[1], "--variant", "full", "--seed", "0",
                    "--out", str(out / "w")]) == 2
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "eval_e3a.py"), "--tiny", "--checkpoint", str(ck), "--model",
                        FAMILIES[0], "--variant", "full", "--seed", "0", "--out", str(tmp_path / "s"), "--task", "depth"],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr[-3000:]
    assert "depth n=1" in p.stdout and "shuffle:" in p.stdout
    # 真模型运行要 --data-root 与带编号的 --out
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "eval_e3a.py"), "--checkpoint", str(ck), "--model", FAMILIES[0],
                        "--variant", "full", "--seed", "0", "--out", str(tmp_path / "x")], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 2 and "--data-root" in p.stderr
    tr = _load_script("train_e3a")
    with pytest.raises(SystemExit):
        tr.parse_args(["--model", FAMILIES[0], "--variant", "full", "--seed", "0", "--out", str(tmp_path / "x"),
                       "--data-root", "/x"])
    assert tr.main(["--tiny", "--model", FAMILIES[0], "--variant", "full", "--seed", "7", "--out", str(tmp_path / "s7")]) == 2


def test_bad_checkpoint_refused(tmp_path):
    """病态的 LoRA（同 tests/test_e3_eval.bad_checkpoint）：合并前后的两条计算路径数值上分家，合并核对不通过，评测拒绝运行。"""
    tr = overfit_trainer("qwen3.5-0.8b", "full", steps=1)
    ad = tr.ad
    mods = dict(ad.model.named_modules())
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name in ad.lora_target_modules():
            m = mods[name]
            a_w, b_w = m.lora_A["default"].weight, m.lora_B["default"].weight
            u = torch.randn(a_w.shape[1], generator=g)
            a_w.copy_(torch.stack([u + 1e-6 * torch.randn(a_w.shape[1], generator=g) for _ in range(a_w.shape[0])]))
            v = torch.randn(b_w.shape[0], generator=g)
            cols = [v if k % 2 == 0 else -v for k in range(b_w.shape[1])]
            b_w.copy_(torch.stack(cols, dim=1) * 1e7)
    ck = tr.save(tmp_path / "bad")
    ev = _load_script("eval_e3a")
    out = tmp_path / "e"
    code = ev.main(["--tiny", "--checkpoint", str(ck), "--model", "qwen3.5-0.8b", "--variant", "full", "--seed", "0",
                    "--out", str(out)])
    assert code == 1
    mdir = out / "eval" / "e3a" / "full" / "qwen3.5-0.8b" / "seed0"
    assert json.loads((mdir / "merge_check.json").read_text())["pass"] is False
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json"]
