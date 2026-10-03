"""DEV-0030 §1、§4：E3b 的评测（src/e3/evaluate.py、eval_scoring.py、merge_check.py；scripts/eval_e3b.py）。CPU fp32 小模型。

- 规则的单元测试：网格编码往返、分段（续跑边界）、匹配计算量的合成规则、配置里钉住的数字；
- 过拟合检查点（DEV-0025 的过拟合流程，8×8 网格；分割 3 条 + 深度 1 条，各自的从零预测与修正对）：三个家族上单轮满分；多轮每轮都调用、
  状态不动（不动点），每轮的输出单独记录与评分；记账字段与 predictions 逐项相加相同；K = 1 与单轮是同一次调用（记账文件逐字节相同）；
- 续跑：中断在一批中间后 --resume，确定的输出（predictions、scores、metrics、README）与一次跑完逐字节相同；评分进程数不改变结果；
- 命令行：--tiny 跑通（训练框架产出的小检查点、tests/fixtures 的评测小数据）、非空目录不加 --resume 拒绝、续跑复用已通过的合并核对；
- 坏检查点（病态的 LoRA：合并前后的两条计算路径数值上分家）：合并核对不通过，评测拒绝运行，不写任何任务目录。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.adapters import factory, registry
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.data.records import LoadResult, Sample
from src.e3 import data as D
from src.e3 import eval_scoring as ES
from src.e3 import evaluate as EV
from src.e3.config import load_config, resolve_run
from src.e3.feedback import feedback_image, upsample_cells
from src.e3.merge_check import load_cti
from src.e3.targets import grid_to_ids, parse_target, seq_len
from src.e3.tokens import add_vocab, attach_vocab
from src.e3.train import Trainer
from src.executors.raster import mask_to_rle
from src.grid.codec import bins_to_depth, grid_to_mask

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()
N = 8  # DEV-0025 的过拟合流程用 8×8（CPU 小模型在 32×32 上逐格过拟合要几百步）
DETERMINISTIC = ("README.md", "config.yaml", "metrics.json", "scores.jsonl")


def _tiny(key):
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _load_script(name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


# ---------------------------------------------------------------------------
# 规则的单元测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [8, 32])
def test_grid_code_round_trip(n):
    rng = np.random.default_rng(n)
    g = rng.integers(0, 2, (n, n))
    c = ES.encode_grid("refseg", g)
    assert len(c) == n * n // 4 and np.array_equal(ES.decode_grid("refseg", c, n), g)
    d = rng.integers(0, 64, (n, n))
    c = ES.encode_grid("depth", d)
    assert len(c) == n * n and np.array_equal(ES.decode_grid("depth", c, n), d)
    assert ES.encode_grid("depth", np.arange(64).reshape(8, 8)) == ES.B64_ALPHABET
    with pytest.raises(ValueError):
        ES.decode_grid("refseg", c, n)


def test_chunk_plan():
    act = [True, False, True, True, False, False, True, True]
    plan = EV.chunk_plan(act, 2)
    assert plan == [(0, 3, [0, 2]), (3, 7, [3, 6]), (7, 8, [7])]
    assert EV.chunk_plan([False] * 3, 2) == [(0, 3, [])]
    assert EV.chunk_plan([True] * 4, 2) == [(0, 2, [0, 1]), (2, 4, [2, 3])]
    for bs in (1, 3, 5):  # 各段首尾相接、覆盖全部，每段至多 bs 个要调用的样本
        p = EV.chunk_plan(act, bs)
        assert p[0][0] == 0 and p[-1][1] == len(act) and all(a[1] == b[0] for a, b in zip(p, p[1:]))
        assert all(len(x[2]) <= bs for x in p) and sum(len(x[2]) for x in p) == sum(act)


def test_vote_rules():
    a = np.array([[1, 0], [1, 0]])
    b = np.array([[0, 0], [1, 1]])
    c = np.array([[0, 1], [1, 1]])
    assert np.array_equal(EV.vote("refseg", [a, b]), a)  # 两票平票：取第一张有效网格（贪心）
    assert np.array_equal(EV.vote("refseg", [None, b, a]), b)  # 贪心失败：第一张有效的是第 1 次采样
    assert np.array_equal(EV.vote("refseg", [a, b, c]), np.array([[0, 0], [1, 1]]))  # 多数票
    assert EV.vote("refseg", [None, None]) is None
    d = EV.vote("depth", [np.array([[10, 20]]), np.array([[11, 23]])])
    assert d.tolist() == [[10, 22]]  # 10.5 → 10、21.5 → 22（np.rint：.5 取偶）


def test_eval_config_pins():
    e = CFG.doc["eval"]
    assert seq_len(CFG.grid_n) == 1057  # max_new_tokens_rule：任务书的 1057
    assert e["multi_turn"]["k_values"] == [1, 2, 4]  # 任务书 K = 1、2、4
    e1 = json.loads((REPO / "configs" / "e1_prompts.json").read_text())
    assert e["batch_size"] == e1["batch_protocol"]["batch_size"] == 64  # 与 E1 冻结的批大小相同（D43）
    assert e["merge_check"]["n"] == load_cti().DEFAULT_N == 200 and e["merge_check"]["checks"] == ["merge"]
    assert e["upsample"]["primary"] == "guided" and e["upsample"]["methods"] == ["guided", "bilinear", "cells"]
    s = EV.resolve_eval(CFG, tiny=False)
    assert (s["batch_size"], s["n_rounds"], s["n_samples"], s["merge_check"]["n"]) == (64, 4, 3, 200)
    t = EV.resolve_eval(CFG, tiny=True)
    assert (t["batch_size"], t["merge_check"]["n"], t["score_workers"]) == (2, 3, 0)
    assert "score_workers" not in EV.identity_settings(s) and "merge_check" not in EV.identity_settings(s)
    assert EV.procedures(s) == ["single", "round2", "round3", "round4", "matched_K2", "matched_K4"]
    assert EV.procedure_stages("multi_K4") == ["round1", "round2", "round3", "round4"]
    assert EV.procedure_stages("matched_K4") == ["round1", "sample1", "sample2", "sample3"]
    assert EV.procedure_stages("multi_K1") == EV.procedure_stages("matched_K1") == EV.procedure_stages("single")


def test_scoring_exact_grids():
    """GT 恰是网格回填的 mask / 每格常数的深度时：cells 满分；双线性会把角磨圆（不是满分），引导上采样不比它差。"""
    h, w = 96, 128
    g = np.zeros((N, N), dtype=np.int64)
    g[2:5, 1:6] = 1
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    gt = grid_to_mask(g, h, w)
    img[gt] = (220, 40, 40)
    out = ES.score_refseg(img, gt, {"p": g, "f": None}, methods=["cells", "bilinear", "guided"], threshold=0.5, n=N)
    assert out["p"]["cells"]["iou"] == 1.0 and out["p"]["cells"]["biou"] == 1.0
    assert 0.9 < out["p"]["bilinear"]["iou"] < 1.0  # 双线性把角磨圆
    assert out["p"]["guided"]["iou"] >= out["p"]["bilinear"]["iou"]  # 物体边缘与图像边缘对齐时，引导滤波把角拉回来
    assert out["p"]["grid"] == {"cell_iou": 1.0, "exact": True}
    assert out["f"]["valid"] is False and out["f"]["cells"]["iou"] == 0.0 and out["f"]["cells"]["union"] == int(gt.sum())
    dg = np.full((N, N), 20, dtype=np.int64)
    dg[:, 4:] = 40
    hd, wd = 480, 640
    depth = bins_to_depth(upsample_cells(dg, hd, wd)).astype(np.float32)
    o = ES.score_depth(np.zeros((hd, wd, 3), np.uint8), depth, {"p": dg, "f": None}, methods=["cells"], n=N,
                       min_valid_fraction=0.5)
    assert o["p"]["cells"]["abs_rel"] < 1e-6 and o["p"]["cells"]["d1"] == 1.0
    assert o["p"]["grid"]["mean_abs_bin_error"] == 0.0 and o["p"]["grid"]["exact_cell_rate"] == 1.0
    assert o["f"] == {"valid": False, "cells": None, "grid": None}


# ---------------------------------------------------------------------------
# 过拟合检查点（三个家族）
# ---------------------------------------------------------------------------


def _png(path: Path, a: np.ndarray) -> str:
    Image.fromarray(a.astype(np.uint8), mode="RGB").save(path)
    return str(path)


def overfit_data(root: Path) -> tuple[list[Sample], list[Sample], dict[str, np.ndarray]]:
    """分割 3 条（96×128，物体画成实色，GT = 网格回填的 mask）、深度 1 条（480×640，GT = 每格常数深度）。"""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    seg, grids = [], {}
    for k in range(3):
        g = np.zeros((N, N), dtype=np.int64)
        r0, c0 = rng.integers(0, N // 2, 2)
        g[r0:r0 + rng.integers(2, N // 2), c0:c0 + rng.integers(2, N // 2)] = 1
        mask = grid_to_mask(g, 96, 128)
        img = np.full((96, 128, 3), 40, dtype=np.uint8)
        img[mask] = (200, 50 + 60 * k, 50)
        sid = f"overfit/seg{k}"
        seg.append(Sample(id=sid, task="refexp", images=(_png(root / f"seg{k}.png", img),), image_keys=(f"overfit:{k}",),
                          fields={"expression": f"object number {k} ", "category": "person" if k == 0 else "dog"},
                          gt={"mask": {"rle": mask_to_rle(mask), "height": 96, "width": 128}}, source="overfit", split="val"))
        grids[sid] = g
    dg = np.full((N, N), 12, dtype=np.int64)
    dg[3:, :] = 30
    dg[5:, 4:] = 50
    depth = bins_to_depth(upsample_cells(dg, 480, 640)).astype(np.float32)
    img = (np.stack([dg * 4, 200 - dg * 3, np.full_like(dg, 90)], -1)).astype(np.uint8)
    img = upsample_cells(img.reshape(N, N, 3)[..., 0], 480, 640)
    rgb = np.stack([img, 255 - img, np.full_like(img, 100)], -1).astype(np.uint8)
    np.savez(root / "depth0.npz", depth_raw=depth)
    dep = [Sample(id="overfit/depth0", task="depth", images=(_png(root / "depth0.png", rgb),), image_keys=("overfit:d0",),
                  fields={"idx": 0}, gt={"depth_npz": str(root / "depth0.npz"), "key": "depth_raw", "units": "meters"},
                  source="overfit", split="test")]
    grids["overfit/depth0"] = dg
    return seg, dep, grids


def train_overfit(key: str, root: Path, *, max_steps: int = 800) -> tuple[Path, list[Sample], list[Sample], dict]:
    """DEV-0025 的过拟合流程（tests/test_e3_train._overfit 的同一套超参），指令换成评测用的指令（make_context，n = 8），每条样本
    另有一条修正对（[原图, GT 网格的叠加图] → GT 网格），训到教师强制逐格正确，存检查点。"""
    seg, dep, grids = overfit_data(root / "data")
    ad = _tiny(key)
    rc = resolve_run(CFG, supervision="grid", tiny=True)
    rc["optim"] = copy.deepcopy(rc["optim"])
    rc["optim"]["lora"]["lr"] = 3e-3
    rc["optim"]["token_rows"]["lr"] = 3e-2
    rc["lora"] = {**rc["lora"], "dropout": 0.0}
    rc["steps"], rc["warmup_steps"], rc["micro_batch_size"] = max_steps, 0, 4
    rc["optim"]["min_lr_ratio"] = 1.0
    tr = Trainer(ad, CFG, rc, supervision="grid", seed=0)
    tr.setup_fresh()
    tok = tr.tokens
    examples = []
    for task, samples in (("refseg", seg), ("depth", dep)):
        ctx = EV.make_context(CFG, tok, task, n=N)
        for j, s in enumerate(samples):
            img = D.load_rgb(s.images[0])
            g = grids[s.id]
            ids = grid_to_ids(g, tok.value_ids(task), tok.row_id, tok.stop_id, n=N)
            for kind, imgs, corr in (("grid", [img], False), ("correction", [img, feedback_image(task, img, g)], True)):
                plan = D.SlotPlan(k=len(examples), step=0, task=task, j=j, index=j, epoch=0, sample_id=s.id, kind=kind)
                examples.append(D.Example(plan, imgs, ctx.instruction(s, correction=corr), ctx.order, ids, [1] * len(ids)))

    def tf_correct() -> bool:
        ad.eval_mode()
        ok = True
        with torch.no_grad():
            for task in ("refseg", "depth"):
                ex = [e for e in examples if e.plan.task == task]
                out = ad.training_forward([tr.build_inputs(e) for e in ex], [e.target_ids for e in ex], enable_thinking=False)
                allowed = tok.allowed_ids(task, "task")
                am = torch.tensor(allowed)[ad.restricted_logits(out.target_hidden, allowed).argmax(-1)]
                ok = ok and bool(torch.equal(am, out.target_ids))
        ad.train_mode()
        return ok

    ad.train_mode()
    ok = False
    for n in range(1, max_steps + 1):
        tr.optimize(examples)
        if n % 10 == 0 and tf_correct():
            ok = True
            break
    ad.eval_mode()
    assert ok, f"{key}：{max_steps} 步内教师强制没有逐格正确"
    ck = ad.save_trained(root / "ck")
    return ck, seg, dep, grids


@pytest.fixture(scope="module", params=FAMILIES)
def overfit(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"overfit_{request.param}")
    ck, seg, dep, grids = train_overfit(request.param, root)
    return {"key": request.param, "ck": ck, "seg": seg, "dep": dep, "grids": grids, "root": root}


def _spec(key: str, ck: Path, *, limit=None) -> EV.EvalSpec:
    e = registry.get_model(key)
    return EV.EvalSpec(model_key=key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=key, tiny=True,
                       dtype="fp32", variant="e3b", seed=0,
                       checkpoint={"manifest_sha256": hashlib.sha256((ck / "manifest.json").read_bytes()).hexdigest()},
                       limit=limit)


def _run(ov, out: Path, task: str, *, settings=None, resume=False):
    ad, _ = factory.load_trained(ov["ck"], device="cpu", merge=True)
    tokens = attach_vocab(ad, CFG.vocab)
    settings = settings or EV.resolve_eval(CFG, tiny=True)
    ctx = EV.make_context(CFG, tokens, task, n=N)
    sel = LoadResult(ov["seg"] if task == "refseg" else ov["dep"], {"set": "overfit"})
    m = EV.run_eval_task(ad, CFG, settings, _spec(ov["key"], ov["ck"]), ctx, sel, out, resume=resume, warmup=False,
                         log=lambda _m: None)
    return m, EV.leaf_dir(out, _spec(ov["key"], ov["ck"]), task), ad, ctx


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_overfit_checkpoint_perfect_single_and_multi_turn(overfit, tmp_path):
    out = tmp_path / "out"
    for task in ("refseg", "depth"):
        m, leaf, ad, ctx = _run(overfit, out, task)
        samples = overfit["seg"] if task == "refseg" else overfit["dep"]
        n = len(samples)
        # 单轮满分（过拟合的样本）
        single = m["single"]
        if task == "refseg":
            assert single["grid"]["exact_rate"] == 1.0
            for k in ("cIoU", "gIoU", "BIoU"):
                assert single["cells"][k]["value"] == 1.0, (task, k)  # 网格逐格正确：按格回填与 GT 逐像素相同
            for meth in ("bilinear", "guided"):  # 上采样把角磨圆，不是满分
                assert single[meth]["valid_rate"] == 1.0 and single[meth]["gIoU"]["value"] > 0.8, (meth, single[meth])
            assert m["headline"]["upsample"] == "guided" and m["headline"]["gIoU"] == single["guided"]["gIoU"]
            assert set(single["cells"]["by_group"]) == {"person", "non_person"}
        else:
            assert single["grid"]["exact_cell_rate"]["value"] == 1.0
            assert single["cells"]["abs_rel"]["value"] < 1e-6 and single["cells"]["d1"]["value"] == 1.0
            assert single["cells"]["d1_strict"]["value"] == 1.0
        # 单轮的定义：与直接调 generate_restricted 的结果相同
        preds = _jsonl(leaf / "predictions.jsonl")
        inp = [ad.build_inputs([D.load_rgb(s.images[0])], ctx.instruction(s, correction=False), ctx.order) for s in samples]
        direct = ad.generate_restricted(inp, list(ctx.allowed), max_new_tokens=seq_len(N), stop_ids=[ctx.tokens.stop_id],
                                        enable_thinking=False)
        for s, p, r in zip(samples, preds, direct):
            g = parse_target(task, r.token_ids, ctx.tokens, n=N).grid
            assert p["grid"] == ES.encode_grid(task, g) == ES.encode_grid(task, overfit["grids"][s.id])
            assert p["n_generated_tokens"] == seq_len(N) and p["stopped"] is True and p["n_images"] == 1
        # 多轮：每轮都调用、两张图、状态不动（不动点），每轮单独记录与评分
        for r in (2, 3, 4):
            st = m["stages"][f"round{r}"]
            assert (st["n_called"], st["n_valid"], st["n_unchanged"], st["n_skipped"]) == (n, n, n, 0)
            recs = _jsonl(leaf / f"predictions_round{r}.jsonl")
            assert all(x["n_images"] == 2 and x["grid"] == x["state"] == p["grid"] for x, p in zip(recs, preds))
            assert m["multi_turn"]["by_round"][str(r)] == single
        assert m["multi_turn"]["by_K"]["1"]["metrics"] == single and m["multi_turn"]["by_K"]["1"]["procedure"] == "single"
        assert m["matched_compute"]["by_K"]["1"]["metrics"] == single
        for k in ("2", "4"):
            assert m["matched_compute"]["by_K"][k]["metrics"][m["headline"]["upsample"]]["valid_rate"] == 1.0
        # 记账：与 predictions 逐项相加相同；K = 1 与单轮是同一次调用（记账文件逐字节相同）
        rec = {st: _jsonl(leaf / EV.stage_file(st)) for st in EV.stage_names(EV.resolve_eval(CFG, tiny=True))}
        v1 = [x["n_visual_tokens"] for x in rec["round1"]]
        assert all(x["n_visual_tokens"] == 2 * v for x, v in zip(rec["round2"], v1))  # 叠加图与原图同尺寸
        cbs = json.loads((leaf / "compute_by_stage.json").read_text())  # 每轮单独的记账
        for st, xs in rec.items():
            for f in ("generated_tokens", "prompt_tokens", "visual_tokens"):
                assert cbs[st][f] == sum(x[f"n_{f}"] for x in xs) == m["stages"][st]["compute"][f], (st, f)
            assert cbs[st]["n_calls"] == n and cbs[st]["call_time_s"] > 0
        single_bytes = (leaf / "compute.json").read_bytes()
        assert (leaf / "compute_multi_K1.json").read_bytes() == single_bytes == (leaf / "compute_matched_K1.json").read_bytes()
        for kk in (1, 2, 4):
            c = json.loads((leaf / f"compute_multi_K{kk}.json").read_text())
            stages = [f"round{r}" for r in range(1, kk + 1)]
            assert (c["n_samples"], c["n_calls"], c["n_rounds"]) == (n, kk * n, kk * n)
            for f in ("generated_tokens", "prompt_tokens", "visual_tokens", "image_pixels"):
                want = sum(x[f"n_{f}"] for st in stages for x in rec[st])
                assert c[f] == want, (task, kk, f)
            assert c["generated_tokens"] == kk * n * seq_len(N)
            assert c["call_time_s"] > 0 and c["wall_time_s"] >= c["call_time_s"]
            mc = json.loads((leaf / f"compute_matched_K{kk}.json").read_text())
            assert (mc["n_calls"], mc["n_rounds"]) == (kk * n, n)  # 采样与贪心互不依赖：同一轮里的调用
            assert mc["visual_tokens"] == kk * sum(v1)
            mm = m["compute"][f"multi_K{kk}"]
            assert mm["ratio_vs_single"]["n_calls"] == kk and mm["file"] == f"compute_multi_K{kk}.json"
            assert "call_time_s" not in mm  # metrics.json 只放计数（墙钟在 compute*.json）
        files = {p.name for p in leaf.iterdir()}
        assert files >= {"predictions.jsonl", "predictions_round4.jsonl", "predictions_sample3.jsonl", "scores.jsonl",
                         "metrics.json", "compute.json", "config.yaml", "meta.json", "README.md", "logs"}
        ident = yaml.safe_load((leaf / "config.yaml").read_text())
        assert ident["checkpoint"]["manifest_sha256"] and ident["context"]["max_new_tokens"] == seq_len(N)
        assert ident["eval"]["k_values"] == [1, 2, 4] and "score_workers" not in ident["eval"]


def test_resume_after_interruption_is_identical(overfit, tmp_path):
    if overfit["key"] != FAMILIES[0]:
        pytest.skip("续跑与评分进程数只在一个家族上核对")
    a, b = tmp_path / "A", tmp_path / "B"
    _, la, _, _ = _run(overfit, a, "refseg")
    _, lb, _, _ = _run(overfit, b, "refseg")
    # 中断在第 3 轮的一批中间（批大小 2，3 条样本：第一批 2 条），之后的阶段与汇总都还没写
    p3 = lb / "predictions_round3.jsonl"
    p3.write_text(p3.read_text().splitlines(keepends=True)[0])
    for f in ("predictions_round4.jsonl", "predictions_sample1.jsonl", "predictions_sample2.jsonl", "predictions_sample3.jsonl",
              "metrics.json", "scores.jsonl", "README.md"):
        (lb / f).unlink()
    with pytest.raises(FileExistsError):
        _run(overfit, b, "refseg")
    _run(overfit, b, "refseg", resume=True)
    for f in DETERMINISTIC + tuple(EV.stage_file(s) for s in EV.stage_names(EV.resolve_eval(CFG, tiny=True))):
        assert (la / f).read_bytes() == (lb / f).read_bytes(), f
    run2 = json.loads((lb / "logs" / "run_002.json").read_text())
    r3 = next(s for s in run2["stages"] if s["stage"] == "round3")
    assert r3["trimmed"] == {"from": 1, "to": 0} and r3["n_calls"] == 3
    # 评分进程数不改变结果（spawn 的进程池）
    settings = {**EV.resolve_eval(CFG, tiny=True), "score_workers": 2}
    ident = yaml.safe_load((la / "config.yaml").read_text())
    ad, _ = factory.load_trained(overfit["ck"], device="cpu", merge=True)
    ctx = EV.make_context(CFG, attach_vocab(ad, CFG.vocab), "refseg", n=N)
    before = {f: (la / f).read_bytes() for f in ("scores.jsonl", "metrics.json")}
    EV.finalize(la, _spec(overfit["key"], overfit["ck"]), ident, ctx, settings, overfit["seg"])
    assert {f: (la / f).read_bytes() for f in before} == before


def test_resume_refuses_changed_identity(overfit, tmp_path):
    if overfit["key"] != FAMILIES[0]:
        pytest.skip("只在一个家族上核对")
    _run(overfit, tmp_path, "depth")
    s = {**EV.resolve_eval(CFG, tiny=True), "batch_size": 1}
    with pytest.raises(ValueError, match="运行身份"):
        _run(overfit, tmp_path, "depth", settings=s, resume=True)


# ---------------------------------------------------------------------------
# 命令行（--tiny）与坏检查点
# ---------------------------------------------------------------------------


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True,
                          cwd=REPO)


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """训练框架（scripts/train_e3b.py --tiny）产出的小检查点（Qwen）。"""
    d = tmp_path_factory.mktemp("tiny_train")
    p = run_script("train_e3b", "--tiny", "--model", FAMILIES[0], "--supervision", "grid", "--seed", "0", "--out", str(d))
    assert p.returncode == 0, p.stderr[-3000:]
    return d / "e3b" / FAMILIES[0] / "seed0" / "ckpt" / "final" / "model"


def test_cli_tiny_and_resume(tiny_ckpt, tmp_path):
    out = tmp_path / "e"
    args = ["--tiny", "--checkpoint", str(tiny_ckpt), "--model", FAMILIES[0], "--seed", "0", "--out", str(out)]
    p = run_script("eval_e3b", *args)
    assert p.returncode == 0, p.stderr[-3000:]
    assert p.stdout.count("single/guided") == 2
    mdir = out / "eval" / "e3b" / FAMILIES[0] / "seed0"
    mc = json.loads((mdir / "merge_check.json").read_text())
    assert mc["pass"] is True and mc["checkpoint"]["manifest_sha256"] and list(mc["checks"]) == ["merge"]
    for task in ("refseg", "depth"):
        leaf = mdir / task
        m = json.loads((leaf / "metrics.json").read_text())
        assert m["n"] == (6 if task == "refseg" else 1) and m["complete"] is True
        assert set(m["compute"]) == {"single", "multi_K1", "multi_K2", "multi_K4", "matched_K1", "matched_K2", "matched_K4"}
    before = {t: {f: (mdir / t / f).read_bytes() for f in DETERMINISTIC} for t in ("refseg", "depth")}
    p = run_script("eval_e3b", *args)
    assert p.returncode == 2 and "--resume" in p.stderr
    p = run_script("eval_e3b", *args, "--resume")
    assert p.returncode == 0, p.stderr[-3000:]
    assert "复用已通过" in p.stderr
    assert {t: {f: (mdir / t / f).read_bytes() for f in DETERMINISTIC} for t in ("refseg", "depth")} == before
    # 命令行的拒绝条件
    assert run_script("eval_e3b", *[x if x != FAMILIES[0] else "qwen3.5-2b" for x in args], "--resume").returncode == 2
    p = run_script("eval_e3b", "--checkpoint", str(tiny_ckpt), "--model", FAMILIES[0], "--seed", "0", "--out", str(tmp_path / "x"))
    assert p.returncode == 2 and "--data-root" in p.stderr


def bad_checkpoint(root: Path) -> Path:
    """病态的 LoRA：A 的两行几乎相同（差 1e-6 的方向）、B 的两列相反且放大 1e7。真实的 ΔW 很小，但合并（先算 B·A）与不合并
    （先算 A·x）两条路径的中间量都很大、舍入误差各不相同，fp32 上也分家：合并核对判不过（教师强制与生成都不一致）。"""
    ad = _tiny(FAMILIES[0])
    add_vocab(ad, CFG.vocab)
    ad.inject_lora(r=2, lora_alpha=2, init_lora_weights=False)
    mods = dict(ad.model.named_modules())
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name in ad.lora_target_modules():
            m = mods[name]
            a_w, b_w = m.lora_A["default"].weight, m.lora_B["default"].weight
            u = torch.randn(a_w.shape[1], generator=g)
            a_w.copy_(torch.stack([u, u + 1e-6 * torch.randn(a_w.shape[1], generator=g)]))
            v = torch.randn(b_w.shape[0], generator=g)
            b_w.copy_(torch.stack([v, -v], dim=1) * 1e7)
    return ad.save_trained(root / "bad_ck")


def test_bad_checkpoint_refused(tmp_path):
    ck = bad_checkpoint(tmp_path)
    ev = _load_script("eval_e3b")
    out = tmp_path / "e"
    code = ev.main(["--tiny", "--checkpoint", str(ck), "--model", FAMILIES[0], "--seed", "0", "--out", str(out)])
    assert code == 1
    mdir = out / "eval" / "e3b" / FAMILIES[0] / "seed0"
    doc = json.loads((mdir / "merge_check.json").read_text())
    assert doc["pass"] is False and doc["failures"]["merge"]
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json"]  # 没有任何任务目录
    # 训练后模型的 T 文本评测同样拒绝
    r1 = _load_script("run_e1")
    out2 = tmp_path / "t"
    code = r1.main(["--tiny", "--model", FAMILIES[0], "--task", "grid16", "--checkpoint", str(ck), "--checkpoint-label",
                    "t_text/e3b/seed0", "--out", str(out2)])
    assert code == 1
    assert sorted(p.name for p in (out2 / "t_text" / "e3b" / "seed0" / FAMILIES[0]).iterdir()) == ["merge_check.json"]


# ---------------------------------------------------------------------------
# 训练后模型的 T 文本评测：run_e1.py --tiny --checkpoint（DEV-0030 §2）
# ---------------------------------------------------------------------------


def test_run_e1_tiny_checkpoint_all_tasks(tiny_ckpt, tmp_path):
    from src.e1.prompts import TASKS

    out = tmp_path / "t"
    args = ["--tiny", "--model", FAMILIES[0], "--task", "all", "--checkpoint", str(tiny_ckpt), "--checkpoint-label",
            "t_text/e3b/seed0", "--out", str(out), "--limit", "2"]  # 每个任务 2 条：全部任务跑通，省时间
    p = run_script("run_e1", *args)
    assert p.returncode == 0, p.stderr[-3000:]
    base = out / "t_text" / "e3b" / "seed0" / FAMILIES[0]
    sha = hashlib.sha256((tiny_ckpt / "manifest.json").read_bytes()).hexdigest()
    assert json.loads((base / "merge_check.json").read_text())["pass"] is True
    for task in TASKS:
        leaf = base / task
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["checkpoint"]["manifest_sha256"] == sha and cfg["checkpoint"]["n_grid_tokens"] == 67
        assert cfg["checkpoint"]["merge_check"]["criteria_version"] == "v1"
        assert cfg["batch_protocol"]["batch_size"] == 4  # --tiny 的批大小；正式运行取冻结的 64，协议不变
        m = json.loads((leaf / "metrics.json").read_text())
        d = m["diagnostics"]["grid_tokens_in_output"]
        recs = _jsonl(leaf / "predictions.jsonl")
        assert d["n"] == m["n"] == len(recs) and d["n_rows"] == sum(1 for r in recs if r["n_grid_tokens"] > 0)
        assert m["three_columns"] is not None and m["complete"] is True
    # 续跑：复用已通过的合并核对，结果不变
    before = (base / "grid16" / "metrics.json").read_bytes()
    p = run_script("run_e1", *args, "--resume")
    assert p.returncode == 0 and "复用已通过" in p.stderr
    assert (base / "grid16" / "metrics.json").read_bytes() == before
    # base 运行：没有 checkpoint 键、没有 n_grid_tokens、没有网格 token 的诊断（与原来逐字节相同的前提）
    p = run_script("run_e1", "--tiny", "--model", FAMILIES[0], "--task", "grid16", "--out", str(tmp_path / "b"))
    assert p.returncode == 0, p.stderr[-3000:]
    leaf = tmp_path / "b" / FAMILIES[0] / "grid16"
    assert "checkpoint" not in yaml.safe_load((leaf / "config.yaml").read_text())
    assert all("n_grid_tokens" not in r for r in _jsonl(leaf / "predictions.jsonl"))
    assert "grid_tokens_in_output" not in json.loads((leaf / "metrics.json").read_text())["diagnostics"]


def test_run_e1_checkpoint_arg_rules(tiny_ckpt, tmp_path):
    r1 = _load_script("run_e1")
    base = ["--tiny", "--model", FAMILIES[0], "--task", "grid16", "--out", str(tmp_path)]
    for extra in (["--checkpoint", str(tiny_ckpt)],  # 没有标签
                  ["--checkpoint", str(tiny_ckpt), "--checkpoint-label", "../x"],
                  ["--checkpoint-label", "a/b"],  # 没有 --checkpoint
                  ["--checkpoint", str(tiny_ckpt), "--checkpoint-label", "a", "--acceptance"],
                  ["--checkpoint", str(tmp_path / "nope"), "--checkpoint-label", "a"]):
        with pytest.raises(SystemExit) as e:
            r1.parse_args(base + extra)
        assert e.value.code == 2, extra
    a = r1.parse_args(base + ["--checkpoint", str(tiny_ckpt), "--checkpoint-label", "t_text/e3b/seed0", "--merge-check-n", "2"])
    assert a.checkpoint_label == "t_text/e3b/seed0" and a.merge_check_n == 2
    # 检查点的构造方式与 --model 不符：报错（退出码 2），不评测
    code = r1.main(["--tiny", "--model", "qwen3.5-2b", "--task", "grid16", "--checkpoint", str(tiny_ckpt), "--checkpoint-label",
                    "x", "--out", str(tmp_path / "y")])
    assert code == 2 and not (tmp_path / "y" / "x").exists()
