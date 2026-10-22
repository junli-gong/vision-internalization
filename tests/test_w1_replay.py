"""DEV-0057：W1 的重放与错误分析（src/w1/replay.py、scripts/w1_replay.py）。

- 重放：真工作台 + 脚本化的假回路（tests._w1_fakes.ScriptedChatAdapter，经 src/w1/loop.run_episodes 生成记录，JSON 往返一次，同
  predictions.jsonl）；重放的返回文字与提交的 mask 与记录逐字、逐位一致；每步每个 mask 的 IoU 与独立算的相同；种子框、第一次贴边、
  最好的 mask、IoU 变化的参照都按规则取。
- 不一致会被标记：改返回文字、改提交的 RLE、改得分；工具清单或原图尺寸不符直接报错。
- 分类：四类的先后与边界值（恰好落在阈值上算达到阈值）、F1 / F2、重放不一致的不进分类、IoU 变化的计数。
- 命令行：run_w1.py --tiny 真跑一小段（回路用假策略适配器 + 真工作台；另一种设置用小型随机模型），加上"自己的框"与 W1-F 两条基线；
  w1_replay.py 重放它们 0 条不一致、输出文件齐全、--workers 2 与 1 逐字节相同、只分类的模式可重跑、改过的记录被标出。
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.records import Sample
from src.e1.scoring import gt_mask
from src.executors.geometry import box_iou, norm1000_to_abs
from src.executors.raster import mask_to_rle, rle_to_mask
from src.metrics.seg import mask_inter_union, sample_iou
from src.w1 import replay as R
from src.w1.config import load_w1_config
from src.w1.loop import run_episodes
from src.w1.scoring import score_episode
from src.w1.workbench import GRABCUT_ITERATIONS, Workbench, grabcut_mask
from tests._w1_fakes import ScriptedChatAdapter, turn

REPO = Path(__file__).resolve().parents[1]
CFG = load_w1_config()
W, H = 80, 60
OBJ = (20, 15, 50, 40)  # 目标：列 20..49、行 15..39（像素边缘 xyxy）
DISTRACTOR = (55, 5, 75, 20)
SEED_BOX = "[190, 170, 690, 750]"  # 目标外扩几个像素（0–1000）
BAD_BOX = "[660, 50, 960, 360]"  # 框住干扰物


def scene() -> Image.Image:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 50, (H, W, 3), dtype=np.uint8)
    x0, y0, x1, y1 = OBJ
    arr[y0:y1, x0:x1] = (220, 60, 40)
    x0, y0, x1, y1 = DISTRACTOR
    arr[y0:y1, x0:x1] = (40, 80, 230)
    return Image.fromarray(arr)


def sample(key: str, path: str = "/nonexistent.png") -> Sample:
    x0, y0, x1, y1 = (float(v) for v in OBJ)
    poly = [[x0, y0, x1, y0, x1, y1, x0, y1]]
    return Sample(id=f"t/val/{key}", task="refexp", images=(path,), image_keys=(f"k:{key}",), fields={"expression": f"object {key}"},
                  gt={"box": [x0, y0, x1, y1], "mask": {"polygons": poly, "height": H, "width": W}}, source="t", split="val")


SCRIPTS = {
    # 好种子 → refine → 腐蚀（改坏）→ 一次报错的调用 → 交腐蚀后的
    "a": [turn(calls=[("grabcut", {"box": SEED_BOX, "out": "a"})]),
          turn(calls=[("refine", {"name": "a", "out": "b"}), ("show", {"name": "b"})]),
          turn(calls=[("morph", {"name": "b", "op": "erode", "radius": "5", "out": "c"})]),
          turn(calls=[("select_superpixels", {"name": "s", "ids": "[1]"})]),
          turn(calls=[("submit_mask", {"name": "c"})])],
    # 没有 grabcut：多边形 → 超像素 → 选超像素 → 没提交（最后一轮只回文字）
    "b": [turn(calls=[("draw_polygon", {"name": "p", "points": "[[260, 260], [610, 260], [610, 650], [260, 650]]"})]),
          turn(calls=[("superpixels", {"n_segments": "30"})]),
          turn(calls=[("select_superpixels", {"name": "s", "points": "[[400, 450]]"})]),
          turn(calls=[("components", {"name": "s"})]),
          turn(calls=[("zoom", {"x1": "0", "y1": "0", "x2": "500", "y2": "500"})]),
          turn(text="Done.")],
    # 种子放在干扰物上，交上去
    "c": [turn(calls=[("grabcut", {"box": BAD_BOX, "out": "m"})]), turn(calls=[("submit_mask", {"name": "m"})])],
}


def make_records() -> tuple[dict[str, dict], dict[str, Sample], Image.Image]:
    im = scene()
    keys = list(SCRIPTS)
    ss = [sample(k) for k in keys]
    p = CFG.protocol("full")
    ad = ScriptedChatAdapter({f"object {k} " : v for k, v in SCRIPTS.items()})
    recs = run_episodes(ad, ss, "refseg", p, CFG.loop_prompts(p), batch_size=4, workbench_factory=Workbench, images=[im] * len(ss))
    out = {}
    for k, s, rec in zip(keys, ss, recs):
        row = {"id": s.id, **{a: b for a, b in rec.to_json().items() if a != "id"}, "score": score_episode("refseg", s, rec)}
        out[k] = json.loads(json.dumps(row))  # 同 predictions.jsonl：JSON 往返一次
    return out, dict(zip(keys, ss)), im


@pytest.fixture(scope="module")
def recs():
    return make_records()


def replay(rec, im, s, **kw):
    return R.replay_episode(rec, im, gt_mask(s), gt_box=s.gt["box"], workbench_factory=Workbench,
                            workbench_kw={"disabled_groups": frozenset()}, **kw)


def iou_of(m, s) -> float:
    return sample_iou(*mask_inter_union(m, gt_mask(s)))


# ---------------------------------------------------------------------------
# 重放
# ---------------------------------------------------------------------------


def test_scripted_records_cover_the_cases(recs):
    r, _, _ = recs
    assert r["a"]["failure"] is None and r["a"]["answer"]["source"] == "submit_mask"
    assert r["b"]["failure"] == "not_submitted"
    assert r["c"]["failure"] is None
    errs = [c for rd in r["a"]["rounds"] for c in rd["tool_calls"] if c["executed"] and c["result"]["error"]]
    assert [c["name"] for c in errs] == ["select_superpixels"]  # 报错的调用也重放，文字一样


def test_replay_matches_record_exactly(recs):
    r, ss, im = recs
    for k in SCRIPTS:
        row = replay(r[k], im, ss[k])
        assert row["replay"] == {"mismatch": False, "reasons": [], "n_calls": len(R.executed_calls(r[k])), "n_image_mismatch": 0}, k
        assert row["tools"] == [c["name"] for c in R.executed_calls(r[k])]
        assert (row["final_inter"], row["final_union"]) == (r[k]["score"]["inter"], r[k]["score"]["union"])
        assert row["final_iou"] == pytest.approx(r[k]["score"]["iou"], abs=0)
    again = replay(r["a"], im, ss["a"])
    assert json.dumps(again, sort_keys=True) == json.dumps(replay(r["a"], im, ss["a"]), sort_keys=True)  # 重放本身确定


def test_replay_per_step_iou_and_derived_fields(recs):
    r, ss, im = recs
    s = ss["a"]
    row = replay(r["a"], im, s, own_box=[19.0, 14.0, 52.0, 41.0], own_box_iou_gt=0.8,
                 w1f_score={"inter": 700, "union": 760, "iou": 700 / 760}, ownbox_score={"inter": 1, "union": 2, "iou": 0.5})
    rgb = np.asarray(im)
    box_abs = norm1000_to_abs(np.asarray(json.loads(SEED_BOX), dtype=np.float64), width=W, height=H)
    m_a = grabcut_mask(rgb, box_abs=box_abs, iterations=GRABCUT_ITERATIONS)
    c0 = row["calls"][0]
    assert c0["tool"] == "grabcut" and c0["written"] == ["a"] and c0["masks"]["a"] == [iou_of(m_a, s), int(m_a.sum())]
    assert c0["deltas"] == [{"mask": "a", "ref": None, "ref_kind": None, "iou_before": None, "iou_after": iou_of(m_a, s)}]
    # 每一步：mask 的 IoU 与面积与独立重算的工作台状态一致
    wb = Workbench(im.copy())
    for c, call in zip(row["calls"], R.executed_calls(r["a"])):
        wb.call(call["name"], dict(call["arguments"]))
        assert c["masks"] == {n: [iou_of(m, s), int(m.sum())] for n, m in wb.masks.items()}, c["k"]
    d_refine, d_morph = row["calls"][1]["deltas"][0], row["calls"][3]["deltas"][0]
    assert (d_refine["mask"], d_refine["ref"], d_refine["ref_kind"]) == ("b", "a", "input")
    assert (d_morph["mask"], d_morph["ref"], d_morph["ref_kind"]) == ("c", "b", "input")
    allv = [(v[0], c["k"], n) for c in row["calls"] for n, v in c["masks"].items()]
    best = max(v for v, _, _ in allv)
    assert row["best_iou"] == best and (row["best_call"], row["best_mask"]) == next((k, n) for v, k, n in allv if v == best)
    assert row["seed_box"] == json.loads(SEED_BOX) and row["seed_box_source"] == "grabcut_box" and row["seed_call"] == 0
    assert row["seed_box_iou_gt"] == box_iou(box_abs.tolist(), s.gt["box"])
    assert row["seed_box_iou_own"] == box_iou(box_abs.tolist(), [19.0, 14.0, 52.0, 41.0])
    assert (row["first_snap_tool"], row["first_snap_call"], row["first_snap_iou"]) == ("grabcut", 0, iou_of(m_a, s))
    assert row["final_iou"] == iou_of(rle_to_mask(r["a"]["answer"]["mask"]["rle"]), s) and row["submitted_mask"] == "c"
    assert (row["w1f_iou"], row["w1f_inter"], row["w1f_union"], row["ownbox_iou"], row["own_box_iou_gt"]) == (700 / 760, 700, 760, 0.5, 0.8)
    assert row["last_mask"] == "c" and row["last_iou"] == row["final_iou"]
    assert row["best_iou"] > 0.9 and row["best_iou"] - row["final_iou"] >= 0.1  # 这个脚本是"改坏"
    assert R.classify_row(row, R.Thresholds()) == "degraded"


def test_replay_without_grabcut_uses_first_mask_bbox(recs):
    r, ss, im = recs
    s = ss["b"]
    row = replay(r["b"], im, s)
    pts = np.asarray(json.loads("[[260, 260], [610, 260], [610, 650], [260, 650]]"), dtype=np.float64)
    wb = Workbench(im.copy())
    wb.call("draw_polygon", {"name": "p", "points": json.dumps(pts.tolist())})
    bb = R.mask_bbox_abs(wb.masks["p"])
    assert row["seed_box_source"] == "mask_bbox:draw_polygon" and row["seed_call"] == 0
    assert row["seed_box"] == R.abs_to_norm1000(bb, W, H)
    assert row["seed_box_iou_gt"] == pytest.approx(box_iou(bb, s.gt["box"]), abs=1e-12)
    assert row["first_snap_tool"] == "select_superpixels"
    d = row["calls"][2]["deltas"][0]
    assert (d["mask"], d["ref"], d["ref_kind"]) == ("s", "p", "working")  # 换了名字：与之前的"当前结果"比
    assert row["failure"] == "not_submitted" and row["final_iou"] == 0.0 and row["final_inter"] == 0
    assert row["best_iou"] > 0 and row["best_mask"] in ("p", "s") and row["last_mask"] == "s"
    assert R.classify_row(row, R.Thresholds()) == "no_submission"


def test_bad_seed_record(recs):
    r, ss, im = recs
    row = replay(r["c"], im, ss["c"])
    assert row["seed_box_iou_gt"] < 0.5 and row["best_iou"] - row["final_iou"] < 0.1
    assert R.classify_row(row, R.Thresholds()) == "bad_seed"


def test_mismatches_are_flagged(recs):
    r, ss, im = recs
    rec = copy.deepcopy(r["a"])
    rec["rounds"][1]["tool_calls"][0]["result"]["text"] += " "
    assert replay(rec, im, ss["a"])["replay"]["reasons"] == ["text@1:refine"]
    rec = copy.deepcopy(r["a"])
    m = rle_to_mask(rec["answer"]["mask"]["rle"])
    m[0, 0] = ~m[0, 0]
    rec["answer"]["mask"]["rle"] = mask_to_rle(m)
    row = replay(rec, im, ss["a"])
    assert row["replay"]["mismatch"] and "submission" in row["replay"]["reasons"]
    rec = copy.deepcopy(r["a"])
    rec["score"]["inter"] += 1
    assert replay(rec, im, ss["a"])["replay"]["reasons"] == ["score"]
    rec = copy.deepcopy(r["b"])  # 没提交的记录：重放却交了 mask
    rec["rounds"][0]["tool_calls"].append({**rec["rounds"][0]["tool_calls"][0], "name": "submit_mask", "arguments": {"name": "p"},
                                           "result": {"text": "Submitted mask 'p' (99 px) as the final answer.", "error": None,
                                                      "images": []}})
    assert set(replay(rec, im, ss["b"])["replay"]["reasons"]) >= {"submission"}
    rec = copy.deepcopy(r["a"])  # 没执行的调用不重放
    rec["rounds"][0]["tool_calls"].append({"name": "delete_mask", "arguments": {"name": "a"}, "raw": "", "counted": False,
                                           "executed": False, "refused": "tool_cap", "result": None})
    assert replay(rec, im, ss["a"])["replay"]["mismatch"] is False


def test_preconditions_raise(recs):
    r, ss, im = recs
    rec = copy.deepcopy(r["a"])
    rec["tools_sha256"] = "0" * 64
    with pytest.raises(R.ReplayError, match="工具清单"):
        replay(rec, im, ss["a"])
    with pytest.raises(R.ReplayError, match="image_size"):
        replay(r["a"], im.resize((W + 1, H)), ss["a"])
    with pytest.raises(R.ReplayError, match="工具清单"):  # 关了组的工作台：清单不同
        R.replay_episode(r["a"], im, gt_mask(ss["a"]), gt_box=ss["a"].gt["box"], workbench_factory=Workbench,
                         workbench_kw={"disabled_groups": frozenset({"eye"})})


def test_workbench_kwargs_follow_run_identity():
    cfg = {"protocol": {"disabled_groups": ["eye"]}, "workbench": {"options": {"view_frames": False}}}
    assert R.workbench_kwargs(cfg) == {"disabled_groups": frozenset({"eye"})}  # 默认值不传（同 W1Config.workbench_options）
    cfg["workbench"]["options"]["view_frames"] = True
    assert R.workbench_kwargs(cfg) == {"disabled_groups": frozenset({"eye"}), "view_frames": True}
    cfg["workbench"]["options"]["bogus"] = 1
    with pytest.raises(R.ReplayError):
        R.workbench_kwargs(cfg)


def test_seed_box_with_view_frames_converts_to_full_image():
    wb = Workbench(scene(), view_frames=True)
    wb.call("zoom", {"x1": "0", "y1": "0", "x2": "500", "y2": "500"})  # 视图 1 = 原图左上四分之一
    args = {"box": "[100, 200, 900, 800]", "view": "1", "out": "m"}
    assert R.seed_from_grabcut(wb, args) == [50.0, 100.0, 450.0, 400.0]
    assert R.seed_from_grabcut(Workbench(scene()), {"box": "[1, 2, 3, 4]", "out": "m"}) == [1.0, 2.0, 3.0, 4.0]
    assert R.seed_from_grabcut(wb, {"mask": "m", "out": "n"}) is None


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------


def row(i, *, final=0.3, best=None, seed=0.8, w1f=0.6, failure=None, mismatch=False, calls=()):
    best = final if best is None else best
    return {"id": f"r{i}", "failure": failure, "final_iou": final, "best_iou": best, "seed_box_iou_gt": seed,
            "seed_box": None if seed is None else [0, 0, 1, 1], "seed_box_iou_own": 0.7, "own_box_iou_gt": 0.9,
            "seed_box_source": None if seed is None else "grabcut_box", "w1f_iou": w1f, "w1f_inter": None if w1f is None else 6,
            "w1f_union": None if w1f is None else 10, "final_inter": int(final * 10), "final_union": 10, "best_inter": int(best * 10),
            "best_union": 10, "best_mask": "m", "last_inter": 2, "last_union": 10, "replay": {"mismatch": mismatch, "reasons": [],
                                                                                         "n_image_mismatch": 0},
            "calls": list(calls)}


@pytest.mark.parametrize("kw, want", [
    (dict(failure="no_answer", final=0.0, best=0.9, seed=0.1), "no_submission"),  # 先判没有提交
    (dict(failure="wrong_submission", final=0.0, best=0.0, seed=None), "no_submission"),
    (dict(final=0.5, best=0.6, seed=0.1), "degraded"),  # 0.6 − 0.5 = 0.0999…：恰好在阈值上算改坏，先于种子
    (dict(final=0.2, best=0.3, seed=0.1), "degraded"),
    (dict(final=0.5, best=0.5999, seed=0.1), "bad_seed"),  # 差一点不到 0.10
    (dict(final=0.2, best=0.25, seed=0.4999), "bad_seed"),
    (dict(final=0.2, best=0.25, seed=None), "bad_seed"),  # 没有种子算种子放错
    (dict(final=0.2, best=0.25, seed=0.5), "snap"),  # 恰好 0.5 不算放错
    (dict(final=0.2, best=0.25, seed=0.9), "snap"),
])
def test_class_order_and_boundaries(kw, want):
    assert R.classify_row(row(0, **kw), R.Thresholds()) == want


def test_failure_definitions_and_boundaries():
    th = R.Thresholds()
    assert R.is_failure(row(0, final=0.5), "F1", th) is False and R.is_failure(row(0, final=0.4999), "F1", th) is True
    assert R.is_failure(row(0, final=0.6, w1f=0.65), "F2", th) is False  # 0.65 − 0.05 = 0.6000…01：恰好在阈值上不算失败
    assert R.is_failure(row(0, final=0.5999, w1f=0.65), "F2", th) is True
    assert R.is_failure(row(0, final=0.0, w1f=None), "F2", th) is None
    with pytest.raises(ValueError):
        R.is_failure(row(0), "F3", th)
    t2 = R.Thresholds(f1_iou=0.3, degrade=0.2, seed_iou=0.7)
    assert R.is_failure(row(0, final=0.35), "F1", t2) is False and R.classify_row(row(0, final=0.5, best=0.6, seed=0.6), t2) == "bad_seed"
    with pytest.raises(ValueError):
        R.Thresholds(degrade=-0.1)


def test_classify_document():
    rows = [row(0, failure="no_answer", final=0.0, best=0.7, seed=0.2, w1f=0.6),
            row(1, final=0.3, best=0.8, seed=0.9, w1f=0.7),
            row(2, final=0.2, best=0.2, seed=0.1, w1f=0.1),  # F1 失败；F2 不失败（W1-F 也差）
            row(3, final=0.4, best=0.4, seed=0.9, w1f=0.9),
            row(4, final=0.9, best=0.9, seed=0.9, w1f=0.8),  # 不失败
            row(5, final=0.0, failure="no_answer", w1f=None),  # F2 不计
            row(6, final=0.1, mismatch=True)]
    doc = R.classify(rows, R.Thresholds(), source={"run": "x"})
    assert doc["n_rows"] == 7 and doc["n_replay_mismatch"] == 1 and doc["n_analyzed"] == 6
    assert doc["replay_mismatch"] == [{"id": "r6", "reasons": []}]
    f1, f2 = doc["failure_defs"]["F1"], doc["failure_defs"]["F2"]
    assert f1["n_failed"] == 5 and {c: x["n"] for c, x in f1["classes"].items()} == {"no_submission": 2, "degraded": 1, "bad_seed": 1,
                                                                                     "snap": 1}
    assert f1["classes"]["degraded"]["ids"] == ["r1"] and f1["classes"]["no_submission"]["share"] == pytest.approx(2 / 5)
    assert f1["classes"]["no_submission"]["median_best_iou"] == pytest.approx(0.35)
    assert f2["n_failed"] == 3 and f2["n_excluded"] == 1
    assert {c: x["n"] for c, x in f2["classes"].items()} == {"no_submission": 1, "degraded": 1, "bad_seed": 0, "snap": 1}
    fb = f1["no_submission_fallback"]
    assert fb["n"] == 2 and fb["ciou_if_best"] == pytest.approx((7 + 0) / 20) and fb["ciou_if_last"] == pytest.approx(4 / 20)
    assert fb["ciou_w1f"] is None  # 有一条没有 W1-F 记录
    assert doc["seed_box"]["paired"]["n"] == 6 and doc["seed_box"]["seed_box_iou_own"]["quantiles"]["p50"] == pytest.approx(0.7)
    assert doc["seed_box"]["paired"]["median_diff_seed_minus_own"] == pytest.approx(-0.05)  # 差 −0.7、0、−0.8、0、0、−0.1
    assert doc["seed_box"]["paired"]["n_seed_worse"] == 3 and doc["seed_box"]["paired"]["acc50_seed"] == pytest.approx(4 / 6)
    md = R.render_md(doc)
    for c in R.CLASSES:
        assert R.CLASS_LABELS[c] in md
    assert "F1：final_iou < 0.5" in md and "不一致 1 条" in md


def test_tool_iou_changes_counts():
    def call(tool, deltas, ok=True):
        return {"tool": tool, "ok": ok, "written": [d[0] for d in deltas],
                "deltas": [{"mask": m, "ref": ref, "ref_kind": kind, "iou_before": b, "iou_after": a} for m, ref, kind, b, a in deltas]}

    calls = [call("grabcut", [("a", None, None, None, 0.6)]), call("refine", [("b", "a", "input", 0.6, 0.7)]),
             call("refine", [("b", "b", "input", 0.7, 0.7)]), call("morph", [("c", "b", "input", 0.7, 0.4)]),
             call("show", []), call("show", [], ok=False)]
    out = R.tool_iou_changes([row(0, calls=calls)], R.Thresholds())
    assert out["refine"] == {"n_executed": 2, "n_ok": 2, "n_wrote": 2, "n_masks_written": 2, "up": 1, "down": 0, "same": 1, "no_ref": 0,
                             "median_delta": pytest.approx(0.05), "mean_delta": pytest.approx(0.05), "median_iou_after": 0.7,
                             "by_ref": {"input": {"n": 2, "median_delta": pytest.approx(0.05)}}}
    assert out["grabcut"]["no_ref"] == 1 and out["grabcut"]["up"] == 0 and out["morph"]["down"] == 1
    assert out["show"]["n_executed"] == 2 and out["show"]["n_ok"] == 1 and out["show"]["n_wrote"] == 0
    loose = R.tool_iou_changes([row(0, calls=calls)], R.Thresholds(delta_eps=0.2))
    assert loose["refine"]["same"] == 2 and loose["morph"]["down"] == 1


# ---------------------------------------------------------------------------
# 命令行：run_w1.py --tiny 真跑一小段，再重放
# ---------------------------------------------------------------------------

QWEN = "qwen3.5-0.8b"
REPLAY = REPO / "scripts" / "w1_replay.py"
RUN_W1 = REPO / "scripts" / "run_w1.py"


def sh(script, args, tmp, ok=True):
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w1replay")
    run = tmp / "0999_w1_tiny"
    base = ["--tiny", "--model", QWEN, "--task", "refseg", "--split", "devcheck", "--out", str(run)]
    sh(RUN_W1, base + ["--setting", "full", "--adapter-factory", "tests._w1_fakes:make_policy_adapter"], tmp)  # 假策略 + 真工作台
    sh(RUN_W1, base + ["--setting", "no_eye"], tmp)  # 小型随机模型 + 真工作台（关掉 eye 组）
    sh(RUN_W1, base + ["--setting", "baseline_ownbox"], tmp)  # 先按 E1 的协议跑框
    sh(RUN_W1, base + ["--setting", "w1f"], tmp)
    return {"tmp": tmp, "run": run}


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_cli_replays_tiny_run_without_mismatch(tiny_run):
    tmp, run = tiny_run["tmp"], tiny_run["run"]
    out = tmp / "replay_full"
    p = sh(REPLAY, ["--run", str(run), "--out", str(out), "--classify"], tmp)
    assert "replay_mismatch=0" in p.stdout
    for f in ("replay.jsonl", "replay_meta.json", "logs/replay_timing.jsonl", "error_analysis.json", "error_analysis.md"):
        assert (out / f).exists(), f
    rows = jsonl(out / "replay.jsonl")
    preds = jsonl(run / QWEN / "refseg" / "full" / "devcheck" / "predictions.jsonl")
    assert [r["id"] for r in rows] == [r["id"] for r in preds] and len(rows) == 3
    assert all(not r["replay"]["mismatch"] for r in rows)
    assert sum(r["n_executed"] for r in rows) == sum(x["n_executed"] for x in preds) > 0
    assert any("grabcut" in r["tools"] for r in rows)
    meta = json.loads((out / "replay_meta.json").read_text())
    assert meta["n_replay_mismatch"] == 0 and meta["source"]["tiny"] is True and meta["source"]["setting"] == "full"
    assert set(meta["source"]["inputs"]) == {"predictions", "config", "w1f", "ownbox", "box"}
    doc = json.loads((out / "error_analysis.json").read_text())
    assert doc["n_analyzed"] == 3 and doc["source"]["replay_sha256"] == meta["replay_sha256"]
    # --workers 2 逐字节相同；已有输出不给 --force 拒绝覆盖
    out2 = tmp / "replay_full_w2"
    sh(REPLAY, ["--run", str(run), "--out", str(out2), "--workers", "2"], tmp)
    assert (out2 / "replay.jsonl").read_bytes() == (out / "replay.jsonl").read_bytes()
    bad = sh(REPLAY, ["--run", str(run), "--out", str(out2)], tmp, ok=False)
    assert bad.returncode == 2 and "--force" in bad.stderr
    # 只分类：只读 replay.jsonl，结果与一起跑的相同；换阈值写进报告
    before = (out / "error_analysis.json").read_bytes()
    sh(REPLAY, ["--classify", "--out", str(out)], tmp)
    assert (out / "error_analysis.json").read_bytes() == before
    sh(REPLAY, ["--classify", "--out", str(out), "--f1-iou", "0.3"], tmp)
    assert json.loads((out / "error_analysis.json").read_text())["thresholds"]["f1_iou"] == 0.3


def test_cli_replays_real_tiny_model_setting(tiny_run):
    tmp, run = tiny_run["tmp"], tiny_run["run"]
    out = tmp / "replay_noeye"
    p = sh(REPLAY, ["--run", str(run), "--out", str(out), "--setting", "no_eye", "--classify"], tmp)
    assert "replay_mismatch=0" in p.stdout
    meta = json.loads((out / "replay_meta.json").read_text())
    assert meta["source"]["workbench_kwargs"] == {"disabled_groups": ["eye"]} and meta["n_rows"] == 3


def test_cli_flags_a_tampered_record(tiny_run):
    tmp, run = tiny_run["tmp"], tiny_run["run"]
    run2 = tmp / "0998_w1_tampered"
    shutil.copytree(run, run2)
    pred = run2 / QWEN / "refseg" / "full" / "devcheck" / "predictions.jsonl"
    recs = jsonl(pred)
    c = next(c for rd in recs[0]["rounds"] for c in rd["tool_calls"] if c["executed"])
    c["result"]["text"] += "!"
    pred.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    out = tmp / "replay_tampered"
    p = sh(REPLAY, ["--run", str(run2), "--out", str(out), "--classify"], tmp)
    assert "replay_mismatch=1" in p.stdout
    doc = json.loads((out / "error_analysis.json").read_text())
    assert doc["n_replay_mismatch"] == 1 and doc["n_analyzed"] == 2 and doc["replay_mismatch"][0]["id"] == recs[0]["id"]
    # 大文件清单里的 sha256 对不上：报错
    (run2 / "LARGE_FILES.manifest.tsv").write_text("path\tbytes\tsha256\tmtime_epoch\n"
                                                   f"{QWEN}/refseg/full/devcheck/predictions.jsonl\t1\t{'0' * 64}\t0\n")
    bad = sh(REPLAY, ["--run", str(run2), "--out", str(tmp / "replay_manifest")], tmp, ok=False)
    assert bad.returncode == 2 and "LARGE_FILES.manifest.tsv" in bad.stderr


def test_cli_argument_rules(tmp_path):
    p = sh(REPLAY, ["--out", str(tmp_path)], tmp_path, ok=False)
    assert p.returncode == 2 and "--classify" in p.stderr
    p = sh(REPLAY, ["--classify", "--out", str(tmp_path)], tmp_path, ok=False)
    assert p.returncode == 2 and "replay.jsonl" in p.stderr
