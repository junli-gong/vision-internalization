"""DEV-0018：E5 的逐条解析与评分、三列汇总、两级门槛（src/e5/scoring.py）。
DEV-0028：派生基线 copy_convex、D59-4 的门槛（三条基线、配对 CI）、预算强制收尾只解析作答段、外部预测对当前 mask 的 IoU。"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from src.e5 import scoring as SC
from src.e5.protocol import eval_items
from src.e5.runner import make_sample
from src.grid.codec import grid_to_text, relief_grid_to_text
from src.synth.api import CURRENT_MASK_BASELINE, current_mask
from src.synth.raster import rasterize
from src.synth.relief import (
    CONVEX_BASELINE,
    CONVEX_PRIOR_AMPLITUDE,
    RELIEF_TARGET_KEY,
    RELIEF_TASKS,
    convex_prior_codes,
    relief_score,
)


@dataclass
class _Res:  # GenerationResult 的替身：parse_generation 只看这三个字段
    text: str
    stopped: bool | None = True
    thinking_unclosed: bool = False


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def test_parse_binary_and_relief_grids():
    g = np.zeros((32, 32), np.uint8)
    g[4:10, 6:12] = 1
    p = SC.parse_answer("binary_grid", _Res("Sure:\n" + grid_to_text(g)), 512, 512)
    assert p.valid and np.array_equal(p.grid, g) and p.mask.shape == (512, 512) and p.mask.sum() == 36 * 256
    codes = g * 17
    p = SC.parse_answer("relief_grid", _Res(relief_grid_to_text(codes)), 512, 512)
    assert p.valid and np.array_equal(p.relief, codes) and np.array_equal(p.grid, g)
    assert SC.parse_answer("binary_grid", _Res(grid_to_text(g), stopped=False), 512, 512).failure == "truncated"
    assert SC.parse_answer("relief_grid", _Res("", thinking_unclosed=True), 512, 512).failure == "thinking_unclosed"
    assert SC.parse_answer("binary_grid", _Res("I cannot do that."), 512, 512).failure == "no_grid"


def test_parse_polygons_and_integer_rasterization():
    p = SC.parse_answer("polygon", _Res('Thinking done.\n```json\n[[250, 250], [500, 250], [500, 750], [250, 750]]\n```'), 512, 512)
    assert p.valid and len(p.polygons) == 1
    want = rasterize([(128, 128), (256, 128), (256, 384), (128, 384)], 512, 512)  # 0–1000 → 像素：×512/1000
    assert np.array_equal(p.mask, want) and want.sum() == 128 * 256
    two = SC.rasterize_polygons([[(0, 0), (10, 0), (10, 10), (0, 10)], [(20, 20), (30, 20), (30, 30)]], 64, 64)
    assert two.sum() == 100 + int(rasterize([(20, 20), (30, 20), (30, 30)], 64, 64).sum())
    q = SC.rasterize_polygons([[(1.3, 1.7), (9.1, 1.7), (9.1, 8.2)]], 16, 16)  # 不在网格上的顶点四舍五入到 1/256 px
    assert np.array_equal(q, rasterize([(round(1.3 * 256) / 256, round(1.7 * 256) / 256), (round(9.1 * 256) / 256, round(1.7 * 256) / 256),
                                        (round(9.1 * 256) / 256, round(8.2 * 256) / 256)], 16, 16))
    assert SC.parse_answer("polygon", _Res("no json"), 512, 512).failure == "no_polygon"


# ---------------------------------------------------------------------------
# 逐条评分与生成器元数据一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["rotation3d", "viewpoint", "occlusion", "imagery"])
def test_baseline_scores_match_generator_meta(task):
    samples = [make_sample(it) for it in eval_items(task, "dev", 2)]
    recs, roles = SC.baseline_records(samples)
    for s in samples:
        for name in s.baselines:
            r = next(x for x in recs[name] if x["index"] == s.index)
            assert r["score"]["iou_full"] == s.meta["baseline_iou"][name]["full"]
            assert r["score"]["iou_grid"] == s.meta["baseline_iou"][name]["grid"]
            if task in RELIEF_TASKS:
                assert r["score"]["relief"]["error"] == s.meta["relief"]["baseline_error"][name]["error"]
            else:
                assert r["score"]["relief"] is None
    if task in RELIEF_TASKS:
        assert roles["gt_affine_upper"] == "reference_upper_bound" and roles["copy_flat"] == "trivial"
        assert roles[CONVEX_BASELINE] == "trivial" and set(recs) == set(SC.baseline_names(samples[0]))
        for s in samples:  # 派生基线 copy_convex：mask 同 copy，起伏 = 冻结 A 的凸起先验
            r = next(x for x in recs[CONVEX_BASELINE] if x["index"] == s.index)
            c = next(x for x in recs["copy"] if x["index"] == s.index)
            assert r["score"]["iou_grid"] == c["score"]["iou_grid"]
            want = relief_score(convex_prior_codes(s.baselines["copy"].grid, CONVEX_PRIOR_AMPLITUDE[task]),
                                s.extras[RELIEF_TARGET_KEY]).error
            assert r["score"]["relief"]["error"] == want
    else:
        assert CONVEX_BASELINE not in recs


def test_perfect_and_failed_predictions():
    s = make_sample(eval_items("rotation3d", "dev", 1)[1])
    good = SC.from_relief(s.extras[RELIEF_TARGET_KEY], *s.target_mask.shape)
    sc = SC.score(s, good, relief_expected=True)
    assert sc["iou_grid"] == 1.0 and sc["relief"]["error"] == 0.0 and sc["iou_full"] < 1.0  # 网格回填到全分辨率有量化误差
    bad = SC.score(s, SC.failed("no_grid"), relief_expected=True)
    assert bad["iou_grid"] == 0.0 and bad["iou_full"] == 0.0 and bad["relief"]["error"] is None
    assert bad["relief"]["reason"] == "invalid_prediction"
    mask_only = SC.score(s, SC.from_mask(s.target_mask), relief_expected=False)
    assert mask_only["iou_full"] == 1.0 and mask_only["relief"] is None


# ---------------------------------------------------------------------------
# 三列
# ---------------------------------------------------------------------------


def _rec(i, iou, *, valid=True, relief=None, flat=False, bin=1, headline=True, groups=None):
    rel = None if relief is False else {"error": relief, "reason": None if relief is not None else "invalid_prediction",
                                        "n_eval_cells": 1, "target_flat": flat}
    return {"id": f"rotation3d/test/{i}", "bin": bin, "groups": groups or {"family": "box"}, "headline": headline,
            "valid": valid, "failure": None if valid else "no_grid",
            "score": {"iou_grid": iou, "iou_full": iou, "relief": rel}}


def test_three_columns():
    recs = [_rec(0, 1.0, relief=0.5), _rec(1, 0.5, relief=1.0), _rec(2, 0.0, valid=False, relief=None),
            _rec(3, 0.25, relief=0.7, flat=True)]
    b = SC.iou_block(recs, "iou_grid", 0)
    assert b["n"] == 4 and b["n_valid"] == 3 and b["valid_rate"] == 0.75
    assert b["strict"] == pytest.approx(1.75 / 4) and b["conditional"] == pytest.approx(1.75 / 3)
    r = SC.relief_block(recs, 0)
    assert r["n"] == 3 and r["n_target_flat"] == 1 and r["n_valid"] == 2
    assert r["strict"] == pytest.approx((0.5 + 1.0 + SC.RELIEF_FAIL_VALUE) / 3) and r["conditional"] == pytest.approx(0.75)
    assert SC.RELIEF_FAIL_VALUE == 2.0
    assert SC.iou_block([], "iou_grid", 0)["strict"] is None  # 空集合报 null
    m = SC.aggregate("rotation3d", recs, n_resamples=20)
    assert m["headline"]["mask_iou_grid"]["strict_ci"]["n_resamples"] == 20
    assert m["parse"]["n_failed"] == 1 and set(m["by_bin"]) == {"1"} and set(m["by_group"]["family"]) == {"box"}


def _noisy(n, center, spread, seed):
    """n 个围绕 center 的值（固定种子），让配对 CI 有宽度。"""
    return list(center + spread * np.random.default_rng(seed).uniform(-1, 1, n))


def _base_sets(n, copy_iou, flat_rel, ga_rel, convex_rel, spread=0.05):
    ci, fl, ga, cv = (_noisy(n, copy_iou, spread, 1), _noisy(n, flat_rel, spread, 2), _noisy(n, ga_rel, spread, 3),
                      _noisy(n, convex_rel, spread, 4))
    return {
        "copy": [_rec(i, ci[i], relief=1.5) for i in range(n)],
        "copy_flat": [_rec(i, ci[i], relief=fl[i]) for i in range(n)],
        "copy_convex": [_rec(i, ci[i], relief=cv[i]) for i in range(n)],
        "gt_affine_upper": [_rec(i, 0.95, relief=ga[i]) for i in range(n)],
    }


ROLES = {"copy": "trivial", "copy_flat": "trivial", "copy_convex": "trivial", "gt_affine_upper": "reference_upper_bound"}


def _system(n, iou, rel, spread=0.05, seed=9):
    io, rl = _noisy(n, iou, spread, seed), _noisy(n, rel, spread, seed + 1)
    return [_rec(i, io[i], relief=rl[i]) for i in range(n)]


def test_gate_d59_three_baselines_with_paired_ci():
    assert SC.JUDGMENT_BASELINES == ("copy_flat", "gt_affine_upper", "copy_convex")
    base = _base_sets(40, 0.7, 1.0, 1.2, 0.9)
    g = SC.gate("rotation3d", _system(40, 0.8, 0.6), base, ROLES, n_resamples=200)
    pre, jud = g["prerequisite"], g["judgment"]
    assert pre["strongest"] == "copy" and pre["pass"] is True and pre["diff_ci"]["low"] > 0  # 同分取名字靠前的；参照线不算
    assert {"system", "system_median", "value", "median", "diff", "diff_ci", "diff_median", "system_better_rate"} <= set(pre)
    assert jud["status"] == "evaluated" and jud["n"] == 40 and set(jud["baselines"]) == set(SC.JUDGMENT_BASELINES)
    for name, c in jud["baselines"].items():
        assert c["beats"] is True and c["diff_ci"]["high"] < 0 and c["diff"] < 0 and c["system_better_rate"] == 1.0, name
        assert c["median"] is not None and c["diff_median"] < 0
    assert jud["pass"] is True and g["imagining"] is True
    assert "pass_vs_flat" not in jud and "flat_beats_gt_affine" not in jud  # 照字面的旧判法已作废（D59-4）
    # 胜过平面与真值仿射、但与凸起先验持平：判定条件不通过
    g = SC.gate("rotation3d", _system(40, 0.8, 0.9), base, ROLES, n_resamples=200)
    jb = g["judgment"]["baselines"]
    assert jb["copy_flat"]["beats"] and jb["gt_affine_upper"]["beats"] and jb["copy_convex"]["beats"] is False
    assert g["judgment"]["pass"] is False and g["imagining"] is False
    # 点估计更低、但配对 CI 跨过 0：不算胜过（只看点估计会判过）
    g = SC.gate("rotation3d", _system(8, 0.8, 0.85, spread=0.6), _base_sets(8, 0.7, 1.0, 1.2, 0.9, spread=0.6), ROLES,
                n_resamples=500)
    c = g["judgment"]["baselines"]["copy_convex"]
    assert c["diff"] < 0 and c["diff_ci"]["high"] >= 0 and c["beats"] is False
    # 先决条件：均值更高、但 CI 跨过 0
    g = SC.gate("rotation3d", _system(8, 0.72, 0.5, spread=0.3), _base_sets(8, 0.7, 1.0, 1.2, 0.9, spread=0.3), ROLES,
                n_resamples=500)
    assert g["prerequisite"]["diff"] > 0 and g["prerequisite"]["pass"] is False and g["imagining"] is False
    # 不做 bootstrap（n_resamples = 0）：判不了，记 null
    g = SC.gate("rotation3d", _system(40, 0.8, 0.6), base, ROLES, n_resamples=0)
    assert g["prerequisite"]["pass"] is None and g["judgment"]["pass"] is None and g["imagining"] is None
    # 全部失败：起伏按 2.0 计
    fails = [_rec(i, 0.0, valid=False, relief=None) for i in range(40)]
    g = SC.gate("rotation3d", fails, base, ROLES, n_resamples=200)
    assert g["judgment"]["system"] == 2.0 and g["judgment"]["pass"] is False and g["imagining"] is False
    # 只给 mask 的系统（外部预测）：判定条件不适用，"想象"判不了（先决条件不过时为 false）
    mask_only = [_rec(i, r["score"]["iou_grid"], relief=False) for i, r in enumerate(_system(40, 0.8, 0.6))]
    g = SC.gate("rotation3d", mask_only, base, ROLES, n_resamples=200)
    assert g["judgment"]["status"] == "not_applicable" and g["prerequisite"]["pass"] is True and g["imagining"] is None
    g = SC.gate("rotation3d", [_rec(i, 0.1, relief=False) for i in range(40)], base, ROLES, n_resamples=200)
    assert g["prerequisite"]["pass"] is False and g["imagining"] is False
    # 三条基线缺一条：报错，不静默跳过
    with pytest.raises(ValueError, match="copy_convex"):
        SC.gate("rotation3d", _system(40, 0.8, 0.6), {k: v for k, v in base.items() if k != "copy_convex"},
                {k: v for k, v in ROLES.items() if k != "copy_convex"}, n_resamples=50)


def test_gate_other_tasks_and_alignment():
    base = _base_sets(30, 0.7, 1.0, 1.2, 0.9)
    gb = SC.gate("viewpoint", _system(30, 0.8, 0.6), base, ROLES, n_resamples=200)  # b 的环绕：同一套判定条件
    assert gb["judgment"]["status"] == "evaluated" and gb["imagining"] is True
    recs = [_rec(i, v, relief=False) for i, v in enumerate(_noisy(30, 0.6, 0.05, 5))]
    cbase = {"visible": [_rec(i, v, relief=False) for i, v in enumerate(_noisy(30, 0.5, 0.05, 6))]}
    gc = SC.gate("occlusion", recs, cbase, {"visible": "trivial"}, n_resamples=200)
    assert gc["judgment"]["status"] == "not_applicable" and gc["prerequisite"]["pass"] is True and gc["imagining"] is True
    with pytest.raises(ValueError):
        SC.gate("occlusion", recs, {"visible": cbase["visible"][:2]}, {"visible": "trivial"}, n_resamples=0)
    only_head = SC.gate("occlusion", recs + [_rec(99, 1.0, relief=False, headline=False)], cbase, {"visible": "trivial"},
                        n_resamples=0)
    assert only_head["n"] == 30  # 只用标题口径的样本


def test_budget_view_parses_only_the_answer():
    """预算强制收尾：只解析 answer_text；作答段写满（stopped=False）判截断；思考段里的草稿网格不看；思考段是否闭合按
    thinking_end 判（DEV-0029）：unclosed → thinking_unclosed，其余三类不看适配器对完整序列的 thinking_unclosed。"""
    g = np.zeros((32, 32), np.uint8)
    g[3:9, 5:11] = 7

    @dataclass
    class _B:
        answer_text: str
        stopped: bool
        thinking_end: str = "natural"
        thinking_unclosed: bool = True  # 适配器对完整序列的判定：E5 只解析作答段，不用它
        text: str = "draft:\n" + relief_grid_to_text(np.ones((32, 32), np.uint8))

    for end in ("natural", "forced", "not_opened"):
        p = SC.parse_answer("relief_grid", SC.budget_view(_B(relief_grid_to_text(g), True, end)), 512, 512)
        assert p.valid and np.array_equal(p.relief, g)
        assert SC.parse_answer("relief_grid", SC.budget_view(_B(relief_grid_to_text(g), False, end)), 512, 512).failure == "truncated"
        assert SC.parse_answer("relief_grid", SC.budget_view(_B("  ", True, end)), 512, 512).failure == "empty"
    # unclosed：思考段里写出停止 token（stopped=True），没有作答段 → thinking_unclosed（不是 empty，也不是截断）
    assert SC.parse_answer("relief_grid", SC.budget_view(_B("", True, "unclosed")), 512, 512).failure == "thinking_unclosed"
    assert SC.budget_view(_B("", True, "unclosed")).thinking_unclosed is True
    assert SC.budget_view(_B("x", True, "forced")).thinking_unclosed is False
    for bad in (None, "closed"):
        with pytest.raises(ValueError, match="thinking_end"):
            SC.budget_view(_B("x", True, bad))


def test_thinking_end_block_four_kinds():
    """按 thinking_end 四类分开汇总（口径同 E1）：四类都列条数（没有的记 0），出现的类给解析、截断、token 数与读数；不是预算强制
    收尾的记录 → None；有的有、有的没有或取值不对 → 报错。"""
    def rec(i, end, *, failure=None, iou=1.0, headline=True):
        return {"id": f"x/{i}", "bin": 0, "groups": {}, "headline": headline, "valid": failure is None, "failure": failure,
                "truncated": failure == "truncated", "thinking_end": end, "n_thinking_tokens": 0 if end == "not_opened" else 10 + i,
                "n_answer_tokens": 0 if end == "unclosed" else 5, "n_inserted_tokens": 2 if end == "forced" else 0,
                "score": {"iou_grid": iou, "iou_full": iou, "relief": None}}

    recs = [rec(0, "forced"), rec(1, "forced", failure="truncated", iou=0.0), rec(2, "natural", headline=False),
            rec(3, "unclosed", failure="thinking_unclosed", iou=0.0)]
    te = SC.thinking_end_block(recs, 0)
    assert te["n"] == 4 and te["counts"] == {"natural": 1, "forced": 2, "not_opened": 0, "unclosed": 1}
    assert set(te["by_kind"]) == {"natural", "forced", "unclosed"}
    f = te["by_kind"]["forced"]
    assert (f["n"], f["n_headline"], f["n_truncated"], f["parse_failure_rate"]) == (2, 2, 1, 0.5)
    assert f["parse"]["failures"] == {"truncated": 1} and f["thinking_tokens"]["max"] == 11
    assert f["headline"]["mask_iou_grid"]["strict"] == 0.5 and f["overall"]["mask_iou_grid"]["valid_rate"] == 0.5
    assert te["by_kind"]["natural"]["n_headline"] == 0 and te["by_kind"]["natural"]["headline"]["mask_iou_grid"]["strict"] is None
    assert te["by_kind"]["unclosed"]["parse"]["failures"] == {"thinking_unclosed": 1}
    assert te["inserted_tokens"]["total"] == 4
    assert SC.thinking_end_block([{k: v for k, v in r.items() if k != "thinking_end"} for r in recs], 0) is None
    with pytest.raises(ValueError, match="有的没有"):
        SC.thinking_end_block(recs + [{k: v for k, v in recs[0].items() if k != "thinking_end"}], 0)
    with pytest.raises(ValueError, match="未知"):
        SC.thinking_end_block(recs + [rec(9, "closed")], 0)
    assert SC.aggregate("occlusion", recs, n_resamples=0)["thinking_end"] == te


def test_current_mask_iou_for_external_predictions():
    for task in ("rotation3d", "viewpoint", "occlusion", "imagery"):
        s = make_sample(eval_items(task, "dev", 1)[0])
        cur = current_mask(s)
        if cur is None:
            sc = SC.score(s, SC.from_mask(s.target_mask), relief_expected=False, current=True)
            assert sc["iou_current_full"] is None and sc["iou_current_grid"] is None
            continue
        sc = SC.score(s, SC.from_mask(cur.mask), relief_expected=False, current=True)  # 只会分割当前位置的系统
        assert sc["iou_current_full"] == 1.0 and sc["iou_current_grid"] == 1.0
        assert sc["iou_full"] == s.meta["baseline_iou"][CURRENT_MASK_BASELINE[task]]["full"]
        bad = SC.score(s, SC.failed("external_missing"), relief_expected=False, current=True)
        assert bad["iou_current_full"] == 0.0 and bad["iou_current_grid"] == 0.0
        assert "iou_current_full" not in SC.score(s, SC.from_mask(cur.mask), relief_expected=False)
    recs = [{"id": f"x/{i}", "bin": 0, "groups": {}, "headline": True, "valid": True, "failure": None,
             "score": {"iou_grid": 0.2, "iou_full": 0.2, "iou_current_grid": 0.9, "iou_current_full": 0.8, "relief": None}}
            for i in range(3)]
    b = SC.blocks(recs, 0)
    assert b["mask_iou_current_grid"]["strict"] == pytest.approx(0.9) and b["mask_iou_current_full"]["n"] == 3
    assert "mask_iou_current_grid" not in SC.blocks([{**r, "score": {**r["score"], "iou_current_grid": None,
                                                                         "iou_current_full": None}} for r in recs], 0)


def test_refusal_diagnostic():
    assert SC.is_refusal({"failure": "no_grid", "text": "I'm sorry, I cannot help with that."})
    assert not SC.is_refusal({"failure": "no_grid", "text": "Row 1 is 0101"})
    assert not SC.is_refusal({"failure": None, "text": "I cannot"})
    assert math.isfinite(SC.RELIEF_FAIL_VALUE)
