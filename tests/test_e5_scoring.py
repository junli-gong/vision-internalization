"""DEV-0018：E5 的逐条解析与评分、三列汇总、两级门槛（src/e5/scoring.py）。"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from src.e5 import scoring as SC
from src.e5.protocol import eval_items
from src.e5.runner import make_sample
from src.grid.codec import grid_to_text, relief_grid_to_text
from src.synth.raster import rasterize
from src.synth.relief import RELIEF_TARGET_KEY


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
            if task == "rotation3d":
                assert r["score"]["relief"]["error"] == s.meta["relief"]["baseline_error"][name]["error"]
            else:
                assert r["score"]["relief"] is None
    if task == "rotation3d":
        assert roles["gt_affine_upper"] == "reference_upper_bound" and roles["copy_flat"] == "trivial"


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


def _base_sets(n, copy_iou, flat_rel, ga_rel, copy_rel=1.5):
    return {
        "copy": [_rec(i, copy_iou, relief=copy_rel) for i in range(n)],
        "copy_flat": [_rec(i, copy_iou, relief=flat_rel) for i in range(n)],
        "gt_affine_upper": [_rec(i, 0.95, relief=ga_rel) for i in range(n)],
    }


ROLES = {"copy": "trivial", "copy_flat": "trivial", "gt_affine_upper": "reference_upper_bound"}


def test_gate_two_levels_on_rotation():
    sys_good = [_rec(i, 0.8, relief=0.9) for i in range(6)]
    g = SC.gate("rotation3d", sys_good, _base_sets(6, 0.7, 1.0, 1.2), ROLES, n_resamples=50)
    assert g["prerequisite"]["pass"] and g["prerequisite"]["strongest"] == "copy"  # 参照线不算；与 copy_flat 同分取 copy
    assert g["judgment"]["status"] == "evaluated" and g["judgment"]["pass"] and g["judgment"]["pass_vs_flat"]
    assert g["judgment"]["flat_beats_gt_affine"] is True and g["imagining"] is True
    # 起伏低于真值仿射、但不低于平面：照字面通过，vs_flat 不通过
    sys_flatish = [_rec(i, 0.8, relief=1.05) for i in range(6)]
    g = SC.gate("rotation3d", sys_flatish, _base_sets(6, 0.7, 1.0, 1.2), ROLES, n_resamples=0)
    assert g["judgment"]["pass"] and g["judgment"]["pass_vs_flat"] is False
    # mask 不胜过原样复制：先决条件不通过
    g = SC.gate("rotation3d", [_rec(i, 0.7, relief=0.5) for i in range(6)], _base_sets(6, 0.7, 1.0, 1.2), ROLES, n_resamples=0)
    assert not g["prerequisite"]["pass"] and g["imagining"] is False
    # 全部失败：起伏按 2.0 计，不会"低于"真值仿射
    fails = [_rec(i, 0.0, valid=False, relief=None) for i in range(6)]
    g = SC.gate("rotation3d", fails, _base_sets(6, 0.7, 1.0, 1.9), ROLES, n_resamples=0)
    assert g["judgment"]["system"] == 2.0 and g["judgment"]["pass"] is False
    # 只给 mask 的系统（思维链、外部）：判定条件不适用
    g = SC.gate("rotation3d", [_rec(i, 0.8, relief=False) for i in range(6)], _base_sets(6, 0.7, 1.0, 1.2), ROLES, n_resamples=0)
    assert g["judgment"]["status"] == "not_applicable" and g["imagining"] is True


def test_gate_other_tasks_and_alignment():
    recs = [_rec(i, 0.6, relief=False) for i in range(4)]
    base = {"copy": [_rec(i, 0.5, relief=False) for i in range(4)]}
    gb = SC.gate("viewpoint", recs, base, {"copy": "trivial"}, n_resamples=0)
    assert gb["judgment"]["status"] == "unavailable" and gb["imagining"] is None and gb["prerequisite"]["pass"]
    gc = SC.gate("occlusion", recs, base, {"copy": "trivial"}, n_resamples=0)
    assert gc["judgment"]["status"] == "not_applicable" and gc["imagining"] is True
    with pytest.raises(ValueError):
        SC.gate("occlusion", recs, {"copy": base["copy"][:2]}, {"copy": "trivial"}, n_resamples=0)
    only_head = SC.gate("occlusion", recs + [_rec(9, 1.0, relief=False, headline=False)], base, {"copy": "trivial"},
                        n_resamples=0)
    assert only_head["n"] == 4  # 只用标题口径的样本


def test_refusal_diagnostic():
    assert SC.is_refusal({"failure": "no_grid", "text": "I'm sorry, I cannot help with that."})
    assert not SC.is_refusal({"failure": "no_grid", "text": "Row 1 is 0101"})
    assert not SC.is_refusal({"failure": None, "text": "I cannot"})
    assert math.isfinite(SC.RELIEF_FAIL_VALUE)
