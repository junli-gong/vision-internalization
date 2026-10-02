"""DEV-0018：a 的目标相对深度（起伏）网格与起伏误差（src/synth/relief.py，D28-13），以及 33 符号的编解码（src/grid/codec.py）。

- 构造例子上与手算一致：格子深度、平均、分档、截断；RMSE / 标准差；没有定义的三种情形；"预测一个平面"恰为 1。
- 仿射连同深度一起取样：mask 与 affine_warp 逐位相同，值是最近邻的源像素。
- 生成器的 a 样本：版本 4；起伏代码与网格一致；元数据里的基线误差与独立复算相同。
- 复算项目负责人 D28-13 的表（同样设置：test 划分、每档 60 条、32×32），并钉住"照字面的判定条件可以被平面满足"这一发现。
"""

from fractions import Fraction
import math

import numpy as np
import pytest

from src.grid import codec as C
from src.grid.codec import (
    RELIEF_LEVEL_SYMBOLS,
    RELIEF_NUM_LEVELS,
    RELIEF_SYMBOLS,
    GridParseFailure,
    parse_relief_grid_text,
    relief_grid_to_text,
)
from src.synth import generate
from src.synth.constants import DEPTH_QUANTUM_M, ROT3D_BINS, TASK_GENERATOR_VERSIONS
from src.synth.relief import (
    FLAT_LEVEL,
    HALF_LEVELS,
    RELIEF_BASELINE_PREFIX,
    RELIEF_TARGET_KEY,
    baseline_relief_key,
    cell_depth_units,
    flat_codes,
    radius_depth_units,
    relief_grid,
    relief_score,
)
from src.synth.render3d import depth_grid
from src.synth.warp2d import affine_warp, affine_warp_values, moment_matched_affine

R_M = 0.25  # 半径 0.25 m = 1024 个深度单位（DEPTH_QUANTUM_M = 2^-12 m）


# ---------------------------------------------------------------------------
# 编解码
# ---------------------------------------------------------------------------


def test_symbol_table():
    assert RELIEF_NUM_LEVELS == 32 and RELIEF_SYMBOLS == ".0123456789ABCDEFGHIJKLMNOPQRSTUV"
    assert RELIEF_LEVEL_SYMBOLS[HALF_LEVELS] == "G" and RELIEF_LEVEL_SYMBOLS[HALF_LEVELS - 1] == "F"


def test_relief_text_roundtrip_and_parse_rules():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, RELIEF_NUM_LEVELS + 1, (8, 8)).astype(np.uint8)
    txt = relief_grid_to_text(codes)
    assert all(len(line) == 8 for line in txt.split("\n"))
    r = parse_relief_grid_text("Here is the grid:\n```\n" + txt + "\n```\nDone.", 8)
    assert r.failure is None and np.array_equal(r.grid, codes)
    spaced = "\n".join(" ".join(line) for line in txt.split("\n"))
    assert np.array_equal(parse_relief_grid_text(spaced, 8).grid, codes)
    lines = txt.split("\n")
    assert parse_relief_grid_text("\n".join(lines[:7]), 8).failure is GridParseFailure.WRONG_ROW_COUNT
    assert parse_relief_grid_text("\n".join(lines[:7] + [lines[7][:7]]), 8).failure is GridParseFailure.WRONG_COL_COUNT
    dirty = lines[:3] + [lines[3][:4] + "x" + lines[3][5:]] + lines[4:]
    assert parse_relief_grid_text("\n".join(dirty), 8).failure is GridParseFailure.ILLEGAL_CHAR
    assert parse_relief_grid_text(txt + "\nok\n" + txt, 8).failure is GridParseFailure.MULTIPLE_BLOCKS
    assert parse_relief_grid_text("no grid here", 8).failure is GridParseFailure.NO_GRID
    # 小写字母与 W–Z 不是符号
    assert parse_relief_grid_text("\n".join(["abcdefgh"] * 8), 8).failure is GridParseFailure.NO_GRID
    assert parse_relief_grid_text("\n".join(["WWWWWWWW"] * 8), 8).failure is GridParseFailure.NO_GRID
    with pytest.raises(ValueError):
        relief_grid_to_text(np.full((2, 2), 33, np.uint8))


def test_binary_parser_unchanged_by_the_relief_addition():
    """原有的 0/1 解析不变：起伏符号里的字母不会让 0/1 网格的解析接受它们。"""
    g = np.eye(4, dtype=np.uint8)
    assert np.array_equal(C.parse_grid_text(C.grid_to_text(g), 4).grid, g)
    assert C.parse_grid_text("\n".join(["0A10"] * 4), 4).failure is not None


# ---------------------------------------------------------------------------
# 起伏网格：构造例子
# ---------------------------------------------------------------------------


def test_cell_depth_units_is_the_rounded_mean_and_matches_depth_grid():
    mask = np.zeros((8, 8), bool)
    mask[0:4, 0:4] = True
    mask[4:6, 4:8] = True  # 第 (1, 1) 格只有一半：占比 0.5 → 网格为 1
    depth = np.zeros((8, 8), np.int64)
    depth[0:4, 0:4] = np.arange(16).reshape(4, 4) + 4000
    depth[4:6, 4:8] = 5001
    g, u = cell_depth_units(depth, mask, 2)
    assert g.tolist() == [[1, 0], [0, 1]]
    assert u[0, 0] == 4008  # (4000 + … + 4015) / 16 = 4007.5 → 取偶 4008
    assert u[1, 1] == 5001 and u[0, 1] == 0
    assert np.array_equal(depth_grid(depth, mask, 2), (u * DEPTH_QUANTUM_M).astype(np.float32))


def _toy(levels_rel_units: list[int], radius_units: int):
    """一行 K 个格子（每格 1 个像素，n = K），格子深度 = 10000 + 给定的相对量（深度单位）。"""
    k = len(levels_rel_units)
    mask = np.ones((k, k), bool)
    depth = np.zeros((k, k), np.int64)
    for j, d in enumerate(levels_rel_units):
        depth[:, j] = 10000 + d
    return depth, mask


def test_relief_levels_by_hand():
    # 4×4 网格，每列一个深度：μ 是 16 格的平均；R = 1024；档宽 2R/32 = 64 个单位
    depth, mask = _toy([-300, -10, 0, 310], 1024)
    rg = relief_grid(depth, mask, 4, R_M)
    assert radius_depth_units(R_M) == 1024 and rg.radius_units == 1024 and rg.n_cells == 16
    mu = Fraction(10000 * 16 + 4 * (-300 - 10 + 0 + 310), 16)  # = 10000
    assert mu == 10000 and Fraction(rg.mean_units_num, rg.n_cells) == mu
    want = [math.floor(Fraction(d, 64)) + 16 for d in (-300, -10, 0, 310)]  # [-5, -1, 0, 4] + 16
    assert want == [11, 15, 16, 20]
    assert rg.codes.tolist() == [[w + 1 for w in want]] * 4
    assert rg.n_clipped == 0 and rg.mean_depth_m() == pytest.approx(10000 * DEPTH_QUANTUM_M)


def test_relief_mean_is_exact_and_clipping():
    depth, mask = _toy([-2000, 0, 0, 1], 1024)  # μ = (−2000 + 1) / 4 = −499.75（相对 10000）
    rg = relief_grid(depth, mask, 4, R_M)
    rel = [Fraction(d) - Fraction(-1999, 4) for d in (-2000, 0, 0, 1)]
    want = [min(max(math.floor(r / 64) + 16, 0), 31) for r in rel]
    assert want[0] == 0 and rg.n_clipped == 4  # 第一列 −1500.25 < −R：截到 0（4 行各一格）
    assert rg.codes[0].tolist() == [w + 1 for w in want]
    depth2, mask2 = _toy([0, 0, 0, 3000], 1024)  # μ = 750：−750 → 档 4（不截），2250 → 档 51 → 截到 31（V）
    rg2 = relief_grid(depth2, mask2, 4, R_M)
    assert rg2.codes[0].tolist() == [5, 5, 5, RELIEF_NUM_LEVELS] and rg2.n_clipped == 4


def test_relief_background_and_empty():
    mask = np.zeros((8, 8), bool)
    rg = relief_grid(np.zeros((8, 8), np.int64), mask, 4, R_M)
    assert rg.n_cells == 0 and not rg.codes.any() and rg.mean_depth_m() is None
    with pytest.raises(ValueError):
        radius_depth_units(0.1)  # 不是 2^-12 的整数倍


def test_relief_error_by_hand():
    t = np.zeros((4, 4), np.uint8)
    t[0, :] = [11, 13, 15, 17]  # 档 10, 12, 14, 16：均值 13，方差 5
    p = np.zeros((4, 4), np.uint8)
    p[0, :3] = [12, 13, 18]  # 档 11, 12, 17；第四格漏掉（不计），另多一格（不计）
    p[3, 3] = 5
    sc = relief_score(p, t)
    d2 = (11 - 10) ** 2 + (12 - 12) ** 2 + (17 - 14) ** 2  # 10
    want = math.sqrt((d2 / 3) / 5)
    assert sc.n_eval_cells == 3 and sc.n_target_cells == 4 and sc.reason is None
    assert sc.error == pytest.approx(want, abs=2**-29)
    assert sc.target_std_levels == pytest.approx(math.sqrt(5), abs=2**-29)
    # 平面恰为 1：常数档 = 目标的平均档
    flat = np.where(t > 0, 14, 0).astype(np.uint8)  # 档 13
    assert relief_score(flat, t).error == 1.0
    assert relief_score(t, t).error == 0.0


def test_relief_error_undefined_cases():
    t = np.zeros((4, 4), np.uint8)
    assert relief_score(t, t).reason == "no_target" and relief_score(t, t).error is None
    t[0, :2] = 7
    assert relief_score(t, t).reason == "flat_target" and relief_score(t, t).error is None
    t[0, 2] = 9
    p = np.zeros_like(t)
    p[3, 3] = 4
    sc = relief_score(p, t)
    assert sc.reason == "no_overlap" and sc.error is None and sc.n_eval_cells == 0
    with pytest.raises(ValueError):
        relief_score(np.zeros((3, 3), np.uint8), t)


def test_flat_codes():
    g = np.eye(3, dtype=np.uint8)
    assert flat_codes(g).tolist() == (np.eye(3, dtype=np.uint8) * (FLAT_LEVEL + 1)).tolist()
    assert RELIEF_LEVEL_SYMBOLS[FLAT_LEVEL] == "G"


# ---------------------------------------------------------------------------
# 仿射连同深度
# ---------------------------------------------------------------------------


def test_affine_warp_values_matches_affine_warp_and_carries_nearest_values():
    rng = np.random.default_rng(1)
    mask = np.zeros((40, 50), bool)
    mask[8:30, 10:22] = True
    mask[20:30, 10:40] = True
    vals = rng.integers(3000, 9000, (40, 50))
    cases = [(((1.0, 0.0), (0.0, 1.0)), (5.0, -2.0)), (((0.7, -0.3), (0.4, 1.1)), (5.0, -2.0)),
             (((-1.0, 0.0), (0.0, 0.5)), (49.0, 10.0))]
    for a, tt in cases:
        m, v = affine_warp_values(mask, vals, a, tt)
        assert np.array_equal(m, affine_warp(mask, a, tt))
        assert not v[~m].any()
    m, v = affine_warp_values(mask, vals, ((1.0, 0.0), (0.0, 1.0)), (3.0, 2.0))  # 整像素平移：值就是平移过去的源像素
    assert np.array_equal(m[2:, 3:], mask[:-2, :-3]) and np.array_equal(v[2:, 3:][m[2:, 3:]], vals[:-2, :-3][mask[:-2, :-3]])
    m0, v0 = affine_warp_values(mask, vals, ((0.0, 0.0), (0.0, 1.0)), (0.0, 0.0))  # 奇异：空
    assert not m0.any() and not v0.any()


# ---------------------------------------------------------------------------
# 生成器的 a 样本
# ---------------------------------------------------------------------------


def _units(depth_m: np.ndarray) -> np.ndarray:
    return np.rint(depth_m.astype(np.float64) / DEPTH_QUANTUM_M).astype(np.int64)


def test_rotation3d_samples_carry_consistent_relief():
    assert TASK_GENERATOR_VERSIONS["rotation3d"] == 4 and TASK_GENERATOR_VERSIONS["viewpoint"] == 4
    for i in range(12):
        s = generate("rotation3d", "val", 40 + i)
        r = s.meta["scene"]["objects"][0]["radius"]
        assert s.meta["generator_version"] == 4
        tgt = s.extras[RELIEF_TARGET_KEY]
        assert tgt.dtype == np.uint8 and tgt.shape == s.target_grid.shape
        assert np.array_equal(tgt > 0, s.target_grid == 1)
        rg = relief_grid(_units(s.extras["depth"]), s.target_mask, s.grid_size, r)
        assert np.array_equal(rg.codes, tgt)
        cp = relief_grid(_units(s.extras["input_depth"]), s.baselines["copy"].mask, s.grid_size, r)
        assert np.array_equal(s.extras[baseline_relief_key("copy")], cp.codes)
        fit = moment_matched_affine(s.baselines["copy"].mask, s.target_mask)
        mw, dw = affine_warp_values(s.baselines["copy"].mask, _units(s.extras["input_depth"]), fit.a, fit.t)
        assert np.array_equal(mw, s.baselines["gt_affine_upper"].mask)
        assert np.array_equal(s.extras[baseline_relief_key("gt_affine_upper")], relief_grid(dw, mw, s.grid_size, r).codes)
        assert np.array_equal(s.extras[baseline_relief_key("copy_flat")], flat_codes(s.baselines["copy"].grid))
        assert np.array_equal(s.baselines["copy_flat"].mask, s.baselines["copy"].mask)
        for name in s.baselines:
            codes = s.extras[RELIEF_BASELINE_PREFIX + name]
            assert np.array_equal(codes > 0, s.baselines[name].grid == 1), name
            sc = relief_score(codes, tgt)
            assert s.meta["relief"]["baseline_error"][name] == {"error": sc.error, "n_eval_cells": sc.n_eval_cells,
                                                                "reason": sc.reason}
        rm = s.meta["relief"]
        assert rm["num_levels"] == 32 and rm["radius_m"] == r and rm["level_width_m"] == 2 * r / 32
        assert rm["target"]["n_cells"] == int(s.target_grid.sum())
        assert s.meta["baseline_roles"]["copy_flat"] == "trivial"


def _table(n_per_bin: int):
    rows = {b: [] for b in range(len(ROT3D_BINS))}
    for i in range(len(ROT3D_BINS) * n_per_bin):
        s = generate("rotation3d", "test", i)
        rows[s.meta["difficulty"]["bin"]].append(s.meta)
    return rows


def test_report_d28_13_table_and_flat_finding():
    """D28-13 的表的复算（同样设置：test 划分、每档 60 条、32×32 网格；样本是任务版本 4 的）。

    钉住两件事（防止过度主张溜回去，见 DEV-0018 完成总结）：
    1. 数值与项目负责人的表同一量级：copy 的网格 IoU 随角度下降，真值仿射的网格 IoU ≥ 0.85，30° 以上真值仿射的起伏误差 > 1；
    2. "预测一个平面"（copy_flat）的起伏误差 ≈ 1.0，在 30° 以上的三档都低于真值仿射：照字面的判定条件可以被平面满足。
    """
    rows = _table(60)
    lines = ["", "bin | copy grid IoU | gt_affine grid IoU | gt_affine relief | copy relief | copy_flat relief | instr relief"]
    head_ga, head_flat = [], []
    for b, ms in rows.items():
        def mean(f):
            v = [f(m) for m in ms]
            v = [x for x in v if x is not None]
            return float(np.mean(v))

        cg = mean(lambda m: m["baseline_iou"]["copy"]["grid"])
        gg = mean(lambda m: m["baseline_iou"]["gt_affine_upper"]["grid"])
        ge = mean(lambda m: m["relief"]["baseline_error"]["gt_affine_upper"]["error"])
        ce = mean(lambda m: m["relief"]["baseline_error"]["copy"]["error"])
        fe = mean(lambda m: m["relief"]["baseline_error"]["copy_flat"]["error"])
        ie = mean(lambda m: m["relief"]["baseline_error"]["instr_affine"]["error"])
        lines.append(f"{ROT3D_BINS[b]} n={len(ms)} | {cg:.3f} | {gg:.3f} | {ge:.3f} | {ce:.3f} | {fe:.3f} | {ie:.3f}")
        assert gg >= 0.85 and gg > cg
        assert 0.9 < fe < 1.15
        if ROT3D_BINS[b][0] >= 30:
            assert ge > 1.0
            head_ga.append(ge)
            head_flat.append(fe)
    print("\n".join(lines))
    assert all(f < g for f, g in zip(head_flat, head_ga))
