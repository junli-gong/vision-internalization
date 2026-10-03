"""DEV-0028（D59-4、D59-5）：凸起先验基线、b（任务版本 5）的起伏与基线、A 的选定、项目负责人 D59-4 的表、当前 mask，
以及 a、b 生成提速前后逐位相同（与旧实现的逐字副本对照）。

- 凸起先验（src/synth/relief.py）：边界距离与整数开方精确；构造例子与手算一致；A = 0 即平面；离边界越远档越小；
  派生基线 copy_convex 不进样本。
- b 的样本：起伏、copy / translate / copy_flat / gt_affine_upper 的起伏代码与独立复算相同，元数据里的基线误差也相同。
- A 只在 val 上选：a 全量复算（600 条）与冻结值、记录一致；用冻结的 A 在 test 上复算项目负责人的表。
- 当前 mask（D59-5）：a、b 的 copy 就是输入图里目标的 mask，c 的 visible 是可见部分，d 的 copy 是第 0 步。
- 提速（rotation3d 单核 ≥ 30 条/秒的任务）：warp2d 的取样、矩匹配搜索、矩、cell_depth_units 与旧实现逐位相同。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.synth import generate
from src.synth.api import CURRENT_MASK_BASELINE, current_mask
from src.synth.constants import DEPTH_QUANTUM_M
from src.synth.relief import (
    CONVEX_BASELINE,
    CONVEX_PRIOR_AMPLITUDE,
    FLAT_LEVEL,
    RELIEF_BASELINE_PREFIX,
    RELIEF_TARGET_KEY,
    RELIEF_TASKS,
    _isqrt,
    baseline_relief_key,
    boundary_distance_sq,
    cell_depth_units,
    convex_prior_codes,
    derived_baselines,
    flat_codes,
    relief_grid,
    relief_score,
)
from src.grid.codec import cell_edges, mask_to_grid
from src.synth import warp2d as W
from src.synth.warp2d import affine_warp_values, moment_matched_affine, shift_mask, shift_values


def _units(depth_m: np.ndarray) -> np.ndarray:
    return np.rint(depth_m.astype(np.float64) / DEPTH_QUANTUM_M).astype(np.int64)


# ---------------------------------------------------------------------------
# 凸起先验
# ---------------------------------------------------------------------------


def _brute_d2(g: np.ndarray) -> np.ndarray:
    """逐格暴力：到最近背景格（网格外一圈也算背景）的格心距离平方。"""
    n0, n1 = g.shape
    pad = np.zeros((n0 + 2, n1 + 2), bool)
    pad[1:-1, 1:-1] = g
    bi, bj = np.nonzero(~pad)
    out = np.zeros(g.shape, np.int64)
    for i, j in zip(*np.nonzero(g)):
        out[i, j] = int(((bi - (i + 1)) ** 2 + (bj - (j + 1)) ** 2).min())
    return out


def test_boundary_distance_and_isqrt_are_exact():
    rng = np.random.default_rng(3)
    for _ in range(20):
        g = np.zeros((12, 12), np.uint8)
        r0, c0 = rng.integers(0, 5, 2)
        g[r0 : r0 + rng.integers(3, 8), c0 : c0 + rng.integers(3, 8)] = 1
        g[rng.integers(0, 12, 6), rng.integers(0, 12, 6)] ^= 1
        assert np.array_equal(boundary_distance_sq(g), _brute_d2(g > 0))
    x = np.array([0, 1, 2, 3, 4, 15, 16, 17, (1 << 26) - 1, 1 << 26, (1 << 51) + 12345, (1 << 52) - 1], np.int64)
    assert _isqrt(x).tolist() == [math.isqrt(int(v)) for v in x]
    with pytest.raises(ValueError):
        _isqrt(np.array([-1]))


def test_convex_prior_by_hand():
    g = np.zeros((7, 7), np.uint8)
    g[1:6, 1:6] = 1  # 5×5 的方块：d² = 1（外圈）、4（中圈）、9（中心）；d = 1, 2, 3 → f = 0, 1/2, 1
    d2 = boundary_distance_sq(g)
    assert d2[1, 1] == 1 and d2[2, 2] == 4 and d2[3, 3] == 9 and d2[2, 3] == 4 and d2[0, 0] == 0
    c = convex_prior_codes(g, 4)  # 档 = 16 − round(4·f)：外圈 16、中圈 14、中心 12（代码 = 档 + 1）
    assert c[1, 1] == FLAT_LEVEL + 1 and c[2, 2] == FLAT_LEVEL - 2 + 1 and c[3, 3] == FLAT_LEVEL - 4 + 1 and c[0, 0] == 0
    c3 = convex_prior_codes(g, 3)  # 3 · 1/2 = 1.5 → 取偶到 2
    assert c3[2, 2] == FLAT_LEVEL - 2 + 1 and c3[3, 3] == FLAT_LEVEL - 3 + 1
    c1 = convex_prior_codes(g, 1)  # 0.5 → 取偶到 0
    assert c1[2, 2] == FLAT_LEVEL + 1 and c1[3, 3] == FLAT_LEVEL
    thin = np.zeros((6, 6), np.uint8)  # 全部贴边（d_max = 1）：f ≡ 0，就是平面
    thin[2, 1:5] = 1
    assert np.array_equal(convex_prior_codes(thin, 7), flat_codes(thin))
    assert boundary_distance_sq(np.ones((5, 5), np.uint8))[2, 2] == 9  # 网格之外都算背景
    assert int(convex_prior_codes(np.ones((32, 32), np.uint8), 40).min()) == 1  # 截到档 0
    assert not convex_prior_codes(np.zeros((4, 4), np.uint8), 3).any()
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            convex_prior_codes(g, bad)


def test_convex_prior_properties_on_samples():
    for task in RELIEF_TASKS:
        for i in range(6):
            s = generate(task, "val", 60 + i)
            g = s.baselines["copy"].grid
            assert np.array_equal(convex_prior_codes(g, 0), flat_codes(g))  # A = 0 ≡ 平面（copy_flat）
            for a in (1, 2, 5):
                c = convex_prior_codes(g, a)
                assert np.array_equal(c > 0, g == 1)
                lev = c[g == 1].astype(int) - 1
                d2 = boundary_distance_sq(g)[g == 1]
                assert lev.max() == FLAT_LEVEL and lev.min() == FLAT_LEVEL - a  # 贴边取平面档，最里面近 A 档
                order = np.argsort(d2, kind="stable")
                assert np.all(np.diff(lev[order]) <= 0)  # 离边界越远越近（档不增）
            der = derived_baselines(s)
            assert set(der) == {CONVEX_BASELINE} and CONVEX_BASELINE not in s.baselines  # 不存进样本（不进摘要）
            pred, codes = der[CONVEX_BASELINE]
            assert pred is s.baselines["copy"] and np.array_equal(codes, convex_prior_codes(g, CONVEX_PRIOR_AMPLITUDE[task]))
    assert derived_baselines(generate("occlusion", "val", 0)) == {}


def test_frozen_amplitudes():
    """A 按任务在 val 上选定后冻结（DEV-0028：a 2 档、b 1 档；改动要先有新的 D 条目）。"""
    assert CONVEX_PRIOR_AMPLITUDE == {"rotation3d": 2, "viewpoint": 1}


def test_amplitude_selection_reproduces_on_val_for_a():
    """a 的选定在 val 上全量复算（每档 200 条、30° 以上三档 = 600 条）：A = 1、2、3 与记录一致，2 最小。
    完整网格搜索（0–12）与 b 的选定用 `python -m src.e5.convex_select select`（b 要生成 800 条，测试里不跑）。"""
    from src.e5 import convex_select as CS

    out = CS.select("rotation3d", grid=(1, 2, 3))
    assert out["split"] == "val" and out["n_samples"] == 600 and out["n_population"] == 600
    assert out["chosen"] == CONVEX_PRIOR_AMPLITUDE["rotation3d"] == 2
    assert [round(out["table"][a], 4) for a in (1, 2, 3)] == [0.9548, 0.9390, 0.9642]
    with pytest.raises(ValueError):
        CS.select("rotation3d", 1, split="test")  # 只在 val 上选
    small = CS.select("viewpoint", 4, grid=(0, 1))  # b 的机制（小样本，只核对结构）
    assert small["split"] == "val" and set(small["table"]) == {0, 1} and small["n_population"] > 0


def test_lead_d59_table_on_test_with_frozen_amplitude():
    """用冻结的 A 在 test 上复算项目负责人 D59-4 的表（a、每档 80 条）。他的表：

        档        平面  真值仿射  凸起先验  凸起先验胜过平面的比例
        15–30°    0.99   0.73     0.91     89%
        30–60°    0.99   1.07     0.92     79%
        60–90°    1.01   1.57     1.01     55%
        90–180°   0.99   1.22     0.96     69%

    平面与真值仿射两列是生成器的性质，与他的数逐个相同（两位小数），说明用的是同一批 test 样本。凸起先验的细节（距离、
    归一化、取整）与他的未必相同，A 是我们在 val 上独立选的：钉住它在每一档都低于平面、胜过平面的样本过半。"""
    from src.e5 import convex_select as CS

    out = CS.report("rotation3d")
    assert out["amplitude"] == 2 and [r["n"] for r in out["rows"]] == [80, 80, 80, 80]
    lead = [(0.99, 0.73), (0.99, 1.07), (1.01, 1.57), (0.99, 1.22)]
    lines = [""]
    for r, (fl, ga) in zip(out["rows"], lead):
        lines.append(f"{r['range']} flat {r['flat_mean']:.3f} gt_affine {r['gt_affine_mean']:.3f} convex {r['convex_mean']:.3f} "
                     f"beats {r['convex_beats_flat']:.0%}")
        assert round(r["flat_mean"], 2) == fl and round(r["gt_affine_mean"], 2) == ga
        assert r["convex_mean"] < r["flat_mean"] and r["convex_beats_flat"] > 0.5
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# b（任务版本 5）的起伏与基线
# ---------------------------------------------------------------------------


def test_shift_values_matches_shift_mask():
    rng = np.random.default_rng(5)
    m = rng.random((20, 30)) < 0.4
    v = rng.integers(1, 9000, (20, 30))
    for dx, dy in ((0, 0), (3, -2), (-7, 5), (30, 0), (0, -20), (29, 19)):
        assert np.array_equal(shift_values(m, dx, dy), shift_mask(m, dx, dy))
        sv = shift_values(v, dx, dy)
        assert sv.dtype == v.dtype and np.array_equal(sv != 0, shift_mask(v != 0, dx, dy))


def test_viewpoint_samples_carry_consistent_relief():
    for i in range(10):
        s = generate("viewpoint", "val", 40 + i)
        t = s.meta["target_obj_id"]
        r = s.meta["scene"]["objects"][t]["radius"]
        assert s.meta["generator_version"] == 5
        assert set(s.baselines) == {"copy", "translate", "copy_flat", "gt_affine_upper"}
        assert s.meta["baseline_roles"] == {"copy": "trivial", "translate": "trivial", "copy_flat": "trivial",
                                            "gt_affine_upper": "reference_upper_bound"}
        tgt = s.extras[RELIEF_TARGET_KEY]
        assert tgt.dtype == np.uint8 and np.array_equal(tgt > 0, s.target_grid == 1)
        assert np.array_equal(relief_grid(_units(s.extras["depth"]), s.target_mask, s.grid_size, r).codes, tgt)
        d0 = _units(s.extras["input_depth"])
        m0 = s.baselines["copy"].mask
        assert np.array_equal(m0, s.extras["input_id_map"] == t + 1)
        assert np.array_equal(s.extras[baseline_relief_key("copy")], relief_grid(d0, m0, s.grid_size, r).codes)
        dx, dy = s.meta["translate_shift_px"]
        mt = shift_mask(m0, dx, dy)
        assert np.array_equal(mt, s.baselines["translate"].mask)
        assert np.array_equal(s.extras[baseline_relief_key("translate")],
                              relief_grid(shift_values(d0, dx, dy), mt, s.grid_size, r).codes)
        fit = moment_matched_affine(m0, s.target_mask)
        mw, dw = affine_warp_values(m0, d0, fit.a, fit.t)
        assert np.array_equal(mw, s.baselines["gt_affine_upper"].mask)
        assert s.meta["gt_affine_upper"] == {"rotation_deg": fit.rotation_deg, "reflect": fit.reflect}
        assert np.array_equal(s.extras[baseline_relief_key("gt_affine_upper")], relief_grid(dw, mw, s.grid_size, r).codes)
        assert np.array_equal(s.extras[baseline_relief_key("copy_flat")], flat_codes(s.baselines["copy"].grid))
        assert np.array_equal(s.baselines["copy_flat"].mask, m0)
        for name in s.baselines:
            codes = s.extras[RELIEF_BASELINE_PREFIX + name]
            assert np.array_equal(codes > 0, s.baselines[name].grid == 1), name
            sc = relief_score(codes, tgt)
            assert s.meta["relief"]["baseline_error"][name] == {"error": sc.error, "n_eval_cells": sc.n_eval_cells,
                                                                "reason": sc.reason}
        rm = s.meta["relief"]
        assert rm["num_levels"] == 32 and rm["radius_m"] == r and rm["target"]["n_cells"] == int(s.target_grid.sum())


def test_report_b_orbit_relief_baselines():
    """b 环绕（标题口径）上的起伏基线，val 划分每档 25 条（只报告）。"""
    from src.e5 import convex_select as CS

    out = CS.report("viewpoint", "val", 25)
    lines = [""] + [f"b orbit {r['range']} n={r['n']}: flat {r['flat_mean']:.3f} gt_affine {r['gt_affine_mean']:.3f} "
                    f"convex(A={out['amplitude']}) {r['convex_mean']:.3f}" for r in out["rows"]]
    print("\n".join(lines))
    assert out["rows"] and all(r["n"] > 0 for r in out["rows"])


# ---------------------------------------------------------------------------
# 当前 mask（D59-5）
# ---------------------------------------------------------------------------


def test_current_mask_is_the_input_state():
    assert CURRENT_MASK_BASELINE == {"rotation3d": "copy", "viewpoint": "copy", "occlusion": "visible", "physics": "copy"}
    for i in range(4):
        a = generate("rotation3d", "val", i)
        assert np.array_equal(current_mask(a).mask, a.extras["input_depth"] > 0)  # 单个实体、没有地板
        b = generate("viewpoint", "val", i)
        assert np.array_equal(current_mask(b).mask, b.extras["input_id_map"] == b.meta["target_obj_id"] + 1)
        c = generate("occlusion", "val", i)
        cm = current_mask(c).mask
        assert cm.any() and not (cm & ~c.target_mask).any() and cm.sum() < c.target_mask.sum()  # 可见部分 ⊂ 非模态
        d = generate("physics", "val", i)
        assert current_mask(d) is d.baselines["copy"]
        assert current_mask(generate("imagery", "val", i)) is None  # e 没有图像


# ---------------------------------------------------------------------------
# 提速前后逐位相同：旧实现的逐字副本（DEV-0018 时的 warp2d / relief）
# ---------------------------------------------------------------------------


def _old_affine_sources(mask, a, t, stride):
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    oh, ow = -(-h // stride), -(-w // stride)
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size == 0:
        return (oh, ow), None, None, None, None
    cols = np.nonzero(m.any(axis=0))[0]
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if det == 0.0:
        return (oh, ow), None, None, None, None
    ia = W.inv2(a)
    xs, ys = (float(cols[0]), float(cols[-1] + 1)), (float(rows[0]), float(rows[-1] + 1))
    px = [(a[0][0] * x + a[0][1] * y) + t[0] for x in xs for y in ys]
    py = [(a[1][0] * x + a[1][1] * y) + t[1] for x in xs for y in ys]
    j_lo = max(math.floor((min(px) - 0.5) / stride) - 1, 0)
    j_hi = min(math.ceil((max(px) - 0.5) / stride) + 1, ow - 1)
    i_lo = max(math.floor((min(py) - 0.5) / stride) - 1, 0)
    i_hi = min(math.ceil((max(py) - 0.5) / stride) + 1, oh - 1)
    if j_hi < j_lo or i_hi < i_lo:
        return (oh, ow), None, None, None, None
    qx = (np.arange(j_lo, j_hi + 1, dtype=np.float64) * stride + 0.5) - t[0]
    qy = (np.arange(i_lo, i_hi + 1, dtype=np.float64) * stride + 0.5) - t[1]
    sx = ia[0][0] * qx[None, :] + ia[0][1] * qy[:, None]
    sy = ia[1][0] * qx[None, :] + ia[1][1] * qy[:, None]
    fx = np.floor(sx)
    fy = np.floor(sy)
    valid = (fx >= 0) & (fx < w) & (fy >= 0) & (fy < h)
    jx = np.where(valid, fx, 0).astype(np.int64)
    iy = np.where(valid, fy, 0).astype(np.int64)
    return (oh, ow), (i_lo, i_hi, j_lo, j_hi), valid & m[iy, jx], iy, jx


def _old_affine_warp(mask, a, t, stride=1):
    shape, win, hit, _iy, _jx = _old_affine_sources(mask, a, t, stride)
    out = np.zeros(shape, dtype=bool)
    if win is not None:
        i_lo, i_hi, j_lo, j_hi = win
        out[i_lo : i_hi + 1, j_lo : j_hi + 1] = hit
    return out


def _old_affine_warp_values(mask, values, a, t):
    v = np.asarray(values)
    shape, win, hit, iy, jx = _old_affine_sources(mask, a, t, 1)
    out = np.zeros(shape, dtype=bool)
    vals = np.zeros(shape, dtype=v.dtype)
    if win is not None:
        i_lo, i_hi, j_lo, j_hi = win
        out[i_lo : i_hi + 1, j_lo : j_hi + 1] = hit
        vals[i_lo : i_hi + 1, j_lo : j_hi + 1] = np.where(hit, v[iy, jx], 0)
    return out, vals


def _old_iou(a, b):
    u = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / u if u else 1.0


def _old_moments(mask):
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    n = int(rows.size)
    r, c = rows.astype(np.int64), cols.astype(np.int64)
    sx, sy = int(c.sum()), int(r.sum())
    sxx, sxy, syy = int((c * c).sum()), int((c * r).sum()), int((r * r).sum())
    return W.Moments(n, sx / n + 0.5, sy / n + 0.5, (n * sxx - sx * sx) / (n * n), (n * sxy - sx * sy) / (n * n),
                     (n * syy - sy * sy) / (n * n))


def _old_mma(src, dst, coarse_deg=10, stride=4, top=3):
    ms, md = _old_moments(src), _old_moments(dst)
    s_half = W.sqrt_spd2(md.sxx, md.sxy, md.syy)
    s_inv = W.inv2(W.sqrt_spd2(ms.sxx, ms.sxy, ms.syy))
    cache = {}

    def params(omega, sigma):
        c, s = W.rot_cs(omega)
        q = ((c, -s * sigma), (s, c * sigma))
        a = W.mul2(W.mul2(s_half, q), s_inv)
        t = (md.mx - (a[0][0] * ms.mx + a[0][1] * ms.my), md.my - (a[1][0] * ms.mx + a[1][1] * ms.my))
        return a, t

    def score(omega, sigma, st):
        key = (omega % 360, sigma, st)
        if key not in cache:
            a, t = params(omega % 360, sigma)
            cache[key] = _old_iou(_old_affine_warp(src, a, t, st), dst[::st, ::st])
        return cache[key]

    coarse = [(om, sg) for sg in (1, -1) for om in range(0, 360, coarse_deg)]
    ranked = sorted(range(len(coarse)), key=lambda k: (-score(*coarse[k], stride), k))
    fine = []
    for k in ranked[:top]:
        om, sg = coarse[k]
        for d in range(-(coarse_deg // 2), coarse_deg // 2 + 1):
            cand = ((om + d) % 360, sg)
            if cand not in fine:
                fine.append(cand)
    fine_ranked = sorted(range(len(fine)), key=lambda k: (-score(*fine[k], 2), k))
    best = None
    for k in fine_ranked[:2]:
        om, sg = fine[k]
        a, t = params(om, sg)
        m = _old_affine_warp(src, a, t, 1)
        v = _old_iou(m, dst)
        if best is None or v > best[5]:
            best = (a, t, om, sg == -1, m, v)
    return best


def _old_cell_depth_units(depth_units, mask, n):
    m = np.asarray(mask, dtype=bool)
    g = mask_to_grid(m, n)
    h, w = m.shape
    re_, ce_ = cell_edges(h, n), cell_edges(w, n)
    du = np.where(m, np.asarray(depth_units, dtype=np.int64), 0)
    s = np.add.reduceat(np.add.reduceat(du, re_[:-1], axis=0), ce_[:-1], axis=1)
    c = np.add.reduceat(np.add.reduceat(m.astype(np.int64), re_[:-1], axis=0), ce_[:-1], axis=1)
    cs = np.maximum(c, 1)
    q, r = np.divmod(s, cs)
    up = (2 * r > cs) | ((2 * r == cs) & (q % 2 == 1))
    return g, np.where(g == 1, q + up.astype(np.int64), 0)


def _old_rasterize_triangles(u, v, inv_z, tris, height, width):
    """DEV-0010 的逐像素候选版本（逐字副本）：包围盒里每个像素中心判三条边。"""
    from src.synth import render3d as R

    h, w = int(height), int(width)
    n_tri = tris.shape[0]
    empty = np.zeros(0, dtype=np.int64)
    x0, y0 = u[tris[:, 0]], v[tris[:, 0]]
    x1, y1 = u[tris[:, 1]], v[tris[:, 1]]
    x2, y2 = u[tris[:, 2]], v[tris[:, 2]]
    a2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    coef = [R._edge_coeffs(x1, y1, x2, y2), R._edge_coeffs(x2, y2, x0, y0), R._edge_coeffs(x0, y0, x1, y1)]
    xmin, xmax = np.minimum(np.minimum(x0, x1), x2), np.maximum(np.maximum(x0, x1), x2)
    ymin, ymax = np.minimum(np.minimum(y0, y1), y2), np.maximum(np.maximum(y0, y1), y2)
    j_lo = np.maximum((xmin - R.HALF) // R.UNITS + 1, 0)
    j_hi = np.minimum((xmax - R.HALF) // R.UNITS, w - 1)
    i_lo = np.maximum((ymin - R.HALF) // R.UNITS + 1, 0)
    i_hi = np.minimum((ymax - R.HALF) // R.UNITS, h - 1)
    nj = np.maximum(j_hi - j_lo + 1, 0)
    ni = np.maximum(i_hi - i_lo + 1, 0)
    cnt = ni * nj
    total = int(cnt.sum())
    if total == 0:
        return empty, empty, empty, np.zeros((2, 0), dtype=np.int64)
    t_idx = np.repeat(np.arange(n_tri, dtype=np.int64), cnt)
    start = np.cumsum(cnt) - cnt
    k = np.arange(total, dtype=np.int64) - start[t_idx]
    njt = nj[t_idx]
    row = i_lo[t_idx] + k // njt
    col = j_lo[t_idx] + k % njt
    px = col * R.UNITS + R.HALF
    py = row * R.UNITS + R.HALF
    es = []
    inside = np.ones(total, dtype=bool)
    for a_, b_, c_, bias in coef:
        e = a_[t_idx] * px + b_[t_idx] * py + c_[t_idx]
        inside &= e >= bias[t_idx]
        es.append(e)
    sel = np.nonzero(inside)[0]
    t_sel, r_sel, c_sel = t_idx[sel], row[sel], col[sel]
    e0, e1, e2 = (e[sel].astype(np.float64) for e in es)
    iz = inv_z[tris[t_sel]]
    num = (e0 * iz[:, 0] + e1 * iz[:, 1]) + e2 * iz[:, 2]
    zq = np.rint(a2[t_sel].astype(np.float64) / num * R.DEPTH_SCALE).astype(np.int64)
    pix = r_sel * w + c_sel
    key = np.sort((pix << R._KEY_PIX_SHIFT) | (zq << R._KEY_Z_SHIFT) | t_sel)
    kp = key >> R._KEY_PIX_SHIFT
    first = np.ones(key.size, dtype=bool)
    first[1:] = kp[1:] != kp[:-1]
    win = key[first]
    return (win >> R._KEY_PIX_SHIFT, (win >> R._KEY_Z_SHIFT) & (R._MAX_DEPTH_UNITS - 1), win & (R._MAX_PRIO - 1),
            np.stack([t_sel, pix]))


def test_span_rasterizer_is_identical_to_the_per_pixel_version():
    """DEV-0028：rasterize_triangles 改为逐行求列区间；与逐像素判边的旧版本在各种三角形上输出逐位相同（含顺序）：
    顶点落在像素中心与格线上（边界归属规则）、水平 / 竖直边、共用边、细长、出画布、退化成一行。"""
    from src.synth import render3d as R

    rng = np.random.default_rng(21)
    for trial in range(60):
        h, w = int(rng.integers(8, 40)), int(rng.integers(8, 40))
        n_v = 30
        step = [R.UNITS // 2, R.UNITS, 1, 37][trial % 4]  # 半像素格点（正好落在像素中心）、整像素、任意
        u = rng.integers(-3 * R.UNITS, (w + 3) * R.UNITS, n_v) // step * step
        v = rng.integers(-3 * R.UNITS, (h + 3) * R.UNITS, n_v) // step * step
        tris = rng.integers(0, n_v, (25, 3))
        a2 = (u[tris[:, 1]] - u[tris[:, 0]]) * (v[tris[:, 2]] - v[tris[:, 0]]) - \
             (v[tris[:, 1]] - v[tris[:, 0]]) * (u[tris[:, 2]] - u[tris[:, 0]])
        tris = np.where((a2 < 0)[:, None], tris[:, [0, 2, 1]], tris)[a2 != 0]  # 30 个顶点上的 25 个三角形：常共用顶点与边
        inv_z = 1.0 / rng.uniform(1.0, 5.0, n_v)
        new, old = R.rasterize_triangles(u, v, inv_z, tris, h, w), _old_rasterize_triangles(u, v, inv_z, tris, h, w)
        for a, b in zip(new, old):
            assert a.dtype == b.dtype and np.array_equal(a, b), trial
    # 真实场景：a、b 的渲染（钉住的摘要也核对了这一点）
    for task in ("rotation3d", "viewpoint"):
        s = generate(task, "train", 900)
        sc = R.Scene3D.from_dict(s.meta["scene"])
        fr = R.render_posed(sc, sc.camera, R.posed_vertices(sc), want_rgb=True, want_amodal=True)
        assert np.array_equal(fr.rgb, s.image)


def test_speedups_are_bitwise_identical_to_the_old_code():
    rng = np.random.default_rng(11)
    pairs = []
    for task in ("rotation3d", "viewpoint"):
        for i in range(4):
            s = generate(task, "train", 500 + i)
            pairs.append((s.baselines["copy"].mask, s.target_mask, _units(s.extras["input_depth"])))
    for src, dst, depth in pairs:
        new = moment_matched_affine(src, dst, values=depth)
        a, t, om, refl, m, v = _old_mma(src, dst)
        assert (new.a, new.t, new.rotation_deg, new.reflect, new.iou) == (a, t, om, refl, v)
        assert np.array_equal(new.mask, m)
        om_, ov_ = _old_affine_warp_values(src, depth, a, t)
        assert np.array_equal(new.mask, om_) and np.array_equal(new.values, ov_)
        assert W.mask_moments(src) == _old_moments(src) and W.mask_moments(dst) == _old_moments(dst)
        for n in (32, 16, 7):
            g_old, u_old = _old_cell_depth_units(depth, src, n)
            g_new, u_new = cell_depth_units(depth, src, n)
            assert np.array_equal(g_old, g_new) and np.array_equal(u_old, u_new)
            assert np.array_equal(cell_depth_units(depth, src, n, g_old)[1], u_old)
    # 随机仿射（含越出画布、奇异、镜像）在随机 mask 上：像与取样的值逐位相同，各种 stride
    for k in range(30):
        h, w = int(rng.integers(20, 90)), int(rng.integers(20, 90))
        mask = rng.random((h, w)) < rng.uniform(0.05, 0.6)
        if k % 7 == 0:
            mask[:] = False
        vals = rng.integers(0, 60000, (h, w))
        a = ((float(rng.normal()), float(rng.normal())), (float(rng.normal()), float(rng.normal())))
        if k % 9 == 0:
            a = ((1.0, 2.0), (0.5, 1.0))  # 奇异
        t = (float(rng.uniform(-30, 30)), float(rng.uniform(-30, 30)))
        for st in (1, 2, 4):
            assert np.array_equal(W.affine_warp(mask, a, t, st), _old_affine_warp(mask, a, t, st))
        nm, nv = affine_warp_values(mask, vals, a, t)
        om_, ov_ = _old_affine_warp_values(mask, vals, a, t)
        assert np.array_equal(nm, om_) and np.array_equal(nv, ov_)
        if mask.any():
            assert W.mask_moments(mask) == _old_moments(mask)
            for n in (5, 8, 20):
                assert np.array_equal(cell_depth_units(vals, mask, n)[1], _old_cell_depth_units(vals, mask, n)[1])
