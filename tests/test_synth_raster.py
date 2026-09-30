"""E5 的精确整数光栅化（src/synth/raster.py，DEV-0009）。

- 构造例子的像素数精确；轴对齐矩形与 box_to_mask 逐位相同；共用边的多边形把像素不重不漏地分开；
- 窗口计算与整幅逐像素判定、与纯有理数（Fraction）参考实现逐位相同；结果只取决于多边形围成的区域；
- 入口检查：只接受 1/256 px 网格上的有限坐标，整数入口只接受 int64（没有浮点路径）；
- 与 pycocotools 的差异只在边界像素上（打印统计）；
- 钉住一组光栅化结果的摘要：在服务器上跑一遍即是跨平台核对。
"""

import hashlib
from fractions import Fraction

import numpy as np
import pytest
import shapely

from src.executors.raster import box_to_mask, polygons_to_mask
from src.synth import shapes as S
from src.synth.constants import SHAPE_FAMILIES
from src.synth.raster import HALF, MAX_ABS_COORD_PX, UNITS, rasterize, rasterize_units, to_units

H, W = 40, 56  # 故意 H ≠ W


def reference_mask(poly, h, w):
    """独立的参考：整幅画布逐像素、逐边判定（向量化的 int64），规则与 raster.py 相同，但不走扫描线、排序与展平。"""
    q = to_units(poly)
    ax, ay = q[:, 0], q[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    sw = ay > by
    x0, y0 = np.where(sw, bx, ax), np.where(sw, by, ay)
    x1, y1 = np.where(sw, ax, bx), np.where(sw, ay, by)
    yy, xx = np.mgrid[:h, :w]
    px = (xx * UNITS + HALF)[..., None]
    py = (yy * UNITS + HALF)[..., None]
    crosses = (y0 < py) & (py <= y1) & ((px - x0) * (y1 - y0) > (py - y0) * (x1 - x0))
    return (crosses.sum(axis=-1) % 2).astype(bool)


def fraction_mask(poly, h, w):
    """纯 Python 有理数实现（Fraction，没有任何浮点运算）：像素中心 (j+1/2, i+1/2) 向左的射线与边的交点数为奇数即在内；
    边只在 min(y) < cy ≤ max(y) 时计数，交点严格在中心左边才计数。只用于小画布。"""
    pts = [(Fraction(float(x)), Fraction(float(y))) for x, y in np.asarray(poly, dtype=np.float64)]
    out = np.zeros((h, w), bool)
    n = len(pts)
    for i in range(h):
        cy = Fraction(2 * i + 1, 2)
        for j in range(w):
            cx = Fraction(2 * j + 1, 2)
            count = 0
            for k in range(n):
                (xa, ya), (xb, yb) = pts[k], pts[(k + 1) % n]
                if ya == yb:
                    continue
                if ya > yb:
                    xa, ya, xb, yb = xb, yb, xa, ya
                if ya < cy <= yb and xa + (cy - ya) * (xb - xa) / (yb - ya) < cx:
                    count += 1
            out[i, j] = count % 2 == 1
    return out


def _random_polygon(rng, h, w, grid, spread=0.6):
    """星形随机多边形（按角度排序，简单但常常不凸），可能部分在画布外；顶点落在 1/grid px 网格上。"""
    n = int(rng.integers(3, 14))
    c = rng.uniform(-10, [w + 10, h + 10])
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.3, spread * max(h, w), n)
    p = np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], axis=1)
    return np.round(p * grid) / grid


# ---------------------------------------------------------------------------
# 构造例子
# ---------------------------------------------------------------------------


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def test_constructed_examples_have_exact_pixel_counts():
    # 整数角点的正方形 [10, 20] × [5, 15]：10 × 10
    m = rasterize(_rect(10, 5, 20, 15), H, W)
    assert m.sum() == 100 and m[5:15, 10:20].all()
    # 角点在像素中心上 [10.5, 20.5] × [5.5, 15.5]：(lo, hi] → 列 11..20、行 6..15
    m = rasterize(_rect(10.5, 5.5, 20.5, 15.5), H, W)
    assert m.sum() == 100 and m[6:16, 11:21].all()
    # 直角三角形 (0,0) (8,0) (0,8)：第 i 行 j ≤ 7 - i，共 8+7+…+1 = 36；斜边上的中心（i + j = 7）归三角形（右边界含）
    tri = rasterize(np.array([[0, 0], [8, 0], [0, 8]], float), H, W)
    assert tri.sum() == 36
    assert all(tri[i, 7 - i] for i in range(8)) and not any(tri[i, 8 - i] for i in range(1, 8))
    # 另一半 (0,0) (8,0) (8,8)：斜边是左边界（不含），第 i 行 j > i，共 7+6+…+0 = 28
    assert rasterize(np.array([[0, 0], [8, 0], [8, 8]], float), H, W).sum() == 28
    # 顶点都在像素中心上的菱形，中心 (4.5, 4.5)、半对角线 4：|x-4.5| + |y-4.5| ≤ 4 按 (lo, hi] 规则共 32 个
    dia = np.array([[4.5, 0.5], [8.5, 4.5], [4.5, 8.5], [0.5, 4.5]])
    m = rasterize(dia, H, W)
    assert m.sum() == 32
    assert np.array_equal(m[:12, :12], fraction_mask(dia, 12, 12)) and not m[12:].any() and not m[:, 12:].any()
    # 完全在画布外、或没有覆盖任何像素中心
    assert not rasterize(_rect(-30, -30, -5, -5), H, W).any()
    assert not rasterize(_rect(2.625, 1.625, 3.375, 2.375), H, W).any()  # 框内没有像素中心（中心在 x.5）


def test_axis_aligned_rectangles_match_box_to_mask():
    """(lo, hi] 与 src.constants.BOX_RASTER_RULE 相同：矩形多边形与 box_to_mask 逐位相同。"""
    rng = np.random.default_rng(0)
    for grid in (1, 2, 256):
        for _ in range(300):
            x = np.sort(np.round(rng.uniform(-3, W + 3, 2) * grid) / grid)
            y = np.sort(np.round(rng.uniform(-3, H + 3, 2) * grid) / grid)
            if x[0] == x[1] or y[0] == y[1]:
                continue
            b = [x[0], y[0], x[1], y[1]]
            assert np.array_equal(rasterize(_rect(*b), H, W), box_to_mask(b, H, W)), b


def test_polygons_sharing_edges_partition_the_pixels():
    """扇形三角剖分：各三角形的 mask 两两不相交，并集等于整个多边形的 mask（边上的像素中心不重不漏）。"""
    rng = np.random.default_rng(1)
    convex = [(f, v.index) for f in ("rectangle", "polygon", "triangle", "circle", "ellipse") for v in S.VARIANTS[f]]
    for k in range(40):
        fam, var = convex[k % len(convex)]
        p = S.pose_polygon(fam, var, rng.uniform(15, 41), rng.uniform(12, 28), rng.uniform(4, 14), rng.uniform(0, 360))
        whole = rasterize(p, H, W)
        parts = [rasterize(np.stack([p[0], p[i], p[i + 1]]), H, W) for i in range(1, len(p) - 1)]
        total = np.sum(parts, axis=0)
        assert total.max() <= 1, (fam, var)
        assert np.array_equal(total.astype(bool), whole), (fam, var)
    # 两个共用一条竖边的正方形、四个共用一个角的正方形
    a, b = rasterize(_rect(10.5, 5.5, 20.5, 15.5), H, W), rasterize(_rect(20.5, 5.5, 30.5, 15.5), H, W)
    assert not (a & b).any() and np.array_equal(a | b, rasterize(_rect(10.5, 5.5, 30.5, 15.5), H, W))
    quads = [rasterize(_rect(x0, y0, x0 + 7.5, y0 + 7.5), H, W) for x0 in (3.5, 11.0) for y0 in (2.5, 10.0)]
    assert np.sum(quads, axis=0).max() == 1 and np.array_equal(np.any(quads, axis=0), rasterize(_rect(3.5, 2.5, 18.5, 17.5), H, W))


# ---------------------------------------------------------------------------
# 与参考实现一致
# ---------------------------------------------------------------------------


def test_window_matches_full_canvas_reference_on_random_polygons():
    rng = np.random.default_rng(2)
    for k in range(600):
        p = _random_polygon(rng, H, W, (1, 2, 256)[k % 3])
        assert np.array_equal(rasterize(p, H, W), reference_mask(p, H, W)), k


def test_pose_polygons_window_matches_full_canvas_reference():
    """生成器实际用的多边形（全部形状族，含部分在画布外的位姿）。"""
    rng = np.random.default_rng(3)
    hh, ww = 96, 160
    for k in range(60):
        fam = SHAPE_FAMILIES[k % len(SHAPE_FAMILIES)]
        v = int(rng.integers(len(S.VARIANTS[fam])))
        cx, cy = rng.uniform(-20, ww + 20), rng.uniform(-20, hh + 20)
        p = S.pose_polygon(fam, v, cx, cy, rng.uniform(5, 50), rng.uniform(0, 360))
        assert np.array_equal(rasterize(p, hh, ww), reference_mask(p, hh, ww)), k


def test_matches_exact_rational_reference():
    """与只用 Fraction 的纯 Python 实现逐位相同：结果就是规则在这些整数输入上的数学值，与平台无关。"""
    rng = np.random.default_rng(4)
    hh, ww = 12, 15
    for k in range(24):
        p = _random_polygon(rng, hh, ww, (1, 2, 256)[k % 3], spread=0.7)
        assert np.array_equal(rasterize(p, hh, ww), fraction_mask(p, hh, ww)), k


def test_result_depends_only_on_the_enclosed_region():
    rng = np.random.default_rng(5)
    for k in range(60):
        p = _random_polygon(rng, H, W, 1)
        ref = rasterize(p, H, W)
        assert np.array_equal(rasterize(p[::-1], H, W), ref)  # 反向
        assert np.array_equal(rasterize(np.roll(p, 3, axis=0), H, W), ref)  # 换起点
        mids = (p + np.roll(p, -1, axis=0)) / 2  # 整数顶点的中点在半像素网格上，仍在 1/256 网格上
        with_mid = np.empty((2 * len(p), 2))
        with_mid[0::2], with_mid[1::2] = p, mids
        assert np.array_equal(rasterize(with_mid, H, W), ref)  # 插入共线顶点


def test_integer_translation_shifts_the_mask_exactly():
    rng = np.random.default_rng(6)
    for _ in range(40):
        p = _random_polygon(rng, 30, 30, 256, spread=0.3)
        p = p - p.min(axis=0) + 20.0  # 放进 [20, 40]²，平移 ±9 后仍离画布边缘很远（np.roll 不会绕回）
        dx, dy = (int(v) for v in rng.integers(-9, 10, 2))
        a = rasterize(p, 80, 80)
        b = rasterize(p + [dx, dy], 80, 80)
        assert np.array_equal(np.roll(np.roll(a, dy, axis=0), dx, axis=1), b)


# ---------------------------------------------------------------------------
# 入口检查：没有浮点路径
# ---------------------------------------------------------------------------


def test_entry_checks():
    with pytest.raises(ValueError, match="网格"):
        rasterize(np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 1.0 / 3.0]]), H, W)  # 1/3 不在 1/256 网格上
    with pytest.raises(ValueError):
        rasterize(np.array([[0.0, 0.0], [10.0, np.nan], [5.0, 5.0]]), H, W)
    with pytest.raises(ValueError):
        rasterize(np.array([[0.0, 0.0], [10.0, 0.0]]), H, W)  # 少于 3 个顶点
    with pytest.raises(ValueError):
        rasterize(np.array([[0.0, 0.0], [2 * MAX_ABS_COORD_PX, 0.0], [5.0, 5.0]]), H, W)
    with pytest.raises(ValueError):
        rasterize(_rect(1, 1, 5, 5), 0, W)
    q = to_units(_rect(1, 1, 5, 5))
    assert q.dtype == np.int64 and np.array_equal(q, np.array([[1, 1], [5, 1], [5, 5], [1, 5]]) * UNITS)
    with pytest.raises(TypeError):
        rasterize_units(q.astype(np.float64), H, W)  # 整数入口拒绝浮点数组
    with pytest.raises(TypeError):
        rasterize_units(q.astype(np.int32), H, W)
    assert np.array_equal(rasterize_units(q, H, W), rasterize(_rect(1, 1, 5, 5), H, W))


# ---------------------------------------------------------------------------
# 与 pycocotools 的差异
# ---------------------------------------------------------------------------


def test_differences_from_pycocotools_are_boundary_pixels_only():
    """生成器式的随机多边形（512² 画布）：与 pycocotools 不同的像素，中心离多边形边界都不到 0.25 px。

    pycocotools 把坐标放大 5 倍取整（顶点偏移 ≤ 0.1 px），沿边描点再取整（≤ 0.1 px），所以不同只出现在边界附近。
    DEV-0009 实测（400 个多边形）：不同的像素占并集的 0.18%、占"中心离边界 ≤ 0.5 px 的像素"的 5.3%，
    两个方向的个数相当（没有系统偏差），离边界最远 0.18 px。
    """
    rng = np.random.default_rng(2026)
    hh = ww = 512
    n_diff = n_union = n_near = n_ours_only = n_coco_only = 0
    worst = 0.0
    for k in range(120):
        fam = SHAPE_FAMILIES[k % len(SHAPE_FAMILIES)]
        v = int(rng.integers(len(S.VARIANTS[fam])))
        r = rng.uniform(10, 140)
        p = S.pose_polygon(fam, v, rng.uniform(r, ww - r), rng.uniform(r, hh - r), r, S.quantize(rng.uniform(0, 360), 1 / 256))
        ours, coco = rasterize(p, hh, ww), polygons_to_mask([p.reshape(-1)], hh, ww)
        g = shapely.polygons(p)
        rows, cols = np.nonzero(ours ^ coco)
        if rows.size:
            worst = max(worst, float(shapely.distance(g.exterior, shapely.points(cols + 0.5, rows + 0.5)).max()))
        x0, x1 = max(int(p[:, 0].min()) - 2, 0), min(int(p[:, 0].max()) + 3, ww)
        y0, y1 = max(int(p[:, 1].min()) - 2, 0), min(int(p[:, 1].max()) + 3, hh)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        n_near += int((shapely.distance(g.exterior, shapely.points(xx.ravel() + 0.5, yy.ravel() + 0.5)) <= 0.5).sum())
        n_diff += rows.size
        n_union += int((ours | coco).sum())
        n_ours_only += int((ours & ~coco).sum())
        n_coco_only += int((coco & ~ours).sum())
    print(
        f"\nexact vs pycocotools (120 polygons, 512x512): differing px {n_diff} = {n_diff / n_union:.4%} of union, "
        f"{n_diff / n_near:.2%} of near-boundary px (center ≤ 0.5 px from edge); ours-only {n_ours_only}, "
        f"coco-only {n_coco_only}; max center-to-edge distance {worst:.3f} px"
    )
    assert worst <= 0.25
    assert n_diff / n_union < 0.005
    assert abs(n_ours_only - n_coco_only) <= 0.2 * n_diff  # 没有系统性地偏大或偏小


# ---------------------------------------------------------------------------
# 钉住：跨平台核对
# ---------------------------------------------------------------------------


def test_pinned_raster_digest():
    """全部形状族、一组固定位姿（含部分在画布外、角点在像素中心上）的光栅化结果的摘要。

    在开发机上红 = 光栅化规则或单位形状变了（生成器版本要加一）；只在服务器上红 = 跨平台不能逐位复现，上报管理员。
    """
    h = hashlib.sha256()
    hh, ww = 64, 96
    for fam in SHAPE_FAMILIES:
        for v in S.VARIANTS[fam]:
            for cx, cy, r, a in ((48.0, 32.0, 25.0, 0.0), (47.5, 31.5, 17.25, 33.0), (90.0, -4.0, 20.0, 211.5), (10.0, 60.0, 9.0, 1.0 / 256)):
                h.update(np.packbits(rasterize(S.pose_polygon(fam, v.index, cx, cy, r, a), hh, ww)).tobytes())
    assert h.hexdigest() == PINNED_RASTER_DIGEST


PINNED_RASTER_DIGEST = "2fbf534f09932248e717b1209fa29d61df3c306b248e2d5fab748ffaed907435"  # 开发机 arm64 macOS，DEV-0009
