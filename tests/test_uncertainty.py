"""按不确定度选区；文件末尾是 DEV-0002 的 E3 执行器端到端彩排（加噪 → 叠加 → 分块拼回 → 上采样 → 选区）。

直接运行 `python -m tests.test_uncertainty` 打印彩排数值。
"""

import inspect
import math

import numpy as np
import pytest
from scipy import ndimage

from src.constants import (
    GRID_SIZE_E3B,
    UNCERTAINTY_ENTROPY_QUANTUM,
    UNCERTAINTY_NUM_REGIONS,
    UNCERTAINTY_PROB_SUM_ATOL,
    UNCERTAINTY_REGION_SIZE,
)
from src.executors.overlay import mask_contour, render_overlay
from src.executors.raster import polygons_to_mask
from src.executors.tiling import make_tiles, stitch
from src.executors.uncertainty import Region, entropy_map, select_uncertain_regions
from src.executors.upsample import bilinear_upsample, guided_upsample, joint_bilateral_upsample
from src.grid.codec import cell_edges, grid_to_mask, mask_to_grid
from src.metrics.seg import boundary_iou, mask_inter_union, mask_to_boundary
from src.noising.mask_noise import corrupt

H, W = 41, 67  # H ≠ W


def _windows(regions):
    return [(r.r0, r.r1, r.c0, r.c1) for r in regions]


def _disjoint(a, b):
    return a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2]


# ---------------------------------------------------------------- 常量与签名


def test_uncertainty_constants_pinned():
    assert (UNCERTAINTY_NUM_REGIONS, UNCERTAINTY_REGION_SIZE) == (4, 64)
    # 注释："64 px 在 640 宽的图上约为 32×32 网格的 3 个格子"
    assert round(UNCERTAINTY_REGION_SIZE / (640 / GRID_SIZE_E3B)) == 3
    assert UNCERTAINTY_ENTROPY_QUANTUM == 2.0**-32 and math.log2(UNCERTAINTY_ENTROPY_QUANTUM).is_integer()
    assert UNCERTAINTY_PROB_SUM_ATOL == 1e-6


def test_signature_has_no_task_parameter():
    # 简报 §4 第 11 条、N8：选区函数只收分布、区域数、边长
    assert list(inspect.signature(select_uncertain_regions).parameters) == ["dist", "k", "size"]
    assert list(inspect.signature(entropy_map).parameters) == ["dist"]


# ---------------------------------------------------------------- 熵


def test_entropy_values_and_binary_equals_two_class():
    p = np.random.default_rng(0).random((H, W))
    p[0, 0], p[0, 1], p[0, 2] = 0.0, 1.0, 0.5
    e = entropy_map(p)
    assert e[0, 0] == 0.0 and e[0, 1] == 0.0 and e[0, 2] == pytest.approx(math.log(2), abs=1e-15)
    two = np.stack([1.0 - p, p], axis=2)
    assert np.array_equal(entropy_map(two), e)  # 逐位相同
    uniform = np.full((H, W, 64), 1 / 64)
    np.testing.assert_allclose(entropy_map(uniform), math.log(64), rtol=1e-14)
    onehot = np.zeros((H, W, 64))
    onehot[..., 7] = 1.0
    assert np.all(entropy_map(onehot) == 0.0)


def _layouts(a):
    """同一组数值的三种内存布局：C 连续、F 连续、从更大数组切出的步长视图。"""
    big = np.zeros((a.shape[0] * 2, a.shape[1] * 3) + a.shape[2:], a.dtype)
    big[::2, ::3] = a
    return [np.ascontiguousarray(a), np.asfortranarray(a), big[::2, ::3]]


def test_results_do_not_depend_on_memory_layout():
    rng = np.random.default_rng(9)
    dist = rng.dirichlet(np.ones(64), size=(H, W))
    ref_e = entropy_map(dist)
    ref_r = select_uncertain_regions(dist, 5, 7)
    for v in _layouts(dist):
        assert np.array_equal(entropy_map(v), ref_e)  # 逐位
        assert select_uncertain_regions(v, 5, 7) == ref_r


def test_entropy_input_checks():
    with pytest.raises(ValueError):
        entropy_map(np.full((3, 4), 1.5))
    with pytest.raises(ValueError):
        entropy_map(np.full((3, 4), np.nan))
    with pytest.raises(ValueError):
        entropy_map(np.full((3, 4, 3), 0.3))  # 和为 0.9
    with pytest.raises(ValueError):
        entropy_map(np.ones((3, 4), dtype=np.int64))
    with pytest.raises(ValueError):
        entropy_map(np.full((3, 4, 1), 1.0))
    ok = np.full((3, 4, 3), 1 / 3)
    ok[0, 0] += [UNCERTAINTY_PROB_SUM_ATOL / 2, 0, 0]
    entropy_map(ok)  # 容差内不报错


# ---------------------------------------------------------------- 选区


def test_highest_entropy_location_is_selected():
    p = np.zeros((H, W))
    p[30, 50] = 0.5  # 唯一不确定的像素
    size = 8
    regs = select_uncertain_regions(p, k=1, size=size)
    # 所有包含它的窗口得分相同，按平局规则取行优先最小的左上角
    assert _windows(regs) == [(30 - size + 1, 31, 50 - size + 1, 51)]
    assert regs[0].mean_entropy == pytest.approx(math.log(2) / size**2, rel=1e-9)
    # 多类情形同理：一个像素分布均匀，其余是 one-hot
    d = np.zeros((H, W, 5))
    d[..., 2] = 1.0
    d[12, 3] = 0.2
    assert _windows(select_uncertain_regions(d, k=1, size=6)) == [(7, 13, 0, 6)]


def test_ties_broken_in_raster_order_with_exact_equal_scores():
    rng = np.random.default_rng(1)
    patch = rng.random((3, 3))  # 非常数内容：浮点求和顺序不同也必须精确打平
    p = np.zeros((H, W))
    p[20:23, 40:43] = patch  # 更靠左，但更靠下
    p[5:8, 60:63] = patch  # 更靠上
    regs = select_uncertain_regions(p, k=2, size=8)
    assert _windows(regs) == [(0, 8, 55, 63), (15, 23, 35, 43)]
    assert regs[0].mean_entropy == regs[1].mean_entropy  # 严格相等


def test_regions_disjoint_in_bounds_and_scores_nonincreasing():
    rng = np.random.default_rng(2)
    maps = [rng.random((H, W)), rng.dirichlet(np.ones(4), size=(H, W)), ndimage.gaussian_filter(rng.random((H, W)), 3)]
    for dist in maps:
        for size in (5, 9, 16):
            regs = select_uncertain_regions(dist, k=50, size=size)
            wins = _windows(regs)
            assert len(regs) >= 1
            for r0, r1, c0, c1 in wins:
                assert 0 <= r0 and r1 <= H and 0 <= c0 and c1 <= W and r1 - r0 == size and c1 - c0 == size
            assert all(_disjoint(a, b) for i, a in enumerate(wins) for b in wins[i + 1 :])
            scores = [r.mean_entropy for r in regs]
            assert all(x >= y for x, y in zip(scores, scores[1:]))
            assert isinstance(regs[0], Region)


def test_k_larger_than_available_returns_fewer():
    # 16×24 图、边长 8：全确定（熵为 0）时仍按行优先铺满，恰好 6 个互不重叠的窗口
    regs = select_uncertain_regions(np.zeros((16, 24)), k=10, size=8)
    assert _windows(regs) == [(0, 8, 0, 8), (0, 8, 8, 16), (0, 8, 16, 24), (8, 16, 0, 8), (8, 16, 8, 16), (8, 16, 16, 24)]
    assert all(r.mean_entropy == 0.0 for r in regs)
    # 10×12、边长 8：选中一个后其余候选全部与它重叠，只返回 1 个
    assert len(select_uncertain_regions(np.random.default_rng(3).random((10, 12)), k=5, size=8)) == 1
    assert select_uncertain_regions(np.zeros((16, 24)), k=0, size=8) == []
    with pytest.raises(ValueError):
        select_uncertain_regions(np.zeros((16, 24)), k=1, size=17)
    with pytest.raises(ValueError):
        select_uncertain_regions(np.zeros((16, 24)), k=-1, size=4)


def test_binary_and_multiclass_inputs_through_one_function():
    rng = np.random.default_rng(4)
    p = ndimage.gaussian_filter(rng.random((H, W)), 2)
    p = (p - p.min()) / (p.max() - p.min())
    two = np.stack([1.0 - p, p], axis=2)
    assert select_uncertain_regions(p, 3, 7) == select_uncertain_regions(two, 3, 7)
    # 深度式的 64 档分布：左右两半各是尖峰，中间一条竖带分布变宽（不连续处），选区应落在竖带上
    d = np.zeros((H, W, 64))
    d[:, :30, 10] = 1.0
    d[:, 34:, 50] = 1.0
    d[:, 30:34, :] = 1 / 64
    for reg in select_uncertain_regions(d, 2, 6):
        assert reg.c0 <= 33 and reg.c1 >= 31  # 与竖带相交


# ---------------------------------------------------------------- 端到端彩排（DEV-0002 验收）


RH, RW = 480, 640  # 与 COCO / NYUv2 相同的尺寸，H ≠ W
GT_POLY = [180.0, 110.0, 330.0, 95.0, 420.0, 170.0, 395.0, 300.0, 290.0, 345.0, 205.0, 280.0, 160.0, 200.0]
NEIGHBOR_POLY = [430.0, 180.0, 540.0, 170.0, 560.0, 300.0, 450.0, 330.0, 410.0, 290.0]


def _area_fraction(mask, n):
    re_, ce = cell_edges(mask.shape[0], n), cell_edges(mask.shape[1], n)
    fg = np.add.reduceat(np.add.reduceat(mask.astype(np.float64), re_[:-1], axis=0), ce[:-1], axis=1)
    return fg / (np.diff(re_)[:, None] * np.diff(ce)[None, :])


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    gt = polygons_to_mask([GT_POLY], RH, RW)
    nb = polygons_to_mask([NEIGHBOR_POLY], RH, RW) & ~gt
    yy, xx = np.mgrid[:RH, :RW] / np.array([RH, RW])[:, None, None]
    img = np.stack([0.35 + 0.3 * xx, 0.45 + 0.2 * yy, 0.5 - 0.2 * xx * yy], axis=2)  # 平滑渐变背景
    img[gt] = [0.75, 0.30, 0.25]
    img[nb] = [0.30, 0.60, 0.35]
    tex = ndimage.gaussian_filter(rng.standard_normal((RH, RW, 3)), (3, 3, 0))
    img = img + 0.06 * tex / tex.std() + 0.04 * rng.standard_normal((RH, RW, 3))
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8), gt, nb


def _iou(a, b):
    i, u = mask_inter_union(a, b)
    return i / u


def rehearse_e3(seed: int = 0) -> dict:
    img, gt, nb = _scene(seed)
    out = {"H": RH, "W": RW, "gt_pixels": int(gt.sum()), "neighbor_pixels": int(nb.sum())}

    # 1. 加噪（s = 0.5）
    noisy, rec = corrupt(gt, 0.5, np.random.default_rng(seed), others=[nb])
    out["1_noisy_iou_vs_gt"] = _iou(noisy, gt)
    out["1_noise_ops"] = [op["op"] for op in rec["ops"]]
    out["1_noise_record"] = rec

    # 2. 叠加渲染
    ov = render_overlay(img, [noisy])
    changed = np.any(ov != img, axis=2)
    out["2_overlay_changed_pixels"] = int(changed.sum())
    out["2_overlay_outside_mask_identical"] = bool(np.array_equal(ov[~noisy], img[~noisy]))
    out["2_contour_pixels"] = int(mask_contour(noisy).sum())

    # 3. 分块后拼回：叠加图（转 float64）和一张 float64 logit 场，都要逐位还原
    tiles = make_tiles(RH, RW)
    cover = np.zeros((RH, RW), np.int64)
    for r0, r1, c0, c1 in tiles:
        cover[r0:r1, c0:c1] += 1
    field = np.random.default_rng(seed + 1).standard_normal((RH, RW, 2)) * 3.0
    ok = []
    for arr in (ov.astype(np.float64), field):
        back = stitch([arr[r0:r1, c0:c1] for r0, r1, c0, c1 in tiles], tiles, RH, RW)
        ok.append(bool(np.array_equal(back.view(np.int64), arr.view(np.int64))))
    out["3_tiles"] = tiles
    out["3_n_tiles"] = len(tiles)
    out["3_max_cover"] = int(cover.max())
    out["3_bitwise_restored"] = all(ok)

    # 4. 低分辨率 logit（GT 的格内前景占比映射成 logit）放大：引导上采样 vs 双线性
    for n in (GRID_SIZE_E3B, 128):
        logit = 10.0 * (_area_fraction(gt, n) - 0.5)
        up_g = guided_upsample(logit, img)
        up_b = bilinear_upsample(logit, RH, RW)
        out[f"4_{n}_boundary_iou_guided"] = boundary_iou(up_g > 0, gt)
        out[f"4_{n}_boundary_iou_bilinear"] = boundary_iou(up_b > 0, gt)
        out[f"4_{n}_boundary_iou_jbu"] = boundary_iou(joint_bilateral_upsample(logit, img) > 0, gt)
        out[f"4_{n}_iou_guided"] = _iou(up_g > 0, gt)
        out[f"4_{n}_iou_bilinear"] = _iou(up_b > 0, gt)
        if n == GRID_SIZE_E3B:
            hard = grid_to_mask(mask_to_grid(gt, n), RH, RW)  # 不细化的硬网格，作参照
            out[f"4_{n}_boundary_iou_hard_grid"] = boundary_iou(hard, gt)
            guided_logit_32 = up_g

    # 5. 在上采样后的概率图上选区
    prob = 1.0 / (1.0 + np.exp(-guided_logit_32))
    regions = select_uncertain_regions(prob)
    band = mask_to_boundary(gt)
    out["5_regions"] = [(r.r0, r.r1, r.c0, r.c1, round(r.mean_entropy, 4)) for r in regions]
    out["5_regions_touch_gt_boundary"] = [bool(band[r.r0 : r.r1, r.c0 : r.c1].any()) for r in regions]
    return out


def test_rehearsal_e3_executor_chain():
    r = rehearse_e3()
    assert 0.3 < r["1_noisy_iou_vs_gt"] < 0.95  # s = 0.5：明显带错但不是全错
    assert r["2_overlay_outside_mask_identical"] and r["2_overlay_changed_pixels"] > 0
    assert r["3_bitwise_restored"] and r["3_max_cover"] >= 3
    for n in (GRID_SIZE_E3B, 128):
        assert r[f"4_{n}_boundary_iou_guided"] > r[f"4_{n}_boundary_iou_bilinear"]
        assert r[f"4_{n}_boundary_iou_jbu"] > r[f"4_{n}_boundary_iou_bilinear"]
    assert len(r["5_regions"]) == UNCERTAINTY_NUM_REGIONS
    assert all(r["5_regions_touch_gt_boundary"])


if __name__ == "__main__":
    for key, value in rehearse_e3().items():
        print(f"{key}: {value}")
