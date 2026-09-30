"""无学习上采样：盒式均值、引导滤波（灰度/彩色）、双线性、引导上采样、联合双边上采样。"""

import inspect
import math

import numpy as np
import pytest
from scipy import ndimage

from src.constants import GRID_SIZE_E3B, GUIDED_FILTER_EPS, GUIDED_FILTER_RADIUS, JBU_RADIUS, JBU_SIGMA_RANGE, JBU_SIGMA_SPATIAL
from src.executors import upsample as U
from src.executors.raster import polygons_to_mask
from src.executors.upsample import (
    bilinear_upsample,
    box_mean,
    guide_to_unit,
    guided_filter,
    guided_upsample,
    joint_bilateral_upsample,
)
from src.grid.codec import cell_edges
from src.metrics.seg import boundary_iou, mask_to_boundary

H, W = 13, 21  # H ≠ W


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- 常量


def test_upsample_constants_pinned():
    assert (GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPS) == (8, 1e-3)
    # 注释："32×32 网格放大到 480×640 是 15–20 倍、128×128 放大 3.75–5 倍；半径 8 即窗口 17 px，约一个格子（20 px）；
    # eps 对应局部标准差约 0.03"
    assert (480 / GRID_SIZE_E3B, 640 / GRID_SIZE_E3B, 480 / 128, 640 / 128) == (15, 20, 3.75, 5)
    assert 2 * GUIDED_FILTER_RADIUS + 1 == 17 and 640 // GRID_SIZE_E3B == 20
    assert abs(math.sqrt(GUIDED_FILTER_EPS) - 0.03) < 0.002
    assert (JBU_SIGMA_SPATIAL, JBU_SIGMA_RANGE, JBU_RADIUS) == (1.0, 0.2, 2)
    # 注释："半径 2 即 5×5 个低分辨率像素，截断在 ±2 sigma"
    assert 2 * JBU_RADIUS + 1 == 5 and JBU_RADIUS == 2 * JBU_SIGMA_SPATIAL


def test_signatures_carry_no_task_identity():
    for fn in (box_mean, guided_filter, bilinear_upsample, guided_upsample, joint_bilateral_upsample):
        assert not any("task" in name for name in inspect.signature(fn).parameters)


# ---------------------------------------------------------------- 盒式均值


def _naive_box_mean(x, r):
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = x[max(0, i - r) : i + r + 1, max(0, j - r) : j + r + 1].mean(axis=(0, 1))
    return out


@pytest.mark.parametrize("r", [0, 1, 3, 30])
def test_box_mean_matches_naive_clipped_window(r):
    x = _rng(r).standard_normal((H, W, 2))
    np.testing.assert_allclose(box_mean(x, r), _naive_box_mean(x, r), rtol=0, atol=1e-12)
    np.testing.assert_allclose(box_mean(x[..., 1], r), _naive_box_mean(x[..., 1], r), rtol=0, atol=1e-12)


# ---------------------------------------------------------------- 引导滤波：参照实现


def _naive_guided(guide, p, r, eps):
    """逐窗口直接按定义算：窗口内总体协方差（先减均值），np.linalg.solve 求 a_k，再对覆盖像素的窗口平均。"""
    g = guide if guide.ndim == 3 else guide[..., None]
    h, w, k = g.shape
    a = np.zeros((h, w, k))
    b = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            sl = (slice(max(0, i - r), i + r + 1), slice(max(0, j - r), j + r + 1))
            gw = g[sl].reshape(-1, k)
            pw = p[sl].reshape(-1)
            mu, pm = gw.mean(axis=0), pw.mean()
            sigma = (gw - mu).T @ (gw - mu) / pw.size
            cov = (gw - mu).T @ (pw - pm) / pw.size
            a[i, j] = np.linalg.solve(sigma + eps * np.eye(k), cov)
            b[i, j] = pm - a[i, j] @ mu
    q = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            sl = (slice(max(0, i - r), i + r + 1), slice(max(0, j - r), j + r + 1))
            q[i, j] = a[sl].reshape(-1, k).mean(axis=0) @ g[i, j] + b[sl].mean()
    return q


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("r,eps", [(1, 1e-2), (2, 1e-4), (4, 1e-1)])
def test_guided_filter_matches_naive_reference(color, r, eps):
    rng = _rng(10 * r + color)
    guide = rng.random((H, W, 3)) if color else rng.random((H, W))
    p = rng.standard_normal((H, W))
    # 容差：两条路径的求和顺序与求逆方式不同；这里 cond(Σ+eps·U) ≤ (3·0.25+eps)/eps ≤ 7.6e3，误差远小于 1e-9
    np.testing.assert_allclose(guided_filter(guide, p, r, eps), _naive_guided(guide, p, r, eps), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- 引导滤波：退化与等价关系


@pytest.mark.parametrize("color", [False, True])
def test_constant_guide_is_exactly_box_mean_applied_twice(color):
    rng = _rng(3)
    p = rng.standard_normal((H, W))
    guide = np.full((H, W, 3), [0.3, 0.7, 0.1]) if color else np.full((H, W), 0.3)
    for r in (1, 2, 5):
        q = guided_filter(guide, p, r, 1e-2)
        # 常数引导图：a ≡ 0，b = 窗口均值，输出 = 窗口均值再取窗口均值，逐位相同
        assert np.array_equal(q, box_mean(box_mean(p, r), r))


def test_constant_guide_is_not_a_single_box_mean():
    # 防止"精确退化为盒式均值滤波（一次）"的说法溜回来：单位冲激经两次盒式均值后支撑宽 4r+1，一次只有 2r+1
    r = 2
    p = np.zeros((H, W))
    p[6, 10] = 1.0
    q = guided_filter(np.full((H, W), 0.5), p, r, 1e-2)
    once = box_mean(p, r)
    assert np.count_nonzero(once) == (2 * r + 1) ** 2
    assert np.count_nonzero(np.abs(q) > 1e-15) == (4 * r + 1) ** 2
    assert not np.allclose(q, once)


def test_color_on_identical_channels_equals_gray_with_eps_over_3():
    rng = _rng(4)
    gray = rng.random((H, W))
    guide3 = np.repeat(gray[..., None], 3, axis=2)
    p = rng.standard_normal((H, W))
    for r, eps in [(1, 1e-2), (3, 1e-3), (2, 3e-1)]:
        color_q = guided_filter(guide3, p, r, eps)
        # Σ = v·11ᵀ，(Σ + eps·U)a = c·1 的解为 a = c/(3v + eps)·1，三通道加起来的系数 = c/(v + eps/3)。
        # 容差：两条路径一边解 3×3 方程、一边做标量除法，舍入不同；引导图在 [0, 1]，
        # cond(Σ+eps·U) ≤ (0.75+eps)/eps ≤ 751，p 的量级 ~3，误差上界约 751 × 3 × 1e-16 × 几十次运算 < 1e-10。
        np.testing.assert_allclose(color_q, guided_filter(gray, p, r, eps / 3.0), rtol=0, atol=1e-10)
        # 同一个 eps 下并不一致（任务书"与灰度版一致"只在 eps 换算后成立）
        assert np.abs(color_q - guided_filter(gray, p, r, eps)).max() > 1e-6


def test_guided_filter_invariant_to_guide_offset_and_channelwise():
    rng = _rng(5)
    guide = rng.random((H, W, 3))
    p = rng.standard_normal((H, W, 19))  # 19 > _CHANNEL_CHUNK，跨块
    q = guided_filter(guide, p, 2, 1e-3)
    # 逐通道单独算与整体算逐位相同（分块只为省内存）
    for c in (0, 7, 8, 18):
        assert np.array_equal(q[..., c], guided_filter(guide, p[..., c], 2, 1e-3))
    # 引导图加常数，数学上输出不变
    np.testing.assert_allclose(guided_filter(guide + 5.0, p, 2, 1e-3), q, rtol=0, atol=1e-9)
    gray = guide[..., 0]
    qg = guided_filter(gray, p, 2, 1e-3)
    for c in (0, 8, 18):
        assert np.array_equal(qg[..., c], guided_filter(gray, p[..., c], 2, 1e-3))


def test_guided_filter_input_checks():
    p = np.zeros((H, W))
    with pytest.raises(ValueError):
        guided_filter(np.zeros((H, W), np.uint8), p, 2, 1e-3)  # 整数引导图：eps 单位不明，拒绝
    with pytest.raises(ValueError):
        guided_filter(np.zeros((H, W)), p, 2, 0.0)
    with pytest.raises(ValueError):
        guided_filter(np.zeros((H, W)), np.zeros((W, H)), 2, 1e-3)
    with pytest.raises(ValueError):
        guided_filter(np.zeros((H, W, 2)), p, 2, 1e-3)
    with pytest.raises(ValueError):
        guided_filter(np.zeros((H, W)), np.full((H, W), np.nan), 2, 1e-3)
    with pytest.raises(ValueError):
        guided_filter(np.zeros((H, W)), p, -1, 1e-3)


# ---------------------------------------------------------------- 双线性


@pytest.mark.parametrize("hw,HW", [((5, 7), (13, 29)), ((32, 32), (90, 140)), ((7, 5), (8, 21)), ((3, 4), (3, 4))])
def test_bilinear_matches_torch_align_corners_false(hw, HW):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    x = _rng(sum(hw)).standard_normal(hw + (3,))
    ours = bilinear_upsample(x, *HW)
    ref = F.interpolate(torch.from_numpy(x).permute(2, 0, 1)[None], size=HW, mode="bilinear", align_corners=False)
    np.testing.assert_allclose(ours, ref[0].permute(1, 2, 0).numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(bilinear_upsample(x[..., 0], *HW), ours[..., 0], rtol=0, atol=0)


def test_bilinear_same_size_is_identity_and_rejects_downsampling():
    x = _rng(6).standard_normal((H, W))
    assert np.array_equal(bilinear_upsample(x, H, W), x)
    with pytest.raises(ValueError):
        bilinear_upsample(x, H - 1, W)
    with pytest.raises(ValueError):
        bilinear_upsample(x, H, W - 1)


# ---------------------------------------------------------------- 引导上采样 vs 双线性（合成阶跃边缘 + 噪声）


def _area_fraction(mask, n):
    re_, ce = cell_edges(mask.shape[0], n), cell_edges(mask.shape[1], n)
    fg = np.add.reduceat(np.add.reduceat(mask.astype(np.float64), re_[:-1], axis=0), ce[:-1], axis=1)
    return fg / (np.diff(re_)[:, None] * np.diff(ce)[None, :])


def _noisy_edge_scene(h, w, seed):
    """不对称多边形物体 + 平滑纹理 + 像素噪声；物体与背景的颜色、形状、位置随种子变化。"""
    rng = _rng(seed)
    cy, cx = h * rng.uniform(0.35, 0.65), w * rng.uniform(0.35, 0.65)
    ang = np.sort(rng.uniform(0, 2 * np.pi, 9))
    rad = rng.uniform(0.18, 0.38, 9) * min(h, w)
    poly = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).reshape(-1).tolist()
    gt = polygons_to_mask([poly], h, w)
    bg = rng.uniform(0.2, 0.8, 3)
    fg = np.clip(bg + rng.choice([-1.0, 1.0], 3) * rng.uniform(0.15, 0.35, 3), 0, 1)
    tex = ndimage.gaussian_filter(rng.standard_normal((h, w, 3)), (3, 3, 0))
    img = np.where(gt[..., None], fg, bg) + 0.08 * tex / tex.std() + 0.05 * rng.standard_normal((h, w, 3))
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8), gt


def _band_errors(pred, gt):
    band = ndimage.binary_dilation(mask_to_boundary(gt), iterations=3)
    return int(np.count_nonzero((pred != gt) & band))


def test_guided_upsample_beats_bilinear_on_noisy_step_edges():
    h, w, n = 120, 200, 16  # 放大 7.5×12.5 倍
    for seed in range(4):
        img, gt = _noisy_edge_scene(h, w, seed)
        logit = 10.0 * (_area_fraction(gt, n) - 0.5)
        guided = guided_upsample(logit, img) > 0
        bilin = bilinear_upsample(logit, h, w) > 0
        biou_g, biou_b = boundary_iou(guided, gt), boundary_iou(bilin, gt)
        assert biou_g > biou_b + 0.03, (seed, biou_g, biou_b)
        assert _band_errors(guided, gt) < _band_errors(bilin, gt)


def test_guided_upsample_is_bilinear_then_guided_filter_with_constants():
    img, _ = _noisy_edge_scene(40, 64, 7)
    low = _rng(8).standard_normal((8, 8, 2))
    ref = guided_filter(img.astype(np.float64) / 255.0, bilinear_upsample(low, 40, 64), GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPS)
    assert np.array_equal(guided_upsample(low, img), ref)
    gray = img[..., 0]
    ref_g = guided_filter(gray / 255.0, bilinear_upsample(low, 40, 64), GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPS)
    assert np.array_equal(guided_upsample(low, gray), ref_g)


def test_guide_to_unit_checks():
    assert guide_to_unit(np.full((2, 3), 255, np.uint8)).max() == 1.0
    with pytest.raises(ValueError):
        guide_to_unit(np.full((2, 3), 2.0))  # 浮点但不在 [0, 1]：不猜量程
    with pytest.raises(ValueError):
        guide_to_unit(np.full((2, 3), 7, np.int64))


# ---------------------------------------------------------------- 联合双边上采样


def _naive_jbu(low, guide01, ss, sr, R):
    h, w = low.shape
    hh, ww = guide01.shape[:2]
    g = guide01 if guide01.ndim == 3 else guide01[..., None]
    out = np.zeros((hh, ww))
    for i in range(hh):
        for j in range(ww):
            py, px = (i + 0.5) * h / hh - 0.5, (j + 0.5) * w / ww - 0.5
            cy = min(max(int(np.floor(py + 0.5)), 0), h - 1)
            cx = min(max(int(np.floor(px + 0.5)), 0), w - 1)
            num = den = 0.0
            for qy in range(cy - R, cy + R + 1):
                for qx in range(cx - R, cx + R + 1):
                    if not (0 <= qy < h and 0 <= qx < w):
                        continue
                    gy, gx = ((2 * qy + 1) * hh) // (2 * h), ((2 * qx + 1) * ww) // (2 * w)
                    f = np.exp(-((qy - py) ** 2 + (qx - px) ** 2) / (2 * ss * ss))
                    d2 = float(np.sum((g[i, j] - g[gy, gx]) ** 2))
                    wt = f * np.exp(-d2 / (2 * sr * sr))
                    num += wt * low[qy, qx]
                    den += wt
            out[i, j] = num / den
    return out


@pytest.mark.parametrize("color", [False, True])
def test_jbu_matches_naive_reference(color):
    rng = _rng(20 + color)
    low = rng.standard_normal((4, 5))
    guide = rng.random((11, 17, 3)) if color else rng.random((11, 17))
    for ss, sr, R in [(1.0, 0.2, 2), (0.5, 0.1, 1), (2.0, 0.5, 3)]:
        ours = joint_bilateral_upsample(low, guide, sigma_spatial=ss, sigma_range=sr, radius=R)
        np.testing.assert_allclose(ours, _naive_jbu(low, guide, ss, sr, R), rtol=1e-12, atol=1e-12)


def test_jbu_special_cases():
    rng = _rng(22)
    guide = rng.random((11, 17, 3))
    # 常数输入 → 常数输出（归一化权重）
    np.testing.assert_allclose(joint_bilateral_upsample(np.full((4, 5), 2.5), guide), 2.5, rtol=0, atol=1e-14)
    # radius=0：只有最近的低分辨率像素，即半像素中心对齐的最近邻放大
    low = rng.standard_normal((4, 5))
    nn = joint_bilateral_upsample(low, guide, radius=0)
    iy = np.clip(np.floor((np.arange(11) + 0.5) * 4 / 11 - 0.5 + 0.5).astype(int), 0, 3)
    ix = np.clip(np.floor((np.arange(17) + 0.5) * 5 / 17 - 0.5 + 0.5).astype(int), 0, 4)
    assert np.array_equal(nn, low[iy][:, ix])
    # sigma_range=inf：值域项消失，与引导图无关
    a = joint_bilateral_upsample(low, guide, sigma_range=np.inf)
    b = joint_bilateral_upsample(low, rng.random((11, 17, 3)), sigma_range=np.inf)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)
    # 多通道逐通道一致
    low3 = rng.standard_normal((4, 5, 3))
    out3 = joint_bilateral_upsample(low3, guide)
    np.testing.assert_allclose(out3[..., 2], joint_bilateral_upsample(low3[..., 2], guide), rtol=0, atol=1e-15)
    # 值域项极小也不下溢成 0/0
    tiny = joint_bilateral_upsample(low, guide, sigma_range=1e-6)
    assert np.all(np.isfinite(tiny))
    with pytest.raises(ValueError):
        joint_bilateral_upsample(low, rng.random((3, 17)))  # 引导图比低分辨率输入还小


def test_jbu_beats_bilinear_on_noisy_step_edges():
    h, w, n = 120, 200, 16
    for seed in range(4):
        img, gt = _noisy_edge_scene(h, w, seed)
        logit = 10.0 * (_area_fraction(gt, n) - 0.5)
        assert boundary_iou(joint_bilateral_upsample(logit, img) > 0, gt) > boundary_iou(bilinear_upsample(logit, h, w) > 0, gt)


def _layouts(a):
    """同一组数值的三种内存布局：C 连续、F 连续、从更大数组切出的步长视图。"""
    big = np.zeros((a.shape[0] * 2, a.shape[1] * 3) + a.shape[2:], a.dtype)
    big[::2, ::3] = a
    return [np.ascontiguousarray(a), np.asfortranarray(a), big[::2, ::3]]


def test_results_do_not_depend_on_memory_layout():
    rng = _rng(30)
    guide = rng.random((H, W, 3))
    p = rng.standard_normal((H, W, 11))
    low = rng.standard_normal((4, 6, 10))
    img = (guide * 255).astype(np.uint8)
    ref = (guided_filter(guide, p, 2, 1e-3), bilinear_upsample(low, H, W), guided_upsample(low, img), joint_bilateral_upsample(low, img))
    for g, pp, lo, im in zip(_layouts(guide), _layouts(p), _layouts(low), _layouts(img)):
        got = (guided_filter(g, pp, 2, 1e-3), bilinear_upsample(lo, H, W), guided_upsample(lo, im), joint_bilateral_upsample(lo, im))
        assert all(np.array_equal(a, b) for a, b in zip(got, ref))  # 逐位


def test_module_uses_only_whitelisted_libraries():
    src = inspect.getsource(U)
    imports = [ln.split()[1] for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert set(imports) <= {"__future__", "numpy", "src.constants"}
