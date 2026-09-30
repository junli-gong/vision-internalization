"""无学习上采样：盒式均值、引导滤波、双线性、引导上采样、联合双边上采样。

只用 numpy（简报 §4 第 2 条白名单：引导滤波、联合双边上采样）。与任务无关：输入是任意通道数的
低分辨率场（logit、概率、嵌入），函数不知道它代表什么。

形状约定：
- 低分辨率场 / 待滤波的场：(h, w) 或 (h, w, C)，输出保持同样的通道布局，dtype float64；
- 引导图：(H, W) 灰度或 (H, W, 3) 彩色。guided_filter 只收浮点引导图，eps 的单位是引导图取值的平方，
  由调用方负责归一化；guided_upsample 和 joint_bilateral_upsample 收 uint8 图（除以 255）或已在 [0, 1] 的浮点图。
- 像素中心：第 i 个像素的中心在 i + 0.5（constants.PIXEL_CENTER_OFFSET），放大时按"半像素中心对齐"映射，
  与 PyTorch F.interpolate(mode="bilinear", align_corners=False) 相同。
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GUIDED_FILTER_EPS,
    GUIDED_FILTER_RADIUS,
    JBU_RADIUS,
    JBU_SIGMA_RANGE,
    JBU_SIGMA_SPATIAL,
    PIXEL_CENTER_OFFSET,
)

# 多通道时每次处理的通道数，只为限制内存。每个通道的运算彼此独立、都是逐元素的 numpy 运算，
# 所以结果与分块方式无关（测试逐位钉住）。
_CHANNEL_CHUNK = 8


# ---------------------------------------------------------------------------
# 输入检查
# ---------------------------------------------------------------------------


def _as_field(x: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    """(h, w) 或 (h, w, C) → (h, w, C) float64；返回 (数组, 原来是否二维)。"""
    a = np.asarray(x)
    if a.dtype.kind not in "biuf":
        raise ValueError(f"{name} 必须是实数数组，收到 dtype {a.dtype}")
    if a.ndim == 2:
        a = a[..., None]
        squeeze = True
    elif a.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"{name} 必须是 (h, w) 或 (h, w, C)，收到形状 {a.shape}")
    if a.shape[0] < 1 or a.shape[1] < 1 or a.shape[2] < 1:
        raise ValueError(f"{name} 形状 {a.shape} 含空维度")
    a = a.astype(np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 含 NaN 或 inf")
    return a, squeeze


def _float_guide(guide: np.ndarray) -> np.ndarray:
    """guided_filter 的引导图：(H, W) 或 (H, W, 3) 浮点 → (H, W, K) float64，K ∈ {1, 3}。"""
    g = np.asarray(guide)
    if g.dtype.kind != "f":
        raise ValueError(
            f"guided_filter 的引导图必须是浮点数组，收到 dtype {g.dtype}；eps 以引导图取值的平方为单位，"
            "请先显式归一化（例如 uint8 图除以 255），或改用 guided_upsample"
        )
    if g.ndim == 2:
        g = g[..., None]
    if g.ndim != 3 or g.shape[2] not in (1, 3):
        raise ValueError(f"引导图必须是 (H, W)、(H, W, 1) 或 (H, W, 3)，收到形状 {np.shape(guide)}")
    g = g.astype(np.float64)
    if not np.all(np.isfinite(g)):
        raise ValueError("引导图含 NaN 或 inf")
    return g


def guide_to_unit(image: np.ndarray) -> np.ndarray:
    """图像 → [0, 1] 浮点引导图，(H, W, K)，K ∈ {1, 3}。

    uint8 除以 255；浮点图必须已在 [0, 1]，否则报错（不猜它的量程）。其他 dtype 报错。
    """
    g = np.asarray(image)
    if g.dtype == np.uint8:
        g = g.astype(np.float64) / 255.0
    elif g.dtype.kind == "f":
        g = g.astype(np.float64)
        if not np.all(np.isfinite(g)) or g.min() < 0.0 or g.max() > 1.0:
            raise ValueError("浮点引导图必须是有限值且落在 [0, 1]；uint8 图请直接传入")
    else:
        raise ValueError(f"引导图必须是 uint8 或 [0, 1] 浮点，收到 dtype {g.dtype}")
    if g.ndim == 2:
        g = g[..., None]
    if g.ndim != 3 or g.shape[2] not in (1, 3):
        raise ValueError(f"引导图必须是 (H, W)、(H, W, 1) 或 (H, W, 3)，收到形状 {np.shape(image)}")
    return g


def _check_radius(radius: int) -> int:
    if isinstance(radius, bool) or int(radius) != radius or radius < 0:
        raise ValueError(f"radius 必须是非负整数，收到 {radius}")
    return int(radius)


# ---------------------------------------------------------------------------
# 盒式均值（He et al. 的 O(N) 实现：沿轴累加和相减，窗口在图边被截断，按窗口内实际像素数归一化）
# ---------------------------------------------------------------------------


def _box_sum_axis(x: np.ndarray, radius: int, axis: int) -> np.ndarray:
    n = x.shape[axis]
    c = np.cumsum(x, axis=axis)
    zero_shape = list(x.shape)
    zero_shape[axis] = 1
    c = np.concatenate([np.zeros(zero_shape, dtype=x.dtype), c], axis=axis)  # c[k] = 前 k 个元素之和
    idx = np.arange(n)
    hi = np.minimum(idx + radius + 1, n)
    lo = np.maximum(idx - radius, 0)
    return np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)


def _box_count(height: int, width: int, radius: int) -> np.ndarray:
    """每个像素的截断窗口内的像素数，形状 (H, W)，float64（精确整数值）。"""
    ih, iw = np.arange(height), np.arange(width)
    nh = np.minimum(ih + radius, height - 1) - np.maximum(ih - radius, 0) + 1
    nw = np.minimum(iw + radius, width - 1) - np.maximum(iw - radius, 0) + 1
    return (nh[:, None] * nw[None, :]).astype(np.float64)


def box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    """以每个像素为中心、边长 2·radius+1 的方窗均值，窗口在图边截断，只对图内像素求平均。

    对应 He et al. 的 boxfilter.m 再除以 N = boxfilter(ones)。x 形状 (H, W) 或 (H, W, ...)，只在前两维上滤波。
    """
    r = _check_radius(radius)
    a = np.asarray(x, dtype=np.float64)
    if a.ndim < 2:
        raise ValueError(f"box_mean 需要至少二维，收到形状 {a.shape}")
    s = _box_sum_axis(_box_sum_axis(a, r, 0), r, 1)
    n = _box_count(a.shape[0], a.shape[1], r)
    return s / n.reshape(n.shape + (1,) * (a.ndim - 2))


# ---------------------------------------------------------------------------
# 引导滤波
# ---------------------------------------------------------------------------


def _guided_gray(g: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """灰度引导：g (H, W)，p (H, W, C)。He et al. TPAMI 2013 Algorithm 1。"""
    mean_i = box_mean(g, r)
    var_i = box_mean(g * g, r) - mean_i * mean_i
    out = np.empty_like(p)
    for c0 in range(0, p.shape[2], _CHANNEL_CHUNK):
        pc = p[..., c0 : c0 + _CHANNEL_CHUNK]
        mean_p = box_mean(pc, r)
        cov_ip = box_mean(g[..., None] * pc, r) - mean_i[..., None] * mean_p
        a = cov_ip / (var_i[..., None] + eps)
        b = mean_p - a * mean_i[..., None]
        out[..., c0 : c0 + _CHANNEL_CHUNK] = box_mean(a, r) * g[..., None] + box_mean(b, r)
    return out


def _guided_color(g: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """彩色引导：g (H, W, 3)，p (H, W, C)。逐像素 3×3 协方差 Σ + eps·U 求逆（显式伴随矩阵，向量化），
    结构同 He et al. 的 guidedfilter_color.m。"""
    mean_i = box_mean(g, r)  # (H, W, 3)
    m0, m1, m2 = mean_i[..., 0], mean_i[..., 1], mean_i[..., 2]
    g0, g1, g2 = g[..., 0], g[..., 1], g[..., 2]
    s00 = box_mean(g0 * g0, r) - m0 * m0 + eps
    s01 = box_mean(g0 * g1, r) - m0 * m1
    s02 = box_mean(g0 * g2, r) - m0 * m2
    s11 = box_mean(g1 * g1, r) - m1 * m1 + eps
    s12 = box_mean(g1 * g2, r) - m1 * m2
    s22 = box_mean(g2 * g2, r) - m2 * m2 + eps
    # 对称 3×3 矩阵的伴随矩阵（6 个独立元素）与行列式
    i00 = s11 * s22 - s12 * s12
    i01 = s02 * s12 - s01 * s22
    i02 = s01 * s12 - s02 * s11
    i11 = s00 * s22 - s02 * s02
    i12 = s02 * s01 - s00 * s12
    i22 = s00 * s11 - s01 * s01
    det = s00 * i00 + s01 * i01 + s02 * i02
    inv = [x[..., None] / det[..., None] for x in (i00, i01, i02, i11, i12, i22)]
    v00, v01, v02, v11, v12, v22 = inv
    out = np.empty_like(p)
    for c0 in range(0, p.shape[2], _CHANNEL_CHUNK):
        pc = p[..., c0 : c0 + _CHANNEL_CHUNK]
        mean_p = box_mean(pc, r)
        cov0 = box_mean(g0[..., None] * pc, r) - m0[..., None] * mean_p
        cov1 = box_mean(g1[..., None] * pc, r) - m1[..., None] * mean_p
        cov2 = box_mean(g2[..., None] * pc, r) - m2[..., None] * mean_p
        a0 = v00 * cov0 + v01 * cov1 + v02 * cov2
        a1 = v01 * cov0 + v11 * cov1 + v12 * cov2
        a2 = v02 * cov0 + v12 * cov1 + v22 * cov2
        b = mean_p - a0 * m0[..., None] - a1 * m1[..., None] - a2 * m2[..., None]
        out[..., c0 : c0 + _CHANNEL_CHUNK] = (
            box_mean(a0, r) * g0[..., None] + box_mean(a1, r) * g1[..., None] + box_mean(a2, r) * g2[..., None] + box_mean(b, r)
        )
    return out


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """引导滤波（He, Sun, Tang）：窗口 ω_k 内拟合 q = a_k·I + b_k，再对覆盖每个像素的窗口的 (a_k, b_k) 求平均。

    guide：(H, W) 灰度或 (H, W, 3) 彩色，浮点；eps 的单位是引导图取值的平方（引导图在 [0, 1] 时即方差）。
    src：(H, W) 或 (H, W, C)，多通道逐通道用同一个引导图滤波。返回 float64，形状同 src。
    radius：窗口半径（像素），窗口边长 2·radius+1，图边截断（box_mean）。eps 必须 > 0。

    实现细节：引导图先逐通道减去它的 (最小值 + 最大值) / 2。数学上输出与引导图的平移无关（b_k 吸收常数），
    这样做一是减轻 E[I²] − E[I]² 的相消误差，二是让常数引导图精确变成全 0：此时 a ≡ 0、b = 窗口均值，
    输出逐位等于 box_mean(box_mean(src))，即盒式均值连做两次（不是一次）。
    彩色版在三个通道相同的引导图上，数学上等于 eps/3 的灰度版（Σ = v·11ᵀ，解得 a = cov/(3v + eps)·1）。
    """
    g = _float_guide(guide)
    p, squeeze = _as_field(src, "src")
    if p.shape[:2] != g.shape[:2]:
        raise ValueError(f"src 空间尺寸 {p.shape[:2]} 与引导图 {g.shape[:2]} 不一致")
    r = _check_radius(radius)
    if not (np.isfinite(eps) and eps > 0):
        raise ValueError(f"eps 必须是正的有限值，收到 {eps}")
    g = g - (g.min(axis=(0, 1)) + g.max(axis=(0, 1))) / 2.0
    if g.shape[2] == 1:
        q = _guided_gray(g[..., 0], p, r, float(eps))
    else:
        q = _guided_color(g, p, r, float(eps))
    return q[..., 0] if squeeze else q


# ---------------------------------------------------------------------------
# 双线性与引导上采样
# ---------------------------------------------------------------------------


def _linear_axis(x: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """沿一个轴做半像素中心对齐的线性插值：src = (dst + 0.5)·n_in/n_out − 0.5，两端夹紧到 [0, n_in − 1]。"""
    n_in = x.shape[axis]
    dst = np.arange(n_out, dtype=np.float64)
    src = (dst + PIXEL_CENTER_OFFSET) * (n_in / n_out) - PIXEL_CENTER_OFFSET
    src = np.clip(src, 0.0, n_in - 1)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = src - i0
    w0 = 1.0 - w1
    shape = [1] * x.ndim
    shape[axis] = n_out
    return np.take(x, i0, axis=axis) * w0.reshape(shape) + np.take(x, i1, axis=axis) * w1.reshape(shape)


def bilinear_upsample(low: np.ndarray, height: int, width: int) -> np.ndarray:
    """双线性放大到 (height, width)，与 PyTorch F.interpolate(mode="bilinear", align_corners=False) 相同的坐标映射。

    low：(h, w) 或 (h, w, C)；要求 height ≥ h、width ≥ w（只做放大，缩小不做抗混叠，不在这里提供）。
    先沿列方向、再沿行方向插值。返回 float64。
    """
    a, squeeze = _as_field(low, "low")
    h, w = a.shape[:2]
    if int(height) != height or int(width) != width or height < h or width < w:
        raise ValueError(f"目标尺寸 ({height}, {width}) 必须是整数且不小于输入 ({h}, {w})")
    out = _linear_axis(_linear_axis(a, int(width), 1), int(height), 0)
    return out[..., 0] if squeeze else out


def guided_upsample(low_logits: np.ndarray, guide_image: np.ndarray) -> np.ndarray:
    """简报 §8.4 的默认上采样：先双线性放大到引导图尺寸，再用引导滤波对齐到图像边缘。

    low_logits：(h, w) 或 (h, w, C)。guide_image：(H, W) 或 (H, W, 3)，uint8 或 [0, 1] 浮点。
    半径与 eps 取 constants.GUIDED_FILTER_RADIUS / GUIDED_FILTER_EPS。返回 float64，(H, W) 或 (H, W, C)。
    """
    g = guide_to_unit(guide_image)
    up = bilinear_upsample(low_logits, g.shape[0], g.shape[1])
    return guided_filter(g if g.shape[2] == 3 else g[..., 0], up, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPS)


# ---------------------------------------------------------------------------
# 联合双边上采样
# ---------------------------------------------------------------------------


def joint_bilateral_upsample(
    low: np.ndarray,
    guide: np.ndarray,
    *,
    sigma_spatial: float = JBU_SIGMA_SPATIAL,
    sigma_range: float = JBU_SIGMA_RANGE,
    radius: int = JBU_RADIUS,
) -> np.ndarray:
    """联合双边上采样（Kopf et al. 2007）：

        S̃_p = (1/k_p) Σ_{q↓∈Ω} S_{q↓} · f(‖p↓ − q↓‖) · g(‖Ĩ_p − Ĩ_q‖)

    - p 是高分辨率（引导图）像素，p↓ 是它在低分辨率图上的连续坐标：(i + 0.5)·h/H − 0.5（半像素中心对齐）；
    - Ω：以离 p↓ 最近的低分辨率像素为中心、(2·radius+1)² 个低分辨率像素，越界的邻居丢弃（不复制边缘）；
    - f：空间高斯，sigma_spatial 以低分辨率像素计；g：值域高斯，sigma_range 以 [0, 1] 引导图计，
      彩色引导图用 RGB 欧氏距离；sigma_range=np.inf 时退化为纯空间高斯插值；
    - Ĩ_q 取低分辨率像素 q↓ 的中心所在的高分辨率像素，下标 ((2q+1)·H) // (2h)（整数运算）；
    - k_p 为权重和。权重在对数域计算并减去每个像素的最大值，值域项再小也不会整体下溢成 0。

    low：(h, w) 或 (h, w, C)；guide：(H, W) 或 (H, W, 3)，uint8 或 [0, 1] 浮点，要求 H ≥ h、W ≥ w。
    返回 float64，(H, W) 或 (H, W, C)。
    """
    a, squeeze = _as_field(low, "low")
    g = guide_to_unit(guide)
    h, w = a.shape[:2]
    big_h, big_w = g.shape[:2]
    if big_h < h or big_w < w:
        raise ValueError(f"引导图尺寸 ({big_h}, {big_w}) 必须不小于低分辨率输入 ({h}, {w})")
    r = _check_radius(radius)
    if not (sigma_spatial > 0 and np.isfinite(sigma_spatial)):
        raise ValueError(f"sigma_spatial 必须是正的有限值，收到 {sigma_spatial}")
    if not sigma_range > 0:
        raise ValueError(f"sigma_range 必须 > 0（可以是 inf），收到 {sigma_range}")

    py = (np.arange(big_h) + PIXEL_CENTER_OFFSET) * (h / big_h) - PIXEL_CENTER_OFFSET
    px = (np.arange(big_w) + PIXEL_CENTER_OFFSET) * (w / big_w) - PIXEL_CENTER_OFFSET
    cy = np.clip(np.floor(py + 0.5).astype(np.int64), 0, h - 1)
    cx = np.clip(np.floor(px + 0.5).astype(np.int64), 0, w - 1)
    gy = ((2 * np.arange(h) + 1) * big_h) // (2 * h)
    gx = ((2 * np.arange(w) + 1) * big_w) // (2 * w)
    g_low = g[gy][:, gx]  # (h, w, K)

    two_ss = 2.0 * sigma_spatial * sigma_spatial
    two_sr = 2.0 * float(sigma_range) * float(sigma_range)
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    logws = []
    for dy, dx in offsets:
        qy, qx = cy + dy, cx + dx
        vy = (qy >= 0) & (qy < h)
        vx = (qx >= 0) & (qx < w)
        qyc, qxc = np.clip(qy, 0, h - 1), np.clip(qx, 0, w - 1)
        log_f = -(((qy - py) ** 2)[:, None] + ((qx - px) ** 2)[None, :]) / two_ss
        d2 = np.sum((g - g_low[qyc][:, qxc]) ** 2, axis=2)
        logw = log_f - d2 / two_sr
        logws.append(np.where(vy[:, None] & vx[None, :], logw, -np.inf))
    log_max = np.max(logws, axis=0)  # 最近邻总在窗口内，所以有限

    num = np.zeros((big_h, big_w, a.shape[2]))
    den = np.zeros((big_h, big_w))
    for (dy, dx), logw in zip(offsets, logws):
        wgt = np.exp(logw - log_max)
        qyc, qxc = np.clip(cy + dy, 0, h - 1), np.clip(cx + dx, 0, w - 1)
        num += wgt[..., None] * a[qyc][:, qxc]
        den += wgt
    out = num / den[..., None]
    return out[..., 0] if squeeze else out
