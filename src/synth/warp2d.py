"""mask 的二维仿射变换、矩与矩匹配（DEV-0010，a、b 的平凡基线与真值仿射上界）。无学习组件。

约定
- 像素 (i, j) 占 [j, j+1) × [i, i+1)，中心 (j + 0.5, i + 0.5)（src.constants.PIXEL_CENTER_OFFSET）。
- 仿射 x' = A·x + t 作用在连续坐标上。affine_warp 用反向映射：输出像素中心 p' 在结果里 ⟺ A⁻¹(p' − t) 落在一个前景像素的方块里
  （floor 取整）。A 奇异（det = 0，如 a 在恰好 90° 时由指令推得的仿射）时，前景的像是零面积的线段，结果为空。
- 矩：前景像素中心的均值与协方差（整数精确求和，再做正确舍入的除法）。

确定性：只用 +、−、×、÷、sqrt、floor（IEEE 正确舍入或精确），三角函数走 dmath.rot_cs；按固定顺序求值，
不用 np.linalg / BLAS，不对浮点数组求和。结果与平台无关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.synth.constants import AFFINE_SEARCH_COARSE_DEG, AFFINE_SEARCH_STRIDE, AFFINE_SEARCH_TOP
from src.synth.dmath import rot_cs

Mat2 = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Moments:
    n: int
    mx: float  # 前景像素中心的均值（像素坐标）
    my: float
    sxx: float  # 协方差（总体口径，除以 n）
    sxy: float
    syy: float


def mask_moments(mask: np.ndarray) -> Moments:
    """前景像素中心的个数、均值、协方差。坐标和、平方和都用 Python 整数精确求和；空 mask 抛 ValueError。"""
    m = np.asarray(mask, dtype=bool)
    box = mask_bbox(m)
    if box is None:
        raise ValueError("空 mask 没有矩")
    r0, r1, c0, c1 = box
    rows, cols = np.nonzero(m[r0 : r1 + 1, c0 : c1 + 1])  # 只在包围盒里找（DEV-0028 提速；坐标加回偏移，整数和逐位相同）
    n = int(rows.size)
    r = rows.astype(np.int64) + r0
    c = cols.astype(np.int64) + c0
    sx, sy = int(c.sum()), int(r.sum())
    sxx, sxy, syy = int((c * c).sum()), int((c * r).sum()), int((r * r).sum())
    return Moments(
        n,
        sx / n + 0.5,
        sy / n + 0.5,
        (n * sxx - sx * sx) / (n * n),
        (n * sxy - sx * sy) / (n * n),
        (n * syy - sy * sy) / (n * n),
    )


def mul2(a: Mat2, b: Mat2) -> Mat2:
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


def inv2(a: Mat2) -> Mat2:
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if det == 0.0:
        raise ZeroDivisionError("奇异矩阵")
    return ((a[1][1] / det, -a[0][1] / det), (-a[1][0] / det, a[0][0] / det))


def sqrt_spd2(sxx: float, sxy: float, syy: float) -> Mat2:
    """对称正定 2×2 矩阵的主平方根（闭式：(S + √det·I) / √(tr + 2√det)）。"""
    det = sxx * syy - sxy * sxy
    if not (det > 0.0 and sxx > 0.0 and syy > 0.0):
        raise ValueError("协方差不是正定的")
    s = math.sqrt(det)
    t = math.sqrt((sxx + syy) + 2.0 * s)
    return (((sxx + s) / t, sxy / t), (sxy / t, (syy + s) / t))


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """前景的包围盒 (首行, 末行, 首列, 末列)（闭区间，整数）；空 mask 为 None。"""
    m = np.asarray(mask, dtype=bool)
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size == 0:
        return None
    cols = np.nonzero(m.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _affine_sources(mask: np.ndarray, a: Mat2, t: tuple[float, float], stride: int,
                    bbox: tuple[int, int, int, int] | None | bool = False):
    """affine_warp 的取样：返回 (输出形状, 窗口 (i_lo, i_hi, j_lo, j_hi) 或 None, 窗口里每个输出像素是否落在前景上,
    源像素的展平下标 行·W + 列)。

    窗口为 None 时结果全空（空 mask、A 奇异、像在画布外）。源下标只在"落在前景上"的位置有意义。
    bbox：调用方已算好的 mask_bbox(mask)（同一个 mask 反复取样时省掉重算；DEV-0028 提速，结果逐位不变）；默认现算。
    """
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    oh, ow = -(-h // stride), -(-w // stride)
    box = mask_bbox(m) if bbox is False else bbox
    if box is None:
        return (oh, ow), None, None, None
    r0, r1, c0, c1 = box
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if det == 0.0:
        return (oh, ow), None, None, None
    ia = inv2(a)
    # 前景包围盒 [c0, c1+1] × [r0, r1+1] 的四个角在变换下的像，决定需要取样的输出范围
    xs, ys = (float(c0), float(c1 + 1)), (float(r0), float(r1 + 1))
    px = [(a[0][0] * x + a[0][1] * y) + t[0] for x in xs for y in ys]
    py = [(a[1][0] * x + a[1][1] * y) + t[1] for x in xs for y in ys]
    # 输出像素 (stride·i', stride·j') 的中心 = (stride·j' + 0.5, stride·i' + 0.5)；多取一格余量，由有效性判定裁掉
    j_lo = max(math.floor((min(px) - 0.5) / stride) - 1, 0)
    j_hi = min(math.ceil((max(px) - 0.5) / stride) + 1, ow - 1)
    i_lo = max(math.floor((min(py) - 0.5) / stride) - 1, 0)
    i_hi = min(math.ceil((max(py) - 0.5) / stride) + 1, oh - 1)
    if j_hi < j_lo or i_hi < i_lo:
        return (oh, ow), None, None, None
    qx = (np.arange(j_lo, j_hi + 1, dtype=np.float64) * stride + 0.5) - t[0]
    qy = (np.arange(i_lo, i_hi + 1, dtype=np.float64) * stride + 0.5) - t[1]
    sx = ia[0][0] * qx[None, :] + ia[0][1] * qy[:, None]
    sy = ia[1][0] * qx[None, :] + ia[1][1] * qy[:, None]
    fx = np.floor(sx)
    fy = np.floor(sy)
    # 落在前景上 ⟺ (floor 后的) 源像素在前景包围盒里且是前景；包围盒外（含画布外）的源像素一定不是前景，
    # 所以按包围盒判有效，再用展平的下标取样（DEV-0028 提速；与按画布判有效、二维下标取样逐位相同：
    # 有效时 fy·W + fx 是 < 2^53 的整数，浮点运算精确）
    valid = (fx >= c0) & (fx <= c1) & (fy >= r0) & (fy <= r1)
    idx = np.where(valid, fy * w + fx, 0).astype(np.intp)
    return (oh, ow), (i_lo, i_hi, j_lo, j_hi), valid & m.reshape(-1)[idx], idx


def affine_warp(mask: np.ndarray, a: Mat2, t: tuple[float, float], stride: int = 1,
                bbox: tuple[int, int, int, int] | None | bool = False) -> np.ndarray:
    """mask 在 x' = A·x + t 下的像，按输出像素中心反向映射取样。

    stride > 1 时只在每隔 stride 个像素的格点上取样（像素 (stride·i, stride·j) 的中心），返回
    (ceil(H/stride), ceil(W/stride)) 的数组，与 mask[::stride, ::stride] 一一对应（矩匹配搜索的粗评估用）。
    bbox：可选的 mask_bbox(mask)，见 _affine_sources。
    """
    shape, win, hit, _idx = _affine_sources(mask, a, t, stride, bbox)
    out = np.zeros(shape, dtype=bool)
    if win is not None:
        i_lo, i_hi, j_lo, j_hi = win
        out[i_lo : i_hi + 1, j_lo : j_hi + 1] = hit
    return out


def affine_warp_values(mask: np.ndarray, values: np.ndarray, a: Mat2, t: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """mask 连同逐像素的值（如深度单位）一起做 affine_warp（全分辨率）：返回 (像 mask, 像上的值)。

    像 mask 与 affine_warp(mask, a, t) 逐位相同；像素 p' 在像里时，它的值 = 反向映射落到的那个源像素的值（最近邻），
    其余位置为 0。值的 dtype 不变（DEV-0018：a 的二维仿射基线的起伏）。
    """
    v = np.asarray(values)
    if v.shape != np.asarray(mask).shape:
        raise ValueError(f"values 的形状 {v.shape} 与 mask 不同")
    shape, win, hit, idx = _affine_sources(mask, a, t, 1)
    out = np.zeros(shape, dtype=bool)
    vals = np.zeros(shape, dtype=v.dtype)
    if win is not None:
        i_lo, i_hi, j_lo, j_hi = win
        out[i_lo : i_hi + 1, j_lo : j_hi + 1] = hit
        vals[i_lo : i_hi + 1, j_lo : j_hi + 1] = np.where(hit, v.reshape(-1)[idx], 0)
    return out, vals


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """整像素平移（向右 dx、向下 dy），移出画布的部分丢掉，不回绕。"""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    out = np.zeros_like(m)
    dx, dy = int(dx), int(dy)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    src = m[max(-dy, 0) : h - max(dy, 0), max(-dx, 0) : w - max(dx, 0)]
    out[max(dy, 0) : max(dy, 0) + src.shape[0], max(dx, 0) : max(dx, 0) + src.shape[1]] = src
    return out


def shift_values(values: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """任意 dtype 的二维数组按 shift_mask 的同一规则整像素平移（向右 dx、向下 dy），移出画布的丢掉，空出的位置填 0
    （DEV-0028：b 的 translate 基线把输入深度图连同 mask 一起平移）。shift_values(mask, …) 与 shift_mask(mask, …) 逐位相同。"""
    v = np.asarray(values)
    h, w = v.shape
    out = np.zeros_like(v)
    dx, dy = int(dx), int(dy)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    src = v[max(-dy, 0) : h - max(dy, 0), max(-dx, 0) : w - max(dx, 0)]
    out[max(dy, 0) : max(dy, 0) + src.shape[0], max(dx, 0) : max(dx, 0) + src.shape[1]] = src
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / u if u else 1.0


@dataclass(frozen=True)
class MomentFit:
    a: Mat2
    t: tuple[float, float]
    rotation_deg: int  # 正交因子 Q = R(rotation_deg) · diag(1, ±1)
    reflect: bool
    mask: np.ndarray  # 变换后的 mask（全分辨率）
    iou: float  # 与目标的 IoU（全分辨率）
    values: np.ndarray | None = None  # 给了 values 时：逐像素的值随 mask 一起变换的结果（= affine_warp_values 的第二项）


def moment_matched_affine(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    coarse_deg: int = AFFINE_SEARCH_COARSE_DEG,
    stride: int = AFFINE_SEARCH_STRIDE,
    top: int = AFFINE_SEARCH_TOP,
    moments: tuple[Moments, Moments] | None = None,
    values: np.ndarray | None = None,
) -> MomentFit:
    """用矩匹配从目标拟合的二维仿射：A = S_dst^{1/2} · Q · S_src^{-1/2}，t = μ_dst − A·μ_src，
    于是 A 把 src 的均值与协方差精确地变成 dst 的。Q ∈ O(2) 不由矩决定，按与 dst 的 IoU 搜索：
    Q = R(ω)·diag(1, σ)，σ ∈ {+1, −1}；ω 先取 coarse_deg 的整数倍（隔 stride 像素抽样评估），
    再在最好的 top 个附近 ±(coarse_deg/2) 度内逐度细化（隔 2 像素抽样），最后对细化后最好的两个按全分辨率 IoU 取最好的。
    同分时取候选顺序靠前的（σ = +1 先于 −1，ω 小的先），结果确定。用了目标，只作"任何二维方法的上界"参考线。
    values：可选的逐像素值（如深度单位，形状同 src）：给了就把它连同 mask 一起做最终的仿射（affine_warp_values），放进
    MomentFit.values（省掉调用方再做一遍同一个仿射；DEV-0028）。
    """
    src = np.asarray(src, dtype=bool)
    dst = np.asarray(dst, dtype=bool)
    # moments：调用方已算好的 (mask_moments(src), mask_moments(dst))（省掉重算）
    ms, md = (mask_moments(src), mask_moments(dst)) if moments is None else moments
    s_half = sqrt_spd2(md.sxx, md.sxy, md.syy)
    s_inv = inv2(sqrt_spd2(ms.sxx, ms.sxy, ms.syy))
    cache: dict[tuple[int, int, int], float] = {}
    # DEV-0028 提速（结果逐位不变，tests 钉住与逐个候选全画布计算的旧做法相同）：src 的包围盒只算一次；
    # 抽样 IoU 只在取样窗口里数交集：|并| = |像| + |dst 抽样| − |交|（整数），窗口外的像全空。
    box = mask_bbox(src)
    dst_sub = {}

    def params(omega: int, sigma: int) -> tuple[Mat2, tuple[float, float]]:
        c, s = rot_cs(omega)
        q = ((c, -s * sigma), (s, c * sigma))
        a = mul2(mul2(s_half, q), s_inv)
        t = (md.mx - (a[0][0] * ms.mx + a[0][1] * ms.my), md.my - (a[1][0] * ms.mx + a[1][1] * ms.my))
        return a, t

    def score(omega: int, sigma: int, st: int) -> float:
        key = (omega % 360, sigma, st)
        if key not in cache:
            a, t = params(omega % 360, sigma)
            if st not in dst_sub:
                d = dst[::st, ::st]
                dst_sub[st] = (d, int(np.count_nonzero(d)))
            d, n_d = dst_sub[st]
            _shape, win, hit, _idx = _affine_sources(src, a, t, st, box)
            if win is None:
                inter = n_h = 0
            else:
                i_lo, i_hi, j_lo, j_hi = win
                inter = int(np.count_nonzero(hit & d[i_lo : i_hi + 1, j_lo : j_hi + 1]))
                n_h = int(np.count_nonzero(hit))
            u = n_h + n_d - inter
            cache[key] = inter / u if u else 1.0
        return cache[key]

    coarse = [(om, sg) for sg in (1, -1) for om in range(0, 360, coarse_deg)]
    ranked = sorted(range(len(coarse)), key=lambda k: (-score(*coarse[k], stride), k))
    fine: list[tuple[int, int]] = []
    half = coarse_deg // 2
    for k in ranked[:top]:
        om, sg = coarse[k]
        for d in range(-half, half + 1):
            cand = ((om + d) % 360, sg)
            if cand not in fine:
                fine.append(cand)
    fine_ranked = sorted(range(len(fine)), key=lambda k: (-score(*fine[k], 2), k))
    # 全分辨率（stride 1）的 IoU 同样只在窗口里数；只对选中的那个生成全分辨率的像（后一个严格更好才换，与原做法相同）
    best = None
    for k in fine_ranked[:2]:
        om, sg = fine[k]
        v = score(om, sg, 1)
        if best is None or v > best[0]:
            best = (v, om, sg)
    v, om, sg = best
    a, t = params(om, sg)
    if values is None:
        return MomentFit(a, t, om, sg == -1, affine_warp(src, a, t, 1, box), v)
    m, vals = affine_warp_values(src, values, a, t)
    return MomentFit(a, t, om, sg == -1, m, v, vals)
