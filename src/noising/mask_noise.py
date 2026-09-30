"""mask 加噪过程（简报 §3.7 第 2 条）：对 GT mask 施加受控扰动，免费制造"带错 mask → 正确 mask"训练对。

只用 numpy、scipy，无学习。给定 rng 的种子，结果确定。

约定：
- mask 是 (H, W) bool（也接受只含 0/1 的整数数组），输出一律是新的 (H, W) bool 数组，不改输入，不越界（不绕回）。
- 严重度 s ∈ [0, 1]：每个算子的扰动量 = s × 系数 × L，L = sqrt(原 mask 面积)（像素），系数见 constants.MASK_NOISE_*。
  s = 0 时每个算子都是恒等变换，corrupt 的输出与输入逐位相同。
- 同一个种子下，随机量（选哪些算子、方向、位置、噪声场）的抽取与 s 无关；按位置做的选择（增删区域的圆心、
  合并哪个邻居、邻居像素的先后）都以原 mask 为参照（ref），也与 s 无关。不同 s 之间只有扰动幅度不同。
- 距离一律用 scipy 的精确欧氏距离变换；图外既不算前景也不算背景：腐蚀不从图边往里吃，膨胀不越出图边。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import ndimage

from src.constants import (
    MASK_NOISE_BLOB_BAND,
    MASK_NOISE_BLOB_COUNT,
    MASK_NOISE_BLOB_RADIUS_MAX,
    MASK_NOISE_DILATE_MAX,
    MASK_NOISE_ERODE_MAX,
    MASK_NOISE_JITTER_AMP_MAX,
    MASK_NOISE_JITTER_SMOOTH,
    MASK_NOISE_MAX_OPS,
    MASK_NOISE_MERGE_MAX,
    MASK_NOISE_P_CLEAN,
    MASK_NOISE_SEVERITY_BETA,
    MASK_NOISE_SHIFT_MAX,
)

# 算子名，同时也是 corrupt 里的施加顺序。平移放在最后，整体移动前面的结果。
OPS = ("add_remove", "merge", "erode", "dilate", "jitter", "translate")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _as_mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    a = np.asarray(mask)
    if a.ndim != 2:
        raise ValueError(f"{name} 必须是二维，收到形状 {a.shape}")
    if a.dtype == bool:
        return a.copy()
    if a.dtype.kind in "iu" and np.all((a == 0) | (a == 1)):
        return a.astype(bool)
    raise ValueError(f"{name} 必须是 bool 或只含 0/1 的整数数组，收到 dtype {a.dtype}")


def _check_severity(severity: float) -> float:
    s = float(severity)
    if not (math.isfinite(s) and 0.0 <= s <= 1.0):
        raise ValueError(f"严重度必须在 [0, 1] 内，收到 {severity}")
    return s


def mask_scale(mask: np.ndarray) -> float:
    """扰动的长度单位 L = sqrt(mask 面积)，像素。空 mask 为 0。"""
    return float(math.sqrt(int(np.count_nonzero(mask))))


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _dist_inside(mask: np.ndarray) -> np.ndarray:
    """前景像素到最近背景像素的欧氏距离（背景像素为 0）。没有背景时（全前景）为 inf：图外不算背景。"""
    if mask.all():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(mask)


def _dist_outside(mask: np.ndarray) -> np.ndarray:
    """背景像素到最近前景像素的欧氏距离（前景像素为 0）。空 mask 时为 inf。"""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(~mask)


# ---------------------------------------------------------------------------
# 算子：签名 (mask, severity, scale, ...) → (新 mask, 参数记录)
# ---------------------------------------------------------------------------


def op_erode(mask: np.ndarray, severity: float, scale: float) -> tuple[np.ndarray, dict]:
    """腐蚀：去掉到背景距离 ≤ r 的前景像素，r = s × MASK_NOISE_ERODE_MAX × L（实数半径，圆盘结构元）。面积不增。"""
    m = _as_mask(mask)
    r = _check_severity(severity) * MASK_NOISE_ERODE_MAX * float(scale)
    rec = {"op": "erode", "radius_px": r}
    if r <= 0.0 or not m.any():
        return m, rec
    return m & (_dist_inside(m) > r), rec


def op_dilate(mask: np.ndarray, severity: float, scale: float) -> tuple[np.ndarray, dict]:
    """膨胀：加入到前景距离 ≤ r 的背景像素，r = s × MASK_NOISE_DILATE_MAX × L。面积不减。"""
    m = _as_mask(mask)
    r = _check_severity(severity) * MASK_NOISE_DILATE_MAX * float(scale)
    rec = {"op": "dilate", "radius_px": r}
    if r <= 0.0 or not m.any():
        return m, rec
    return m | (_dist_outside(m) <= r), rec


def op_jitter(mask: np.ndarray, severity: float, scale: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """边界抖动：符号距离 φ（前景为正）加上平滑随机场 A·n 后取 > 0。

    n = 白噪声经 sigma = MASK_NOISE_JITTER_SMOOTH × L 的高斯平滑、再除以全图标准差；A = s × MASK_NOISE_JITTER_AMP_MAX × L。
    边界沿法向移动约 A·n。白噪声总是整张抽取（与 s、mask 无关），保证同种子下不同 s 用同一个场。
    """
    m = _as_mask(mask)
    s = _check_severity(severity)
    field = rng.standard_normal(m.shape)
    sigma = MASK_NOISE_JITTER_SMOOTH * float(scale)
    amp = s * MASK_NOISE_JITTER_AMP_MAX * float(scale)
    rec = {"op": "jitter", "amplitude_px": amp, "smooth_sigma_px": sigma}
    if amp <= 0.0 or not m.any() or m.all():
        return m, rec
    n = ndimage.gaussian_filter(field, sigma) if sigma > 0 else field
    std = float(n.std())
    if std > 0:
        n = n / std
    phi = np.where(m, _dist_inside(m), -_dist_outside(m))
    return (phi + amp * n) > 0, rec


def op_add_remove(
    mask: np.ndarray, severity: float, scale: float, rng: np.random.Generator, ref: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """随机增删圆形区域。

    个数在 MASK_NOISE_BLOB_COUNT 闭区间里均匀取；每个区域以 1/2 概率为"增"或"删"：
    - 增：圆心均匀取自参照物体外、到它的距离 ≤ MASK_NOISE_BLOB_BAND × L 的像素，圆内像素置 1；
    - 删：圆心均匀取自参照物体内的像素，圆内像素置 0。
    参照物体 ref 默认就是 mask；corrupt 传原 mask。
    半径 = s × MASK_NOISE_BLOB_RADIUS_MAX × L × u，u ~ U[0.5, 1)；圆 = 到圆心距离 < 半径的像素（半径 0 时为空）。
    圆心、个数、增删、u 的抽取与 s 无关，每个区域固定抽 3 个随机数；没有候选圆心时该区域跳过（记录里注明）。
    """
    m = _as_mask(mask)
    r = m if ref is None else _as_mask(ref, "ref")
    if r.shape != m.shape:
        raise ValueError(f"ref 形状 {r.shape} 与 mask {m.shape} 不一致")
    s = _check_severity(severity)
    lo, hi = MASK_NOISE_BLOB_COUNT
    n_blobs = int(rng.integers(lo, hi + 1))
    height, width = m.shape
    inside = np.flatnonzero(r)
    band = np.flatnonzero(~r & (_dist_outside(r) <= MASK_NOISE_BLOB_BAND * float(scale)))
    out = m.copy()
    blobs = []
    for _ in range(n_blobs):
        add = bool(rng.random() < 0.5)
        u_pos = float(rng.random())
        u_size = float(rng.uniform(0.5, 1.0))
        cand = band if add else inside
        kind = "add" if add else "remove"
        if cand.size == 0:
            blobs.append({"kind": kind, "skipped": "no candidate center"})
            continue
        idx = int(cand[min(int(u_pos * cand.size), cand.size - 1)])
        cy, cx = divmod(idx, width)
        radius = s * MASK_NOISE_BLOB_RADIUS_MAX * float(scale) * u_size
        blobs.append({"kind": kind, "center_rc": [int(cy), int(cx)], "radius_px": radius})
        if radius <= 0.0:
            continue
        ext = int(math.ceil(radius))
        r0, r1 = max(0, cy - ext), min(height, cy + ext + 1)
        c0, c1 = max(0, cx - ext), min(width, cx + ext + 1)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 < radius * radius
        if add:
            out[r0:r1, c0:c1] |= disk
        else:
            out[r0:r1, c0:c1] &= ~disk
    return out, {"op": "add_remove", "blobs": blobs}


def shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """整数平移，dy 沿行向下为正、dx 沿列向右为正；移出图外的像素丢弃，空出的位置填 0（不绕回）。"""
    m = np.asarray(mask, dtype=bool)
    height, width = m.shape
    out = np.zeros_like(m)
    if abs(dy) >= height or abs(dx) >= width:
        return out
    src = m[max(0, -dy) : height - max(0, dy), max(0, -dx) : width - max(0, dx)]
    out[max(0, dy) : max(0, dy) + src.shape[0], max(0, dx) : max(0, dx) + src.shape[1]] = src
    return out


def op_translate(mask: np.ndarray, severity: float, scale: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """平移：方向角 θ ~ U[0, 2π)（总是抽取），距离 d = s × MASK_NOISE_SHIFT_MAX × L，
    位移 dy = round(d·sinθ)、dx = round(d·cosθ)（四舍五入，.5 进位），见 shift_mask。"""
    m = _as_mask(mask)
    s = _check_severity(severity)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    d = s * MASK_NOISE_SHIFT_MAX * float(scale)
    dy, dx = _round_half_up(d * math.sin(theta)), _round_half_up(d * math.cos(theta))
    return shift_mask(m, dy, dx), {"op": "translate", "dy": dy, "dx": dx, "distance_px": d, "angle_rad": theta}


def _nearest_neighbor(m: np.ndarray, others: Sequence[np.ndarray]) -> tuple[int, np.ndarray, np.ndarray] | None:
    """在 others 里找离 m 最近的物体（只看它不在 m 里的像素）；平局取下标小的。返回 (下标, 其像素, 到 m 的距离场)。"""
    if not m.any():
        return None
    cands = [(i, o & ~m) for i, o in enumerate(others)]
    cands = [(i, c) for i, c in cands if c.any()]
    if not cands:
        return None
    dist = _dist_outside(m)
    best = min(cands, key=lambda ic: (float(dist[ic[1]].min()), ic[0]))
    return best[0], best[1], dist


def op_merge(mask: np.ndarray, severity: float, others: Sequence[np.ndarray], ref: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """与邻近物体合并：取 others 里离参照物体最近的一个，把它不在参照物体里的像素按到参照物体的距离由近到远排序
    （距离相同按行优先），前 floor(s × MASK_NOISE_MERGE_MAX × 其像素数) 个并入 mask。s = 1 且系数为 1 时整个邻近物体并入。
    参照物体 ref 默认就是 mask；corrupt 传原 mask。参照物体为空、或 others 里没有与它不重合的像素时不做任何事（记录里注明）。"""
    m = _as_mask(mask)
    r = m if ref is None else _as_mask(ref, "ref")
    s = _check_severity(severity)
    os_ = [_as_mask(o, f"others[{i}]") for i, o in enumerate(others)]
    for name, o in [(f"others[{i}]", o) for i, o in enumerate(os_)] + [("ref", r)]:
        if o.shape != m.shape:
            raise ValueError(f"{name} 形状 {o.shape} 与 mask {m.shape} 不一致")
    found = _nearest_neighbor(r, os_)
    if found is None:
        return m, {"op": "merge", "skipped": "no neighbor"}
    idx, pix_mask, dist = found
    pix = np.flatnonzero(pix_mask)
    order = np.argsort(dist.flat[pix], kind="stable")
    n_take = int(math.floor(s * MASK_NOISE_MERGE_MAX * pix.size))
    out = m.copy()
    out.flat[pix[order[:n_take]]] = True
    return out, {"op": "merge", "neighbor": idx, "n_pixels": n_take, "neighbor_pixels": int(pix.size)}


# ---------------------------------------------------------------------------
# 噪声日程与组合
# ---------------------------------------------------------------------------


def sample_severity(rng: np.random.Generator) -> float:
    """按噪声日程采样严重度：以概率 MASK_NOISE_P_CLEAN 返回 0.0，否则 s ~ Beta(MASK_NOISE_SEVERITY_BETA)。"""
    if rng.random() < MASK_NOISE_P_CLEAN:
        return 0.0
    a, b = MASK_NOISE_SEVERITY_BETA
    return float(rng.beta(a, b))


def corrupt(mask: np.ndarray, severity: float, rng: np.random.Generator, others: Sequence[np.ndarray] = ()) -> tuple[np.ndarray, dict]:
    """对 mask 施加一组扰动，返回 (带噪 mask, 记录)。

    - 可用算子：OPS 全部，"merge" 只在 others 里有与 mask 不重合的非空物体时可用。
    - 个数 n ~ U{1, …, min(MASK_NOISE_MAX_OPS, 可用数)}，从可用算子里不放回地抽 n 个，按 OPS 的顺序施加，都用同一个 s。
    - 长度单位 L 取原 mask 的面积算，所有算子共用；增删区域与合并按原 mask 选位置（ref）。
    - 记录：{"severity", "scale_px", "ops": [每个算子的参数]}，全是 Python 原生类型，可直接写 JSON。
    - 空 mask 原样返回（没有物体可扰动；增删区域没有候选圆心，其余算子幅度为 0）。
    """
    m = _as_mask(mask)
    s = _check_severity(severity)
    os_ = [_as_mask(o, f"others[{i}]") for i, o in enumerate(others)]
    for i, o in enumerate(os_):
        if o.shape != m.shape:
            raise ValueError(f"others[{i}] 形状 {o.shape} 与 mask {m.shape} 不一致")
    scale = mask_scale(m)
    has_neighbor = _nearest_neighbor(m, os_) is not None
    available = [op for op in OPS if op != "merge" or has_neighbor]
    n_ops = int(rng.integers(1, min(MASK_NOISE_MAX_OPS, len(available)) + 1))
    picked = sorted(int(i) for i in rng.choice(len(available), size=n_ops, replace=False))
    record = {"severity": s, "scale_px": scale, "ops": []}
    out = m
    for op in (available[i] for i in picked):
        if op == "add_remove":
            out, rec = op_add_remove(out, s, scale, rng, ref=m)
        elif op == "merge":
            out, rec = op_merge(out, s, os_, ref=m)
        elif op == "erode":
            out, rec = op_erode(out, s, scale)
        elif op == "dilate":
            out, rec = op_dilate(out, s, scale)
        elif op == "jitter":
            out, rec = op_jitter(out, s, scale, rng)
        else:
            out, rec = op_translate(out, s, scale, rng)
        record["ops"].append(rec)
    return out, record
