"""E3a 的 GT 与分辨率换算（DEV-0034 §2）：训练时 GT 降到 128 × 128，评测时 128 × 128 的输出回到原图分辨率。全部无学习（numpy）。

**GT → 128 × 128**（损失在 128 × 128 上算）：
- 标签（语义分割、指代分割）：最近邻，取每个目标格中心所在的源像素：src = min(n_in − 1, floor((i + 0.5) · n_in / n_out))
  （像素中心对齐，与 src/constants.PIXEL_CENTER_OFFSET 的约定相同）。语义分割之后再 reduce_zero_label（与 E2 相同）。
- 深度、法向：有效像素的面积平均。目标格 i 覆盖源坐标 [i · n_in / n_out, (i + 1) · n_in / n_out)，源像素 y 覆盖 [y, y + 1)，
  权重 = 两者重叠的长度（area_weights，沿两个轴可分离）。mean = Σ w·m·v / Σ w·m（m = 有效掩码），有效面积比例 = Σ w·m / 格面积；
  比例不足 min_valid_fraction 的格子不监督。深度在 log 空间平均（几何平均：损失与评测都在 log 空间），法向平均后再单位化（平均向量的
  范数 < NORMAL_MIN_NORM 的格子不监督）。缩小与放大都是同一个公式（源图小于 128 时每格只落在一两个源像素里）。

**128 × 128 → 原图**（评测；简报 §8.4 的无学习上采样）：to_original。原图两边都不小于 128：直接 guided_upsample（先双线性放大、
再以原图为引导做引导滤波）或 bilinear_upsample；某一边小于 128：先沿那一边面积平均降到原图尺寸（area_resize），再交给同一个函数
（它们只做放大）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.constants import ADE_IGNORE_INDEX, ADE_REDUCE_ZERO_LABEL
from src.e2.tasks import depth_train_valid
from src.executors.upsample import bilinear_upsample, guided_upsample
from src.metrics.seg import reduce_zero_label

NORMAL_MIN_NORM = 1e-6  # 法向平均向量的范数下限（相反的法向平均后接近 0，方向无定义）


class NoSupervision(ValueError):
    """这条样本在 128 × 128 上没有可监督的像素（深度、法向的有效格子为 0，语义分割全是忽略值）。"""


# ---------------------------------------------------------------------------
# 采样矩阵
# ---------------------------------------------------------------------------


def nearest_indices(n_in: int, n_out: int) -> np.ndarray:
    """最近邻的源下标：每个目标格中心 (i + 0.5) · n_in / n_out 所在的源像素（整数运算，无浮点误差）。"""
    if n_in < 1 or n_out < 1:
        raise ValueError(f"尺寸必须为正：{n_in} → {n_out}")
    i = np.arange(n_out, dtype=np.int64)
    return np.minimum(n_in - 1, ((2 * i + 1) * n_in) // (2 * n_out))


def area_weights(n_in: int, n_out: int) -> np.ndarray:
    """[n_out, n_in] 的重叠长度矩阵（float64）：目标格 [i·n_in/n_out, (i+1)·n_in/n_out) 与源像素 [y, y+1) 的重叠。每行之和 =
    n_in / n_out（格的长度），每列之和 = 1。"""
    if n_in < 1 or n_out < 1:
        raise ValueError(f"尺寸必须为正：{n_in} → {n_out}")
    lo = np.arange(n_out, dtype=np.float64) * n_in / n_out
    hi = np.arange(1, n_out + 1, dtype=np.float64) * n_in / n_out
    y = np.arange(n_in, dtype=np.float64)
    return np.clip(np.minimum(hi[:, None], y[None, :] + 1.0) - np.maximum(lo[:, None], y[None, :]), 0.0, None)


def _separable(ah: np.ndarray, v: np.ndarray, aw: np.ndarray) -> np.ndarray:
    """(H, W, C) → (h, w, C)：每个通道 ah @ v[..., c] @ awᵀ（两次矩阵乘；通道之间互相独立）。"""
    return np.stack([ah @ v[..., c] @ aw.T for c in range(v.shape[2])], axis=-1)


def area_mean(values: np.ndarray, valid: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """有效像素的面积平均：values (H, W) 或 (H, W, C)，valid (H, W) bool → (mean (n_out, n_out[, C])，有效面积比例 (n_out, n_out))。
    没有有效像素的格子 mean 记 0（比例为 0）。"""
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(valid, dtype=bool)
    if m.shape != v.shape[:2]:
        raise ValueError(f"valid {m.shape} 与 values {v.shape[:2]} 尺寸不同")
    h, w = m.shape
    ah, aw = area_weights(h, n_out), area_weights(w, n_out)
    mf = m.astype(np.float64)
    den = ah @ mf @ aw.T
    vz = np.where(m[..., None] if v.ndim == 3 else m, v, 0.0)
    if v.ndim == 2:
        num = ah @ vz @ aw.T
    else:
        num = _separable(ah, vz, aw)
    cell = (h / n_out) * (w / n_out)
    d = den if v.ndim == 2 else den[..., None]
    mean = np.divide(num, d, out=np.zeros_like(num), where=np.broadcast_to(d > 0, num.shape))
    return mean, den / cell


def area_resize(field: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """(h, w[, C]) → (out_h, out_w[, C]) 的面积平均（全部像素有效）。只用来把 128 的输出降到较小的原图边上。"""
    f = np.asarray(field, dtype=np.float64)
    ah, aw = area_weights(f.shape[0], out_h), area_weights(f.shape[1], out_w)
    ah = ah / ah.sum(axis=1, keepdims=True)
    aw = aw / aw.sum(axis=1, keepdims=True)
    return ah @ f @ aw.T if f.ndim == 2 else _separable(ah, f, aw)


# ---------------------------------------------------------------------------
# 各任务的 128 × 128 目标
# ---------------------------------------------------------------------------


@dataclass
class DenseTarget:
    """一条样本在 out × out 上的目标（numpy）。refseg：mask (out, out) float32 0/1；semseg：label (out, out) int64（忽略值
    ADE_IGNORE_INDEX）；depth：log_depth (out, out) float32 + valid bool；normals：normal (3, out, out) float32 单位向量 + valid。"""

    task: str
    values: np.ndarray
    valid: np.ndarray | None
    info: dict[str, Any]

    @property
    def n_supervised(self) -> int:
        if self.task == "semseg":
            return int(np.count_nonzero(self.values != ADE_IGNORE_INDEX))
        if self.task == "refseg":
            return int(self.values.size)
        return int(np.count_nonzero(self.valid))


def refseg_target(mask: np.ndarray, out: int) -> DenseTarget:
    m = np.asarray(mask, dtype=bool)
    small = m[np.ix_(nearest_indices(m.shape[0], out), nearest_indices(m.shape[1], out))].astype(np.float32)
    return DenseTarget("refseg", small, None, {"fg_fraction": float(small.mean()), "src_hw": list(m.shape)})


def semseg_target(label: np.ndarray, out: int) -> DenseTarget:
    lab = np.asarray(label)
    small = lab[np.ix_(nearest_indices(lab.shape[0], out), nearest_indices(lab.shape[1], out))]
    red = reduce_zero_label(small) if ADE_REDUCE_ZERO_LABEL else small.astype(np.int64)
    red = red.astype(np.int64)
    n = int(np.count_nonzero(red != ADE_IGNORE_INDEX))
    if n == 0:
        raise NoSupervision("语义分割：128 × 128 上全是忽略值")
    return DenseTarget("semseg", red, None, {"n_labeled": n, "src_hw": list(lab.shape)})


def depth_target(depth_m: np.ndarray, out: int, *, min_valid_fraction: float) -> DenseTarget:
    d = np.asarray(depth_m, dtype=np.float64)
    valid = depth_train_valid(d)
    logd = np.log(np.where(valid, d, 1.0))
    mean, frac = area_mean(logd, valid, out)
    ok = frac >= float(min_valid_fraction)
    if not ok.any():
        raise NoSupervision("深度：没有有效面积过半的格子")
    return DenseTarget("depth", np.where(ok, mean, 0.0).astype(np.float32), ok,
                       {"n_valid_cells": int(ok.sum()), "src_hw": list(d.shape)})


def normals_target(normal: np.ndarray, valid: np.ndarray, out: int, *, min_valid_fraction: float) -> DenseTarget:
    n = np.asarray(normal, dtype=np.float64)
    mean, frac = area_mean(n, valid, out)
    norm = np.linalg.norm(mean, axis=-1)
    ok = (frac >= float(min_valid_fraction)) & (norm >= NORMAL_MIN_NORM)
    if not ok.any():
        raise NoSupervision("法向：没有有效面积过半的格子")
    unit = np.where(ok[..., None], mean / np.maximum(norm, NORMAL_MIN_NORM)[..., None], 0.0)
    return DenseTarget("normals", np.ascontiguousarray(unit.transpose(2, 0, 1)).astype(np.float32), ok,
                       {"n_valid_cells": int(ok.sum()), "src_hw": list(valid.shape)})


# ---------------------------------------------------------------------------
# 128 × 128 → 原图
# ---------------------------------------------------------------------------

UPSAMPLE_METHODS: tuple[str, ...] = ("guided", "bilinear")


def to_original(field: np.ndarray, image: np.ndarray, method: str) -> np.ndarray:
    """(h, w) 或 (h, w, C) 的场 → 原图 (H, W[, C])（float64；规则见模块说明）。image 是 uint8 的 (H, W, 3) 原图（guided 的引导图）。"""
    f = np.asarray(field, dtype=np.float64)
    big_h, big_w = int(image.shape[0]), int(image.shape[1])
    h, w = f.shape[:2]
    if big_h < h or big_w < w:
        f = area_resize(f, min(h, big_h), min(w, big_w))
    if method == "guided":
        return guided_upsample(f, image)
    if method == "bilinear":
        return bilinear_upsample(f, big_h, big_w)
    raise ValueError(f"未知的上采样 {method!r}（可用：{UPSAMPLE_METHODS}）")


__all__ = [
    "DenseTarget",
    "NORMAL_MIN_NORM",
    "NoSupervision",
    "UPSAMPLE_METHODS",
    "area_mean",
    "area_resize",
    "area_weights",
    "depth_target",
    "nearest_indices",
    "normals_target",
    "refseg_target",
    "semseg_target",
    "to_original",
]
