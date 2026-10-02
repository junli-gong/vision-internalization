"""深度网格的加噪过程（E3b 深度修正对，简报 §3.7 第 2 条、§8.5）。只用 numpy、scipy，无学习；给定 rng 的种子，结果确定。

src/noising/mask_noise.py 只处理二值 mask，深度网格（N×N 档号）没有现成的过程，这里按同一套约定另写一个（任务书 §2 的"深度的
修正对"；扰动种类与日程属于简报 §9 的自由探索，系数暂定，写在 configs/e3b_train.json 的 depth_noise，E3b 预注册前定稿）：
- 严重度 s ∈ [0, 1]，由调用方用 src/noising/mask_noise.sample_severity 按同一个噪声日程采样（10% 干净对 + Beta(1, 1)）。s = 0 时
  每个算子都是恒等变换，输出与输入逐位相同。
- 同一个种子下，随机量（选哪些算子、方向、位置、噪声场）的抽取与 s 无关，不同 s 之间只有幅度不同（与 mask_noise 相同）。
- 算子（OPS 的顺序也是施加顺序；个数 n ~ U{1, …, min(max_ops, 3)}，不放回地抽，按 OPS 的顺序施加，都用同一个 s）：
  - shift：整体尺度误差，全图档号加同一个偏移 round(s × shift_max_bins × z)，z ~ N(0, 1)；
  - field：平滑的局部误差，白噪声经 sigma = field_smooth_cells 的高斯平滑、除以全图标准差，乘 s × field_amp_max_bins，四舍五入后加上；
  - blobs：区域错误，个数在 blob_count 闭区间里均匀取；每个区域是圆心均匀、半径 s × blob_radius_max_cells × u（u ~ U[0.5, 1)）的
    圆盘（格中心到圆心的距离 < 半径），圆内的格子换成同一个档号（均匀取自 [0, DEPTH_NUM_BINS)）。
  每一步之后裁剪到 [0, DEPTH_NUM_BINS − 1]。
- 记录：{"severity", "ops": [每个算子的参数]}，全是 Python 原生类型，可直接写 JSON。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from src.constants import DEPTH_NUM_BINS

OPS = ("shift", "field", "blobs")


def _check_severity(severity: float) -> float:
    s = float(severity)
    if not (math.isfinite(s) and 0.0 <= s <= 1.0):
        raise ValueError(f"严重度必须在 [0, 1] 内，收到 {severity}")
    return s


def _round_half_up(x: np.ndarray | float) -> np.ndarray:
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


def _clip(g: np.ndarray) -> np.ndarray:
    return np.clip(g, 0, DEPTH_NUM_BINS - 1).astype(np.int64)


def op_shift(g: np.ndarray, s: float, p: Mapping[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    z = float(rng.standard_normal())
    off = int(_round_half_up(s * float(p["shift_max_bins"]) * z))
    return _clip(g + off), {"op": "shift", "offset_bins": off}


def op_field(g: np.ndarray, s: float, p: Mapping[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    field = rng.standard_normal(g.shape)
    sigma = float(p["field_smooth_cells"])
    amp = s * float(p["field_amp_max_bins"])
    rec = {"op": "field", "amplitude_bins": amp, "smooth_sigma_cells": sigma}
    if amp <= 0.0:
        return g.copy(), rec
    n = ndimage.gaussian_filter(field, sigma, mode="nearest") if sigma > 0 else field
    std = float(n.std())
    if std > 0:
        n = n / std
    return _clip(g + _round_half_up(amp * n)), rec


def op_blobs(g: np.ndarray, s: float, p: Mapping[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    lo, hi = (int(v) for v in p["blob_count"])
    k = int(rng.integers(lo, hi + 1))
    n_rows, n_cols = g.shape
    out = g.copy()
    blobs = []
    yy, xx = np.mgrid[0:n_rows, 0:n_cols]
    for _ in range(k):
        cy, cx = float(rng.uniform(0, n_rows)), float(rng.uniform(0, n_cols))
        u = float(rng.uniform(0.5, 1.0))
        val = int(rng.integers(0, DEPTH_NUM_BINS))
        radius = s * float(p["blob_radius_max_cells"]) * u
        blobs.append({"center_rc": [cy, cx], "radius_cells": radius, "bin": val})
        if radius <= 0.0:
            continue
        disk = (yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 < radius * radius
        out[disk] = val
    return out, {"op": "blobs", "blobs": blobs}


_FN = {"shift": op_shift, "field": op_field, "blobs": op_blobs}


def corrupt_depth_grid(grid: np.ndarray, severity: float, rng: np.random.Generator,
                       params: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    """对深度档号网格施加一组扰动，返回 (带噪网格 int64，记录)。规则见模块说明；params 是配置的 depth_noise 节。"""
    g = np.asarray(grid)
    if g.ndim != 2 or not np.issubdtype(g.dtype, np.integer):
        raise ValueError(f"深度网格必须是二维整数档号，收到 dtype {g.dtype}、形状 {g.shape}")
    if g.size and (int(g.min()) < 0 or int(g.max()) >= DEPTH_NUM_BINS):
        raise ValueError("深度档号越界")
    s = _check_severity(severity)
    max_ops = int(params["max_ops"])
    if not 1 <= max_ops:
        raise ValueError(f"max_ops 必须 ≥ 1，收到 {max_ops}")
    n_ops = int(rng.integers(1, min(max_ops, len(OPS)) + 1))
    picked = sorted(int(i) for i in rng.choice(len(OPS), size=n_ops, replace=False))
    out = g.astype(np.int64)
    rec: dict[str, Any] = {"severity": s, "ops": []}
    for name in (OPS[i] for i in picked):
        out, r = _FN[name](out, s, params, rng)
        rec["ops"].append(r)
    return out, rec


__all__ = ["OPS", "corrupt_depth_grid"]
