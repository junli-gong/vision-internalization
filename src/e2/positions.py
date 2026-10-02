"""E2 按光栅位置分开的指标（DEV-0024；D53-3 的 I2）：划分的定义。

为什么分开报：三个家族在 LLM 内部，图像 token 之间都是按光栅顺序的因果注意力，第 (r, c) 格从第 1 层起只看得到光栅顺序在它之前的
图像 token，所以"信息经过 LLM 增加或丢失"可能随位置变化。标题指标（整张图）不变，另外按两种划分各报一遍标题指标（点估计）：

- half（上半 / 下半）：像素 (y, x) 的中心在 y + 0.5，归上半当且仅当 y + 0.5 < H/2（即 2y + 1 < H；H 为奇数时正中一行归下半）。
- raster_quartile（token 光栅下标的四分位 q1–q4）：像素中心落在 LLM token 网格 (H_t, W_t) 的第 r = ⌊(y + 0.5)·H_t / H⌋ 行、
  第 c = ⌊(x + 0.5)·W_t / W⌋ 列（三家族都把整张图缩放进视觉编码器，不裁不补边，token 网格均匀覆盖整张图；D53 审查 §3），光栅下标
  k = r·W_t + c ∈ [0, H_t·W_t)，四分位 q = ⌊4k / (H_t·W_t)⌋（q1 是最先的四分之一）。网格行数不是 4 的倍数时边界落在行中间，照算。
- 深度、语义分割、法向按像素划分：每张图的每个区域只用落在该区域的像素算逐样本统计（与整图的统计同一公式，tests 钉住）。
- 指代分割按目标质心划分整条样本：质心 = GT 掩码像素中心的平均 (ȳ, x̄)（像素单位），用同样的规则判上下半与光栅四分位；
  GT 掩码为空的样本不进任何区域（计数）。
网格取 LLM 层的 token 网格（ImageTokenInfo.grid_hw）；pre_proj 等参考线的探针也按它划分，所有探针比较的是同一组像素 / 样本。
"""

from __future__ import annotations

from typing import Any

import numpy as np

HALVES: tuple[str, ...] = ("top", "bottom")
QUARTILES: tuple[str, ...] = ("q1", "q2", "q3", "q4")
SPLITS: dict[str, tuple[str, ...]] = {"half": HALVES, "raster_quartile": QUARTILES}
SPLIT_LABELS = {"half": "上半 / 下半（像素中心所在的行）", "raster_quartile": "token 光栅下标的四分位"}
REGION_TITLES = {"top": "上半", "bottom": "下半", "q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4"}
DEFINITION = (
    "half：像素中心 y + 0.5 < H/2 归上半，否则下半；raster_quartile：像素中心所在 LLM token 格 (r, c) 的光栅下标 k = r·W_t + c，"
    "q = ⌊4k / (H_t·W_t)⌋；深度 / 语义分割 / 法向按像素划分，指代分割按 GT 掩码质心划分整条样本（src/e2/positions.py）"
)


def _check(hw: Any, grid_hw: Any) -> tuple[int, int, int, int]:
    H, W = (int(v) for v in hw)
    ht, wt = (int(v) for v in grid_hw)
    if min(H, W, ht, wt) < 1:
        raise ValueError(f"图像尺寸 {hw} 与 token 网格 {grid_hw} 都必须为正")
    return H, W, ht, wt


def cell_index(n_pix: int, n_tok: int) -> np.ndarray:
    """一个维度上每个像素（中心 i + 0.5）落在第几个 token 格：⌊(2i + 1)·n_tok / (2·n_pix)⌋，整数运算。"""
    i = np.arange(int(n_pix), dtype=np.int64)
    return ((2 * i + 1) * int(n_tok)) // (2 * int(n_pix))


def pixel_labels(hw: Any, grid_hw: Any) -> dict[str, np.ndarray]:
    """GT 分辨率 (H, W) 上每个像素在两种划分里的区域下标：{"half": int64 H×W（0 上 / 1 下），"raster_quartile": int64 H×W（0–3）}。"""
    H, W, ht, wt = _check(hw, grid_hw)
    half_row = (2 * np.arange(H, dtype=np.int64) + 1 >= H).astype(np.int64)
    k = cell_index(H, ht)[:, None] * wt + cell_index(W, wt)[None, :]
    return {"half": np.broadcast_to(half_row[:, None], (H, W)), "raster_quartile": (4 * k) // (ht * wt)}


def point_labels(cy: float, cx: float, hw: Any, grid_hw: Any) -> dict[str, int]:
    """连续坐标（像素单位，原点在左上角；像素 (y, x) 的中心是 (y + 0.5, x + 0.5)）在两种划分里的区域下标。
    对像素中心与 pixel_labels 逐个相同（tests 钉住）。"""
    H, W, ht, wt = _check(hw, grid_hw)
    if not (0.0 <= cy <= H and 0.0 <= cx <= W):
        raise ValueError(f"坐标 ({cy}, {cx}) 不在图像 {H}×{W} 内")
    r = min(int(np.floor(cy * ht / H)), ht - 1)
    c = min(int(np.floor(cx * wt / W)), wt - 1)
    return {"half": int(2 * cy >= H), "raster_quartile": (4 * (r * wt + c)) // (ht * wt)}


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """二值掩码的质心（像素中心的平均，像素单位）；掩码为空时 None。"""
    ys, xs = np.nonzero(np.asarray(mask))
    if ys.size == 0:
        return None
    return float(ys.mean()) + 0.5, float(xs.mean()) + 0.5


__all__ = [
    "DEFINITION",
    "HALVES",
    "QUARTILES",
    "REGION_TITLES",
    "SPLITS",
    "SPLIT_LABELS",
    "cell_index",
    "mask_centroid",
    "pixel_labels",
    "point_labels",
]
