"""叠加渲染：把 mask 以 alpha 混合画在原图上，外加实色轮廓（无学习；只用 numpy、scipy）。

简报 §3.8：叠加图是反馈，原图像素始终来自原图；§12：训练和评测用同一套渲染参数。所以 render_overlay
不接受任何渲染参数，颜色、alpha、轮廓一律取自 src/constants.py，训练与评测按构造一致。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

from src.constants import (
    OVERLAY_ALPHA_DEN,
    OVERLAY_ALPHA_NUM,
    OVERLAY_CONTOUR_WIDTH,
    OVERLAY_DRAW_CONTOUR,
    OVERLAY_PALETTE,
)

_STRUCT_3X3 = np.ones((3, 3), dtype=bool)


def _as_bool_mask(m: np.ndarray, shape: tuple[int, int], k: int) -> np.ndarray:
    a = np.asarray(m)
    if a.shape != shape:
        raise ValueError(f"第 {k} 个 mask 形状 {a.shape} 与图像 {shape} 不一致")
    if a.dtype == bool:
        return a
    if a.dtype.kind in "iu" and np.all((a == 0) | (a == 1)):
        return a.astype(bool)
    raise ValueError(f"第 {k} 个 mask 必须是 bool 或只含 0/1 的整数数组，收到 dtype {a.dtype}")


def _as_mask_list(masks: Sequence[np.ndarray] | np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            raise ValueError("masks 是单个二维数组，含义不明；单个 mask 请写成 [mask]，多个 mask 可以堆成 (K, H, W)")
        if masks.ndim != 3:
            raise ValueError(f"masks 数组必须是 (K, H, W)，收到形状 {masks.shape}")
    return [_as_bool_mask(m, shape, k) for k, m in enumerate(masks)]


def mask_contour(mask: np.ndarray, width: int = OVERLAY_CONTOUR_WIDTH) -> np.ndarray:
    """轮廓 = mask 减去"3×3 全 1 核腐蚀 width 次"的结果，图外按背景（scipy border_value=0）。

    width=1 即 mask 减 3×3 腐蚀：mask 内、8 邻域里至少有一个背景像素（含图外）的像素。
    贴着图边的 mask 像素也算轮廓。width 必须是正整数（scipy 的 iterations<1 表示"腐蚀到不再变化"，这里不允许）。
    """
    m = np.asarray(mask)
    if m.dtype != bool or m.ndim != 2:
        raise ValueError(f"mask 必须是二维 bool 数组，收到 dtype {m.dtype}、形状 {m.shape}")
    if isinstance(width, bool) or int(width) != width or width < 1:
        raise ValueError(f"轮廓宽度必须是正整数，收到 {width}")
    eroded = ndimage.binary_erosion(m, structure=_STRUCT_3X3, iterations=int(width), border_value=0)
    return m & ~eroded


def render_overlay(image: np.ndarray, masks: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """把 masks 按顺序叠加到 image 上，返回新的 (H, W, 3) uint8 数组，不改输入。

    - image：(H, W, 3) uint8 RGB。masks：若干 (H, W) 的 bool（或只含 0/1 的整数）数组，也可以是 (K, H, W) 数组。
    - 第 k 个 mask 用 OVERLAY_PALETTE[k]；mask 数超过调色板颜色数时报错。
    - 不在任何 mask 里的像素与原图逐位相同。
    - 填充：被至少一个 mask 覆盖的像素只混合一次，用覆盖它的最后一个 mask 的颜色（后画的在上），
      逐通道 out = (原值·(DEN−NUM) + 颜色·NUM + DEN//2) // DEN，alpha = NUM/DEN，四舍五入、恰为 .5 时进位。
    - 轮廓（OVERLAY_DRAW_CONTOUR 为真时）：填充之后按顺序把每个 mask 的轮廓（mask_contour）涂成该 mask 的颜色，
      不混合；轮廓相交处后画的在上。
    """
    img = np.asarray(image)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image 必须是 (H, W, 3) uint8，收到 dtype {img.dtype}、形状 {img.shape}")
    ms = _as_mask_list(masks, img.shape[:2])
    palette = np.asarray(OVERLAY_PALETTE, dtype=np.int64)
    if len(ms) > len(palette):
        raise ValueError(f"mask 有 {len(ms)} 个，调色板只有 {len(palette)} 种颜色")
    num, den = int(OVERLAY_ALPHA_NUM), int(OVERLAY_ALPHA_DEN)

    out = img.copy()
    owner = np.full(img.shape[:2], -1, dtype=np.int64)
    for k, m in enumerate(ms):
        owner[m] = k
    covered = owner >= 0
    if np.any(covered):
        base = img[covered].astype(np.int64)
        color = palette[owner[covered]]
        out[covered] = ((base * (den - num) + color * num + den // 2) // den).astype(np.uint8)
    if OVERLAY_DRAW_CONTOUR:
        for k, m in enumerate(ms):
            out[mask_contour(m, OVERLAY_CONTOUR_WIDTH)] = palette[k].astype(np.uint8)
    return out
