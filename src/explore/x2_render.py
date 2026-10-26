"""X-2 的修正叠加图（DEV-0069）：一张图，原图上画当前 mask（半透明填充 + 内侧轮廓 + 外侧深色描边）与提出的框。

风格与 W2 的挑选渲染一致（src/w2/render.py，separate 布局的单张）：
- 尺寸：src/w2/render.separate_size(原图宽高, style["separate_max_wh"])（横图不超过 640×480，竖图对调；不放大）；
- 缩放：原图与 mask 用 W2 的同一对函数（_resize_rgb、_resize_mask：缩小 INTER_AREA、放大 INTER_LINEAR，mask 按面积占比 ≥ 0.5 取 1）；
- 叠加：W2 的 _overlay（整数有理数的 alpha 混合，填充、轮廓、描边的颜色与宽度取 style，即冻结的 configs/w2.json 的 render.style）；
- 不写编号（只有一张图）。
- 框：在叠加之后画，颜色 BOX_RGB（Okabe–Ito 的 vermillion，与蓝填充、黄轮廓、黑描边都分得开），线宽 BOX_PX 个最终图上的像素。
  框按缩放比例换到最终图上：列 c₀ = floor(x₀ · tw / W)、c₁ = ceil(x₁ · tw / W) − 1（行同理），夹到图内；线画在 [c₀, c₁] 的边界上、
  向内 BOX_PX 个像素（框太窄时整块涂满）。
确定性：同样的输入逐位得到同样的图（同 W2 的渲染）。

只 import 不含学习的模块（numpy、OpenCV 的缩放、PIL；src/w2/render 也一样）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from src.w2 import render as WR

RENDER_VERSION = "x2-overlay-v1"
BOX_RGB: tuple[int, int, int] = (213, 94, 0)  # Okabe–Ito "vermillion"
BOX_PX = 2


def overlay_size(width: int, height: int, style: Mapping[str, Any]) -> tuple[int, int]:
    """最终图的 (宽, 高)：W2 separate 布局的单图尺寸。"""
    st = WR.resolve_style(style)
    return WR.separate_size(int(width), int(height), st["separate_max_wh"])


def _box_pixels(box: Sequence[float], w: int, h: int, tw: int, th: int) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = (float(v) for v in box)
    sx, sy = tw / float(w), th / float(h)
    c0, c1 = max(int(math.floor(x0 * sx)), 0), min(int(math.ceil(x1 * sx)) - 1, tw - 1)
    r0, r1 = max(int(math.floor(y0 * sy)), 0), min(int(math.ceil(y1 * sy)) - 1, th - 1)
    if c1 < c0 or r1 < r0:
        return None
    return c0, r0, c1, r1


def draw_box(rgb: np.ndarray, px_box: tuple[int, int, int, int] | None, *, color: Sequence[int] = BOX_RGB, width: int = BOX_PX) -> np.ndarray:
    """在 rgb（已是最终图的分辨率）上画框线（返回新数组）。px_box = (c₀, r₀, c₁, r₁)，含端点。"""
    out = rgb.copy()
    if px_box is None:
        return out
    c0, r0, c1, r1 = px_box
    t = int(width)
    col = np.asarray(color, np.uint8)
    out[r0 : min(r0 + t, r1 + 1), c0 : c1 + 1] = col
    out[max(r1 - t + 1, r0) : r1 + 1, c0 : c1 + 1] = col
    out[r0 : r1 + 1, c0 : min(c0 + t, c1 + 1)] = col
    out[r0 : r1 + 1, max(c1 - t + 1, c0) : c1 + 1] = col
    return out


def render_overlay(image: Image.Image, mask: np.ndarray, box: Sequence[float] | None, *, style: Mapping[str, Any]) -> Image.Image:
    """修正叠加图（模块说明）。image：PIL 原图；mask：(H, W) bool 的当前 mask；box：原图像素的提出框（None = 不画）；
    style：W2 的渲染样式（configs/w2.json 的 render.style，键与缺省值以 src/w2/render 为准）。返回一张 PIL RGB 图。"""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image 必须是 PIL.Image，收到 {type(image).__name__}")
    st = WR.resolve_style(style)
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    m = np.asarray(mask)
    if m.dtype != bool or m.shape != (h, w):
        raise ValueError(f"mask 必须是 (H, W) = {(h, w)} 的 bool 数组，收到 {m.dtype} {m.shape}")
    tw, th = WR.separate_size(w, h, st["separate_max_wh"])
    small = WR._resize_rgb(rgb, tw, th)
    out = WR._overlay(small, WR._resize_mask(m, tw, th), st)
    if box is not None:
        out = draw_box(out, _box_pixels(box, w, h, tw, th))
    return Image.fromarray(out)


__all__ = ["BOX_PX", "BOX_RGB", "RENDER_VERSION", "draw_box", "overlay_size", "render_overlay"]
