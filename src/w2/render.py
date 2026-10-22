"""W2 的挑选图渲染（DEV-0058；D113；接口约定见 docs/w2_plan.md §2，样式、尺寸与视觉 token 数见 docs/w2_tools.md §5）。

render_selection(image, candidates, layout=, style=) → 交给 LLM 的图（PIL，RGB）：
- layout = "grid"：一张拼图。每格是原图（等比缩放）叠上一个候选；可选地第 0 格放不叠加的原图作参照（style["reference"]）。
  格子按行优先排：[参照,] 1, 2, …, K。拼图的长边 style["grid_long_side"]（默认 640，与 E1 的图像口径一致：640×480 在
  PixelBudget(300) 下恰为 300 个视觉 token）。
- layout = "separate"：[参照,] 然后 K 张图，每张是原图叠一个候选；每张不超过 style["separate_max_wh"]（横图 640×480，竖图 480×640），
  不放大。
- 叠加：半透明填充（整数有理数混合，逐位确定）+ mask 内侧的实色轮廓 + mask 外侧一圈深色描边（任何背景上轮廓都看得清）。
  填充够淡，物体本身的边缘透过填充仍然可见。颜色取 Okabe–Ito 色盲友好调色板（蓝填充、黄轮廓：蓝–黄对比在红绿色盲下保留）。
- 编号：从 1 开始，与 generate_candidates 的顺序一致；写在每格（每张）左上角的黑底白字小方块里，OpenCV 的 Hershey 矢量字体
  （不依赖 FreeType，跨机器一致，同 W1 工作台）。参照默认不写字（style["reference_label"] = ""）。
- 缩放：先把原图与 mask 缩到格子大小（缩小用 cv2.INTER_AREA，放大用 cv2.INTER_LINEAR；mask 按面积占比 ≥ 0.5 取 1），
  再在格子分辨率上画填充、轮廓与编号，所以线宽与字号是最终图上的像素。
- 确定性：同样的输入逐位得到同样的图（同一环境；OpenCV 版本不同时缩放与抗锯齿可能差几个灰度级）。

style 可以只给一部分键，其余取 DEFAULT_STYLE；未知的键报错。resolve_style 返回补全后的样式（写进运行身份）。
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from src.executors.overlay import mask_contour

LAYOUTS = ("grid", "separate")
# 建议的默认布局（DEV-0059 的 configs/w2.json 定，管理员可改）：PixelBudget(300) 下任何一张图都被缩放到约 300 个视觉 token，
# 所以 3×3 的拼图每格只有约 33 个 token；separate 每个候选各有约 300 个（K = 8 加参照共约 2,700 个）。见 docs/w2_tools.md §5.3。
DEFAULT_LAYOUT = "separate"
ALPHA_DEN = 1000  # 填充的 alpha 按 round(alpha·1000)/1000 的整数有理数混合
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_STRUCT_3X3 = np.ones((3, 3), dtype=bool)

# 默认样式（DEV-0059 抄进 configs/w2.json 的 render.style；docs/w2_tools.md §5 逐项列出，测试核对）
DEFAULT_STYLE: Mapping[str, Any] = MappingProxyType({
    "fill_rgb": (0, 114, 178),  # Okabe–Ito "blue"
    "fill_alpha": 0.4,
    "contour_rgb": (240, 228, 66),  # Okabe–Ito "yellow"
    "contour_px": 2,  # mask 内侧的轮廓宽度（最终图上的像素）
    "halo_rgb": (0, 0, 0),
    "halo_px": 1,  # mask 外侧的深色描边宽度；0 = 不画
    "label_text_rgb": (255, 255, 255),
    "label_bg_rgb": (0, 0, 0),
    "grid_label_px": 22,  # 网格每格的编号字高（数字的像素高度）
    "separate_label_px": 32,  # 单图的编号字高
    "label_pad_px": 4,  # 编号四周的留白
    "reference": True,  # 是否放一张不叠加的原图作参照（grid：第 0 格；separate：第一张）
    "reference_label": "",  # 参照上写的字；"" = 不写
    "grid_long_side": 640,
    "grid_cols": 0,  # 0 = 自动：ceil(sqrt(格数))
    "gutter_px": 4,  # 格子之间的间隔
    "gutter_rgb": (255, 255, 255),
    "separate_max_wh": (640, 480),  # 单图的上限（横图宽×高；竖图对调）
})

_RGB_KEYS = ("fill_rgb", "contour_rgb", "halo_rgb", "label_text_rgb", "label_bg_rgb", "gutter_rgb")
_INT_RANGES = {"contour_px": (0, 16), "halo_px": (0, 16), "grid_label_px": (6, 128), "separate_label_px": (6, 256),
               "label_pad_px": (0, 32), "grid_long_side": (64, 4096), "grid_cols": (0, 16), "gutter_px": (0, 64)}


def _int(key: str, v: Any) -> int:
    lo, hi = _INT_RANGES[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not lo <= v <= hi:
        raise ValueError(f"style[{key!r}] 必须是 {lo}..{hi} 的整数，收到 {v!r}")
    return int(v)


def resolve_style(style: Mapping[str, Any] | None) -> dict[str, Any]:
    """DEFAULT_STYLE 补全并校验；返回 JSON 可写的 dict（颜色、尺寸写成列表）。"""
    style = dict(style or {})
    extra = set(style) - set(DEFAULT_STYLE)
    if extra:
        raise ValueError(f"未知的样式键 {sorted(extra)}；可选 {sorted(DEFAULT_STYLE)}")
    s = {**DEFAULT_STYLE, **style}
    out: dict[str, Any] = {}
    for k in _RGB_KEYS:
        v = s[k]
        if not isinstance(v, (list, tuple)) or len(v) != 3 or any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 255 for c in v):
            raise ValueError(f"style[{k!r}] 必须是 3 个 0..255 的整数，收到 {v!r}")
        out[k] = [int(c) for c in v]
    a = s["fill_alpha"]
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not 0 <= a <= 1:
        raise ValueError(f"style['fill_alpha'] 必须在 [0, 1] 内，收到 {a!r}")
    out["fill_alpha"] = float(a)
    for k in _INT_RANGES:
        out[k] = _int(k, s[k])
    for k in ("reference",):
        if not isinstance(s[k], bool):
            raise ValueError(f"style[{k!r}] 必须是 true / false，收到 {s[k]!r}")
        out[k] = s[k]
    if not isinstance(s["reference_label"], str) or len(s["reference_label"]) > 16 or not s["reference_label"].isascii():
        raise ValueError(f"style['reference_label'] 必须是至多 16 个 ASCII 字符的字符串，收到 {s['reference_label']!r}")
    out["reference_label"] = s["reference_label"]
    mwh = s["separate_max_wh"]
    if not isinstance(mwh, (list, tuple)) or len(mwh) != 2 or any(isinstance(c, bool) or not isinstance(c, int) or not 16 <= c <= 4096 for c in mwh):
        raise ValueError(f"style['separate_max_wh'] 必须是两个 16..4096 的整数 [宽, 高]，收到 {mwh!r}")
    out["separate_max_wh"] = [int(c) for c in mwh]
    return out


# ---------------------------------------------------------------------------
# 单格
# ---------------------------------------------------------------------------


def _resize_rgb(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = img.shape[:2]
    if (tw, th) == (w, h):
        return img.copy()
    interp = cv2.INTER_AREA if tw <= w and th <= h else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(img), (tw, th), interpolation=interp)


def _resize_mask(mask: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = mask.shape
    if (tw, th) == (w, h):
        return mask.copy()
    interp = cv2.INTER_AREA if tw <= w and th <= h else cv2.INTER_LINEAR
    return cv2.resize(mask.astype(np.float32), (tw, th), interpolation=interp) >= 0.5


def _overlay(rgb: np.ndarray, mask: np.ndarray, st: Mapping[str, Any]) -> np.ndarray:
    """半透明填充 + 内侧轮廓 + 外侧描边（都在 rgb 的分辨率上），返回新数组。"""
    out = rgb.copy()
    if not mask.any():
        return out
    num = int(round(st["fill_alpha"] * ALPHA_DEN))
    base = out[mask].astype(np.int64)
    color = np.asarray(st["fill_rgb"], np.int64)
    out[mask] = ((base * (ALPHA_DEN - num) + color * num + ALPHA_DEN // 2) // ALPHA_DEN).astype(np.uint8)
    if st["halo_px"]:
        ring = ndimage.binary_dilation(mask, structure=_STRUCT_3X3, iterations=st["halo_px"]) & ~mask
        out[ring] = st["halo_rgb"]
    if st["contour_px"]:
        out[mask_contour(mask, st["contour_px"])] = st["contour_rgb"]
    return out


def _label(img: np.ndarray, text: str, label_px: int, st: Mapping[str, Any]) -> None:
    """左上角的黑底白字编号（原地画）。label_px 是数字的像素高度；字号不超过格子短边的 1/4。"""
    if not text:
        return
    h, w = img.shape[:2]
    px = max(6, min(label_px, min(h, w) // 4))
    thick = max(1, int(round(px / 9)))
    (_, h1), _ = cv2.getTextSize("0", _FONT, 1.0, thick)
    scale = px / h1
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    pad = st["label_pad_px"]
    x1, y1 = min(tw + 2 * pad + thick, w), min(th + 2 * pad + thick, h)
    img[:y1, :x1] = st["label_bg_rgb"]
    cv2.putText(img, text, (pad, pad + th), _FONT, scale, tuple(st["label_text_rgb"]), thick, cv2.LINE_AA)


def _tile(rgb_small: np.ndarray, mask: np.ndarray | None, text: str, label_px: int, st: Mapping[str, Any]) -> np.ndarray:
    th, tw = rgb_small.shape[:2]
    out = rgb_small.copy() if mask is None else _overlay(rgb_small, _resize_mask(mask, tw, th), st)
    _label(out, text, label_px, st)
    return out


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------


def grid_shape(n_cells: int, cols: int = 0) -> tuple[int, int]:
    """(行, 列)：cols = 0 时取 ceil(sqrt(n))，行 = ceil(n / 列)。"""
    if n_cells < 1:
        raise ValueError("至少要 1 格")
    c = cols or int(math.ceil(math.sqrt(n_cells)))
    return int(math.ceil(n_cells / c)), c


def grid_tile_size(w: int, h: int, rows: int, cols: int, long_side: int, gutter: int) -> tuple[int, int]:
    """格子尺寸 (宽, 高)：等比缩放原图，使拼图的长边（含间隔）不超过 long_side 且尽量接近。"""
    if cols * w >= rows * h:
        tw = max(1, (long_side - (cols - 1) * gutter) // cols)
        th = max(1, int(math.floor(h * tw / w + 0.5)))
    else:
        th = max(1, (long_side - (rows - 1) * gutter) // rows)
        tw = max(1, int(math.floor(w * th / h + 0.5)))
    return tw, th


def separate_size(w: int, h: int, max_wh: Sequence[int]) -> tuple[int, int]:
    """单图尺寸 (宽, 高)：横图（w ≥ h）不超过 max_wh，竖图不超过对调后的 max_wh；不放大。"""
    mw, mh = (max_wh[0], max_wh[1]) if w >= h else (max_wh[1], max_wh[0])
    s = min(1.0, mw / w, mh / h)
    if s == 1.0:
        return w, h
    return max(1, int(math.floor(w * s + 0.5))), max(1, int(math.floor(h * s + 0.5)))


def render_selection(image: Image.Image, candidates: Sequence[Any], *, layout: str, style: Mapping[str, Any]) -> list[Image.Image]:
    """挑选图（模块说明）。candidates 是 generate_candidates 的输出（只用 .mask），编号 1..K 按这个顺序。返回 PIL 图的列表：
    grid 一张；separate 为 K 张（style["reference"] 时前面多一张原图）。"""
    if layout not in LAYOUTS:
        raise ValueError(f"layout 必须是 {LAYOUTS} 之一，收到 {layout!r}")
    st = resolve_style(style)
    if not isinstance(image, Image.Image):
        raise TypeError(f"image 必须是 PIL.Image，收到 {type(image).__name__}")
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if not candidates:
        raise ValueError("candidates 为空")
    masks = []
    for i, cand in enumerate(candidates, start=1):
        m = np.asarray(cand.mask)
        if m.dtype != bool or m.shape != (h, w):
            raise ValueError(f"候选 {i} 的 mask 必须是 (H, W) = {(h, w)} 的 bool 数组，收到 {m.dtype} {m.shape}")
        masks.append(m)
    cells: list[tuple[np.ndarray | None, str]] = [(None, st["reference_label"])] if st["reference"] else []
    cells += [(m, str(i)) for i, m in enumerate(masks, start=1)]

    if layout == "separate":
        tw, th = separate_size(w, h, st["separate_max_wh"])
        small = _resize_rgb(rgb, tw, th)
        return [Image.fromarray(_tile(small, m, text, st["separate_label_px"], st)) for m, text in cells]

    rows, cols = grid_shape(len(cells), st["grid_cols"])
    g = st["gutter_px"]
    tw, th = grid_tile_size(w, h, rows, cols, st["grid_long_side"], g)
    small = _resize_rgb(rgb, tw, th)
    canvas = np.empty((rows * th + (rows - 1) * g, cols * tw + (cols - 1) * g, 3), np.uint8)
    canvas[:] = st["gutter_rgb"]
    for k, (m, text) in enumerate(cells):
        r, c = divmod(k, cols)
        y0, x0 = r * (th + g), c * (tw + g)
        canvas[y0 : y0 + th, x0 : x0 + tw] = _tile(small, m, text, st["grid_label_px"], st)
    return [Image.fromarray(canvas)]


def visual_tokens(adapter: Any, images: Sequence[Image.Image]) -> list[int]:
    """这些图在适配器当前的图像口径下各占多少个视觉 token：用适配器自己的图像处理（build_inputs → 图像处理器）现算，
    取每张图在 input_ids 里的视觉 token 段长度。适配器需已加载（服务器上的真模型或开发机上的 tiny 适配器都行）。"""
    inputs = adapter.build_inputs(list(images), "x", "image_first")
    return [int(info.span.end - info.span.start) for info in inputs.images]


__all__ = [
    "ALPHA_DEN",
    "DEFAULT_LAYOUT",
    "DEFAULT_STYLE",
    "LAYOUTS",
    "grid_shape",
    "grid_tile_size",
    "render_selection",
    "resolve_style",
    "separate_size",
    "visual_tokens",
]
