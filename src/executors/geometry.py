"""几何计算与坐标格式换算（无学习；只用 numpy、shapely）。

内部坐标约定见 src/constants.py：绝对像素、浮点、xyxy，原点左上。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.constants import NORM1000_SCALE


def check_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    if not np.all(np.isfinite([x0, y0, x1, y1])):
        raise ValueError(f"框坐标非有限值: {box}")
    if x1 < x0 or y1 < y0:
        raise ValueError(f"框坐标 x1<x0 或 y1<y0: {box}")
    return x0, y0, x1, y1


def box_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = check_box(box)
    return (x1 - x0) * (y1 - y0)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 xyxy 框的 IoU，连续坐标，面积 = (x1-x0)*(y1-y0)，不加 1。

    对齐 pytorch/vision torchvision/ops/boxes.py 的 box_iou（连续坐标，无 +1）。
    注意 MAttNet / lichengunc/refer 系列的 computeIoU 用离散 +1 约定，与此不同。
    框非法（非有限、x1<x0、y1<y0）或并集为 0 时抛 ValueError。
    """
    ax0, ay0, ax1, ay1 = check_box(a)
    bx0, by0, bx1, by1 = check_box(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    if union <= 0:
        raise ValueError("两个框面积都为 0，IoU 未定义")
    return inter / union


def _to_shape(polygons: Sequence[Sequence[float]] | Sequence[np.ndarray]):
    polys = []
    for poly in polygons:
        arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        p = Polygon(arr)
        if not p.is_valid:
            raise ValueError("多边形不合法（自交等）；本函数不做修复")
        polys.append(p)
    if not polys:
        raise ValueError("polygons 为空")
    return unary_union(polys)


def polygon_iou(a: Sequence[Sequence[float]] | Sequence[np.ndarray], b: Sequence[Sequence[float]] | Sequence[np.ndarray]) -> float:
    """两组多边形（各自取并集）的面积 IoU，用 shapely 在连续坐标上精确计算。

    每组是多边形列表，每个多边形为扁平 [x0, y0, ...] 或 (k, 2) 数组。
    多边形不合法（自交等）时抛 ValueError，不静默修复；并集为 0 时抛 ValueError。
    """
    sa, sb = _to_shape(a), _to_shape(b)
    union = sa.union(sb).area
    if union <= 0:
        raise ValueError("并集面积为 0，IoU 未定义")
    return sa.intersection(sb).area / union


def _scale_xy(coords: Sequence[float] | np.ndarray, sx: float, sy: float) -> np.ndarray:
    """对 x、y 交替排列的坐标（框 xyxy、点 xy、扁平多边形）逐轴缩放。最后一维长度必须为偶数。"""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.shape[-1] % 2 != 0:
        raise ValueError(f"最后一维必须是 x, y 交替，长度为偶数；收到形状 {arr.shape}")
    out = arr.copy()
    out[..., 0::2] *= sx
    out[..., 1::2] *= sy
    return out


def _check_wh(width: float, height: float) -> None:
    if not (width > 0 and height > 0):
        raise ValueError(f"原图宽高必须为正: W={width}, H={height}")


def abs_to_norm1000(coords, *, width: float, height: float) -> np.ndarray:
    """绝对像素 → 0–1000 归一化：x' = x / W * 1000，y' = y / H * 1000（浮点，不取整）。"""
    _check_wh(width, height)
    return _scale_xy(coords, NORM1000_SCALE / width, NORM1000_SCALE / height)


def norm1000_to_abs(coords, *, width: float, height: float) -> np.ndarray:
    """0–1000 归一化 → 绝对像素。"""
    _check_wh(width, height)
    return _scale_xy(coords, width / NORM1000_SCALE, height / NORM1000_SCALE)


def abs_to_unit(coords, *, width: float, height: float) -> np.ndarray:
    """绝对像素 → 0–1：x' = x / W，y' = y / H。"""
    _check_wh(width, height)
    return _scale_xy(coords, 1.0 / width, 1.0 / height)


def unit_to_abs(coords, *, width: float, height: float) -> np.ndarray:
    """0–1 → 绝对像素。"""
    _check_wh(width, height)
    return _scale_xy(coords, float(width), float(height))
