"""光栅化与 RLE 编解码（无学习；只用 numpy、pycocotools）。

mask 一律是形状 (H, W) 的 bool 数组。坐标约定见 src/constants.py。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pycocotools import mask as mask_utils

from src.constants import PIXEL_CENTER_OFFSET


def _check_hw(height: int, width: int) -> None:
    if int(height) != height or int(width) != width or height <= 0 or width <= 0:
        raise ValueError(f"非法图像尺寸 H={height}, W={width}")


def box_to_mask(box_xyxy: Sequence[float], height: int, width: int) -> np.ndarray:
    """框 → mask：像素中心 (j+0.5, i+0.5) 满足 x0 < cx <= x1、y0 < cy <= y1 即为 1。

    边界归属 (lo, hi] 见 constants.BOX_RASTER_RULE。框可以部分或全部落在图外，图外部分被忽略。
    """
    _check_hw(height, width)
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    if not np.all(np.isfinite([x0, y0, x1, y1])):
        raise ValueError(f"框坐标非有限值: {box_xyxy}")
    if x1 < x0 or y1 < y0:
        raise ValueError(f"框坐标 x1<x0 或 y1<y0: {box_xyxy}")
    cx = np.arange(width) + PIXEL_CENTER_OFFSET
    cy = np.arange(height) + PIXEL_CENTER_OFFSET
    in_x = (cx > x0) & (cx <= x1)
    in_y = (cy > y0) & (cy <= y1)
    return in_y[:, None] & in_x[None, :]


def _to_coco_polygons(polygons: Sequence[Sequence[float]] | Sequence[np.ndarray]) -> list[list[float]]:
    out = []
    for poly in polygons:
        arr = np.asarray(poly, dtype=np.float64).reshape(-1)
        if arr.size % 2 != 0:
            raise ValueError("多边形坐标个数必须为偶数（x, y 交替）")
        if arr.size < 6:
            raise ValueError("多边形至少需要 3 个顶点")
        if not np.all(np.isfinite(arr)):
            raise ValueError("多边形坐标含非有限值")
        out.append(arr.tolist())
    if not out:
        raise ValueError("polygons 为空")
    return out


def polygons_to_mask(polygons: Sequence[Sequence[float]] | Sequence[np.ndarray], height: int, width: int) -> np.ndarray:
    """多边形 → mask，走 pycocotools（与 COCO/RefCOCO GT 同一套规则）。

    polygons：COCO 格式的多边形列表，每个是扁平的 [x0, y0, x1, y1, ...] 或形状 (k, 2) 的数组。
    多个多边形取并集，与 pycocotools COCO.annToMask 相同（frPyObjects + merge）。
    """
    _check_hw(height, width)
    rles = mask_utils.frPyObjects(_to_coco_polygons(polygons), int(height), int(width))
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def mask_to_rle(mask: np.ndarray) -> dict:
    """mask → COCO 压缩 RLE，counts 转成 str，可直接写 JSON。"""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask 必须是二维，收到形状 {mask.shape}")
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": rle["counts"].decode("ascii")}


def rle_to_mask(rle: dict) -> np.ndarray:
    """COCO RLE → mask。接受压缩 RLE（counts 为 str/bytes）和未压缩 RLE（counts 为整数列表）。"""
    h, w = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, list):
        rle_c = mask_utils.frPyObjects({"size": [h, w], "counts": counts}, h, w)
    else:
        rle_c = {"size": [h, w], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    return mask_utils.decode(rle_c).astype(bool)
