"""分块与重叠拼接（无学习；只用 numpy）。

窗口格式见 constants.PIXEL_WINDOW_FORMAT：(r0, r1, c0, c1)，半开，arr[r0:r1, c0:c1] 即块内容。
与任务无关：块里放什么（logit、概率、嵌入）由调用方决定。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.constants import TILE_OVERLAP, TILE_SIZE

Window = tuple[int, int, int, int]


def _check_positive_int(v: int, name: str) -> int:
    if isinstance(v, bool) or int(v) != v or v <= 0:
        raise ValueError(f"{name} 必须是正整数，收到 {v}")
    return int(v)


def _axis_starts(size: int, tile: int, stride: int) -> list[int]:
    """一个轴上各块的起点，逐字照 mmsegmentation EncoderDecoder.slide_inference：

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        y1 = h_idx * h_stride;  y2 = min(y1 + h_crop, h_img);  y1 = max(y2 - h_crop, 0)
    """
    n = max(size - tile + stride - 1, 0) // stride + 1
    starts = []
    for i in range(n):
        end = min(i * stride + tile, size)
        starts.append(max(end - tile, 0))
    return starts


def make_tiles(height: int, width: int, tile: int = TILE_SIZE, overlap: int = TILE_OVERLAP) -> list[Window]:
    """覆盖整张图的块坐标列表，先行后列（行优先）排列，结果确定。

    步长 = tile - overlap；除最后一块外相邻块恰好重叠 overlap 像素，最后一块贴边对齐（与前一块的重叠可能更大）；
    不填充：图比块小的那一维只有一块，块长就是图长。要求 0 ≤ overlap < tile。
    """
    h = _check_positive_int(height, "height")
    w = _check_positive_int(width, "width")
    t = _check_positive_int(tile, "tile")
    if isinstance(overlap, bool) or int(overlap) != overlap or not 0 <= overlap < t:
        raise ValueError(f"overlap 必须是整数且 0 ≤ overlap < tile={t}，收到 {overlap}")
    stride = t - int(overlap)
    rows = _axis_starts(h, t, stride)
    cols = _axis_starts(w, t, stride)
    return [(r0, min(r0 + t, h), c0, min(c0 + t, w)) for r0 in rows for c0 in cols]


def stitch(tile_values: Sequence[np.ndarray], tiles: Sequence[Window], height: int, width: int) -> np.ndarray:
    """把各块的值拼回 (height, width[, ...])：重叠处取平均，权重 = 覆盖该像素的块数。

    tile_values[i] 的形状必须是 (r1-r0, c1-c0) 加上各块一致的尾部维度（如通道 C）；值必须有限。
    按块的给定顺序逐块做增量平均 m ← m + (v − m)/k（k 为到这一块为止覆盖该像素的块数），数学上等于 Σv / k；
    某块在某像素上的值与当前均值相同时均值保持不动。所以从同一张图切出的块拼回时逐位还原，常数拼出同一个常数。
    有像素没被任何块覆盖时报错。返回 float64。
    """
    h = _check_positive_int(height, "height")
    w = _check_positive_int(width, "width")
    if len(tile_values) != len(tiles):
        raise ValueError(f"块值 {len(tile_values)} 个，块坐标 {len(tiles)} 个，数量不一致")
    if len(tiles) == 0:
        raise ValueError("没有任何块")
    tail = np.shape(tile_values[0])[2:]
    mean = np.zeros((h, w) + tail, dtype=np.float64)
    count = np.zeros((h, w), dtype=np.int64)
    expand = (slice(None), slice(None)) + (None,) * len(tail)
    for i, (vals, win) in enumerate(zip(tile_values, tiles)):
        r0, r1, c0, c1 = (int(x) for x in win)
        if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
            raise ValueError(f"第 {i} 块坐标 {win} 越界或为空（图 {h}×{w}）")
        v = np.asarray(vals)
        if v.dtype.kind not in "biuf":
            raise ValueError(f"第 {i} 块的值必须是实数，收到 dtype {v.dtype}")
        v = v.astype(np.float64)
        if v.shape != (r1 - r0, c1 - c0) + tail:
            raise ValueError(f"第 {i} 块的值形状 {v.shape} 与窗口 {(r1 - r0, c1 - c0) + tail} 不一致")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"第 {i} 块的值含 NaN 或 inf")
        k = count[r0:r1, c0:c1] + 1
        m = mean[r0:r1, c0:c1]
        kk = k[expand]
        updated = np.where(v == m, m, m + (v - m) / kk)
        mean[r0:r1, c0:c1] = np.where(kk == 1, v, updated)
        count[r0:r1, c0:c1] = k
    if np.any(count == 0):
        rr, cc = np.nonzero(count == 0)
        raise ValueError(f"有 {rr.size} 个像素没被任何块覆盖，例如 ({rr[0]}, {cc[0]})")
    return mean
