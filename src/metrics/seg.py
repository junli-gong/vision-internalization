"""分割指标：cIoU、gIoU、Boundary IoU、mIoU。

mask 输入可以是 bool 或 0/1 整数数组，形状 (H, W)，预测与 GT 形状必须相同。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage

from src.constants import ADE_IGNORE_INDEX, BOUNDARY_IOU_DILATION_RATIO


def _pair(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.shape != g.shape or p.ndim != 2:
        raise ValueError(f"pred 形状 {p.shape} 与 gt 形状 {g.shape} 不一致或不是二维")
    return p, g


def mask_inter_union(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    """单样本前景的 (交, 并) 像素数。"""
    p, g = _pair(pred, gt)
    return int(np.count_nonzero(p & g)), int(np.count_nonzero(p | g))


def sample_iou(inter: int, union: int) -> float:
    """单样本 IoU；并集为 0（预测与 GT 都为空）时记 1.0。

    对齐 dvlab-research/LISA train_ds.py validate() 的 `acc_iou[union_i == 0] += 1.0  # no-object target`。
    LISA 用 inter / (union + 1e-5)；这里精确相除，差别 ≤ 1e-5 / union（union ≥ 1 像素时可忽略）。
    """
    if union == 0:
        return 1.0
    return inter / union


def ciou(stats: Sequence[tuple[int, int]]) -> float | None:
    """cIoU = 累计交 / 累计并。空集合或累计并为 0 时返回 None。

    对齐 dvlab-research/LISA train_ds.py validate()：`intersection_meter.sum / (union_meter.sum + 1e-10)`
    的前景类（iou_class[1]）；这里不加 1e-10。
    stats：每个样本的 (交, 并)，来自 mask_inter_union。
    """
    if len(stats) == 0:
        return None
    total_i = sum(i for i, _ in stats)
    total_u = sum(u for _, u in stats)
    if total_u == 0:
        return None
    return total_i / total_u


def giou(stats: Sequence[tuple[int, int]]) -> float | None:
    """gIoU = 逐样本 IoU 的平均（并集为 0 的样本记 1.0，见 sample_iou）。空集合返回 None。

    对齐 dvlab-research/LISA train_ds.py validate() 的 `acc_iou_meter.avg[1]`。
    """
    if len(stats) == 0:
        return None
    return float(np.mean([sample_iou(i, u) for i, u in stats]))


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = BOUNDARY_IOU_DILATION_RATIO) -> np.ndarray:
    """mask → 边界带（bool）。逐位对齐 bowenc0221/boundary-iou-api
    boundary_iou/utils/boundary_utils.py 的 mask_to_boundary：

        dilation = int(round(dilation_ratio * sqrt(h**2 + w**2)))，小于 1 取 1（Python round，银行家舍入）
        四周零填充 1 像素 → 3×3 全 1 核腐蚀 dilation 次 → 去掉填充 → mask - 腐蚀结果

    cv2.erode 默认把图外视为最大值，所以只有那一圈零填充向内传播；这等价于 scipy
    binary_erosion(border_value=0)（图外全为 0，最近的图外零点必在那一圈上）。测试里与朴素实现
    （以及装了 opencv 时的 cv2）逐位对照。
    """
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"mask 必须是二维，收到形状 {m.shape}")
    h, w = m.shape
    dilation = int(round(dilation_ratio * np.sqrt(h**2 + w**2)))
    if dilation < 1:
        dilation = 1
    eroded = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), iterations=dilation, border_value=0)
    return m & ~eroded


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = BOUNDARY_IOU_DILATION_RATIO) -> float:
    """单样本 Boundary IoU = |B(gt) ∩ B(pred)| / |B(gt) ∪ B(pred)|。

    边界带用 bowenc0221/boundary-iou-api 的 mask_to_boundary（见上）；IoU 按 Cheng et al. 2021
    (arXiv:2103.16562) 的定义在两个边界带之间计算。该仓库没有单样本 boundary_iou 函数（它的
    Boundary AP 走 COCOeval），所以这一步没有逐函数的官方对照。
    两个边界带都为空（两个 mask 都为空）时是 0/0；这里记 1.0，与 gIoU 的空并集约定一致（待裁）。
    """
    p, g = _pair(pred, gt)
    bp = mask_to_boundary(p, dilation_ratio)
    bg = mask_to_boundary(g, dilation_ratio)
    union = int(np.count_nonzero(bp | bg))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(bp & bg)) / union


def mean_boundary_iou(values: Sequence[float]) -> float | None:
    """逐样本 Boundary IoU 的平均。空集合返回 None。"""
    if len(values) == 0:
        return None
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# 语义分割 mIoU（ADE20K）
# ---------------------------------------------------------------------------


def reduce_zero_label(label: np.ndarray) -> np.ndarray:
    """ADE20K 标签的 reduce_zero_label：0 → 255（忽略），其余减 1，原来的 255 仍为 255。

    对齐 open-mmlab/mmsegmentation mmseg/datasets/transforms/loading.py LoadAnnotations 的
    `gt_semantic_seg[gt_semantic_seg == 0] = 255; gt_semantic_seg = gt_semantic_seg - 1;
    gt_semantic_seg[gt_semantic_seg == 254] = 255`。
    """
    lab = np.asarray(label).astype(np.int64)
    out = lab - 1
    out[lab == 0] = ADE_IGNORE_INDEX
    out[lab == ADE_IGNORE_INDEX] = ADE_IGNORE_INDEX
    return out


def intersect_and_union(pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_index: int = ADE_IGNORE_INDEX) -> tuple[np.ndarray, np.ndarray]:
    """单张图逐类的 (交, 并) 像素数，形状各为 (num_classes,)。

    对齐 open-mmlab/mmsegmentation mmseg/evaluation/metrics/iou_metric.py IoUMetric.intersect_and_union：
    去掉 label == ignore_index 的像素；histc 只统计 [0, num_classes-1] 内的值（越界预测不计入任何类）；
    并 = 预测面积 + 标签面积 - 交。label 应已做过 reduce_zero_label。
    """
    p = np.asarray(pred).astype(np.int64)
    lab = np.asarray(label).astype(np.int64)
    if p.shape != lab.shape:
        raise ValueError(f"pred 形状 {p.shape} 与 label 形状 {lab.shape} 不一致")
    keep = lab != ignore_index
    p, lab = p[keep], lab[keep]

    def hist(x: np.ndarray) -> np.ndarray:
        x = x[(x >= 0) & (x < num_classes)]
        return np.bincount(x, minlength=num_classes)

    area_inter = hist(p[p == lab])
    area_pred = hist(p)
    area_label = hist(lab)
    return area_inter, area_pred + area_label - area_inter


def mean_iou(per_image: Iterable[tuple[np.ndarray, np.ndarray]]) -> float | None:
    """mIoU：按类在整个数据集上累计交与并，IoU = 累计交 / 累计并，并集为 0 的类不计（nanmean）。

    对齐 open-mmlab/mmsegmentation mmseg/evaluation/metrics/iou_metric.py
    IoUMetric.total_area_to_metrics 的 `iou = total_area_intersect / total_area_union` 与 np.nanmean。
    per_image：每张图 intersect_and_union 的结果。空集合或所有类并集都为 0 时返回 None。
    """
    items = list(per_image)
    if not items:
        return None
    total_i = np.sum([i for i, _ in items], axis=0).astype(np.float64)
    total_u = np.sum([u for _, u in items], axis=0).astype(np.float64)
    valid = total_u > 0
    if not np.any(valid):
        return None
    return float(np.mean(total_i[valid] / total_u[valid]))
