"""分割指标：cIoU、gIoU、Boundary IoU、mIoU。

mask 输入只接受 bool 或取值在 {0, 1} 的整数数组（src.grid.codec.as_binary_mask），形状 (H, W)，
预测与 GT 形状必须相同。预测为 None 表示解析失败：该样本 IoU 记 0，cIoU 累计 (0, |GT|)，
GT 为空时也记 0（D10）；预测是真实的空 mask 且 GT 也为空时，按空并集约定记 1.0。
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence

import numpy as np
from scipy import ndimage

from src.constants import ADE_IGNORE_INDEX, ADE_NUM_CLASSES, BOUNDARY_IOU_DILATION_RATIO
from src.grid.codec import as_binary_mask


class MaskStats(NamedTuple):
    """单样本前景的交、并像素数；failed=True 表示预测解析失败（预测为 None）。"""

    inter: int
    union: int
    failed: bool = False


def _pair(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = as_binary_mask(pred, "pred")
    g = as_binary_mask(gt, "gt")
    if p.shape != g.shape or p.ndim != 2:
        raise ValueError(f"pred 形状 {p.shape} 与 gt 形状 {g.shape} 不一致或不是二维")
    return p, g


def mask_inter_union(pred: np.ndarray | None, gt: np.ndarray) -> MaskStats:
    """单样本前景的 (交, 并, failed)。pred 为 None 时返回 (0, |GT|, True)。"""
    if pred is None:
        g = as_binary_mask(gt, "gt")
        if g.ndim != 2:
            raise ValueError(f"gt 必须是二维，收到形状 {g.shape}")
        return MaskStats(0, int(np.count_nonzero(g)), True)
    p, g = _pair(pred, gt)
    return MaskStats(int(np.count_nonzero(p & g)), int(np.count_nonzero(p | g)))


def _unpack(s: Sequence) -> tuple[int, int, bool]:
    if len(s) == 2:
        return int(s[0]), int(s[1]), False
    return int(s[0]), int(s[1]), bool(s[2])


def sample_iou(inter: int, union: int, failed: bool = False) -> float:
    """单样本 IoU。解析失败记 0.0；否则并集为 0（预测与 GT 都为空）时记 1.0。

    对齐 dvlab-research/LISA train_ds.py validate() 的 `acc_iou[union_i == 0] += 1.0  # no-object target`。
    LISA 用 inter / (union + 1e-5)；这里精确相除，差别 ≤ 1e-5 / union（union ≥ 1 像素时可忽略）。
    """
    if failed:
        return 0.0
    if union == 0:
        return 1.0
    return inter / union


def ciou(stats: Sequence[Sequence]) -> float | None:
    """cIoU = 累计交 / 累计并。空集合或累计并为 0 时返回 None。

    对齐 dvlab-research/LISA train_ds.py validate()：`intersection_meter.sum / (union_meter.sum + 1e-10)`
    的前景类（iou_class[1]）；这里不加 1e-10。
    stats：每个样本的 MaskStats 或 (交, 并)，来自 mask_inter_union。解析失败的样本按 (0, |GT|) 计入。
    """
    if len(stats) == 0:
        return None
    parts = [_unpack(s) for s in stats]
    total_u = sum(u for _, u, _ in parts)
    if total_u == 0:
        return None
    return sum(i for i, _, _ in parts) / total_u


def giou(stats: Sequence[Sequence]) -> float | None:
    """gIoU = 逐样本 IoU 的平均（见 sample_iou：解析失败记 0，空并集记 1.0）。空集合返回 None。

    对齐 dvlab-research/LISA train_ds.py validate() 的 `acc_iou_meter.avg[1]`。
    """
    if len(stats) == 0:
        return None
    return float(np.mean([sample_iou(*_unpack(s)) for s in stats]))


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = BOUNDARY_IOU_DILATION_RATIO) -> np.ndarray:
    """mask → 边界带（bool）。逐位对齐 bowenc0221/boundary-iou-api
    boundary_iou/utils/boundary_utils.py 的 mask_to_boundary：

        dilation = int(round(dilation_ratio * sqrt(h**2 + w**2)))，小于 1 取 1（Python round，银行家舍入）
        四周零填充 1 像素 → 3×3 全 1 核腐蚀 dilation 次 → 去掉填充 → mask - 腐蚀结果

    cv2.erode 默认把图外视为最大值，所以只有那一圈零填充向内传播；这等价于 scipy
    binary_erosion(border_value=0)（图外全为 0，最近的图外零点必在那一圈上）。测试里与朴素实现
    （以及装了 opencv 时的 cv2）逐位对照。
    """
    m = as_binary_mask(mask)
    if m.ndim != 2:
        raise ValueError(f"mask 必须是二维，收到形状 {m.shape}")
    h, w = m.shape
    dilation = int(round(dilation_ratio * np.sqrt(h**2 + w**2)))
    if dilation < 1:
        dilation = 1
    eroded = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), iterations=dilation, border_value=0)
    return m & ~eroded


def boundary_iou(pred: np.ndarray | None, gt: np.ndarray, dilation_ratio: float = BOUNDARY_IOU_DILATION_RATIO) -> float:
    """单样本 Boundary IoU = |B(gt) ∩ B(pred)| / |B(gt) ∪ B(pred)|。

    边界带用 bowenc0221/boundary-iou-api 的 mask_to_boundary（见上）；IoU 按 Cheng et al. 2021
    (arXiv:2103.16562) 的定义在两个边界带之间计算。该仓库没有单样本 boundary_iou 函数（它的
    Boundary AP 走 COCOeval），所以这一步没有逐函数的官方对照。
    两个边界带都为空（两个 mask 都为空）时是 0/0；这里记 1.0，与 gIoU 的空并集约定一致（D10(3)）。
    pred 为 None（解析失败）时记 0.0，GT 为空也一样。
    """
    if pred is None:
        as_binary_mask(gt, "gt")
        return 0.0
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


def intersect_and_union(
    pred: np.ndarray,
    label: np.ndarray,
    num_classes: int = ADE_NUM_CLASSES,
    ignore_index: int = ADE_IGNORE_INDEX,
    *,
    reduce_zero: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """单张图逐类的 (交, 并) 像素数，形状各为 (num_classes,)。

    reduce_zero 是必填的关键字参数，没有默认值（D12，N1）：
    - reduce_zero=True：label 是原始 ADE20K 标注，这里先做 reduce_zero_label。ADE 调用方传
      reduce_zero=constants.ADE_REDUCE_ZERO_LABEL（对齐 mmseg ADE20KDataset）。
    - reduce_zero=False：label 已经是 0..num_classes-1 加忽略值（已 reduce 过的 ADE 标签、二类或其他数据集）。
    已 reduce 过的标签再传 True 会减两次，完美预测的 mIoU 算成 0，所以不给默认值。
    之后对齐 open-mmlab/mmsegmentation mmseg/evaluation/metrics/iou_metric.py IoUMetric.intersect_and_union：
    去掉 label == ignore_index 的像素；histc 只统计 [0, num_classes-1] 内的值（越界预测不计入任何类）；
    并 = 预测面积 + 标签面积 - 交。
    """
    p = np.asarray(pred).astype(np.int64)
    lab = np.asarray(label).astype(np.int64)
    if p.shape != lab.shape:
        raise ValueError(f"pred 形状 {p.shape} 与 label 形状 {lab.shape} 不一致")
    if reduce_zero:
        lab = reduce_zero_label(lab)
    keep = lab != ignore_index
    p, lab = p[keep], lab[keep]

    def hist(x: np.ndarray) -> np.ndarray:
        x = x[(x >= 0) & (x < num_classes)]
        return np.bincount(x, minlength=num_classes)

    area_inter = hist(p[p == lab])
    area_pred = hist(p)
    area_label = hist(lab)
    return area_inter, area_pred + area_label - area_inter


def miou_from_totals(total_inter: np.ndarray, total_union: np.ndarray) -> float | None:
    """由逐类累计的交、并算 mIoU：IoU = 交 / 并，并集为 0 的类不计（nanmean）。所有类并集都为 0 时返回 None。

    对齐 open-mmlab/mmsegmentation mmseg/evaluation/metrics/iou_metric.py
    IoUMetric.total_area_to_metrics 的 `iou = total_area_intersect / total_area_union` 与 np.nanmean。
    交为 0、并 > 0 的类计 IoU = 0，不剔除。
    """
    ti = np.asarray(total_inter, dtype=np.float64)
    tu = np.asarray(total_union, dtype=np.float64)
    valid = tu > 0
    if not np.any(valid):
        return None
    return float(np.mean(ti[valid] / tu[valid]))


def mean_iou(per_image: Iterable[tuple[np.ndarray, np.ndarray]]) -> float | None:
    """mIoU：按类在整个数据集上累计交与并，再用 miou_from_totals。

    per_image：每张图 intersect_and_union 的结果。空集合或所有类并集都为 0 时返回 None。
    """
    items = list(per_image)
    if not items:
        return None
    return miou_from_totals(np.sum([i for i, _ in items], axis=0), np.sum([u for _, u in items], axis=0))


def pixel_correct_total(
    pred: np.ndarray,
    label: np.ndarray,
    num_classes: int = ADE_NUM_CLASSES,
    ignore_index: int = ADE_IGNORE_INDEX,
    *,
    reduce_zero: bool,
) -> tuple[int, int]:
    """单张图的 (预测正确的像素数, 参与评测的像素数)，像素准确率用。DEV-0023 新增（E2 探针）。

    参与评测的像素与 intersect_and_union 相同：reduce_zero 的规则相同（必填，没有默认值），去掉 label == ignore_index，
    只计 label ∈ [0, num_classes)；正确 = 其中 pred == label 的像素，恰为 intersect_and_union 的逐类交之和
    （tests/test_e2_metrics.py 钉住）。
    """
    p = np.asarray(pred).astype(np.int64)
    lab = np.asarray(label).astype(np.int64)
    if p.shape != lab.shape:
        raise ValueError(f"pred 形状 {p.shape} 与 label 形状 {lab.shape} 不一致")
    if reduce_zero:
        lab = reduce_zero_label(lab)
    keep = (lab != ignore_index) & (lab >= 0) & (lab < num_classes)
    return int(np.count_nonzero(p[keep] == lab[keep])), int(np.count_nonzero(keep))


def pixel_accuracy(per_image: Iterable[tuple[int, int]]) -> float | None:
    """像素准确率 = Σ正确 / Σ参与评测的像素（逐像素汇总）。空集合或分母为 0 返回 None。

    对齐 open-mmlab/mmsegmentation mmseg/evaluation/metrics/iou_metric.py IoUMetric.total_area_to_metrics 的
    `all_acc = total_area_intersect.sum() / total_area_label.sum()`（aAcc）。per_image：每张图 pixel_correct_total 的结果。
    """
    items = list(per_image)
    total = sum(int(t) for _, t in items)
    if not items or total == 0:
        return None
    return sum(int(c) for c, _ in items) / total
