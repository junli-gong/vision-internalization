"""框定位指标：逐样本 IoU 与 Acc@IoU0.5。"""

from __future__ import annotations

from typing import Sequence

from src.constants import BOX_ACC_IOU_THRESHOLD
from src.executors.geometry import box_iou


def box_sample_iou(pred: Sequence[float] | None, gt: Sequence[float]) -> float:
    """单个样本的框 IoU。pred 为 None（解析失败）记 0.0；解析失败率由解析器另行统计。

    IoU 定义对齐 pytorch/vision torchvision/ops/boxes.py 的 box_iou（连续坐标，不加 1），
    实现见 src.executors.geometry.box_iou。
    """
    if pred is None:
        return 0.0
    return box_iou(pred, gt)


def box_acc_at_iou(ious: Sequence[float], threshold: float = BOX_ACC_IOU_THRESHOLD) -> float | None:
    """Acc@IoU：IoU ≥ threshold 记对，返回正确比例。空集合返回 None。

    对齐 QwenLM/Qwen-VL eval_mm/evaluate_grounding.py 的判据 `if iou >= 0.5:`。
    """
    if len(ious) == 0:
        return None
    return sum(1 for v in ious if v >= threshold) / len(ious)
