"""开放词汇检测 mAP：包一层 pycocotools 的 COCOeval。"""

from __future__ import annotations

import contextlib
import copy
import io
from typing import Any, Sequence

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.executors.geometry import check_box


def detection_map(gt_coco: dict, preds: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    """COCO 风格 bbox mAP，返回 {"AP": AP@[.5:.95], "AP50": AP@.5}。

    直接调用 cocodataset/cocoapi PythonAPI/pycocotools/cocoeval.py 的 COCOeval(iouType="bbox")
    evaluate/accumulate/summarize，取 stats[0]、stats[1]。
    gt_coco：COCO 格式的 GT 字典（images、annotations、categories；bbox 为 xywh，与标注文件一致）。
    preds：每条 {"image_id", "category_id", "bbox_xyxy"（内部约定：绝对像素 xyxy）, 可选 "score"}。
      无分数的预测统一记 score=1.0（语言模型输出框时通常没有置信度）。注意 COCOeval 按分数做
      稳定排序（mergesort）：全部同分时，accumulate 先按图像 id 顺序、再按同一图内 preds 的输入顺序排，
      所以 AP 依赖同一图内预测的顺序；调用方应固定这个顺序（如按模型输出顺序）。
    GT 没有图像或没有标注时返回 None。有 GT 但预测为空时，AP、AP50 记 0.0
    （pycocotools 的 loadRes 对空列表会抛 IndexError，所以不进 COCOeval；按"什么都没检出"如实记 0）。
    COCOeval 给出的 -1（某项未定义）转成 None。
    """
    if not gt_coco.get("images") or not gt_coco.get("annotations"):
        return None
    if len(preds) == 0:
        return {"AP": 0.0, "AP50": 0.0}
    gt = copy.deepcopy(gt_coco)
    for ann in gt["annotations"]:
        ann.setdefault("iscrowd", 0)
        ann.setdefault("area", float(ann["bbox"][2]) * float(ann["bbox"][3]))
    dets = []
    for p in preds:
        x0, y0, x1, y1 = check_box(p["bbox_xyxy"])
        score = p.get("score")
        dets.append(
            {
                "image_id": p["image_id"],
                "category_id": p["category_id"],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": 1.0 if score is None else float(score),
            }
        )
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ap, ap50 = float(ev.stats[0]), float(ev.stats[1])
    return {"AP": None if ap < 0 else ap, "AP50": None if ap50 < 0 else ap50}
