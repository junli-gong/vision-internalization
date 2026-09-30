"""开放词汇检测：COCO 风格 mAP（包一层 pycocotools 的 COCOeval），以及 IoU=0.5 下的单一工作点 P/R/F1。

语言模型输出的框通常没有置信度，mAP 依赖的排序没有定义。COCOeval 在同分时按图像 id 顺序、再按图内输入顺序
打破平局，所以同一组预测只改图像编号 AP 就会变（独立审查实测）。这里的处理（addendum-1 第 3 条、D10(5)）：
- 预测全部有分数：照常跑一次 COCOeval。
- 预测全部没有分数：做 K = DETECTION_TIEBREAK_K 次随机打破平局——每次给全部预测（图内与跨图一起）
  一个随机置换作为分数，跑 COCOeval；报告 K 次的平均，另报最小值和最大值。种子固定，所以只要 preds 的
  列表顺序不变，结果与图像编号无关。置换成对抽取（随机置换 + 它的逆序，antithetic）：每一对预测的先后
  在 K 次里正反各占一半，消掉了"哪条预测排在列表前面"带来的蒙特卡洛偏向（两条预测时这种偏向可达 0.13）。
- 有的有分数、有的没有：抛 ValueError，不猜。
标题数字用 mAP 还是 F1@0.5 待 N9 裁定，两者都给。
"""

from __future__ import annotations

import contextlib
import copy
import io
from typing import Any, Callable, Sequence

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

from src.constants import DETECTION_OPERATING_IOU, DETECTION_TIEBREAK_K, DETECTION_TIEBREAK_SEED
from src.executors.geometry import check_box

_AP_KEYS = ("AP", "AP50", "AP75")  # COCOeval.stats[0], [1], [2]


def _prepare_gt(gt_coco: dict) -> dict:
    gt = copy.deepcopy(gt_coco)
    for ann in gt["annotations"]:
        ann.setdefault("iscrowd", 0)
        ann.setdefault("area", float(ann["bbox"][2]) * float(ann["bbox"][3]))
    return gt


def _to_xywh(preds: Sequence[dict[str, Any]], image_ids: set) -> list[list[float]]:
    out = []
    for p in preds:
        if p["image_id"] not in image_ids:
            raise ValueError(f"预测的 image_id={p['image_id']!r} 不在 GT 里")
        x0, y0, x1, y1 = check_box(p["bbox_xyxy"])
        out.append([x0, y0, x1 - x0, y1 - y0])
    return out


def _cocoeval_stats(gt: dict, preds: Sequence[dict[str, Any]], xywh: list[list[float]], scores: np.ndarray) -> np.ndarray:
    dets = [
        {"image_id": p["image_id"], "category_id": p["category_id"], "bbox": b, "score": float(s)}
        for p, b, s in zip(preds, xywh, scores)
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return np.asarray(ev.stats[:3], dtype=np.float64)


def _tiebreak_scores(n: int, k: int, seed: int) -> list[np.ndarray]:
    """k 组互不相同的分数 (perm+1)/n。置换成对抽取：随机置换 π 之后紧跟逆序 n−1−π；k 为奇数时最后一组单独抽。"""
    rng = np.random.default_rng(seed)
    perms: list[np.ndarray] = []
    while len(perms) < k:
        perm = rng.permutation(n)
        perms.append(perm)
        if len(perms) < k:
            perms.append(n - 1 - perm)
    return [(perm + 1) / n for perm in perms]


def operating_point(gt: dict, preds: Sequence[dict[str, Any]], xywh: list[list[float]], iou_thr: float) -> dict[str, Any]:
    """单一工作点：把全部预测当作"检出"，在 IoU ≥ iou_thr 下数 TP/FP/FN。

    按 (图像, 类别) 做最大基数匹配（scipy linear_sum_assignment），结果与预测顺序无关。
    IoU 用 pycocotools mask.iou（与 COCOeval 相同，连续坐标、xywh）。crowd GT 与 COCOeval 一样当作忽略区：
    不计入 GT 数；没匹配上非 crowd GT、但与某个 crowd 区域的 IoU（crowd 口径：交/预测面积）≥ 阈值的预测不算 FP。
    P = TP/(TP+FP)（没有有效预测时为 None），R = TP/n_gt，F1 = 2TP/(2TP+FP+n_gt−TP)。
    """
    groups: dict[tuple, dict[str, list]] = {}
    for ann in gt["annotations"]:
        g = groups.setdefault((ann["image_id"], ann["category_id"]), {"gt": [], "crowd": [], "det": []})
        g["crowd" if ann.get("iscrowd", 0) else "gt"].append(list(map(float, ann["bbox"])))
    for p, b in zip(preds, xywh):
        groups.setdefault((p["image_id"], p["category_id"]), {"gt": [], "crowd": [], "det": []})["det"].append(b)
    tp = fp = n_gt = 0
    for g in groups.values():
        n_gt += len(g["gt"])
        dets = g["det"]
        if not dets:
            continue
        matched = np.zeros(len(dets), bool)
        if g["gt"]:
            ious = np.asarray(mask_utils.iou(dets, g["gt"], [0] * len(g["gt"])))
            adj = ious >= iou_thr
            rows, cols = linear_sum_assignment(-adj.astype(np.float64))
            hit = adj[rows, cols]
            matched[rows[hit]] = True
            tp += int(hit.sum())
        rest = [d for d, m in zip(dets, matched) if not m]
        if rest and g["crowd"]:
            ious_c = np.asarray(mask_utils.iou(rest, g["crowd"], [1] * len(g["crowd"])))
            fp += int((ious_c.max(axis=1) < iou_thr).sum())
        else:
            fp += len(rest)
    n_pred = tp + fp
    return {
        "P50": None if n_pred == 0 else tp / n_pred,
        "R50": None if n_gt == 0 else tp / n_gt,
        "F1_50": None if n_gt == 0 else 2 * tp / (n_pred + n_gt),
        "n_tp": tp,
        "n_fp": fp,
        "n_gt": n_gt,
    }


def detection_map(
    gt_coco: dict,
    preds: Sequence[dict[str, Any]],
    *,
    n_tiebreaks: int = DETECTION_TIEBREAK_K,
    seed: int = DETECTION_TIEBREAK_SEED,
) -> dict[str, Any] | None:
    """COCO 风格 bbox AP（AP@[.5:.95]、AP50、AP75）与 IoU=DETECTION_OPERATING_IOU 下的 P/R/F1。

    AP 直接调用 cocodataset/cocoapi PythonAPI/pycocotools/cocoeval.py 的 COCOeval(iouType="bbox")
    evaluate/accumulate/summarize，取 stats[0]、stats[1]、stats[2]。
    gt_coco：COCO 格式的 GT 字典（images、annotations、categories；bbox 为 xywh，与标注文件一致）。
    preds：每条 {"image_id", "category_id", "bbox_xyxy"（内部约定：绝对像素 xyxy）, 可选 "score"}。
    分数处理见模块说明：全无分数时取 n_tiebreaks 次随机打破平局的平均，AP_min/AP_max 等给出范围。
    返回里 n_tiebreaks 为实际次数（有分数时为 0）。
    GT 没有图像或没有标注时返回 None。有 GT 但预测为空时 AP 各项记 0.0（pycocotools 的 loadRes 对空列表会抛
    IndexError，所以不进 COCOeval；按"什么都没检出"如实记 0），P50 为 None。COCOeval 给出的 -1（未定义）转成 None。
    """
    if not gt_coco.get("images") or not gt_coco.get("annotations"):
        return None
    if n_tiebreaks < 1:
        raise ValueError("n_tiebreaks 至少为 1")
    gt = _prepare_gt(gt_coco)
    xywh = _to_xywh(preds, {img["id"] for img in gt["images"]})
    out: dict[str, Any] = {}
    if len(preds) == 0:
        for k in _AP_KEYS:
            out[k] = 0.0
            out[f"{k}_min"] = out[f"{k}_max"] = 0.0
        out["n_tiebreaks"] = 0
    else:
        has = [p.get("score") is not None for p in preds]
        if all(has):
            runs = _cocoeval_stats(gt, preds, xywh, np.array([float(p["score"]) for p in preds]))[None, :]
            out["n_tiebreaks"] = 0
        elif not any(has):
            runs = np.stack([_cocoeval_stats(gt, preds, xywh, s) for s in _tiebreak_scores(len(preds), n_tiebreaks, seed)])
            out["n_tiebreaks"] = n_tiebreaks
        else:
            raise ValueError("预测里有的有分数、有的没有；不猜怎么混排")
        neg = runs < 0
        if np.any(neg.any(axis=0) != neg.all(axis=0)):  # -1 只取决于 GT，不该随打破平局变化
            raise RuntimeError("COCOeval 的 -1 哨兵在不同次打破平局间不一致")
        for j, k in enumerate(_AP_KEYS):
            col = runs[:, j]
            undefined = bool(np.all(col < 0))
            out[k] = None if undefined else float(col.mean())
            out[f"{k}_min"] = None if undefined else float(col.min())
            out[f"{k}_max"] = None if undefined else float(col.max())
    out.update(operating_point(gt, preds, xywh, DETECTION_OPERATING_IOU))
    return out


def detection_bootstrap_statistic(
    gt_coco: dict,
    preds: Sequence[dict[str, Any]],
    key: str,
    *,
    n_tiebreaks: int = 1,
    seed: int = DETECTION_TIEBREAK_SEED,
) -> tuple[np.ndarray, Callable[[np.ndarray], float]]:
    """按图像重采样的 bootstrap 路径：返回 (图像下标数组, statistic)，直接交给 stats.bootstrap_ci：

        idx, stat = detection_bootstrap_statistic(gt, preds, "AP50")
        ci = bootstrap_ci(idx, statistic=stat, n_resamples=200)

    statistic 收到一组（有放回的）图像下标，把被抽中的图像连同它们的标注与预测重新编号后调用 detection_map，
    返回 key 对应的值（"AP"、"AP50"、"AP75"、"F1_50" 等）；值为 None 时返回 nan（bootstrap_ci 会因此报错）。
    每次重采样都跑 COCOeval，代价高：n_tiebreaks 默认 1（重采样本身已覆盖打破平局的随机性），
    重采样次数由调用方用 bootstrap_ci 的 n_resamples 控制。
    """
    gt = _prepare_gt(gt_coco)
    images = list(gt["images"])
    anns_by_img: dict[Any, list] = {}
    for ann in gt["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    preds_by_img: dict[Any, list] = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    def statistic(sample: np.ndarray) -> float:
        new_images, new_anns, new_preds = [], [], []
        for j, k in enumerate(np.asarray(sample).astype(np.int64).tolist()):
            img = images[k]
            new_id = j + 1
            new_images.append({**img, "id": new_id})
            for ann in anns_by_img.get(img["id"], []):
                new_anns.append({**ann, "id": len(new_anns) + 1, "image_id": new_id})
            for p in preds_by_img.get(img["id"], []):
                new_preds.append({**p, "image_id": new_id})
        r = detection_map(
            {**gt, "images": new_images, "annotations": new_anns}, new_preds, n_tiebreaks=n_tiebreaks, seed=seed
        )
        v = None if r is None else r[key]
        return float("nan") if v is None else float(v)

    return np.arange(len(images)), statistic
