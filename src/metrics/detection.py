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

使用约定与口径（D12）：
- 预测列表按规范顺序传入：先按 image_id、同一张图内保持模型的输出顺序（稳定排序），用 canonical_order。
  antithetic 平局打破的结果依赖列表顺序（蒙特卡洛误差量级，复核实测 sd 约 0.0008）；规范顺序让它只取决于
  图像编号与模型输出，不取决于调用方怎么拼列表。
- AP 与 F1 对"幻觉类别"的口径不同（N6）：COCOeval 忽略全集里没有 GT 的类别（包括类别 id 不在 categories
  里的框），这些框不影响 AP；operating_point 把它们算作 FP，会拉低 P 与 F1。
- 检测 bootstrap（detection_bootstrap_statistic）的每次重采样默认只做 n_tiebreaks=1 次平局打破（N5），
  所以它的 estimate 不是标题数字。标题 AP 取 detection_map（K = DETECTION_TIEBREAK_K），CI 取 bootstrap；
  bootstrap 分布里混进了单次打破平局的噪声，CI 偏宽，方向是保守的。
- 两个系统比 F1 的配对路径（N8）：operating_point 返回逐图计数 per_image，per_image_counts 取成数组，
  交给 stats.paired_bootstrap_ci / stats.is_significant，statistic 用 stats.f1_statistic。
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


def canonical_order(preds: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """预测的规范顺序：按 image_id 稳定排序，同一张图内保持原来的（模型输出）顺序。"""
    return sorted(preds, key=lambda p: p["image_id"])


def operating_point(gt: dict, preds: Sequence[dict[str, Any]], xywh: list[list[float]], iou_thr: float) -> dict[str, Any]:
    """单一工作点：把全部预测当作"检出"，在 IoU ≥ iou_thr 下数 TP/FP/FN。

    按 (图像, 类别) 做最大基数匹配（scipy linear_sum_assignment），结果与预测顺序无关。
    IoU 用 pycocotools mask.iou（与 COCOeval 相同，连续坐标、xywh）。crowd GT 与 COCOeval 一样当作忽略区：
    不计入 GT 数；没匹配上非 crowd GT、但与某个 crowd 区域的 IoU（crowd 口径：交/预测面积）≥ 阈值的预测不算 FP。
    P = TP/(TP+FP)（没有有效预测时为 None），R = TP/n_gt，F1 = 2TP/(2TP+FP+n_gt−TP)。
    只有一个框同时与两个以上同类 GT 的 IoU ≥ 阈值时，最大匹配才与 COCOeval 的按分数贪心不同，
    这时给出贪心 TP 的上界（D12(3)(4)）。
    另返回 per_image：GT 里每张图的 (image_id, tp, fp, n_gt)，按 image_id 排序（没有预测也没有 GT 的图记 0）。
    """
    groups: dict[tuple, dict[str, list]] = {}
    for ann in gt["annotations"]:
        g = groups.setdefault((ann["image_id"], ann["category_id"]), {"gt": [], "crowd": [], "det": []})
        g["crowd" if ann.get("iscrowd", 0) else "gt"].append(list(map(float, ann["bbox"])))
    for p, b in zip(preds, xywh):
        groups.setdefault((p["image_id"], p["category_id"]), {"gt": [], "crowd": [], "det": []})["det"].append(b)
    counts = {img["id"]: [0, 0, 0] for img in gt["images"]}  # image_id -> [tp, fp, n_gt]
    for (image_id, _), g in groups.items():
        c = counts[image_id]
        c[2] += len(g["gt"])
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
            c[0] += int(hit.sum())
        rest = [d for d, m in zip(dets, matched) if not m]
        if rest and g["crowd"]:
            ious_c = np.asarray(mask_utils.iou(rest, g["crowd"], [1] * len(g["crowd"])))
            c[1] += int((ious_c.max(axis=1) < iou_thr).sum())
        else:
            c[1] += len(rest)
    per_image = [(image_id, *counts[image_id]) for image_id in sorted(counts)]
    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    n_gt = sum(c[2] for c in counts.values())
    n_pred = tp + fp
    return {
        "P50": None if n_pred == 0 else tp / n_pred,
        "R50": None if n_gt == 0 else tp / n_gt,
        "F1_50": None if n_gt == 0 else 2 * tp / (n_pred + n_gt),
        "n_tp": tp,
        "n_fp": fp,
        "n_gt": n_gt,
        "per_image": per_image,
    }


def per_image_counts(result: dict[str, Any]) -> tuple[tuple, np.ndarray]:
    """detection_map / operating_point 结果里的 per_image → (image_id 元组, (n_images, 3) 的 (tp, fp, n_gt) 数组)。

    两个系统配对前先比较 image_id 元组是否相同（同一份 GT 时必然相同）。
    """
    rows = result["per_image"]
    return tuple(r[0] for r in rows), np.array([r[1:] for r in rows], dtype=np.int64).reshape(-1, 3)


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
    重采样次数由调用方用 bootstrap_ci 的 n_resamples 控制。它的 estimate 不是标题数字（见模块说明，N5）。
    预测的 image_id 必须都在 GT 里，否则与 detection_map 一样抛 ValueError（N3）。
    """
    gt = _prepare_gt(gt_coco)
    _to_xywh(preds, {img["id"] for img in gt["images"]})
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
