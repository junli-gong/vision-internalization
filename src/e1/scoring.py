"""E1 的逐条解析、评分、诊断与汇总（DEV-0014 任务书 §2）。指标一律用已合入的 src/metrics 与 src/stats，不另写。

逐条（score_generation）：生成结果 → parse_generation（思考段未闭合、截断、空回答先判；E1 协议：截断一律判失败）→ 本任务的解析
→ 逐条得分，写成 predictions.jsonl 的一行。坐标一律在原图尺寸上：框与多边形由解析层按原图宽高换成绝对像素，网格用
grid_to_mask 回填到原图。

汇总（aggregate）：只读 predictions.jsonl 的记录与样本的 GT（不读内存里的生成结果），所以续跑后再算一遍逐字节相同。
- 定位：Acc@IoU0.5（metrics.box）；多边形：cIoU（ratio_of_sums）与 gIoU（metrics.seg）；网格：gIoU 与 cIoU；
- 计数、点对、多选：准确率；
- 检测：F1@0.5 为标题（metrics.detection.operating_point，逐图 (tp, fp, n_gt) 做 bootstrap，stats.f1_statistic），
  随机打破平局后平均的 mAP 作补充（detection_map，K = DETECTION_TIEBREAK_K）；编造类别 = 预测的类别名（normalize_phrase 后）
  不在给定类别表里，按框计比例；没有标签的框单列，同样记为 FP；
- 每个任务：bootstrap 95% CI（按样本重采样；检测按图），ParseTally 分原因，截断率，思考段未闭合率，生成 token 数分布，
  实际看到的像素数分布（D28-3），坐标诊断（全部坐标 ≤1 且含小数的条数，D22-2 / D30）。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from src.constants import BOX_ACC_IOU_THRESHOLD, GRID_SIZE_E1
from src.data.records import Sample
from src.e1.tasks import groups_of
from src.executors.raster import polygons_to_mask, rle_to_mask
from src.grid.codec import GridParseFailure, grid_to_mask, parse_grid_text
from src.metrics.box import box_acc_at_iou, box_sample_iou
from src.metrics.detection import (
    canonical_order,
    detection_bootstrap_statistic,
    detection_map,
    per_image_counts,
)
from src.metrics.seg import ciou, giou, mask_inter_union, sample_iou
from src.parsing import (
    ParseFailure,
    ParseResult,
    ParseTally,
    normalize_phrase,
    parse_choice,
    parse_count,
    parse_generation,
    parse_named_choice,
    parse_polygons,
)
from src.stats import CI, bootstrap_ci, f1_statistic, ratio_of_sums

COORD_TASKS = frozenset({"refexp_box", "refexp_poly", "ovd"})
MCQ_TASKS = frozenset({"rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial"})
MASK_TASKS = frozenset({"refexp_poly", "grid16"})
UNKNOWN_CATEGORY_ID = -1  # 编造类别与无标签的框：不在 GT 的类别表里，COCOeval 忽略、F1 记 FP（D12 N6）

_FAILURES = {e.value: e for e in list(ParseFailure) + list(GridParseFailure)}
if len(_FAILURES) != len(ParseFailure) + len(GridParseFailure):
    raise RuntimeError("ParseFailure 与 GridParseFailure 有重名的取值")


def failure_enum(value: str | None):
    """predictions.jsonl 里的失败原因字符串 → 枚举（ParseTally 只收枚举）。"""
    if value is None:
        return None
    try:
        return _FAILURES[value]
    except KeyError:
        raise ValueError(f"未知的失败原因 {value!r}") from None


# ---------------------------------------------------------------------------
# 多选题：先认字母，认不出再按选项原文
# ---------------------------------------------------------------------------


class MCQParse(NamedTuple):
    """value：选项字母；route："letter"（parse_choice 认出）或 "option_text"（按选项原文匹配上）。"""

    value: str | None
    route: str | None
    failure: ParseFailure | None


def parse_mcq(text: str, *, letters: Sequence[str], options: Sequence[str] | None) -> MCQParse:
    """parse_choice(text, letters)；只有它判 no_choice 时，再用 parse_named_choice(text, options)（规范化后整段相等，或恰好一个
    选项原文作为整词短语出现），匹配上就取该选项的字母。选项原文规范化后有重复（parse_named_choice 抛 ValueError）时不做第二步。"""
    r = parse_choice(text, letters)
    if r.failure is None:
        return MCQParse(r.value, "letter", None)
    if r.failure is not ParseFailure.NO_CHOICE or not options:
        return MCQParse(None, None, r.failure)
    try:
        n = parse_named_choice(text, list(options))
    except ValueError:
        return MCQParse(None, None, r.failure)
    if n.failure is not None:
        return MCQParse(None, None, n.failure)
    return MCQParse(letters[list(options).index(n.value)], "option_text", None)


# ---------------------------------------------------------------------------
# 坐标诊断（D22-2 / D30）：方括号里的数全部 ≤ 1，且至少一个带小数点
# ---------------------------------------------------------------------------

_BRACKET = re.compile(r"\[[^\[\]]*\]")
_NUM = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")


def coord_all_le1_decimal(text: str) -> bool:
    """回答里所有方括号内的数（JSON 的 [x1, y1, x2, y2]、多边形的点、InternVL 的 <box>[[…]]</box> 都在方括号里）全部 ≤ 1，
    且至少有一个带小数点：疑似 0–1 归一化坐标被当成 0–1000（D22-2 的风险）。没有数时为 False。与家族无关。"""
    nums = [m.group() for b in _BRACKET.findall(text) for m in _NUM.finditer(b)]
    if not nums:
        return False
    return all(float(v) <= 1.0 for v in nums) and any("." in v for v in nums)


# ---------------------------------------------------------------------------
# GT
# ---------------------------------------------------------------------------


def gt_mask(sample: Sample) -> np.ndarray:
    m = sample.gt["mask"]
    h, w = int(m["height"]), int(m["width"])
    if "polygons" in m:
        return polygons_to_mask(m["polygons"], h, w)
    rle = dict(m["rle"])
    rle.setdefault("size", [h, w])
    return rle_to_mask(rle)


def _round_trip(x: Any) -> Any:
    """元组 → 列表（JSON 写出与读回同形）。"""
    if isinstance(x, (list, tuple)):
        return [_round_trip(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# 逐条
# ---------------------------------------------------------------------------


def score_generation(task: str, parser: str, sample: Sample, result: Any, *, adapter: Any, image_size: tuple[int, int],
                     extras: Mapping[str, Any]) -> dict[str, Any]:
    """一条生成结果 → predictions.jsonl 里的 failure / parsed / score / 诊断字段（不含公共字段）。

    image_size = 原图 (W, H)。adapter 只用它的 parse_one_box / parse_boxes（本家族的原生框格式，不需要模型）。
    """
    w, h = image_size
    out: dict[str, Any] = {}
    if parser == "one_box":
        p = parse_generation(result, adapter.parse_one_box, width=w, height=h)
        out["failure"] = None if p.failure is None else p.failure.value
        out["parsed"] = None if p.failure else {"box": list(p.box)}
        iou = box_sample_iou(None if p.failure else p.box, sample.gt["box"])
        out["score"] = {"iou": iou, "correct": iou >= BOX_ACC_IOU_THRESHOLD}
    elif parser == "polygon":
        p = parse_generation(result, parse_polygons, width=w, height=h)
        out["failure"] = None if p.failure is None else p.failure.value
        out["parsed"] = None if p.failure else {"polygons": _round_trip(p.value)}
        g = gt_mask(sample)
        if g.shape != (h, w):
            raise ValueError(f"{sample.id}：GT mask 形状 {g.shape} 与原图 (H, W) = {(h, w)} 不符")
        pred = None if p.failure else polygons_to_mask(list(p.value), h, w)
        st = mask_inter_union(pred, g)
        out["score"] = {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}
    elif parser == "grid16":
        p = parse_generation(result, parse_grid_text, n=GRID_SIZE_E1)
        out["failure"] = None if p.failure is None else p.failure.value
        out["parsed"] = None if p.failure else {"grid": ["".join(str(int(v)) for v in row) for row in p.grid.tolist()]}
        g = gt_mask(sample)
        if g.shape != (h, w):
            raise ValueError(f"{sample.id}：GT mask 形状 {g.shape} 与原图 (H, W) = {(h, w)} 不符")
        pred = None if p.failure else grid_to_mask(p.grid, h, w)
        st = mask_inter_union(pred, g)
        out["score"] = {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}
    elif parser == "count":
        p = parse_generation(result, parse_count)
        out["failure"] = None if p.failure is None else p.failure.value
        out["parsed"] = None if p.failure else {"count": p.value}
        gt = int(sample.gt["count"])
        out["score"] = {"gt": gt, "correct": p.failure is None and p.value == gt}
    elif parser == "choice":
        p = parse_generation(result, parse_mcq, letters=list(extras["letters"]), options=list(extras["options"]))
        out["failure"] = None if p.failure is None else p.failure.value
        out["parsed"] = None if p.failure else {"letter": p.value, "route": p.route}
        gt = sample.gt["answer"] if task != "rel_depth_nyu" else sample.gt["closer"]
        out["score"] = {"gt": gt, "correct": p.failure is None and p.value == gt}
    elif parser == "boxes":
        p = parse_generation(result, adapter.parse_boxes, width=w, height=h)
        out["failure"] = None if p.failure is None else p.failure.value
        names = {normalize_phrase(n): cid for n, cid in zip(sample.fields["categories"], sample.fields["category_ids"])}
        if p.failure:
            out["parsed"] = None
            out["score"] = {"n_pred": 0, "n_fabricated": 0, "n_unlabeled": 0}
        else:
            cids = [UNKNOWN_CATEGORY_ID if lab is None else names.get(normalize_phrase(lab), UNKNOWN_CATEGORY_ID)
                    for lab in p.labels]
            n_unl = sum(1 for lab in p.labels if lab is None)
            out["parsed"] = {"boxes": _round_trip(p.boxes), "labels": list(p.labels), "category_ids": cids}
            out["score"] = {"n_pred": len(cids), "n_fabricated": sum(1 for c in cids if c == UNKNOWN_CATEGORY_ID) - n_unl,
                            "n_unlabeled": n_unl}
    else:
        raise ValueError(f"未知的解析函数 {parser!r}")
    if task in COORD_TASKS:
        out["coord_le1_decimal"] = coord_all_le1_decimal(result.text)
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def ci_dict(ci: CI | None) -> dict[str, Any] | None:
    if ci is None:
        return None
    return {"low": ci.low, "high": ci.high, "level": ci.level, "n_resamples": ci.n_resamples}


def _mean_ci(values: Sequence[float]) -> dict[str, Any]:
    """逐样本值（0/1 或 IoU）的均值与 bootstrap CI；空集合报 null。"""
    if not values:
        return {"value": None, "ci": None, "n": 0}
    ci = bootstrap_ci(np.asarray(values, dtype=np.float64))
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(values)}


def _ratio_ci(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    if not pairs or sum(u for _, u in pairs) == 0:
        return {"value": None, "ci": None, "n": len(pairs)}
    arr = np.asarray(pairs, dtype=np.float64)
    try:
        ci = bootstrap_ci(arr, statistic=ratio_of_sums)
    except ValueError:  # 某次重采样的累计并为 0（极小样本）
        return {"value": ratio_of_sums(arr), "ci": None, "n": len(pairs)}
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(pairs)}


def distribution(values: Iterable[float | int]) -> dict[str, Any] | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    arr = np.asarray(v, dtype=np.float64)
    top = Counter(v).most_common()
    top.sort(key=lambda kv: (-kv[1], kv[0]))
    q = np.percentile(arr, [5, 50, 95])
    return {"n": len(v), "min": min(v), "p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2]), "max": max(v),
            "mean": float(arr.mean()), "n_distinct": len(set(v)), "top": [[k, c] for k, c in top[:10]]}


def _rate(k: int, n: int) -> float | None:
    return None if n == 0 else k / n


def _groups(task: str, samples: Sequence[Sample]) -> dict[str, dict[str, list[int]]]:
    """分档维度 → 取值 → 样本下标（取值按字符串排序，保证 JSON 键序固定）。"""
    out: dict[str, dict[str, list[int]]] = {}
    for k, s in enumerate(samples):
        for dim, val in groups_of(task, s).items():
            out.setdefault(dim, {}).setdefault(val, []).append(k)
    return {d: dict(sorted(v.items())) for d, v in sorted(out.items())}


def _ovd_gt(samples: Sequence[Sample]) -> dict[str, Any]:
    """一组检测样本（同一数据集）的 COCO 格式 GT：图像只含这些样本，标注取自加载器的 gt（xyxy → xywh）。"""
    images, anns, cats = [], [], {}
    for s in samples:
        w, h = s.fields["image_size"]
        iid = int(s.fields["image_id"])
        images.append({"id": iid, "width": int(w), "height": int(h)})
        for b in s.gt["boxes"]:
            x0, y0, x1, y1 = b["box"]
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": int(b["category_id"]),
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": int(b["iscrowd"])})
        for name, cid in zip(s.fields["categories"], s.fields["category_ids"]):
            cats[int(cid)] = name
    return {"images": images, "annotations": anns, "categories": [{"id": k, "name": v} for k, v in sorted(cats.items())]}


def _ovd_dataset_metrics(samples: Sequence[Sample], recs: Sequence[Mapping[str, Any]], *, map_ci_resamples: int) -> dict:
    preds = []
    for s, r in zip(samples, recs):
        if r["parsed"] is None:
            continue
        for box, cid in zip(r["parsed"]["boxes"], r["parsed"]["category_ids"]):
            preds.append({"image_id": int(s.fields["image_id"]), "category_id": int(cid), "bbox_xyxy": box})
    preds = canonical_order(preds)
    n_pred = sum(r["score"]["n_pred"] for r in recs)
    n_fab = sum(r["score"]["n_fabricated"] for r in recs)
    n_unl = sum(r["score"]["n_unlabeled"] for r in recs)
    out: dict[str, Any] = {"n_images": len(samples), "n_pred_boxes": n_pred, "n_fabricated": n_fab, "n_unlabeled": n_unl,
                           "fabricated_rate": _rate(n_fab, n_pred), "unlabeled_rate": _rate(n_unl, n_pred)}
    gt = _ovd_gt(samples)
    res = detection_map(gt, preds)
    if res is None:
        out.update(F1_50=None, note="GT 没有标注")
        return out
    ids, counts = per_image_counts(res)
    f1 = {"value": res["F1_50"], "ci": None, "n": len(ids)}
    if counts.shape[0] and counts.sum() > 0:
        try:
            ci = bootstrap_ci(counts.astype(np.float64), statistic=f1_statistic)
            f1["ci"] = ci_dict(ci)
        except ValueError:
            pass
    out.update(F1_50=f1, P50=res["P50"], R50=res["R50"], n_tp=res["n_tp"], n_fp=res["n_fp"], n_gt=res["n_gt"])
    for k in ("AP", "AP50", "AP75"):
        out[k] = {"mean": res[k], "min": res[f"{k}_min"], "max": res[f"{k}_max"]}
    out["n_tiebreaks"] = res["n_tiebreaks"]
    out["AP_ci"] = None
    if map_ci_resamples > 0 and preds:
        idx, stat = detection_bootstrap_statistic(gt, preds, "AP")
        try:
            out["AP_ci"] = {**ci_dict(bootstrap_ci(idx, statistic=stat, n_resamples=map_ci_resamples)),
                            "note": "每次重采样只打破一次平局（N5），CI 偏宽"}
        except ValueError:
            out["AP_ci"] = None
    return out


def per_sample_metric(task: str, recs: Sequence[Mapping[str, Any]]):
    """标题指标的逐样本取值（acceptance 的配对比较也用它）：返回 (名字, 逐样本数组, statistic)。检测返回 None（按图另算）。"""
    if task == "refexp_box":
        return "acc@0.5", np.array([float(r["score"]["correct"]) for r in recs]), None
    if task == "refexp_poly":
        return "cIoU", np.array([[r["score"]["inter"], r["score"]["union"]] for r in recs], dtype=np.float64), ratio_of_sums
    if task == "grid16":
        return "gIoU", np.array([r["score"]["iou"] for r in recs]), None
    if task == "ovd":
        return None
    return "accuracy", np.array([float(r["score"]["correct"]) for r in recs]), None


def aggregate(task: str, samples: Sequence[Sample], records: Sequence[Mapping[str, Any]], *, has_gt: bool,
              map_ci_resamples: int = 0) -> dict[str, Any]:
    """predictions.jsonl 的记录（与 samples 同序）→ metrics.json 的内容（不含运行标识，调用方补）。"""
    if [s.id for s in samples] != [r["id"] for r in records]:
        raise ValueError("records 与 samples 的 id 不同序或不一致")
    n = len(records)
    tally = ParseTally(failure_enum(r["failure"]) for r in records)
    n_trunc = sum(1 for r in records if r["truncated"])
    n_unc = sum(1 for r in records if r["thinking_unclosed"])
    m: dict[str, Any] = {
        "n": n,
        "has_gt": has_gt,
        "headline": [],
        "parse": tally.to_dict(),
        "truncation": {"n_truncated": n_trunc, "rate": _rate(n_trunc, n)},
        "thinking_unclosed": {"n": n_unc, "rate": _rate(n_unc, n)},
        "diagnostics": {
            "generated_tokens": distribution(r["n_generated_tokens"] for r in records),
            "visual_tokens": distribution(r["n_visual_tokens"] for r in records),
            "image_pixels": distribution(r["n_image_pixels"] for r in records),
        },
        "breakdown": {},
    }
    if task in COORD_TASKS:
        k = sum(1 for r in records if r.get("coord_le1_decimal"))
        m["diagnostics"]["coord_all_le1_with_decimal"] = {"n": k, "rate": _rate(k, n)}
    groups = _groups(task, samples)

    if task == "ovd":
        by_ds: dict[str, list[int]] = groups.get("dataset", {})
        per_ds = {}
        if has_gt:
            for ds, idx in by_ds.items():
                per_ds[ds] = _ovd_dataset_metrics([samples[i] for i in idx], [records[i] for i in idx],
                                                  map_ci_resamples=map_ci_resamples)
        else:
            for ds, idx in by_ds.items():
                rs = [records[i] for i in idx]
                n_pred = sum(r["score"]["n_pred"] for r in rs)
                n_fab = sum(r["score"]["n_fabricated"] for r in rs)
                per_ds[ds] = {"n_images": len(idx), "n_pred_boxes": n_pred, "n_fabricated": n_fab,
                              "fabricated_rate": _rate(n_fab, n_pred), "note": "没有 GT，只核对格式"}
        m["breakdown"]["dataset"] = per_ds
        if has_gt:
            for ds in ("coco_val2017",):
                if ds in per_ds and per_ds[ds].get("F1_50"):
                    m["headline"].append({"name": "F1@0.5", "scope": ds, **per_ds[ds]["F1_50"]})
            odinw = {k: v for k, v in per_ds.items() if k.startswith("odinw13/")}
            vals = [v["F1_50"]["value"] for v in odinw.values() if v.get("F1_50") and v["F1_50"]["value"] is not None]
            if odinw:
                m["headline"].append({"name": "F1@0.5", "scope": "odinw13_mean_over_datasets",
                                      "value": float(np.mean(vals)) if vals else None, "ci": None,
                                      "n": len(vals), "note": "各数据集 F1 的算术平均（ODinW 惯例），不给 CI；各数据集的 CI 见 breakdown"})
                aps = [v["AP"]["mean"] for v in odinw.values() if v.get("AP") and v["AP"]["mean"] is not None]
                m["odinw13_mean_AP"] = float(np.mean(aps)) if aps else None
        n_pred = sum(r["score"]["n_pred"] for r in records)
        n_fab = sum(r["score"]["n_fabricated"] for r in records)
        m["fabricated"] = {"n_pred_boxes": n_pred, "n_fabricated": n_fab, "rate": _rate(n_fab, n_pred),
                           "n_unlabeled": sum(r["score"]["n_unlabeled"] for r in records)}
        return m

    if not has_gt:
        return m
    name, values, stat = per_sample_metric(task, records)
    if task == "refexp_poly":
        pairs = [(r["score"]["inter"], r["score"]["union"]) for r in records]
        m["headline"].append({"name": "cIoU", "scope": "all", **_ratio_ci(pairs)})
        m["headline"].append({"name": "gIoU", "scope": "all", **_mean_ci([r["score"]["iou"] for r in records])})
    elif task == "grid16":
        pairs = [(r["score"]["inter"], r["score"]["union"]) for r in records]
        m["headline"].append({"name": "gIoU", "scope": "all", **_mean_ci([r["score"]["iou"] for r in records])})
        m["headline"].append({"name": "cIoU", "scope": "all", **_ratio_ci(pairs)})
    else:
        m["headline"].append({"name": name, "scope": "all", **_mean_ci(list(values))})
    if task == "refexp_box":
        m["mean_iou"] = float(np.mean([r["score"]["iou"] for r in records])) if records else None
        assert m["headline"][0]["value"] == box_acc_at_iou([r["score"]["iou"] for r in records])  # 与 metrics.box 同一判据

    for dim, vals in groups.items():
        out = {}
        for val, idx in vals.items():
            rs = [records[i] for i in idx]
            if task in MASK_TASKS:
                out[val] = {"cIoU": _ratio_ci([(r["score"]["inter"], r["score"]["union"]) for r in rs]),
                            "gIoU": _mean_ci([r["score"]["iou"] for r in rs])}
            else:
                out[val] = {name: _mean_ci([float(r["score"]["correct"]) for r in rs])}
            out[val]["parse_failure_rate"] = ParseTally(failure_enum(r["failure"]) for r in rs).failure_rate
        m["breakdown"][dim] = out

    if task in MASK_TASKS:  # 与 metrics.seg 同一口径的核对
        st = [(r["score"]["inter"], r["score"]["union"], r["failure"] is not None) for r in records]
        gi = giou(st)
        assert gi is None or math.isclose(gi, float(np.mean([r["score"]["iou"] for r in records])), rel_tol=0, abs_tol=1e-12)
        ci_ = ciou(st)
        hv = next(h for h in m["headline"] if h["name"] == "cIoU")["value"]
        assert (ci_ is None) == (hv is None) and (ci_ is None or math.isclose(ci_, hv, rel_tol=1e-12))
    if task == "count":
        rules = Counter(r["extras"]["object_rule"] for r in records)
        m["breakdown"]["object_rule"] = {
            k: {"n": rules[k], "accuracy": _rate(sum(1 for r in records if r["extras"]["object_rule"] == k
                                                     and r["score"]["correct"]), rules[k])}
            for k in sorted(rules)}
    if task in MCQ_TASKS:
        routes = Counter((r["parsed"] or {}).get("route") for r in records if r["parsed"])
        m["parse_routes"] = {k: routes[k] for k in sorted(routes)}
    if task == "spatial_cvbench":
        src = m["breakdown"].get("source", {})
        acc = {k: src[k]["accuracy"]["value"] for k in ("ADE20K", "COCO", "Omni3D") if k in src}
        m["cvbench_official_combined"] = (
            0.5 * ((acc["ADE20K"] + acc["COCO"]) / 2 + acc["Omni3D"]) if len(acc) == 3 else None)
        m["cvbench_official_combined_note"] = ("CV-Bench README：1/2 · ((acc_2D_ade + acc_2D_coco)/2 + acc_3D_omni)；"
                                               "三个来源有缺时为 null")
    return m


__all__ = [
    "COORD_TASKS",
    "MCQParse",
    "MCQ_TASKS",
    "UNKNOWN_CATEGORY_ID",
    "aggregate",
    "coord_all_le1_decimal",
    "distribution",
    "failure_enum",
    "gt_mask",
    "parse_mcq",
    "per_sample_metric",
    "score_generation",
]
