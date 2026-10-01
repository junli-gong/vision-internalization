"""批处理验收（D28-5；DEV-0014 任务书 §3，DEV-0015 §2）：开发子集上，同一模型与任务，逐条（batch = 1，走 generate）跑一遍、
批量跑一遍，比较两遍的 predictions.jsonl。

报告（每个任务一行）：
- 解析后答案的一致率，三种口径都算（N15；规则见各 _agree_* 函数与 AGREEMENT_RULES）：
  (a) bitwise 逐位（D32 现行）；(b) iou90 两个预测的 IoU ≥ 0.9；(c) outcome 按判定结果（指标层面的对错 / TP、FP 是否相同）。
  每种口径各自判通过：一致率 ≥ AGREEMENT_MIN，且配对差 95% CI 含 0（有指标时）。all_pass 用 --agreement 选定的口径
  （默认 DEFAULT_AGREEMENT = bitwise，项目负责人答复 N15 后再改）；三种口径的 all_pass 也都写出。
- 两遍标题指标之差，以及按样本配对的 bootstrap CI（stats.paired_bootstrap_ci，与指标同一个统计量）。判据：配对差的 95% CI 含 0
  （D32-2）；另报批量那遍的指标是否落在逐条那遍自身的 bootstrap CI 内，供参考；
- raw_text 逐字相同的比例。
检测的开发子集没有 GT（src/e1/tasks）：指标差记 null；(c) 口径要 GT，记 null（该口径下这一行不判，列进 not_evaluated）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.constants import BOX_ACC_IOU_THRESHOLD, DETECTION_OPERATING_IOU
from src.data.records import Sample
from src.e1.scoring import ovd_gt, per_sample_metric
from src.executors.geometry import box_iou
from src.executors.raster import polygons_to_mask
from src.grid.codec import grid_to_mask
from src.metrics.detection import operating_point
from src.stats import bootstrap_ci, paired_bootstrap_ci

AGREEMENT_MIN = 0.98  # D28-5
AGREE_MASK_IOU = 0.9  # 多边形与网格：两遍解析出的 mask（原图尺寸）IoU ≥ 此值算一致
AGREE_BOX_IOU = 0.9  # (b) 口径：两遍的框（检测按输出顺序逐框配对）IoU ≥ 此值算一致（N15 (b)：与多边形、网格同一个数）
AGREEMENTS: tuple[str, ...] = ("bitwise", "iou90", "outcome")
DEFAULT_AGREEMENT = "bitwise"  # D32；项目负责人答复 N15 后再改
AGREEMENT_LABELS = {"bitwise": "(a) 逐位", "iou90": f"(b) IoU ≥ {AGREE_BOX_IOU}", "outcome": "(c) 按判定结果"}
AGREEMENT_RULES = {
    "bitwise": "两遍都解析失败 → 一致，一遍失败一遍成功 → 不一致；定位：框逐位相等；检测：(标签, 框) 列表逐位相等（顺序也相同）；"
               f"计数、点对、多选：值相等；多边形、网格：mask IoU ≥ {AGREE_MASK_IOU}（D32-2）",
    "iou90": f"两遍都解析失败 → 一致，一遍失败一遍成功 → 不一致；定位：两框 IoU ≥ {AGREE_BOX_IOU}；检测：框数与标签序列相同，"
             f"按输出顺序逐框配对、每对 IoU ≥ {AGREE_BOX_IOU}；计数、点对、多选：值相等；多边形、网格：mask IoU ≥ {AGREE_MASK_IOU}",
    "outcome": f"定位：IoU {BOX_ACC_IOU_THRESHOLD} 下两遍的对错相同（解析失败记错，所以两遍都失败、或一遍失败一遍成功但错，都算一致）；"
               f"检测：每张图在 IoU {DETECTION_OPERATING_IOU} 下的 TP 与 FP 计数相同（要 GT；没有 GT 记 null）；"
               f"计数、点对、多选：值相等（两遍都失败 → 一致，一遍失败一遍成功 → 不一致）；多边形、网格：mask IoU ≥ {AGREE_MASK_IOU}",
}
OUTCOME_NO_GT_NOTE = ("检测的开发子集（RefCOCO 开发子集的图）没有 GT：落地的 refcoco_unc.json 只含被指代的标注，不是这些图的全部 "
                      "COCO 标注（src/data/land/refcoco.build_eval_json），算不出每张图的 TP/FP")


def _mask(task: str, rec: Mapping[str, Any]) -> np.ndarray:
    w, h = rec["image_size"]
    if task == "refexp_poly":
        return polygons_to_mask(rec["parsed"]["polygons"], h, w)
    grid = np.array([[int(c) for c in row] for row in rec["parsed"]["grid"]], dtype=np.uint8)
    return grid_to_mask(grid, h, w)


def _mask_agree(task: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    ma, mb = _mask(task, a), _mask(task, b)
    union = int(np.count_nonzero(ma | mb))
    iou = 1.0 if union == 0 else int(np.count_nonzero(ma & mb)) / union
    return iou >= AGREE_MASK_IOU


def _value_agree(task: str, pa: Mapping[str, Any], pb: Mapping[str, Any]) -> bool:
    if task == "count":
        return pa["count"] == pb["count"]
    return pa["letter"] == pb["letter"]


def answers_agree(task: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """(a) 逐位（D32-2）：两遍对同一条样本的解析结果是否一致。
    - 两遍都解析失败（任何原因）→ 一致；一遍失败一遍成功 → 不一致；
    - 定位：解析出的框逐位相等；检测：(标签, 框) 的列表逐位相等（顺序也要相同）；
    - 计数、点对、多选：解析出的值相等；
    - 多边形、网格：两个预测按原图尺寸光栅化，mask IoU ≥ AGREE_MASK_IOU（两个都是空 mask 记 IoU = 1）。
    """
    fa, fb = a["failure"] is not None, b["failure"] is not None
    if fa or fb:
        return fa and fb
    pa, pb = a["parsed"], b["parsed"]
    if task == "refexp_box":
        return pa["box"] == pb["box"]
    if task == "ovd":
        return list(zip(pa["labels"], pa["boxes"])) == list(zip(pb["labels"], pb["boxes"]))
    if task in ("refexp_poly", "grid16"):
        return _mask_agree(task, a, b)
    return _value_agree(task, pa, pb)


def _pair_iou(x: Sequence[float], y: Sequence[float]) -> float:
    """两个预测框的 IoU；逐位相同记 1（含两个面积为 0 的相同框），并集为 0 而不相同记 0。"""
    if list(x) == list(y):
        return 1.0
    try:
        return box_iou(x, y)
    except ValueError:
        return 0.0


def answers_agree_iou(task: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """(b) IoU ≥ AGREE_BOX_IOU：框也按 IoU 判，其余同 (a)。检测按输出顺序逐框配对（框数与标签序列要相同）。"""
    fa, fb = a["failure"] is not None, b["failure"] is not None
    if fa or fb:
        return fa and fb
    pa, pb = a["parsed"], b["parsed"]
    if task == "refexp_box":
        return _pair_iou(pa["box"], pb["box"]) >= AGREE_BOX_IOU
    if task == "ovd":
        if list(pa["labels"]) != list(pb["labels"]) or len(pa["boxes"]) != len(pb["boxes"]):
            return False
        return all(_pair_iou(x, y) >= AGREE_BOX_IOU for x, y in zip(pa["boxes"], pb["boxes"]))
    return answers_agree(task, a, b)


def _ovd_tp_fp(samples: Sequence[Sample], recs: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """每张图在 IoU = DETECTION_OPERATING_IOU 下的 (TP, FP)，与样本同序。按数据集（source）分组建 GT（与 metrics 同一份
    scoring.ovd_gt），operating_point 与标题 F1@0.5 同一个函数；解析失败的图没有预测（TP = FP = 0）。"""
    out: dict[str, tuple[int, int]] = {}
    by_src: dict[str, list[int]] = {}
    for k, s in enumerate(samples):
        by_src.setdefault(s.source, []).append(k)
    for idx in by_src.values():
        ss, rs = [samples[k] for k in idx], [recs[k] for k in idx]
        gt = ovd_gt(ss)
        preds, xywh = [], []
        for s, r in zip(ss, rs):
            if r["parsed"] is None:
                continue
            for box, cid in zip(r["parsed"]["boxes"], r["parsed"]["category_ids"]):
                x0, y0, x1, y1 = box
                preds.append({"image_id": int(s.fields["image_id"]), "category_id": int(cid), "bbox_xyxy": box})
                xywh.append([x0, y0, x1 - x0, y1 - y0])
        per = {row[0]: (int(row[1]), int(row[2])) for row in operating_point(gt, preds, xywh, DETECTION_OPERATING_IOU)["per_image"]}
        for s in ss:
            out[s.id] = per[int(s.fields["image_id"])]
    return [out[s.id] for s in samples]


def agreement_flags(task: str, mode: str, samples: Sequence[Sample], single: Sequence[Mapping[str, Any]],
                    batched: Sequence[Mapping[str, Any]], *, has_gt: bool) -> list[bool] | None:
    """一种口径下逐条是否一致；该口径算不出（(c) 的检测、定位没有 GT）时返回 None。"""
    if mode not in AGREEMENTS:
        raise ValueError(f"agreement 必须是 {AGREEMENTS} 之一，收到 {mode!r}")
    pairs = list(zip(single, batched))
    if mode == "bitwise":
        return [answers_agree(task, a, b) for a, b in pairs]
    if mode == "iou90":
        return [answers_agree_iou(task, a, b) for a, b in pairs]
    if task == "refexp_box":
        if not has_gt:
            return None
        return [bool(a["score"]["correct"]) == bool(b["score"]["correct"]) for a, b in pairs]
    if task == "ovd":
        if not has_gt:
            return None
        return [x == y for x, y in zip(_ovd_tp_fp(samples, single), _ovd_tp_fp(samples, batched))]
    return [answers_agree(task, a, b) for a, b in pairs]


def _ci(ci) -> dict[str, Any] | None:
    return None if ci is None else {"estimate": ci.estimate, "low": ci.low, "high": ci.high, "level": ci.level,
                                    "n_resamples": ci.n_resamples}


def compare(task: str, samples: Sequence[Sample], single: Sequence[Mapping[str, Any]],
            batched: Sequence[Mapping[str, Any]], *, has_gt: bool, batch_size: int,
            agreement: str = DEFAULT_AGREEMENT) -> dict[str, Any]:
    """一个任务的验收结果（字典，键序固定）。answer_agreement、n_disagree、disagree_ids、pass 取 agreement 选定的口径；
    三种口径的明细在 agreement_by_mode。"""
    if agreement not in AGREEMENTS:
        raise ValueError(f"agreement 必须是 {AGREEMENTS} 之一，收到 {agreement!r}")
    if [r["id"] for r in single] != [r["id"] for r in batched] or [r["id"] for r in single] != [s.id for s in samples]:
        raise ValueError("两遍的样本不同或不同序")
    n = len(single)
    same_raw = [a["raw_text"] == b["raw_text"] for a, b in zip(single, batched)]
    same_fail = sum(1 for a, b in zip(single, batched) if a["failure"] == b["failure"])
    out: dict[str, Any] = {
        "task": task, "n": n, "batch_size": batch_size, "agreement": agreement,
        "answer_agreement": None, "n_disagree": None, "disagree_ids": None,
        "raw_text_equal_rate": None if n == 0 else sum(same_raw) / n,
        "failure_reason_equal_rate": None if n == 0 else same_fail / n,
        "metric": None, "metric_single": None, "metric_batched": None, "diff": None, "diff_ci": None,
        "diff_ci_contains_0": None, "batched_within_single_ci": None, "single_ci": None, "metric_note": None,
    }
    spec = per_sample_metric(task, single) if (has_gt and n) else None
    if spec is None:
        out["metric_note"] = "没有 GT（检测的开发子集）或空集合：只比一致率与 raw_text" if n else "空集合"
    else:
        name, a, stat = spec
        _, b, _ = per_sample_metric(task, batched)
        f = stat if stat is not None else (lambda x: float(np.mean(x)))
        out["metric"] = name
        out["metric_single"], out["metric_batched"] = float(f(a)), float(f(b))
        out["diff"] = out["metric_single"] - out["metric_batched"]
        try:
            dci = paired_bootstrap_ci(a, b, statistic=stat)
            sci = bootstrap_ci(a, statistic=stat)
        except ValueError as e:  # 统计量在某次重采样上无定义（如 cIoU 的累计并为 0）
            out["metric_note"] = f"CI 无定义：{e}"
        else:
            out["diff_ci"], out["single_ci"] = _ci(dci), _ci(sci)
            out["diff_ci_contains_0"] = dci.contains(0.0)
            out["batched_within_single_ci"] = sci.contains(out["metric_batched"])
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in AGREEMENTS:
        flags = agreement_flags(task, mode, samples, single, batched, has_gt=has_gt)
        if flags is None:
            by_mode[mode] = {"rate": None, "n_disagree": None, "disagree_ids": None, "pass": None,
                             "note": OUTCOME_NO_GT_NOTE if task == "ovd" else "没有 GT，算不出对错"}
            continue
        rate = None if n == 0 else sum(flags) / n
        checks = [rate is not None and rate >= AGREEMENT_MIN]
        if out["diff_ci_contains_0"] is not None:
            checks.append(out["diff_ci_contains_0"])
        by_mode[mode] = {"rate": rate, "n_disagree": n - sum(flags),
                         "disagree_ids": [r["id"] for r, ok in zip(single, flags) if not ok][:50], "pass": all(checks),
                         "note": None}
    sel = by_mode[agreement]
    out.update(answer_agreement=sel["rate"], n_disagree=sel["n_disagree"], disagree_ids=sel["disagree_ids"])
    out["agreement_by_mode"] = by_mode
    out["pass"] = sel["pass"]
    return out


def overall(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    """一种口径下的总判定：判了的行全部通过才算通过；这一口径下不判（pass 为 null）的任务列进 not_evaluated。
    一行都没判时 all_pass 为 null。"""
    judged = [r for r in rows if r["agreement_by_mode"][mode]["pass"] is not None]
    return {"all_pass": all(r["agreement_by_mode"][mode]["pass"] for r in judged) if judged else None,
            "n_judged": len(judged), "not_evaluated": [r["task"] for r in rows if r not in judged]}


def _f(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def summary_markdown(model_key: str, rows: Sequence[Mapping[str, Any]], *, header: Mapping[str, Any],
                     agreement: str = DEFAULT_AGREEMENT) -> str:
    lines = [f"# 批处理验收 · {model_key}", "",
             f"- 判据（D28-5）：解析后答案的一致率 ≥ {AGREEMENT_MIN}，且两遍指标的配对差 95% CI 含 0。一致率按三种口径各算一遍（N15）；"
             f"本次的 all_pass 用 **{AGREEMENT_LABELS[agreement]}**（--agreement {agreement}）。"]
    for mode in AGREEMENTS:
        lines.append(f"- {AGREEMENT_LABELS[mode]}（{mode}）：{AGREEMENT_RULES[mode]}")
    lines += [f"- {dict(header)}", ""]
    head = ["任务", "n", "批大小"]
    for mode in AGREEMENTS:
        head += [f"{AGREEMENT_LABELS[mode]} 一致率", f"{AGREEMENT_LABELS[mode]} 通过"]
    head += ["raw_text 逐字相同", "指标", "逐条", "批量", "差", "配对差 95% CI", "CI 含 0", "批量落在逐条 CI 内"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        ci = r["diff_ci"]
        ci_s = f"[{_f(ci['low'])}, {_f(ci['high'])}]" if ci else "—"
        cells = [r["task"], str(r["n"]), str(r["batch_size"])]
        for mode in AGREEMENTS:
            m = r["agreement_by_mode"][mode]
            cells += [_f(m["rate"]), _f(m["pass"])]
        cells += [_f(r["raw_text_equal_rate"]), r["metric"] or "—", _f(r["metric_single"]), _f(r["metric_batched"]),
                  _f(r["diff"]), ci_s, _f(r["diff_ci_contains_0"]), _f(r["batched_within_single_ci"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| 口径 | all_pass | 判了的任务数 | 不判的任务 |", "|---|---|---|---|"]
    for mode in AGREEMENTS:
        o = overall(rows, mode)
        mark = "（本次采用）" if mode == agreement else ""
        lines.append(f"| {AGREEMENT_LABELS[mode]}{mark} | {_f(o['all_pass'])} | {o['n_judged']} | "
                     f"{'、'.join(o['not_evaluated']) or '—'} |")
    notes = [f"- {r['task']}：{r['metric_note']}" for r in rows if r.get("metric_note")]
    notes += [f"- {r['task']}（{AGREEMENT_LABELS[mode]}）：{r['agreement_by_mode'][mode]['note']}"
              for r in rows for mode in AGREEMENTS if r["agreement_by_mode"][mode]["note"]]
    if notes:
        lines += ["", "注："] + notes
    return "\n".join(lines) + "\n"


__all__ = [
    "AGREEMENTS",
    "AGREEMENT_LABELS",
    "AGREEMENT_MIN",
    "AGREEMENT_RULES",
    "AGREE_BOX_IOU",
    "AGREE_MASK_IOU",
    "DEFAULT_AGREEMENT",
    "agreement_flags",
    "answers_agree",
    "answers_agree_iou",
    "compare",
    "overall",
    "summary_markdown",
]
