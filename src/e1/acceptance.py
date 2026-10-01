"""批处理验收（D28-5；DEV-0014 任务书 §3）：开发子集上，同一模型与任务，逐条（batch = 1，走 generate）跑一遍、批量跑一遍，
比较两遍的 predictions.jsonl。

报告（每个任务一行）：
- 解析后答案的一致率（判据见 answers_agree），必须 ≥ AGREEMENT_MIN；
- 两遍标题指标之差，以及按样本配对的 bootstrap CI（stats.paired_bootstrap_ci，与指标同一个统计量）。判据：配对差的 95% CI 含 0
  （"指标差落在 CI 内"的实现解释，见完成总结）；另报批量那遍的指标是否落在逐条那遍自身的 bootstrap CI 内，供参考；
- raw_text 逐字相同的比例。
检测的开发子集没有 GT（src/e1/tasks），只比一致率与 raw_text，指标差记 null 并注明。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.data.records import Sample
from src.e1.scoring import per_sample_metric
from src.executors.raster import polygons_to_mask
from src.grid.codec import grid_to_mask
from src.stats import bootstrap_ci, paired_bootstrap_ci

AGREEMENT_MIN = 0.98  # D28-5
AGREE_MASK_IOU = 0.9  # 多边形与网格：两遍解析出的 mask（原图尺寸）IoU ≥ 此值算一致


def _mask(task: str, rec: Mapping[str, Any]) -> np.ndarray:
    w, h = rec["image_size"]
    if task == "refexp_poly":
        return polygons_to_mask(rec["parsed"]["polygons"], h, w)
    grid = np.array([[int(c) for c in row] for row in rec["parsed"]["grid"]], dtype=np.uint8)
    return grid_to_mask(grid, h, w)


def answers_agree(task: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """两遍对同一条样本的解析结果是否一致（规则写进完成总结与预注册）：
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
        ma, mb = _mask(task, a), _mask(task, b)
        union = int(np.count_nonzero(ma | mb))
        iou = 1.0 if union == 0 else int(np.count_nonzero(ma & mb)) / union
        return iou >= AGREE_MASK_IOU
    if task == "count":
        return pa["count"] == pb["count"]
    return pa["letter"] == pb["letter"]


def _ci(ci) -> dict[str, Any] | None:
    return None if ci is None else {"estimate": ci.estimate, "low": ci.low, "high": ci.high, "level": ci.level,
                                    "n_resamples": ci.n_resamples}


def compare(task: str, samples: Sequence[Sample], single: Sequence[Mapping[str, Any]],
            batched: Sequence[Mapping[str, Any]], *, has_gt: bool, batch_size: int) -> dict[str, Any]:
    """一个任务的验收结果（字典，键序固定）。"""
    if [r["id"] for r in single] != [r["id"] for r in batched] or [r["id"] for r in single] != [s.id for s in samples]:
        raise ValueError("两遍的样本不同或不同序")
    n = len(single)
    agree = [answers_agree(task, a, b) for a, b in zip(single, batched)]
    same_raw = [a["raw_text"] == b["raw_text"] for a, b in zip(single, batched)]
    same_fail = sum(1 for a, b in zip(single, batched) if a["failure"] == b["failure"])
    out: dict[str, Any] = {
        "task": task, "n": n, "batch_size": batch_size,
        "answer_agreement": None if n == 0 else sum(agree) / n, "n_disagree": n - sum(agree),
        "disagree_ids": [r["id"] for r, ok in zip(single, agree) if not ok][:50],
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
    checks = [out["answer_agreement"] is not None and out["answer_agreement"] >= AGREEMENT_MIN]
    if out["diff_ci_contains_0"] is not None:
        checks.append(out["diff_ci_contains_0"])
    out["pass"] = all(checks)
    return out


def _f(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def summary_markdown(model_key: str, rows: Sequence[Mapping[str, Any]], *, header: Mapping[str, Any]) -> str:
    lines = [f"# 批处理验收 · {model_key}", "",
             f"- 判据（D28-5）：解析后答案的一致率 ≥ {AGREEMENT_MIN}，且两遍指标的配对差 95% CI 含 0；多边形与网格按 mask IoU ≥ "
             f"{AGREE_MASK_IOU} 判一致。", f"- {dict(header)}", "",
             "| 任务 | n | 批大小 | 答案一致率 | raw_text 逐字相同 | 指标 | 逐条 | 批量 | 差 | 配对差 95% CI | CI 含 0 | 批量落在逐条 CI 内 | 通过 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = r["diff_ci"]
        ci_s = f"[{_f(ci['low'])}, {_f(ci['high'])}]" if ci else "—"
        lines.append(f"| {r['task']} | {r['n']} | {r['batch_size']} | {_f(r['answer_agreement'])} | {_f(r['raw_text_equal_rate'])} | "
                     f"{r['metric'] or '—'} | {_f(r['metric_single'])} | {_f(r['metric_batched'])} | {_f(r['diff'])} | {ci_s} | "
                     f"{_f(r['diff_ci_contains_0'])} | {_f(r['batched_within_single_ci'])} | {_f(r['pass'])} |")
    notes = [f"- {r['task']}：{r['metric_note']}" for r in rows if r.get("metric_note")]
    if notes:
        lines += ["", "注："] + notes
    return "\n".join(lines) + "\n"


__all__ = ["AGREEMENT_MIN", "AGREE_MASK_IOU", "answers_agree", "compare", "summary_markdown"]
