"""W1 的评分与汇总（DEV-0051）：指标一律用 E1 的函数（src/e1/scoring、src/metrics/seg、src/metrics/box、src/stats），不另写。

- 逐条（score_episode）：refseg = 提交的 mask 与 GT mask 的 inter / union / iou（metrics.seg.mask_inter_union、sample_iou；失败记
  (0, |GT|)）；refexp_box = 框 IoU 与 IoU ≥ 0.5（metrics.box.box_sample_iou、constants.BOX_ACC_IOU_THRESHOLD；失败记 0）；count = 与 GT
  是否相等（失败记错）。score 的键与 E1 的 predictions.jsonl 相同，所以 E1 的 metric_of 与配对比较能直接读。
- 汇总（aggregate_w1）：只读 predictions.jsonl 的记录（续跑后再算一遍逐字节相同）。标题指标 = E1 的 metric_of（refseg：cIoU 与 gIoU；
  refexp_box：acc@0.5；count：accuracy），带 bootstrap 95% CI；另报失败原因、结束方式、轮数与工具调用的分布、各工具的调用次数、
  思考段的结束方式、token 数分布。refseg 另与 metrics.seg.ciou / giou 核对同一口径。
- 配对（paired_ciou）：两种设置在同一批 id 上的 cIoU 差（a − b）与配对 bootstrap 95% CI（stats.paired_bootstrap_ci，
  statistic = ratio_of_sums）。id 集合不同就报错；顺序按 a。scripts/w1_gate.py 用它算 D103 §三的条件 1。
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np

from src.constants import BOX_ACC_IOU_THRESHOLD
from src.data.records import Sample
from src.e1.scoring import ci_dict, distribution, gt_mask, metric_of
from src.metrics.box import box_sample_iou
from src.metrics.seg import ciou, giou, mask_inter_union, sample_iou
from src.stats import paired_bootstrap_ci, ratio_of_sums

HEADLINE: Mapping[str, tuple[str, ...]] = {"refseg": ("cIoU", "gIoU"), "refexp_box": ("acc@0.5",), "count": ("accuracy",)}


def score_episode(task: str, sample: Sample, rec: Any) -> dict[str, Any]:
    """一条 EpisodeRecord → score（与 E1 同形）。回路的 failure 不为 None 时按失败记分。"""
    ok = rec.failure is None
    if task == "refseg":
        g = gt_mask(sample)
        w, h = rec.image_size
        if g.shape != (h, w):
            raise ValueError(f"{sample.id}：GT mask 形状 {g.shape} 与原图 (H, W) = {(h, w)} 不符")
        if ok and rec.mask is None:
            raise RuntimeError(f"{sample.id}：没有失败却没有 mask")
        st = mask_inter_union(rec.mask if ok else None, g)
        return {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}
    if task == "refexp_box":
        iou = box_sample_iou(rec.box if ok else None, sample.gt["box"])
        return {"iou": iou, "correct": iou >= BOX_ACC_IOU_THRESHOLD}
    if task == "count":
        gt = int(sample.gt["count"])
        return {"gt": gt, "correct": ok and rec.count == gt}
    raise ValueError(f"未知任务 {task!r}")


def headline(task: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": name, "scope": "all", **metric_of(name, records)} for name in HEADLINE[task]]


def _counts(values) -> dict[str, int]:
    c = Counter("ok" if v is None else str(v) for v in values)
    return {k: c[k] for k in sorted(c)}


def aggregate_w1(task: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """predictions.jsonl 的回路记录 → metrics.json 的主体（不含运行标识）。"""
    n = len(records)
    m: dict[str, Any] = {"n": n, "headline": headline(task, records) if records else []}
    m["failures"] = _counts(r["failure"] for r in records)
    m["n_failed"] = sum(1 for r in records if r["failure"] is not None)
    m["ended_by"] = _counts(r["ended_by"] for r in records)
    m["rounds"] = distribution(r["n_rounds"] for r in records)
    m["tool_calls"] = {"counted": distribution(r["n_tool_calls"] for r in records),
                       "executed": distribution(r["n_executed"] for r in records),
                       "n_refused": sum(r["n_refused"] for r in records),
                       "n_tool_call_errors": sum(r["n_tool_call_errors"] for r in records)}
    names: Counter[str] = Counter()
    refused: Counter[str] = Counter()
    errors = 0
    for r in records:
        for rd in r["rounds"]:
            for c in rd["tool_calls"]:
                if c["executed"]:
                    names[c["name"]] += 1
                    errors += int(bool(c["result"]["error"]))
                else:
                    refused[c["refused"]] += 1
    m["tool_usage"] = {"executed_by_name": {k: names[k] for k in sorted(names)},
                       "refused_by_reason": {k: refused[k] for k in sorted(refused)}, "n_tool_errors": errors}
    m["thinking_end"] = _counts(te for r in records for te in r["thinking_end"])
    for k in ("prompt", "generated", "inserted", "visual"):
        m.setdefault("tokens", {})[k] = distribution(r["tokens"][k] for r in records)
    m["answer_source"] = _counts((r["answer"] or {}).get("source") for r in records)
    if task == "refseg" and records:  # 与 metrics.seg 同一口径的核对（同 E1 aggregate）
        st = [(r["score"]["inter"], r["score"]["union"], r["failure"] is not None) for r in records]
        gi = giou(st)
        assert gi is None or math.isclose(gi, float(np.mean([r["score"]["iou"] for r in records])), rel_tol=0, abs_tol=1e-12)
        ci_ = ciou(st)
        hv = m["headline"][0]["value"]
        assert (ci_ is None) == (hv is None) and (ci_ is None or math.isclose(ci_, hv, rel_tol=1e-12))
    if task == "refexp_box" and records:
        m["mean_iou"] = float(np.mean([r["score"]["iou"] for r in records]))
    return m


def _by_id(recs: Sequence[Mapping[str, Any]], what: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for r in recs:
        if r["id"] in out:
            raise ValueError(f"{what}：id {r['id']} 重复")
        out[r["id"]] = r
    return out


def aligned_pairs(recs_a: Sequence[Mapping[str, Any]], recs_b: Sequence[Mapping[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """按 id 对齐两组记录（顺序按 a），返回 (ids, a 的 (n, 2) 交并, b 的 (n, 2) 交并)。id 集合不同就报错。"""
    a, b = _by_id(recs_a, "a"), _by_id(recs_b, "b")
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise ValueError(f"两组记录的 id 集合不同：只在 a 里 {len(only_a)} 条（如 {only_a[:3]}），只在 b 里 {len(only_b)} 条（如 {only_b[:3]}）")
    ids = [r["id"] for r in recs_a]
    pa = np.array([[a[i]["score"]["inter"], a[i]["score"]["union"]] for i in ids], dtype=np.float64)
    pb = np.array([[b[i]["score"]["inter"], b[i]["score"]["union"]] for i in ids], dtype=np.float64)
    return ids, pa, pb


def exact_ciou(recs: Sequence[Mapping[str, Any]]) -> Fraction:
    """cIoU 的精确值 Σinter / Σunion（整数相除，没有浮点误差）。"""
    inter = sum(int(r["score"]["inter"]) for r in recs)
    union = sum(int(r["score"]["union"]) for r in recs)
    if union == 0:
        raise ValueError("累计并为 0，cIoU 无定义")
    return Fraction(inter, union)


def paired_ciou(recs_a: Sequence[Mapping[str, Any]], recs_b: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """两种设置在同一批 id 上的 cIoU 差 a − b 与配对 bootstrap 95% CI（ratio_of_sums；重采样数、水平与种子取 src/constants）。"""
    ids, pa, pb = aligned_pairs(recs_a, recs_b)
    ci = paired_bootstrap_ci(pa, pb, statistic=ratio_of_sums)
    if ci is None:
        return {"n": 0, "a": None, "b": None, "diff": None, "ci": None}
    return {"n": len(ids), "a": ratio_of_sums(pa), "b": ratio_of_sums(pb), "diff": ci.estimate, "ci": ci_dict(ci)}


__all__ = [
    "HEADLINE",
    "aggregate_w1",
    "aligned_pairs",
    "exact_ciou",
    "headline",
    "paired_ciou",
    "score_episode",
]
