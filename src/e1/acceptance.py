"""批处理验收。

v3（DEV-0017 §3；D36-1；协议写在 configs/e1_prompts.json 的 acceptance 与 batch_protocol；scripts/run_e1.py --acceptance）：
开发子集上，A1、A2 = 冻结的批配置跑两遍，B = 换一种批组成（composition_b）跑一遍。
- (i) rerun_identical：每个任务 A1 与 A2 的 predictions.jsonl 逐字节相同（predictions.jsonl 不含墙钟）。
- (ii) noise_floor：任务按输出类型合成组（acceptance.groups），组内样本在 A1 与 B 之间逐条配对，统计量（池内 0/1 均值、cIoU、
  gIoU、F1@0.5）之差的配对 bootstrap 95% CI 落在 [−δ, +δ] 以内。这个差与 CI 就是数值噪声底。
- 通过 = 全部任务过 (i) 且能判的组全部过 (ii)；不再以 batch-1 为参照。下面 v2 的三种一致口径在 A1 与 B 之间照算，只作参考。

v2（D28-5；DEV-0014 任务书 §3，DEV-0015 §2；保留作 v3 的参考列）：开发子集上，同一模型与任务，逐条（batch = 1，走 generate）
跑一遍、批量跑一遍，比较两遍的 predictions.jsonl。

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

import hashlib
from pathlib import Path

from src.constants import BOX_ACC_IOU_THRESHOLD, DETECTION_OPERATING_IOU
from src.data.records import Sample
from src.data.subsets import seeded_sample
from src.e1.scoring import ovd_per_image_counts, per_sample_metric
from src.executors.geometry import box_iou
from src.executors.raster import polygons_to_mask
from src.grid.codec import grid_to_mask
from src.stats import bootstrap_ci, f1_statistic, paired_bootstrap_ci, ratio_of_sums

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
OUTCOME_NO_GT_NOTE = ("检测的开发核对（RefCOCO 开发子集的图）没有 GT：这些图在 instances_train2014 里的全部标注"
                      "（coco_train2014_e1dev，DEV-0017）没有落地，落地的 refcoco 文件只含被指代的标注，算不出每张图的 TP/FP")


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
    """每张图在 IoU = DETECTION_OPERATING_IOU 下的 (TP, FP)，与样本同序（scoring.ovd_per_image_counts 去掉 n_GT）。"""
    return [(tp, fp) for tp, fp, _ in ovd_per_image_counts(samples, recs)]


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
        if not n:
            out["metric_note"] = "空集合"
        elif task == "ovd" and has_gt:
            out["metric_note"] = "检测的指标差按图另算（F1@0.5：验收 v3 的 reference_columns 与 detection 组）"
        else:
            out["metric_note"] = "没有 GT（检测的开发子集）或空集合：只比一致率与 raw_text"
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


# ---------------------------------------------------------------------------
# v3（DEV-0017；D36-1）：冻结批配置重跑逐位相同 + 两种批组成之间 ±δ（数值噪声底）
# ---------------------------------------------------------------------------

PASSES_V3: tuple[str, ...] = ("A1", "A2", "B")  # A1、A2：冻结的批配置；B：composition_b


def group_of(task: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """任务所属的组（acceptance.groups；配置校验保证每个任务恰好在一组里）。"""
    for g in protocol["groups"]:
        if task in g["tasks"]:
            return dict(g)
    raise KeyError(f"{task} 不在任何验收组里")


def take_samples(task: str, samples: Sequence[Sample], protocol: Mapping[str, Any]) -> tuple[list[Sample], dict[str, Any]]:
    """按组的 take 取样本（acceptance.take_rule）：take 是整数且样本多于它时，按 sha256(salt|<purpose>:<组名>|id) 升序取前
    take 个（src/data/subsets.seeded_sample），保持原样本顺序；否则全取。返回 (样本, 记录)。"""
    g = group_of(task, protocol)
    st = {"group": g["name"], "take": g["take"], "n_before": len(samples)}
    if g["take"] == "all" or len(samples) <= g["take"]:
        st["n"] = len(samples)
        return list(samples), st
    purpose = f"{protocol['take_salt_purpose']}:{g['name']}"
    chosen = set(seeded_sample([s.id for s in samples], g["take"], purpose))
    out = [s for s in samples if s.id in chosen]
    st.update(n=len(out), purpose=purpose)
    return out, st


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def rerun_identical(task: str, a1: Sequence[Mapping[str, Any]], a2: Sequence[Mapping[str, Any]], *,
                    sha_a1: str | None = None, sha_a2: str | None = None) -> dict[str, Any]:
    """(i)：同一批配置的两遍逐位相同。逐条比整条记录（predictions.jsonl 的一行，不含墙钟）；给了两个文件的 sha256 时
    还要求文件逐字节相同。"""
    if [r["id"] for r in a1] != [r["id"] for r in a2]:
        raise ValueError(f"{task}：A1 与 A2 的样本不同或不同序")
    diff = [x["id"] for x, y in zip(a1, a2) if x != y]
    same_file = None if sha_a1 is None or sha_a2 is None else sha_a1 == sha_a2
    identical = not diff and same_file is not False
    return {"task": task, "n": len(a1), "n_diff": len(diff), "diff_ids": diff[:50],
            "raw_text_diff": sum(1 for x, y in zip(a1, a2) if x["raw_text"] != y["raw_text"]),
            "predictions_sha256": {"A1": sha_a1, "A2": sha_a2}, "file_identical": same_file, "identical": identical}


def group_values(group: Mapping[str, Any], parts: Sequence[tuple[str, Sequence[Sample], Sequence[Mapping[str, Any]]]]):
    """一组的逐样本值与统计量：parts = [(任务, 样本, 记录)]（组内各任务按组定义的顺序拼接）。
    accuracy_pooled：各任务逐条 0/1 拼成一列，均值；cIoU：(交, 并)，ratio_of_sums；gIoU：逐条 IoU，均值；
    F1@0.5：逐图 (TP, FP, n_GT)，f1_statistic。返回 (逐样本数组, statistic；None = 均值)。"""
    metric = group["metric"]
    if metric == "accuracy_pooled":
        arrs = [per_sample_metric(t, rs)[1] for t, _, rs in parts]
        return (np.concatenate(arrs) if arrs else np.zeros(0)), None
    if len(parts) != 1:
        raise ValueError(f"组 {group['name']} 的统计量 {metric} 只用于单个任务")
    task, ss, rs = parts[0]
    if metric == "cIoU":
        return per_sample_metric(task, rs)[1], ratio_of_sums
    if metric == "gIoU":
        return per_sample_metric(task, rs)[1], None
    if metric == "F1@0.5":
        return np.asarray(ovd_per_image_counts(ss, rs), dtype=np.float64).reshape(-1, 3), f1_statistic
    raise ValueError(f"未知的组统计量 {metric!r}")


def noise_floor(group: Mapping[str, Any], delta: float, parts_a: Sequence[tuple[str, Sequence[Sample], Sequence[Mapping]]],
                parts_b: Sequence[tuple[str, Sequence[Sample], Sequence[Mapping]]]) -> dict[str, Any]:
    """(ii)：一组在 A1 与 B 之间的指标差（A1 − B）与配对 bootstrap 95% CI；CI ⊂ [−δ, +δ] 即通过。
    样本在两遍之间按 id 对齐（同一份选择，只有批组成不同）。空组或统计量无定义时 pass 为 null，写明原因。"""
    for (ta, sa, ra), (tb, sb, rb) in zip(parts_a, parts_b):
        if ta != tb or [r["id"] for r in ra] != [r["id"] for r in rb] or [s.id for s in sa] != [r["id"] for r in ra]:
            raise ValueError(f"组 {group['name']}：A1 与 B 的样本不同或不同序（{ta}）")
    n = sum(len(ra) for _, _, ra in parts_a)
    n_min = group.get("n_min")
    out: dict[str, Any] = {"group": group["name"], "tasks": [t for t, _, _ in parts_a], "metric": group["metric"],
                           "metric_class": group["metric_class"], "delta": delta, "n": n, "n_min": n_min,
                           "n_below_min": None if n_min is None else n < n_min,
                           "metric_a1": None, "metric_b": None, "diff": None, "diff_ci": None, "within_delta": None,
                           "pass": None, "note": None}
    if n == 0:
        out["note"] = "空组"
        return out
    a, stat = group_values(group, parts_a)
    b, _ = group_values(group, parts_b)
    f = stat if stat is not None else (lambda x: float(np.mean(x)))
    out["metric_a1"], out["metric_b"] = float(f(a)), float(f(b))
    if not (np.isfinite(out["metric_a1"]) and np.isfinite(out["metric_b"])):
        out["note"] = "统计量无定义（如 cIoU 的并为 0、检测没有 GT 框）"
        out["metric_a1"] = out["metric_a1"] if np.isfinite(out["metric_a1"]) else None
        out["metric_b"] = out["metric_b"] if np.isfinite(out["metric_b"]) else None
        return out
    out["diff"] = out["metric_a1"] - out["metric_b"]
    try:
        ci = paired_bootstrap_ci(a, b, statistic=stat)
    except ValueError as e:
        out["note"] = f"CI 无定义：{e}"
        return out
    out["diff_ci"] = _ci(ci)
    out["within_delta"] = bool(-delta <= ci.low and ci.high <= delta)
    out["pass"] = out["within_delta"]
    return out


def reference_columns(task: str, samples: Sequence[Sample], a1: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]],
                      *, has_gt: bool, batch_size_b: int) -> dict[str, Any]:
    """A1 与 B 之间的参考列（不参与判定）：三种一致口径的一致率、raw_text 逐字相同的比例、本任务标题指标之差与配对 CI。"""
    row = compare(task, samples, a1, b, has_gt=has_gt, batch_size=batch_size_b)
    out = {"agreement_rate": {m: row["agreement_by_mode"][m]["rate"] for m in AGREEMENTS},
           "raw_text_equal_rate": row["raw_text_equal_rate"], "metric": row["metric"], "metric_a1": row["metric_single"],
           "metric_b": row["metric_batched"], "diff": row["diff"], "diff_ci": row["diff_ci"], "note": row["metric_note"]}
    if task == "ovd" and has_gt and samples:  # compare 不算检测的指标差；这里按图算 F1@0.5（与 detection 组同一个统计量）
        nf = noise_floor({"name": "ovd", "metric": "F1@0.5", "metric_class": "accuracy", "n_min": None}, 0.0,
                         [(task, samples, a1)], [(task, samples, b)])
        out.update(metric="F1@0.5", metric_a1=nf["metric_a1"], metric_b=nf["metric_b"], diff=nf["diff"],
                   diff_ci=nf["diff_ci"], note=nf["note"])
    return out


def overall_v3(task_rows: Sequence[Mapping[str, Any]], group_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """总判定：(i) 全部任务逐位相同；(ii) 能判的组全部在 ±δ 以内。没有一组能判时 all_pass 为 null。"""
    rerun_ok = all(r["rerun"]["identical"] for r in task_rows)
    judged = [g for g in group_rows if g["pass"] is not None]
    groups_ok = all(g["pass"] for g in judged)
    return {"all_pass": (rerun_ok and groups_ok) if judged else None, "rerun_all_identical": rerun_ok,
            "rerun_not_identical": [r["task"] for r in task_rows if not r["rerun"]["identical"]],
            "groups_judged": [g["group"] for g in judged], "groups_failed": [g["group"] for g in judged if not g["pass"]],
            "groups_not_evaluated": [g["group"] for g in group_rows if g["pass"] is None],
            "groups_below_n_min": [g["group"] for g in group_rows if g["n_below_min"]]}


def summary_markdown_v3(model_key: str, task_rows: Sequence[Mapping[str, Any]], group_rows: Sequence[Mapping[str, Any]], *,
                        header: Mapping[str, Any], protocol: Mapping[str, Any], batch_protocol: Mapping[str, Any]) -> str:
    o = overall_v3(task_rows, group_rows)
    lines = [f"# 批处理验收 v3 · {model_key}", "",
             f"- 规则（configs/e1_prompts.json 的 acceptance，v{protocol['version']}）：{protocol['rule']}",
             f"- 本次 A1、A2：batch_size = {header['batch_size_a']}，{batch_protocol['padding_side']} 填充，"
             f"顺序 {batch_protocol['order']}，{batch_protocol['letter_batches']}；B：batch_size = {header['batch_size_b']}。"
             f"{'是' if header['frozen'] else '不是'}协议的冻结值（A {batch_protocol['batch_size']}、"
             f"B {protocol['composition_b']['batch_size']}）",
             f"- composition_b 的规则：{protocol['composition_b']['rule']}",
             f"- δ：{protocol['delta']}；组与合并规则见下表与配置的 groups；{protocol['take_rule']}",
             f"- {dict(header)}", "",
             "## 判定", "",
             f"- all_pass：**{_f(o['all_pass'])}**；(i) 重跑逐位相同：{_f(o['rerun_all_identical'])}"
             f"（不同的任务：{'、'.join(o['rerun_not_identical']) or '—'}）；(ii) 不通过的组：{'、'.join(o['groups_failed']) or '—'}；"
             f"不判的组：{'、'.join(o['groups_not_evaluated']) or '—'}；n 低于 n_min 的组：{'、'.join(o['groups_below_n_min']) or '—'}",
             "", "## 数值噪声底（A1 与 B 之间，按组）", "",
             "| 组 | 任务 | 统计量 | n | n_min | A1 | B | 差（A1 − B） | 配对差 95% CI | δ | CI ⊂ ±δ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in group_rows:
        ci = g["diff_ci"]
        ci_s = f"[{_f(ci['low'])}, {_f(ci['high'])}]" if ci else "—"
        lines.append(f"| {g['group']} | {'、'.join(g['tasks'])} | {g['metric']} | {g['n']} | {_f(g['n_min'])} | "
                     f"{_f(g['metric_a1'])} | {_f(g['metric_b'])} | {_f(g['diff'])} | {ci_s} | {_f(g['delta'])} | "
                     f"{_f(g['within_delta'])} |")
    head = ["任务", "组", "n", "(i) A1 = A2 逐位", "不同的条数"] + [f"{AGREEMENT_LABELS[m]} 一致率（A1 vs B）" for m in AGREEMENTS]
    head += ["raw_text 逐字相同（A1 vs B）", "指标", "A1", "B", "差", "配对差 95% CI"]
    lines += ["", "## 逐任务（(i) 的判定；A1 与 B 之间的一致率与指标差只作参考）", "",
              "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in task_rows:
        ref, ci = r["reference"], r["reference"]["diff_ci"]
        cells = [r["task"], r["group"], str(r["n"]), _f(r["rerun"]["identical"]), str(r["rerun"]["n_diff"])]
        cells += [_f(ref["agreement_rate"][m]) for m in AGREEMENTS]
        cells += [_f(ref["raw_text_equal_rate"]), ref["metric"] or "—", _f(ref["metric_a1"]), _f(ref["metric_b"]),
                  _f(ref["diff"]), f"[{_f(ci['low'])}, {_f(ci['high'])}]" if ci else "—"]
        lines.append("| " + " | ".join(cells) + " |")
    notes = [f"- 组 {g['group']}：{g['note']}" for g in group_rows if g["note"]]
    notes += [f"- {r['task']}：{r['reference']['note']}" for r in task_rows if r["reference"]["note"]]
    if notes:
        lines += ["", "注："] + notes
    return "\n".join(lines) + "\n"


__all__ = [
    "PASSES_V3",
    "file_sha256",
    "group_of",
    "group_values",
    "noise_floor",
    "overall_v3",
    "reference_columns",
    "rerun_identical",
    "summary_markdown_v3",
    "take_samples",
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
