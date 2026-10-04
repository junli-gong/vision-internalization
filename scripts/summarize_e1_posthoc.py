#!/usr/bin/env python
"""E1 事后读数的汇总（DEV-0033；D66；E1 预注册 §10）：有效性检查、格式强制的配对比较、B200 用时估算。只读运行目录，不跑模型。

三个子命令（目录都是 run_e1.py 的 --out，即 experiments/E1/<NNNN_slug>；predictions.jsonl 也可以是 .jsonl.gz）：

1. validity —— §10.1 的有效性检查（规则与阈值取 configs/e1_posthoc.json 的 debiased_letter.validity_check）：
    python scripts/summarize_e1_posthoc.py validity --debiased experiments/E1/<去偏读数的运行> \\
        --free experiments/E1/0027_e1_main [--main experiments/E1/0027_e1_main] --out-json <json> --out-md <md>
   逐格（模型 × 任务）：在 free 子集的 id（<free>/<模型>/<任务>__free）上，去偏读数（<debiased>/<模型>/<任务>__debiased）的点估计、
   随机水平（逐题 1/k 的平均）、自由作答的有效率与严格读数；两条规则同时满足就标为"该模型的即时读数不反映能力"（flag）。另列全量的
   去偏读数与原字母读数（同一次运行的 s = 0）及其配对差。给了 --main 时，核对 s = 0 的答案与主评测（<main>/<模型>/<任务>）的字母读数
   逐条一致的比例（同一批组成，预期 1.0；不一致的 id 列出）。

2. forced —— §10.2 的配对比较：
    python scripts/summarize_e1_posthoc.py forced --forced experiments/E1/<格式强制的运行> --main experiments/E1/0027_e1_main \\
        --think experiments/E1/0028_e1_think --out-json <json> --out-md <md>
   逐格（模型 × 读法）：格式强制（<forced>/<模型>/<任务>__forced_<读法>）与同一批 id 的主评测（<main>/<模型>/<任务>）、思考子集
   （<think>/<模型>/<任务>__think）逐条配对：严格读数（cIoU、gIoU）与有效率各自的值与 bootstrap 95% CI、配对差（格式强制 − 对方）与
   配对 bootstrap 95% CI；条件读数各自在自己的有效样本上算（值与 CI），另给两边都有效的样本上的配对差。

3. estimate —— 按 RUN-0027（与 RUN-0028）的实测吞吐推算两项在 B200 上的用时（GPU 小时）：
    python scripts/summarize_e1_posthoc.py estimate --main experiments/E1/0027_e1_main [--out-json <json>] [--out-md <md>]
   去偏读数：每个模型 × 任务，主评测字母读数的墙钟（logs/run_*.json 的 wall_s，含建输入；另报只算生成的 compute.json）× Σk / n
   （读数次数是原来的 Σk / n 倍）。格式强制（Qwen 四个尺寸）：从主评测 grid16 与 refexp_poly 的 logs/timing.jsonl 估出批大小 64 下
   每个解码步的墙钟（整批墙钟 / 批里最长的生成长度，取中位数），网格两种读法 = 批数 × 271 步，多边形预填按上限 1024 步算（上界）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.e1.posthoc import GRID_FIXED_TOKENS, POSTHOC_PATH, load_posthoc_config  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e1.scoring import ci_dict, is_valid  # noqa: E402
from src.stats import bootstrap_ci, paired_bootstrap_ci, ratio_of_sums  # noqa: E402

LETTER_TASKS = ("rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial")
FORCED = {"prefill": "refexp_poly", "restricted": "grid16", "structured": "grid16"}
MODEL_ORDER = ("qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "gemma-4-e2b-it", "gemma-4-e4b-it",
               "internvl3.5-2b-hf", "internvl3.5-4b-hf")
POLY_MAX_NEW = 1024  # 多边形预填的上限同 E1 多边形（configs/e1_prompts.json 的 refexp_poly.max_new_tokens；estimate 只作上界）
# 有效性检查的比较容差：读数都是 n 条上的比例，0.7 − 0.5 这类差在浮点下会落在阈值下面一点点；1e-9 远小于一条样本（n = 200 时 0.005）
RULE_EPS = 1e-9


def read_records(leaf: Path) -> list[dict[str, Any]] | None:
    """<leaf>/predictions.jsonl（或 .jsonl.gz）；不存在时 None。"""
    f = leaf / "predictions.jsonl"
    if f.exists():
        text = f.read_text(encoding="utf-8")
    elif Path(str(f) + ".gz").exists():
        with gzip.open(str(f) + ".gz", "rt", encoding="utf-8") as g:
            text = g.read()
    else:
        return None
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def models_in(*roots: Path) -> list[str]:
    seen = {p.name for r in roots for p in r.iterdir() if p.is_dir() and not p.name.startswith((".", "logs"))}
    return [m for m in MODEL_ORDER if m in seen] + sorted(seen - set(MODEL_ORDER))


def _mean_ci(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"value": None, "ci": None, "n": 0}
    ci = bootstrap_ci(np.asarray(values, dtype=np.float64))
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(values)}


def _ratio_ci(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    if not pairs or sum(u for _, u in pairs) == 0:
        return {"value": None, "ci": None, "n": len(pairs)}
    arr = np.asarray(pairs, dtype=np.float64)
    try:
        ci = bootstrap_ci(arr, statistic=ratio_of_sums)
    except ValueError:
        return {"value": ratio_of_sums(arr), "ci": None, "n": len(pairs)}
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(pairs)}


def _paired(a: Sequence[Any], b: Sequence[Any], *, ratio: bool = False) -> dict[str, Any]:
    """配对差 a − b（同一批样本、同序）：均值类或 ratio_of_sums（cIoU）；某次重采样无定义时 CI 为 null。"""
    if not a:
        return {"value": None, "ci": None, "n": 0}
    av, bv = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    stat = ratio_of_sums if ratio else None
    try:
        ci = paired_bootstrap_ci(av, bv, statistic=stat)
    except ValueError:
        pa, pb = (ratio_of_sums(av), ratio_of_sums(bv)) if ratio else (float(av.mean()), float(bv.mean()))
        return {"value": pa - pb, "ci": None, "n": len(a), "note": "某次重采样上统计量无定义，不给 CI"}
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(a)}


def _f(v: Any) -> str:
    return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _ci(c: Mapping[str, Any] | None) -> str:
    return f"[{_f(c['low'])}, {_f(c['high'])}]" if c else "—"


def _configs() -> tuple[Any, Any]:
    prompts = load_prompt_config(PROMPTS_PATH)
    return prompts, load_posthoc_config(prompts, POSTHOC_PATH)


# ---------------------------------------------------------------------------
# 1. 有效性检查（§10.1）
# ---------------------------------------------------------------------------


def validity_cell(task: str, deb: Sequence[Mapping[str, Any]], free: Sequence[Mapping[str, Any]],
                  thresholds: Mapping[str, float], main: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """一格（模型 × 任务）的有效性检查。deb：去偏读数的全量记录；free：free 子集的记录；main：主评测的字母读数（可选，核对用）。"""
    by_id = {r["id"]: r for r in deb}
    missing = [r["id"] for r in free if r["id"] not in by_id]
    if missing:
        raise KeyError(f"{task}：free 的 {len(missing)} 个 id 不在去偏读数里，例如 {missing[:3]}")
    on = [by_id[r["id"]] for r in free]
    deb_acc = float(np.mean([float(r["score"]["correct"]) for r in on])) if on else None
    rnd = float(np.mean([1.0 / int(r["k"]) for r in on])) if on else None
    free_valid = float(np.mean([float(is_valid(task, r)) for r in free])) if free else None
    free_strict = float(np.mean([float(bool((r.get("score") or {}).get("correct"))) for r in free])) if free else None
    rule1 = deb_acc is not None and deb_acc <= rnd + thresholds["near_random_margin"] + RULE_EPS
    gap = None if free_strict is None or deb_acc is None else free_strict - deb_acc
    rule2 = (free_valid is not None and free_valid >= thresholds["free_valid_rate_min"] - RULE_EPS
             and gap is not None and gap >= thresholds["free_minus_debiased_min"] - RULE_EPS)
    a = [float(r["score"]["correct"]) for r in deb]
    b = [float(r["score"]["original_correct"]) for r in deb]
    out: dict[str, Any] = {
        "task": task, "n_free": len(free),
        "debiased_on_free": deb_acc, "random_level_on_free": rnd,
        "original_letter_on_free": float(np.mean([float(r["score"]["original_correct"]) for r in on])) if on else None,
        "free_valid_rate": free_valid, "free_strict": free_strict, "free_minus_debiased": gap,
        "rule1_near_random": bool(rule1), "rule2_free_valid_and_ahead": bool(rule2),
        "flag_not_reflecting_ability": bool(rule1 and rule2),
        "n_eval": len(deb), "debiased_eval": _mean_ci(a), "original_letter_eval": _mean_ci(b),
        "debiased_minus_original_eval": _paired(a, b),
    }
    if main is not None:
        m_by = {r["id"]: r for r in main}
        common = [r for r in deb if r["id"] in m_by]
        dis = [r["id"] for r in common if r["parsed"]["original_letter"] != m_by[r["id"]]["parsed"]["letter"]]
        out["order0_vs_main_letter"] = {"n": len(common), "n_agree": len(common) - len(dis),
                                        "rate": (len(common) - len(dis)) / len(common) if common else None,
                                        "disagree_ids": dis[:20]}
    return out


def cmd_validity(a) -> dict[str, Any]:
    prompts, posthoc = _configs()
    th = posthoc.validity_thresholds()
    deb_root, free_root = Path(a.debiased), Path(a.free)
    main_root = Path(a.main) if a.main else None
    cells = []
    for model in models_in(deb_root):
        for task in LETTER_TASKS:
            deb = read_records(deb_root / model / f"{task}__debiased")
            free = read_records(free_root / model / f"{task}__free")
            if deb is None or free is None:
                cells.append({"model": model, "task": task, "missing": [n for n, x in (("debiased", deb), ("free", free))
                                                                         if x is None]})
                continue
            main = read_records(main_root / model / task) if main_root is not None else None
            cells.append({"model": model, **validity_cell(task, deb, free, th, main)})
    flagged = [f"{c['model']}/{c['task']}" for c in cells if c.get("flag_not_reflecting_ability")]
    return {"kind": "validity", "rule": posthoc.debiased["validity_check"]["rule"], "thresholds": th,
            "posthoc_config_sha256": posthoc.sha256, "prompt_config_sha256": prompts.sha256,
            "inputs": {"debiased": str(deb_root), "free": str(free_root), "main": str(main_root) if main_root else None},
            "cells": cells, "flagged": flagged, "n_flagged": len(flagged)}


def validity_md(doc: Mapping[str, Any]) -> str:
    th = doc["thresholds"]
    lines = ["# E1 事后读数：去偏字母读数的有效性检查（§10.1，事后）", "",
             f"规则：两条同时满足即标为\"该模型的即时读数不反映能力\"——(1) free 的 id 上去偏读数 ≤ 随机水平 + {th['near_random_margin']}；"
             f"(2) 自由作答有效率 ≥ {th['free_valid_rate_min']} 且自由作答严格读数 − 去偏读数 ≥ {th['free_minus_debiased_min']}。", "",
             "| 模型 | 任务 | n(free) | 去偏@free | 随机 | 原字母@free | 自由有效率 | 自由严格 | 差 | 规则1 | 规则2 | 标记 | 去偏(全量) | 95% CI | 原字母(全量) | 去偏−原字母 | 配对 95% CI | s=0 与主评测一致 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in doc["cells"]:
        if "missing" in c:
            lines.append(f"| {c['model']} | {c['task']} | 缺 {c['missing']} |" + " |" * 15)
            continue
        agree = c.get("order0_vs_main_letter")
        lines.append(
            f"| {c['model']} | {c['task']} | {c['n_free']} | {_f(c['debiased_on_free'])} | {_f(c['random_level_on_free'])} | "
            f"{_f(c['original_letter_on_free'])} | {_f(c['free_valid_rate'])} | {_f(c['free_strict'])} | {_f(c['free_minus_debiased'])} | "
            f"{'是' if c['rule1_near_random'] else '否'} | {'是' if c['rule2_free_valid_and_ahead'] else '否'} | "
            f"{'**不反映能力**' if c['flag_not_reflecting_ability'] else '—'} | {_f(c['debiased_eval']['value'])} | "
            f"{_ci(c['debiased_eval']['ci'])} | {_f(c['original_letter_eval']['value'])} | "
            f"{_f(c['debiased_minus_original_eval']['value'])} | {_ci(c['debiased_minus_original_eval']['ci'])} | "
            f"{'—' if agree is None else _f(agree['rate'])} |")
    lines += ["", f"被标记的格子（{doc['n_flagged']}）：{', '.join(doc['flagged']) or '无'}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. 格式强制的配对比较（§10.2）
# ---------------------------------------------------------------------------


def _iou_rows(task: str, recs: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    return {"pairs": [(float(r["score"]["inter"]), float(r["score"]["union"])) for r in recs],
            "iou": [float(r["score"]["iou"]) for r in recs],
            "valid": [float(is_valid(task, r)) for r in recs]}


def _side(task: str, recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x = _iou_rows(task, recs)
    ok = [r for r in recs if is_valid(task, r)]
    y = _iou_rows(task, ok)
    return {"n": len(recs), "strict": {"cIoU": _ratio_ci(x["pairs"]), "gIoU": _mean_ci(x["iou"])},
            "valid_rate": _mean_ci(x["valid"]), "n_valid": len(ok),
            "conditional": {"cIoU": _ratio_ci(y["pairs"]), "gIoU": _mean_ci(y["iou"])}}


def forced_cell(task: str, forced: Sequence[Mapping[str, Any]], others: Mapping[str, Sequence[Mapping[str, Any]] | None]
                ) -> dict[str, Any]:
    """一格（模型 × 读法）：格式强制与每个对照（主评测、思考子集）在同一批 id 上逐条配对。"""
    ids = [r["id"] for r in forced]
    out: dict[str, Any] = {"task": task, "n": len(ids), "forced": _side(task, forced), "vs": {}}
    fx = _iou_rows(task, forced)
    for name, recs in others.items():
        if recs is None:
            out["vs"][name] = {"missing": True}
            continue
        by = {r["id"]: r for r in recs}
        miss = [i for i in ids if i not in by]
        if miss:
            raise KeyError(f"{task}：{name} 里缺 {len(miss)} 个格式强制的 id，例如 {miss[:3]}")
        o = [by[i] for i in ids]
        ox = _iou_rows(task, o)
        both = [k for k in range(len(ids)) if fx["valid"][k] and ox["valid"][k]]
        out["vs"][name] = {
            "other": _side(task, o),
            "paired_diff": {  # 格式强制 − 对方
                "cIoU": _paired(fx["pairs"], ox["pairs"], ratio=True),
                "gIoU": _paired(fx["iou"], ox["iou"]),
                "valid_rate": _paired(fx["valid"], ox["valid"]),
                "conditional_both_valid": {
                    "n": len(both),
                    "cIoU": _paired([fx["pairs"][k] for k in both], [ox["pairs"][k] for k in both], ratio=True),
                    "gIoU": _paired([fx["iou"][k] for k in both], [ox["iou"][k] for k in both])}}}
    return out


def cmd_forced(a) -> dict[str, Any]:
    prompts, posthoc = _configs()
    froot, mroot = Path(a.forced), Path(a.main)
    troot = Path(a.think) if a.think else None
    cells = []
    for model in models_in(froot):
        for variant, task in FORCED.items():
            fr = read_records(froot / model / f"{task}__forced_{variant}")
            if fr is None:
                continue
            others = {"main": read_records(mroot / model / task)}
            if troot is not None:
                others["think"] = read_records(troot / model / f"{task}__think")
            cells.append({"model": model, "variant": variant, **forced_cell(task, fr, others)})
    return {"kind": "forced", "posthoc_config_sha256": posthoc.sha256, "prompt_config_sha256": prompts.sha256,
            "rule": posthoc.forced["report_rule"],
            "inputs": {"forced": str(froot), "main": str(mroot), "think": str(troot) if troot else None}, "cells": cells}


def forced_md(doc: Mapping[str, Any]) -> str:
    lines = ["# E1 事后诊断：格式强制（§10.2，事后）", "",
             "严格读数（cIoU、gIoU）、有效率与条件读数；配对差 = 格式强制 − 对照（同一批 id，配对 bootstrap 95% CI）。", "",
             "| 模型 | 读法 | 任务 | n | 有效率 | 严格 cIoU | 严格 gIoU | 条件 cIoU | 条件 gIoU | 对照 | 对照有效率 | 对照严格 cIoU | 对照严格 gIoU | Δ有效率 [CI] | Δ严格 cIoU [CI] | Δ严格 gIoU [CI] | 两边都有效 n | Δ条件 gIoU [CI] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in doc["cells"]:
        f = c["forced"]
        head = (f"| {c['model']} | {c['variant']} | {c['task']} | {c['n']} | {_f(f['valid_rate']['value'])} | "
                f"{_f(f['strict']['cIoU']['value'])} | {_f(f['strict']['gIoU']['value'])} | {_f(f['conditional']['cIoU']['value'])} | "
                f"{_f(f['conditional']['gIoU']['value'])}")
        for name, v in c["vs"].items():
            if v.get("missing"):
                lines.append(head + f" | {name} | 缺 |" + " |" * 7)
                continue
            o, d = v["other"], v["paired_diff"]
            cb = d["conditional_both_valid"]
            lines.append(head + f" | {name} | {_f(o['valid_rate']['value'])} | {_f(o['strict']['cIoU']['value'])} | "
                         f"{_f(o['strict']['gIoU']['value'])} | {_f(d['valid_rate']['value'])} {_ci(d['valid_rate']['ci'])} | "
                         f"{_f(d['cIoU']['value'])} {_ci(d['cIoU']['ci'])} | {_f(d['gIoU']['value'])} {_ci(d['gIoU']['ci'])} | "
                         f"{cb['n']} | {_f(cb['gIoU']['value'])} {_ci(cb['gIoU']['ci'])} |")
    return "\n".join(lines + [""])


# ---------------------------------------------------------------------------
# 3. 用时估算（B200，按 RUN-0027 实测吞吐）
# ---------------------------------------------------------------------------


def _run_wall(leaf: Path) -> float | None:
    vals = [json.loads(p.read_text()).get("wall_s") for p in sorted((leaf / "logs").glob("run_*.json"))]
    vals = [v for v in vals if v is not None]
    return float(sum(vals)) if vals else None


def _step_times(leaf: Path, recs: Sequence[Mapping[str, Any]]) -> list[float]:
    """每批：整批墙钟 / 批里最长的生成长度（每个解码步的墙钟）。"""
    gen = {r["id"]: int(r["n_generated_tokens"]) for r in recs}
    batches: dict[tuple[int, int], dict[str, Any]] = {}
    f = leaf / "logs" / "timing.jsonl"
    if not f.exists():
        return []
    for line in f.read_text().splitlines():
        t = json.loads(line)
        b = batches.setdefault((int(t["run"]), int(t["batch_index"])), {"wall": float(t["batch_wall_time_s"]), "ids": []})
        b["ids"].append(t["id"])
    out = []
    for b in batches.values():
        steps = max(gen.get(i, 0) for i in b["ids"])
        if steps > 0 and len(b["ids"]) >= 32:  # 只用满批附近的（批大小 64；末尾的小批不代表）
            out.append(b["wall"] / steps)
    return out


def cmd_estimate(a) -> dict[str, Any]:
    root = Path(a.main)
    n_forced = int(a.forced_n)
    batch = int(a.batch_size)
    rows = []
    for model in models_in(root):
        for task in LETTER_TASKS:
            leaf = root / model / task
            recs = read_records(leaf)
            if recs is None:
                continue
            n = len(recs)
            sum_k = sum(len(r["extras"]["letters"]) for r in recs)
            comp = json.loads((leaf / "compute.json").read_text())
            wall = _run_wall(leaf)
            factor = sum_k / n
            rows.append({"kind": "debiased", "model": model, "task": task, "n": n, "sum_k": sum_k, "factor": factor,
                         "letter_generate_s": comp["call_time_s"], "letter_wall_s": wall,
                         "debiased_generate_s": comp["call_time_s"] * factor,
                         "debiased_wall_s": None if wall is None else wall * factor})
        if not model.startswith("qwen3.5"):
            continue
        steps = []
        for task in ("grid16", "refexp_poly"):
            recs = read_records(root / model / task)
            if recs is not None:
                steps += _step_times(root / model / task, recs)
        if steps:
            st = float(statistics.median(steps))
            n_batches = -(-n_forced // batch)
            rows.append({"kind": "forced", "model": model, "step_s_median": st, "n_step_samples": len(steps),
                         "n_batches": n_batches,
                         "restricted_s": n_batches * GRID_FIXED_TOKENS * st, "structured_s": n_batches * GRID_FIXED_TOKENS * st,
                         "prefill_upper_s": n_batches * POLY_MAX_NEW * st})
    deb_wall = sum(r["debiased_wall_s"] or 0.0 for r in rows if r["kind"] == "debiased")
    deb_gen = sum(r["debiased_generate_s"] for r in rows if r["kind"] == "debiased")
    forced = sum(r["restricted_s"] + r["structured_s"] + r["prefill_upper_s"] for r in rows if r["kind"] == "forced")
    return {"kind": "estimate", "source": str(root), "rows": rows,
            "totals_gpu_hours": {"debiased_wall": deb_wall / 3600, "debiased_generate_only": deb_gen / 3600,
                                 "forced_upper": forced / 3600},
            "note": "去偏读数 = 主评测字母读数的墙钟 × Σk/n（每条样本读 k 次；含建输入与预热，不含加载模型）；格式强制按批大小 "
                    f"{batch}、每个解码步的中位墙钟外推：网格两种读法各 {GRID_FIXED_TOKENS} 步，多边形预填按上限 {POLY_MAX_NEW} 步（上界）；"
                    "都不含加载权重与预热。"}


def estimate_md(doc: Mapping[str, Any]) -> str:
    lines = ["# E1 事后读数的 B200 用时估算（按 RUN-0027 实测吞吐）", "", doc["note"], "",
             "| 模型 | 任务 | n | Σk | 倍数 | 字母读数墙钟 s | 去偏读数墙钟 s（估） | 只算生成 s（估） |", "|---|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        if r["kind"] == "debiased":
            lines.append(f"| {r['model']} | {r['task']} | {r['n']} | {r['sum_k']} | {r['factor']:.3f} | {_f(r['letter_wall_s'])} | "
                         f"{_f(r['debiased_wall_s'])} | {_f(r['debiased_generate_s'])} |")
    lines += ["", "| 模型 | 每步墙钟中位数 s | 批数 | 网格 restricted s | 网格 structured s | 多边形预填（上界）s |", "|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        if r["kind"] == "forced":
            lines.append(f"| {r['model']} | {r['step_s_median']:.5f} | {r['n_batches']} | {r['restricted_s']:.1f} | "
                         f"{r['structured_s']:.1f} | {r['prefill_upper_s']:.1f} |")
    t = doc["totals_gpu_hours"]
    lines += ["", f"合计（GPU 小时）：去偏读数 {t['debiased_wall']:.3f}（只算生成 {t['debiased_generate_only']:.3f}）；"
              f"格式强制（上界）{t['forced_upper']:.3f}。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validity", help="§10.1 的有效性检查")
    v.add_argument("--debiased", required=True, help="去偏读数的运行目录（run_e1.py --subset debiased 的 --out）")
    v.add_argument("--free", required=True, help="free 子集的运行目录（RUN-0027：experiments/E1/0027_e1_main）")
    v.add_argument("--main", help="主评测的运行目录（核对 s = 0 与字母读数逐条一致；可选）")
    f = sub.add_parser("forced", help="§10.2 的配对比较")
    f.add_argument("--forced", required=True, help="格式强制的运行目录（run_e1.py --subset forced 的 --out）")
    f.add_argument("--main", required=True, help="主评测的运行目录（RUN-0027）")
    f.add_argument("--think", help="思考子集的运行目录（RUN-0028）")
    e = sub.add_parser("estimate", help="按 RUN-0027 的实测吞吐推算 B200 用时")
    e.add_argument("--main", required=True, help="主评测的运行目录（RUN-0027）")
    e.add_argument("--forced-n", type=int, default=200, help="格式强制每个任务的样本数（默认 200）")
    e.add_argument("--batch-size", type=int, default=64, help="批大小（默认 64，冻结值）")
    for s in (v, f, e):
        s.add_argument("--out-json", help="写出 JSON")
        s.add_argument("--out-md", help="写出 Markdown 表")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    a = parse_args(argv)
    doc, md = {"validity": (cmd_validity, validity_md), "forced": (cmd_forced, forced_md),
               "estimate": (cmd_estimate, estimate_md)}[a.cmd]
    out = doc(a)
    text = md(out)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.out_md:
        Path(a.out_md).write_text(text, encoding="utf-8")
    if not (a.out_json or a.out_md):
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
