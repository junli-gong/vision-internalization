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
   --posthoc experiments/E1/0030_e1_posthoc（DEV-0036）：另按 RUN-0030 的实测推算对调标记（= 同一模型 NYU 去偏读数的实测墙钟，读数
   次数与批组成相同）与结构强制的多边形（预填目录每步的中位墙钟 × 批数 × 769 步 × 两个变体，上界）。

DEV-0036（D72-2、D72-3；事后）：
- validity 加 --letter-swap <对调标记的运行目录>：NYU 点对的有效性检查改用对调标记的去偏读数；另出 D72-2 的标题表（headline_d72：NYU =
  对调标记，BLINK = 自由作答 112 条，CV-Bench、EmbSpatial = 循环移位去偏），每格带有效性检查的标记。模型目录只认登记过的模型键或含任务
  叶目录的目录（probe_scripts 之类不再当成模型）。
- ownbox ——"模型自己的框"基线与并列表（规则见 configs/e1_posthoc.json 的 own_box；要 GT mask，所以给 --data-root；不跑模型）：
    python scripts/summarize_e1_posthoc.py ownbox --main experiments/E1/0027_e1_main --think experiments/E1/0028_e1_think \\
        --forced experiments/E1/0030_e1_posthoc --forced-poly experiments/E1/<结构强制的多边形的运行> --data-root $PROJ/data \\
        --out-json <json> --out-md <md>
  对每个模型、refexp_poly 与 grid16 的思考子集 id：按 id 对上 RUN-0027 refexp_box 的框并核对同一句话（渲染出的定位指令逐字相同、原图
  宽高相同），框光栅化（多边形：四个角当多边形；网格：再填进 16×16 网格）后与 GT 算 cIoU、gIoU；主评测、思考、预填 / 受限、结构强制
  各读数在同一批 id 上并列，减去"自己的框"的配对差与配对 bootstrap CI；开思考的框只在交集上。
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

from src.adapters import registry  # noqa: E402
from src.e1.poly_forced import near_rect_summary, near_rectangle  # noqa: E402
from src.e1.posthoc import GRID_FIXED_TOKENS, POSTHOC_PATH, load_posthoc_config  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, TASKS, load_prompt_config  # noqa: E402
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


def is_model_dir(p: Path) -> bool:
    """运行目录下的模型目录：登记过的模型键，或里面至少有一个以任务名开头的叶目录（<任务>[__…]）。probe_scripts、logs 这类
    非模型目录不算（RUN-0030 的汇总多出"缺"的行，DEV-0036 顺手修）。"""
    if not p.is_dir() or p.name.startswith((".", "logs")):
        return False
    if p.name in registry.MODELS:
        return True
    return any(c.is_dir() and c.name.split("__")[0] in TASKS for c in p.iterdir())


def models_in(*roots: Path) -> list[str]:
    seen = {p.name for r in roots for p in r.iterdir() if is_model_dir(p)}
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


def _letter_source(task: str, model: str, deb_root: Path, ls_root: Path | None) -> tuple[str, list[dict[str, Any]] | None]:
    """一格用哪种字母读数：给了 --letter-swap 时 NYU 点对用对调标记的去偏读数（D72-2），其余照旧用循环移位的去偏读数。"""
    if task == "rel_depth_nyu" and ls_root is not None:
        return "letter_swap", read_records(ls_root / model / f"{task}__letter_swap")
    return "letter_debiased", read_records(deb_root / model / f"{task}__debiased")


def cmd_validity(a) -> dict[str, Any]:
    prompts, posthoc = _configs()
    th = posthoc.validity_thresholds()
    deb_root, free_root = Path(a.debiased), Path(a.free)
    main_root = Path(a.main) if a.main else None
    ls_root = Path(a.letter_swap) if getattr(a, "letter_swap", None) else None
    models = models_in(deb_root, *([ls_root] if ls_root is not None else []))
    cells = []
    for model in models:
        for task in LETTER_TASKS:
            src, deb = _letter_source(task, model, deb_root, ls_root)
            free = read_records(free_root / model / f"{task}__free")
            if deb is None or free is None:
                cells.append({"model": model, "task": task, "readout": src,
                              "missing": [n for n, x in ((src, deb), ("free", free)) if x is None]})
                continue
            main = read_records(main_root / model / task) if main_root is not None else None
            cell = {"model": model, "readout": src, **validity_cell(task, deb, free, th, main)}
            if src == "letter_swap":  # 被替换的循环移位读数照列，供对照
                cyc = read_records(deb_root / model / f"{task}__debiased")
                if cyc is not None:
                    cell["superseded_cyclic_debiased_eval"] = _mean_ci([float(r["score"]["correct"]) for r in cyc])
            cells.append(cell)
    flagged = [f"{c['model']}/{c['task']}" for c in cells if c.get("flag_not_reflecting_ability")]
    return {"kind": "validity", "rule": posthoc.debiased["validity_check"]["rule"], "thresholds": th,
            "posthoc_config_sha256": posthoc.sha256, "prompt_config_sha256": prompts.sha256,
            "inputs": {"debiased": str(deb_root), "free": str(free_root), "main": str(main_root) if main_root else None,
                       "letter_swap": str(ls_root) if ls_root else None},
            "cells": cells, "flagged": flagged, "n_flagged": len(flagged),
            "headline_d72": headline_d72(posthoc, models, cells, deb_root, free_root, ls_root)}


def headline_d72(posthoc: Any, models: Sequence[str], cells: Sequence[Mapping[str, Any]], deb_root: Path, free_root: Path,
                 ls_root: Path | None) -> dict[str, Any]:
    """D72-2 的标题表（事后）：NYU 点对 = 对调标记的去偏读数（全量），BLINK = 自由作答（free，112 条即全量），CV-Bench、EmbSpatial =
    循环移位的去偏读数。每格另记有效性检查的标记（§10.1 的两条规则；NYU 用对调标记的读数）。"""
    src_of = dict(posthoc.headline["tasks"])
    flag = {(c["model"], c["task"]): c.get("flag_not_reflecting_ability") for c in cells}
    rows = []
    for model in models:
        row: dict[str, Any] = {"model": model, "cells": {}}
        for task in LETTER_TASKS:
            src = src_of[task]
            e: dict[str, Any] = {"readout": src}
            if src == "free":
                recs = read_records(free_root / model / f"{task}__free")
                full = read_records(deb_root / model / f"{task}__debiased")
                if recs is None:
                    e["missing"] = "free"
                else:
                    e.update(_mean_ci([float(bool((r.get("score") or {}).get("correct"))) for r in recs]))
                    e["valid_rate"] = float(np.mean([float(is_valid(task, r)) for r in recs])) if recs else None
                    if full is not None:  # free 是不是全量（BLINK：think 子集 112 条 = 评测集全部）
                        e["n_eval"] = len(full)
                        e["covers_eval"] = {r["id"] for r in recs} == {r["id"] for r in full}
            else:
                root, leaf = (ls_root, f"{task}__letter_swap") if src == "letter_swap" else (deb_root, f"{task}__debiased")
                recs = read_records(root / model / leaf) if root is not None else None
                if recs is None:
                    e["missing"] = src
                else:
                    e.update(_mean_ci([float(r["score"]["correct"]) for r in recs]))
                    e["random_level"] = float(np.mean([1.0 / int(r["k"]) for r in recs])) if recs else None
            e["flag_not_reflecting_ability"] = flag.get((model, task))
            row["cells"][task] = e
        rows.append(row)
    return {"rule": posthoc.headline["rule"], "sources": src_of, "rows": rows}


def validity_md(doc: Mapping[str, Any]) -> str:
    th = doc["thresholds"]
    lines = ["# E1 事后读数：去偏字母读数的有效性检查（§10.1，事后）", "",
             f"规则：两条同时满足即标为\"该模型的即时读数不反映能力\"——(1) free 的 id 上去偏读数 ≤ 随机水平 + {th['near_random_margin']}；"
             f"(2) 自由作答有效率 ≥ {th['free_valid_rate_min']} 且自由作答严格读数 − 去偏读数 ≥ {th['free_minus_debiased_min']}。"
             + ("NYU 点对用对调标记的去偏读数（letter_swap，D72-2）。" if doc["inputs"].get("letter_swap") else ""), "",
             "| 模型 | 任务 | 读数 | n(free) | 去偏@free | 随机 | 原字母@free | 自由有效率 | 自由严格 | 差 | 规则1 | 规则2 | 标记 | 去偏(全量) | 95% CI | 原字母(全量) | 去偏−原字母 | 配对 95% CI | s=0 与主评测一致 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in doc["cells"]:
        if "missing" in c:
            lines.append(f"| {c['model']} | {c['task']} | {c['readout']} | 缺 {c['missing']} |" + " |" * 15)
            continue
        agree = c.get("order0_vs_main_letter")
        lines.append(
            f"| {c['model']} | {c['task']} | {c['readout']} | {c['n_free']} | {_f(c['debiased_on_free'])} | {_f(c['random_level_on_free'])} | "
            f"{_f(c['original_letter_on_free'])} | {_f(c['free_valid_rate'])} | {_f(c['free_strict'])} | {_f(c['free_minus_debiased'])} | "
            f"{'是' if c['rule1_near_random'] else '否'} | {'是' if c['rule2_free_valid_and_ahead'] else '否'} | "
            f"{'**不反映能力**' if c['flag_not_reflecting_ability'] else '—'} | {_f(c['debiased_eval']['value'])} | "
            f"{_ci(c['debiased_eval']['ci'])} | {_f(c['original_letter_eval']['value'])} | "
            f"{_f(c['debiased_minus_original_eval']['value'])} | {_ci(c['debiased_minus_original_eval']['ci'])} | "
            f"{'—' if agree is None else _f(agree['rate'])} |")
    lines += ["", f"被标记的格子（{doc['n_flagged']}）：{', '.join(doc['flagged']) or '无'}", ""]
    sup = [c for c in doc["cells"] if c.get("superseded_cyclic_debiased_eval")]
    if sup:
        lines += ["NYU 点对被替换的循环移位去偏读数（全量，供对照）：" + "；".join(
            f"{c['model']} {_f(c['superseded_cyclic_debiased_eval']['value'])}" for c in sup), ""]
    hd = doc.get("headline_d72")
    if hd:
        names = {"letter_swap": "对调标记", "free": "自由作答", "letter_debiased": "循环移位去偏"}
        lines += ["## D72-2 的标题表（事后）", "", hd["rule"], "",
                  "每格：读数 [95% CI]（n）；BLINK 另给自由作答的有效率；† = 有效性检查标为\"不反映能力\"。", "",
                  "| 模型 | " + " | ".join(f"{t}（{names[hd['sources'][t]]}）" for t in LETTER_TASKS) + " |",
                  "|---|" + "---|" * len(LETTER_TASKS)]
        for row in hd["rows"]:
            cells = []
            for t in LETTER_TASKS:
                e = row["cells"][t]
                if "missing" in e:
                    cells.append(f"缺 {e['missing']}")
                    continue
                s = f"{_f(e['value'])} {_ci(e['ci'])}（{e['n']}）"
                if e["readout"] == "free":
                    s += f"，有效率 {_f(e.get('valid_rate'))}" + ("" if e.get("covers_eval", True) else "，**不是全量**")
                if e.get("flag_not_reflecting_ability"):
                    s += " †"
                cells.append(s)
            lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
        lines.append("")
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
# 4. "模型自己的框"基线与并列表（DEV-0036；D72-3；configs/e1_posthoc.json 的 own_box）
# ---------------------------------------------------------------------------

POLY_READOUTS = (("main", "{main}", "refexp_poly"), ("think", "{think}", "refexp_poly__think"),
                 ("prefill", "{forced}", "refexp_poly__forced_prefill"),
                 ("poly_structured", "{forced_poly}", "refexp_poly__poly_structured"),
                 ("poly_structured_min8", "{forced_poly}", "refexp_poly__poly_structured_min8"))
GRID_READOUTS = (("main", "{main}", "grid16"), ("think", "{think}", "grid16__think"),
                 ("restricted", "{forced}", "grid16__forced_restricted"), ("structured", "{forced}", "grid16__forced_structured"))
OWN_BOX, OWN_BOX_THINK = "own_box", "own_box_think"


def box_polygon(box: Sequence[float]) -> list[list[float]]:
    """框（绝对像素 xyxy）→ 四个角的多边形 (x0, y0)、(x1, y0)、(x1, y1)、(x0, y1)（扁平，COCO 格式）。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def own_box_record(task: str, box_rec: Mapping[str, Any], gt: Any) -> dict[str, Any]:
    """一条"自己的框"（光栅化规则见 own_box.raster_rule）：多边形对照用 E1 多边形的光栅化，网格对照再经 mask_to_grid 填进 16×16 网格、
    grid_to_mask 回填。返回与预测记录同形的伪记录（failure、text、parsed、score.inter / union / iou），有效与否同框记录。"""
    from src.constants import GRID_SIZE_E1
    from src.executors.raster import polygons_to_mask
    from src.grid.codec import grid_to_mask, mask_to_grid
    from src.metrics.seg import mask_inter_union, sample_iou

    h, w = gt.shape
    box = (box_rec.get("parsed") or {}).get("box")
    pred, parsed = None, None
    if box is not None:
        m = polygons_to_mask(box_polygon(box), h, w)
        if task == "refexp_poly":
            pred = m
            x0, y0, x1, y1 = (float(v) for v in box)
            parsed = {"polygons": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]}
        else:
            grid = mask_to_grid(m, GRID_SIZE_E1)
            pred = grid_to_mask(grid, h, w)
            parsed = {"grid": ["".join(str(int(v)) for v in row) for row in grid.tolist()]}
    st = mask_inter_union(pred, gt)
    return {"id": box_rec["id"], "failure": box_rec["failure"], "text": box_rec.get("text", ""),
            "raw_text": box_rec.get("raw_text", ""), "parsed": parsed, "box": box,
            "score": {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}}


def match_boxes(samples: Sequence[Any], box_recs: Sequence[Mapping[str, Any]] | None, prompts: Any, family: str,
                gts: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]]]:
    """按 id 对上框记录，并核对是同一句话（规则见 own_box.id_match_rule）。返回 (对上的 {id: 框记录}, {原因: [id]})。"""
    by = {r["id"]: r for r in (box_recs or [])}
    ok: dict[str, Mapping[str, Any]] = {}
    bad: dict[str, list[str]] = {"missing_id": [], "instruction_differs": [], "image_size_differs": []}
    for s in samples:
        r = by.get(s.id)
        if r is None:
            bad["missing_id"].append(s.id)
            continue
        if r["instruction"] != prompts.render("refexp_box", family, s).text:
            bad["instruction_differs"].append(s.id)
            continue
        h, w = gts[s.id].shape
        if [int(v) for v in r["image_size"]] != [w, h]:
            bad["image_size_differs"].append(s.id)
            continue
        ok[s.id] = r
    return ok, bad


def _poly_of(r: Mapping[str, Any]) -> Any:
    return (r.get("parsed") or {}).get("polygons")


def ownbox_row(task: str, name: str, recs: Sequence[Mapping[str, Any]], base: Sequence[Mapping[str, Any]] | None,
               nr: Mapping[str, Any], base_name: str = OWN_BOX) -> dict[str, Any]:
    """一行：n、有效率、严格与条件读数（_side）、近矩形占比（多边形），以及减去 base（同一批 id、同序）的配对差。"""
    row: dict[str, Any] = {"readout": name, **_side(task, recs)}
    if task == "refexp_poly":
        row["near_rectangle"] = near_rect_summary([near_rectangle(_poly_of(r), **nr) for r in recs],
                                                  [is_valid(task, r) for r in recs])
    if base is not None:
        if [r["id"] for r in recs] != [r["id"] for r in base]:
            raise ValueError(f"{task}/{name}：与 {base_name} 不是同一批 id、同序")
        x, y = _iou_rows(task, recs), _iou_rows(task, base)
        row["minus"] = base_name
        row["paired_diff"] = {"cIoU": _paired(x["pairs"], y["pairs"], ratio=True), "gIoU": _paired(x["iou"], y["iou"]),
                              "valid_rate": _paired(x["valid"], y["valid"])}
    return row


def ownbox_cell(task: str, samples: Sequence[Any], gts: Mapping[str, Any], prompts: Any, family: str,
                box_main: Sequence[Mapping[str, Any]] | None, box_think: Sequence[Mapping[str, Any]] | None,
                readouts: Mapping[str, Sequence[Mapping[str, Any]] | None], nr: Mapping[str, Any]) -> dict[str, Any]:
    """一个模型 × 一个目标任务：对上框、算"自己的框"，再把各读数在对上的 id 上并列（与"自己的框"配对）；开思考的框只在交集上。"""
    matched, unmatched = match_boxes(samples, box_main, prompts, family, gts)
    ids = [s.id for s in samples if s.id in matched]
    own = [own_box_record(task, matched[i], gts[i]) for i in ids]
    out: dict[str, Any] = {"task": task, "n_ids": len(samples), "n_matched": len(ids),
                           "unmatched": {k: {"n": len(v), "ids": v[:20]} for k, v in unmatched.items()},
                           "own_box_valid": {"n_valid": sum(1 for r in own if is_valid("refexp_box", matched[r["id"]]))},
                           "rows": [ownbox_row(task, OWN_BOX, own, None, nr)], "missing": {}}
    for name, recs in readouts.items():
        if recs is None:
            out["missing"][name] = "没有这个读数的运行目录"
            continue
        by = {r["id"]: r for r in recs}
        miss = [i for i in ids if i not in by]
        if miss:
            out["missing"][name] = f"缺 {len(miss)} 条（例如 {miss[:3]}）"
            continue
        out["rows"].append(ownbox_row(task, name, [by[i] for i in ids], own, nr))
    # 开思考的框：只在交集上（与同一批交集 id 上的思考读数、以及自己的框配对）
    if box_think is not None:
        tm, tun = match_boxes([s for s in samples if s.id in matched], box_think, prompts, family, gts)
        tids = [i for i in ids if i in tm]
        own_t = [own_box_record(task, tm[i], gts[i]) for i in tids]
        tb: dict[str, Any] = {"n_intersection": len(tids), "n_not_in_think_box_subset": len(tun["missing_id"]),
                              "n_instruction_differs": len(tun["instruction_differs"]),
                              "n_image_size_differs": len(tun["image_size_differs"])}
        if tids:
            own_by = {r["id"]: r for r in own}
            tb["row"] = ownbox_row(task, OWN_BOX_THINK, own_t, [own_by[i] for i in tids], nr)
            think = readouts.get("think")
            if think is not None and all(i in {r["id"] for r in think} for i in tids):
                tby = {r["id"]: r for r in think}
                tb["think_on_intersection"] = ownbox_row(task, "think", [tby[i] for i in tids], own_t, nr, base_name=OWN_BOX_THINK)
        out["think_box"] = tb
    return out


def _ownbox_paths(a) -> Any:
    from src.e1.tasks import DataPaths, default_nyu_cache

    if a.tiny:  # 事后读数的开发机彩排用 tests/fixtures/e1_posthoc（与 run_e1 --tiny 的事后子集同一份）
        base = DataPaths.posthoc_fixtures(nyu_cache=default_nyu_cache())
    else:
        import os

        root = a.data_root or os.environ.get("VISINT_DATA_ROOT")
        if not root:
            raise SystemExit("ownbox 要 --data-root（或 VISINT_DATA_ROOT）取 GT mask；开发机彩排用 --tiny")
        base = DataPaths(root=Path(root), nyu_cache=default_nyu_cache())
    kw = {k: Path(v) for k, v in (("dev_subsets", a.dev_subsets), ("eval_subsets", a.eval_subsets)) if v}
    return DataPaths(root=base.root, dev_subsets=kw.get("dev_subsets", base.dev_subsets),
                     eval_subsets=kw.get("eval_subsets", base.eval_subsets), nyu_pairs=base.nyu_pairs,
                     nyu_cache=base.nyu_cache, strict=base.strict)


def cmd_ownbox(a) -> dict[str, Any]:
    from src.e1.scoring import gt_mask
    from src.e1.tasks import select

    prompts, posthoc = _configs()
    ob = posthoc.own_box
    nr = posthoc.near_rect_thresholds()
    paths = _ownbox_paths(a)
    roots = {"main": Path(a.main), "think": Path(a.think) if a.think else None,
             "forced": Path(a.forced) if a.forced else None, "forced_poly": Path(a.forced_poly) if a.forced_poly else None}
    sels = {t: select(t, ob["sample_subset"], paths) for t in ob["targets"]}
    gts = {t: {s.id: gt_mask(s) for s in sel.samples} for t, sel in sels.items()}
    models = [m for m in models_in(roots["main"]) if (a.model is None or m in a.model)]
    out_models = []
    for model in models:
        family = registry.get_model(model).family
        box_main = read_records(roots["main"] / model / ob["box_task"])
        box_think = read_records(roots["think"] / model / f"{ob['box_task']}__think") if roots["think"] else None
        cells = {}
        for task in ob["targets"]:
            spec = POLY_READOUTS if task == "refexp_poly" else GRID_READOUTS
            readouts = {}
            for name, root_key, leaf in spec:
                root = roots[root_key.strip("{}")]
                readouts[name] = read_records(root / model / leaf) if root is not None else None
            cells[task] = ownbox_cell(task, sels[task].samples, gts[task], prompts, family, box_main, box_think, readouts, nr)
        out_models.append({"model": model, "family": family, "has_box_main": box_main is not None,
                           "has_box_think": box_think is not None, "tasks": cells})
    return {"kind": "ownbox", "posthoc_config_sha256": posthoc.sha256, "prompt_config_sha256": prompts.sha256,
            "rules": {k: ob[k] for k in ("id_match_rule", "think_rule", "raster_rule", "valid_rule", "report_rule")},
            "near_rectangle": nr,
            "inputs": {k: (str(v) if v is not None else None) for k, v in roots.items()},
            "data": {"root": str(paths.root), "eval_subsets": str(paths.eval_subsets)}, "models": out_models}


def _row_md(r: Mapping[str, Any], n_override: int | None = None) -> str:
    s = r["strict"]
    c = r["conditional"]
    near = r.get("near_rectangle")
    near_s = "—" if near is None else f"{_f(near['share_of_all'])} / {_f(near['share_of_valid'])}"
    d = r.get("paired_diff")
    ds = ("— | —" if d is None else
          f"{_f(d['cIoU']['value'])} {_ci(d['cIoU']['ci'])} | {_f(d['gIoU']['value'])} {_ci(d['gIoU']['ci'])}")
    return (f"| {r['readout']} | {n_override if n_override is not None else r['n']} | {_f(r['valid_rate']['value'])} | "
            f"{_f(s['cIoU']['value'])} {_ci(s['cIoU']['ci'])} | {_f(s['gIoU']['value'])} {_ci(s['gIoU']['ci'])} | "
            f"{_f(c['cIoU']['value'])} | {_f(c['gIoU']['value'])} | {near_s} | {r.get('minus', '—')} | {ds} |")


def ownbox_md(doc: Mapping[str, Any]) -> str:
    nr = doc["near_rectangle"]
    lines = ["# E1 事后诊断：\"模型自己的框\"基线与格式强制、思考的并列表（D72-3，事后）", "",
             f"- 对 id 与核对同一句话：{doc['rules']['id_match_rule']}",
             f"- 光栅化：{doc['rules']['raster_rule']}",
             f"- 近矩形：恰好 {nr['n_points']} 个点、面积 / 外接框面积 ≥ {nr['min_area_ratio']}；列出占全部 / 占有效。",
             "- 配对差 = 本行 − \"减去\"列的读数（同一批 id，配对 bootstrap 95% CI）。own_box_think 只在开思考的框与目标任务的交集上。", ""]
    head = ("| 读数 | n | 有效率 | 严格 cIoU [CI] | 严格 gIoU [CI] | 条件 cIoU | 条件 gIoU | 近矩形（全部 / 有效） | 减去 | "
            "Δ cIoU [CI] | Δ gIoU [CI] |")
    for m in doc["models"]:
        lines += [f"## {m['model']}", ""]
        for task, c in m["tasks"].items():
            un = "、".join(f"{k} {v['n']}" for k, v in c["unmatched"].items() if v["n"]) or "无"
            lines += [f"### {task}（思考子集的 id {c['n_ids']} 条，对上 {c['n_matched']} 条；对不上：{un}）", "", head,
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            lines += [_row_md(r) for r in c["rows"]]
            tb = c.get("think_box")
            if tb:
                if tb.get("row"):
                    lines.append(_row_md(tb["row"]))
                if tb.get("think_on_intersection"):
                    r = dict(tb["think_on_intersection"])
                    r["readout"] = "think（交集上）"
                    lines.append(_row_md(r))
            if c["missing"]:
                lines.append("")
                lines.append("缺的读数：" + "；".join(f"{k}：{v}" for k, v in c["missing"].items()))
            if tb is not None:
                lines += ["", f"开思考的框与目标任务的 id 交集：{tb['n_intersection']} 条。"]
            lines.append("")
    return "\n".join(lines)


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
    totals = {"debiased_wall": deb_wall / 3600, "debiased_generate_only": deb_gen / 3600, "forced_upper": forced / 3600}
    if getattr(a, "posthoc", None):
        rows += dev0036_estimate_rows(Path(a.posthoc), n_forced, batch)
        totals["letter_swap"] = sum(r["estimate_s"] for r in rows if r["kind"] == "letter_swap") / 3600
        totals["forced_poly_upper"] = sum(r["upper_s"] for r in rows if r["kind"] == "forced_poly") / 3600
    return {"kind": "estimate", "source": str(root), "rows": rows, "totals_gpu_hours": totals,
            "note": "去偏读数 = 主评测字母读数的墙钟 × Σk/n（每条样本读 k 次；含建输入与预热，不含加载模型）；格式强制按批大小 "
                    f"{batch}、每个解码步的中位墙钟外推：网格两种读法各 {GRID_FIXED_TOKENS} 步，多边形预填按上限 {POLY_MAX_NEW} 步（上界）；"
                    "都不含加载权重与预热。"
                    + ("DEV-0036（--posthoc，按 RUN-0030 实测）：对调标记 = 同一模型 RUN-0030 的 rel_depth_nyu__debiased 实测墙钟（读数次数"
                       "同为 2 × n，批组成同；另加换标记的图在 CPU 上渲染与核对，不在内）；结构强制的多边形 = RUN-0030 预填目录每个解码步的中位"
                       f"墙钟 × 批数 × {POLY_STRUCTURED_MAX_TOKENS} 步（语法允许的最长生成，上界）× 两个变体。" if getattr(a, "posthoc", None) else "")}


# 结构强制的多边形在 Qwen3.5 的切分下最长的生成（含停止 token）：1 + 64 × 10 + 63 × 2 + 1 + 1（tests 用 PolyGrammar 钉住）
POLY_STRUCTURED_MAX_TOKENS = 769


def dev0036_estimate_rows(posthoc_root: Path, n_forced: int, batch: int) -> list[dict[str, Any]]:
    """按 RUN-0030（B200）的实测推算 DEV-0036 两项新读数的用时（见 cmd_estimate 的说明）。"""
    rows: list[dict[str, Any]] = []
    for model in models_in(posthoc_root):
        leaf = posthoc_root / model / "rel_depth_nyu__debiased"
        recs = read_records(leaf)
        wall = _run_wall(leaf) if recs is not None else None
        if wall is not None:
            rows.append({"kind": "letter_swap", "model": model, "n": len(recs), "n_readings": sum(int(r["k"]) for r in recs),
                         "ref": "RUN-0030 rel_depth_nyu__debiased 的 run 墙钟", "ref_wall_s": wall, "estimate_s": wall})
        pre = posthoc_root / model / "refexp_poly__forced_prefill"
        precs = read_records(pre)
        if precs is None:
            continue
        steps = _step_times(pre, precs)
        if not steps:
            continue
        st = float(statistics.median(steps))
        n_batches = -(-n_forced // batch)
        one = n_batches * POLY_STRUCTURED_MAX_TOKENS * st
        rows.append({"kind": "forced_poly", "model": model, "step_s_median": st, "n_step_samples": len(steps),
                     "n_batches": n_batches, "max_tokens": POLY_STRUCTURED_MAX_TOKENS, "per_variant_upper_s": one,
                     "upper_s": 2 * one, "prefill_ref_wall_s": _run_wall(pre)})
    return rows


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
    if "letter_swap" in t:
        lines += ["## DEV-0036：对调标记与结构强制的多边形（按 RUN-0030 实测）", "",
                  "| 模型 | 对调标记：读数次数 | RUN-0030 NYU 去偏实测 s（= 估算） | 多边形：每步墙钟中位数 s | 批数 | 每个变体上界 s | 两个变体上界 s |",
                  "|---|---|---|---|---|---|---|"]
        by: dict[str, dict[str, Any]] = {}
        for r in doc["rows"]:
            if r["kind"] in ("letter_swap", "forced_poly"):
                by.setdefault(r["model"], {})[r["kind"]] = r
        for m, d in by.items():
            ls, fp = d.get("letter_swap"), d.get("forced_poly")
            lines.append(f"| {m} | {ls['n_readings'] if ls else '—'} | {_f(ls['estimate_s']) if ls else '—'} | "
                         f"{(format(fp['step_s_median'], '.5f')) if fp else '—'} | {fp['n_batches'] if fp else '—'} | "
                         f"{_f(fp['per_variant_upper_s']) if fp else '—'} | {_f(fp['upper_s']) if fp else '—'} |")
        lines += ["", f"合计（GPU 小时，不含加载权重）：对调标记 {t['letter_swap']:.3f}；结构强制的多边形（上界）{t['forced_poly_upper']:.3f}。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validity", help="§10.1 的有效性检查与 D72-2 的标题表")
    v.add_argument("--debiased", required=True, help="去偏读数的运行目录（run_e1.py --subset debiased 的 --out）")
    v.add_argument("--free", required=True, help="free 子集的运行目录（RUN-0027：experiments/E1/0027_e1_main）")
    v.add_argument("--main", help="主评测的运行目录（核对 s = 0 与字母读数逐条一致；可选）")
    v.add_argument("--letter-swap", help="对调标记的去偏读数的运行目录（run_e1.py --subset letter_swap 的 --out；D72-2：NYU 点对改用它）")
    o = sub.add_parser("ownbox", help="\"模型自己的框\"基线与并列表（D72-3）")
    o.add_argument("--main", required=True, help="主评测的运行目录（RUN-0027：refexp_box 的框、refexp_poly 与 grid16 的主评测）")
    o.add_argument("--think", help="思考子集的运行目录（RUN-0028：开思考的框与思考读数）")
    o.add_argument("--forced", help="格式强制的运行目录（RUN-0030：预填、受限、结构强制的网格）")
    o.add_argument("--forced-poly", help="结构强制的多边形的运行目录（run_e1.py --subset forced_poly 的 --out）")
    o.add_argument("--data-root", help="落地目录（GT mask；默认环境变量 VISINT_DATA_ROOT）")
    o.add_argument("--tiny", action="store_true", help="开发机彩排：数据取 tests/fixtures/e1_posthoc")
    o.add_argument("--eval-subsets", help="评测子集清单（默认 configs/e1_eval_subsets.json）")
    o.add_argument("--dev-subsets", help="开发子集清单（默认 configs/e1_dev_subsets.json）")
    o.add_argument("--model", nargs="+", help="只汇总这些模型（默认 --main 下的全部）")
    f = sub.add_parser("forced", help="§10.2 的配对比较")
    f.add_argument("--forced", required=True, help="格式强制的运行目录（run_e1.py --subset forced 的 --out）")
    f.add_argument("--main", required=True, help="主评测的运行目录（RUN-0027）")
    f.add_argument("--think", help="思考子集的运行目录（RUN-0028）")
    e = sub.add_parser("estimate", help="按 RUN-0027 的实测吞吐推算 B200 用时")
    e.add_argument("--main", required=True, help="主评测的运行目录（RUN-0027）")
    e.add_argument("--forced-n", type=int, default=200, help="格式强制每个任务的样本数（默认 200）")
    e.add_argument("--batch-size", type=int, default=64, help="批大小（默认 64，冻结值）")
    e.add_argument("--posthoc", help="RUN-0030 的运行目录：另按它的实测推算 DEV-0036 的对调标记与结构强制的多边形")
    for s in (v, f, e, o):
        s.add_argument("--out-json", help="写出 JSON")
        s.add_argument("--out-md", help="写出 Markdown 表")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    a = parse_args(argv)
    doc, md = {"validity": (cmd_validity, validity_md), "forced": (cmd_forced, forced_md),
               "estimate": (cmd_estimate, estimate_md), "ownbox": (cmd_ownbox, ownbox_md)}[a.cmd]
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
