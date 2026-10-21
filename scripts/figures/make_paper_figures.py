#!/usr/bin/env python
"""论文用图表（DEV-0052，D103 §五）：E1 差距图、E2 逐层曲线、E3 对照表、干扰随尺寸的曲线。

只读结果文件，不跑模型，不改任何结果；脚本里不写任何读数（数字一律从结果文件读）。一个命令生成全部：

    python scripts/figures/make_paper_figures.py                       # 输出到 docs/figures/
    python scripts/figures/make_paper_figures.py --root <结果树> --out <目录>   # 测试用的假结果树

- 每张图出 PDF（矢量，TrueType 字体）与 PNG（300 dpi）；表出 LaTeX（booktabs）与 Markdown；
  另写 docs/figures/README.md（数据来源路径、口径、生成时间与 commit、缺的数据与一致性核对）
  与 docs/figures/data/*.json（图表上每个数字的来源记录）。
- 数据补齐（9B 的 RUN-0056、0.8B 秩 128 的 RUN-0054）之后重跑同一个命令即可：目录出现就自动纳入；
  没有时照常出图，README 与图注写明缺（missing）。
- 口径只在一处定义：§13 的 8 个 T 文本读数、L/P/R 的算法、标题读数的取法都 import 自
  scripts/mgr/interference.py；E2 的标题指标取法 import 自 scripts/mgr/e2_summary.py。本文件只放
  "数据在哪"（路径）与"怎么画"（英文标签、配色、尺寸）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]


def _load_script(alias: str, rel: str):
    spec = importlib.util.spec_from_file_location(alias, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ITF = _load_script("_paperfig_interference", "scripts/mgr/interference.py")  # §13 的定义（L_ITEMS、size_block、summary…）
E2S = _load_script("_paperfig_e2_summary", "scripts/mgr/e2_summary.py")  # E2 标题指标（headline_of）

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------------------------------------
# 数据在哪（相对 --root；只有路径与命名，没有读数）
# ---------------------------------------------------------------------------------------------

QWEN = (("0.8B", "qwen3.5-0.8b"), ("2B", "qwen3.5-2b"), ("4B", "qwen3.5-4b"), ("9B", "qwen3.5-9b"))

# E1（G1、RUN-0027；结构强制与自己的框：RUN-0033，D72-3、D81-2）
E1_MAIN = "experiments/E1/0027_e1_main"
E1_POSTHOC2 = "experiments/E1/0033_e1_posthoc2"
E1_OWNBOX = E1_POSTHOC2 + "/ownbox.json"
# 其他家族：InternVL3.5 作参照画进同一张图（都能作答，有效率 1.0）；Gemma 4 不画（定位以拒答为主，见 README）
E1_REFERENCE = (("InternVL3.5-2B", "internvl3.5-2b-hf"), ("InternVL3.5-4B", "internvl3.5-4b-hf"))
E1_EXCLUDED = (("Gemma-4-E2B", "gemma-4-e2b-it"), ("Gemma-4-E4B", "gemma-4-e4b-it"))
# (键, 组, 坐标轴标签, 来源, 任务目录或 ownbox 行, 指标名)
E1_READOUTS = (
    ("grounding", "where", "Grounding\nAcc@0.5", "main", "refexp_box", "acc@0.5"),
    ("own_box", "where", "Own box\ncIoU", "ownbox", "own_box", "cIoU"),
    ("polygon", "outline", "Polygon\ncIoU", "main", "refexp_poly", "cIoU"),
    ("grid16", "outline", "Grid 16×16\ngIoU", "main", "grid16", "gIoU"),
    ("poly_structured", "outline", "Structured\npolygon cIoU", "ownbox", "poly_structured", "cIoU"),
)
E1_PAIRED = (("poly_structured", "structured, ≥3 vertices"), ("poly_structured_min8", "structured, ≥8 vertices"))

# E2（G2、RUN-0029）：逐探针的原始 metrics.json（带每层的 CI）；汇总 JSON 只用来核对
E2_RAW = "experiments/E2/0029_e2_qwen/results_0029/0029_e2_qwen"
E2_SUMMARY = "experiments/E2/0029_e2_qwen/e2_summary.json"
E2_TASKS = (  # (任务, 面板标题, 纵轴标签, 数据集说明)
    ("depth", "Depth", "AbsRel ↓", "NYUv2"),
    ("normals", "Surface normals", "Mean angular error (°) ↓", "NYUv2"),
    ("semseg", "Semantic segmentation", "mIoU ↑", "ADE20K val"),
    ("refseg", "Referring segmentation", "cIoU ↑", "RefCOCO val, instruction first"),
)

# E3（预注册 §12、§13；D93-1：效应都在机内算，表里注明机器）
INTERFERENCE_JSON = "docs/analysis/e3_interference_by_size.json"
E3_SIZES = (
    {"size": "0.8B", "model": "qwen3.5-0.8b", "machine": "hpcgpu07", "seeds": tuple(ITF.SEEDS), "runs": "RUN-0050",
     "dirs": ("experiments/E3/0050_hpc_eval_08b_hpcgpu07",),
     "base": ("experiments/E3/0050_hpc_eval_08b_hpcgpu07/qwen3.5-0.8b",)},
    {"size": "2B", "model": "qwen3.5-2b", "machine": "B200", "seeds": tuple(ITF.SEEDS),
     "runs": "RUN-0038, RUN-0039, RUN-0046; base: RUN-0027, RUN-0030, RUN-0033",
     "dirs": ("experiments/E3/0038_e3_eval_2b_s0", "experiments/E3/0039_e3b_diag_2b_s0", "experiments/E3/0046_e3_eval_2b_h1"),
     "base": ("experiments/E1/0027_e1_main/qwen3.5-2b", "experiments/E1/0030_e1_posthoc/qwen3.5-2b",
              "experiments/E1/0033_e1_posthoc2/qwen3.5-2b")},
    {"size": "4B", "model": "qwen3.5-4b", "machine": "hpcgpu06", "seeds": tuple(ITF.SEEDS), "runs": "RUN-0052",
     "dirs": ("experiments/E3/0052_hpc_eval_4b_hpcgpu06",),
     "base": ("experiments/E3/0052_hpc_eval_4b_hpcgpu06/qwen3.5-4b",)},
    # 9B：每个变体 1 个种子（E3 预注册 §3），RUN-0056
    {"size": "9B", "model": "qwen3.5-9b", "machine": "hpcgpu06", "seeds": (0,), "runs": "RUN-0056",
     "dirs": ("experiments/E3/0056_hpc_9b_eval_hpcgpu06",),
     "base": ("experiments/E3/0056_hpc_9b_eval_hpcgpu06/qwen3.5-9b",)},
)
# 0.8B 秩 128（§13a 及 D99-1 的修订；RUN-0054）：C3 与 E3b+C3 都是 r128（变体名仍是 c3、e3b_c3，靠目录区分）；
# P 用 E3b（r32，RUN-0050）减 E3b+C3（r128）。
E3_R128 = {"size": "0.8B", "label": "0.8B r128", "model": "qwen3.5-0.8b", "machine": "hpcgpu07", "seeds": tuple(ITF.SEEDS),
           "runs": "RUN-0054 (E3b r32 from RUN-0050)",
           "dirs": ("experiments/E3/0054_r128_eval_hpcgpu07",), "e3b_dirs": ("experiments/E3/0050_hpc_eval_08b_hpcgpu07",)}

E3_ARMS = (("base", "Base"), ("c3", "C3"), ("e3b_c3", "E3b+C3"), ("e3b", "E3b"))
E3_GRID = (("refseg", "cIoU", "RefSeg", "cIoU ↑", 100.0), ("depth", "abs_rel", "Depth", "AbsRel ↓", 1.0))
# §13 的 8 个 T 文本读数：任务清单取自 interference.L_ITEMS；这里只给英文表头
TTEXT_LABELS = {
    "refexp_box": ("Ground.", "Acc@0.5 ↑"),
    "refexp_poly": ("Polygon", "cIoU ↑"),
    "count": ("Count", "Acc ↑"),
    "ovd": ("Detect.", "F1@0.5 ↑"),
    "rel_depth_nyu__letter_swap": ("NYU pairs", "swap ↑"),
    "rel_depth_blink__debiased": ("BLINK", "debiased ↑"),
    "spatial_cvbench__debiased": ("CV-Bench", "debiased ↑"),
    "spatial_embspatial__debiased": ("EmbSpatial", "debiased ↑"),
}
INTERFERENCE_PANELS = (  # (量, 纵轴标签, 标题)
    ("L", r"$L$ (points)", "(a) Language"),
    ("P_seg", r"$P_\mathrm{seg}$ (ΔcIoU)", "(b) Pixel: RefSeg"),
    ("P_depth", r"$P_\mathrm{depth}$ (ΔAbsRel)", "(c) Pixel: depth"),
    ("R_seg", r"$R_\mathrm{seg}$ (fraction)", "(d) Relative: RefSeg"),
)

# ---------------------------------------------------------------------------------------------
# 怎么画：尺寸、字体、配色（dataviz 参考调色板；validate_palette.js 在白底上校验通过）
# ---------------------------------------------------------------------------------------------

SINGLE_COL_IN = 3.3
DOUBLE_COL_IN = 6.9
DPI = 300
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"  # 分类槽 1、2（两色全对 CVD ΔE 24.7、常规 33.6）
SIZE_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # Qwen 0.8B→9B，有序单色阶（--ordinal 通过）
REF_RAMP = ("#f89c7b", "#bb4714")  # InternVL3.5 2B→4B，第二个有序色阶取槽 2 的色相（--ordinal 通过）
INK, INK_2, MUTED, GRIDLINE, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f3f2ee"
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": DPI,
    "figure.dpi": 100,
}

MISSING = "missing"
SEED_OFFSET, EXTRA_OFFSET = 0.93, 1.2  # 干扰图：各种子相对平均值、秩 128 相对 0.8B 的横向错开（对数轴上的倍数）


# ---------------------------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------------------------


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _rel(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _point(v, ci=None, n=None, source=None) -> dict:
    ci = ci or {}
    return {"value": v, "low": ci.get("low"), "high": ci.get("high"), "level": ci.get("level"),
            "n_resamples": ci.get("n_resamples"), "n": n, "source": source}


def e1_main_headline(root: Path, model: str, task: str, name: str):
    """E1 metrics.json 的标题读数（严格读数，scope = all），按名字取，带 95% CI。"""
    f = root / E1_MAIN / model / task / "metrics.json"
    d = _load(f)
    if d is None:
        return None
    for h in d.get("headline") or []:
        if isinstance(h, dict) and h.get("name") == name and h.get("scope", "all") == "all":
            return _point(h.get("value"), h.get("ci"), h.get("n"), _rel(root, f))
    return None


def _ownbox_task(ownbox: dict | None, model: str, task: str = "refexp_poly"):
    for m in (ownbox or {}).get("models") or []:
        if m.get("model") == model:
            return (m.get("tasks") or {}).get(task)
    return None


def _ownbox_row(task: dict | None, readout: str):
    for r in (task or {}).get("rows") or []:
        if r.get("readout") == readout:
            return r
    return None


def collect_e1(root: Path) -> dict:
    ownbox_path = root / E1_OWNBOX
    ownbox = _load(ownbox_path)
    models = [(lab, m, "qwen") for lab, m in QWEN] + [(lab, m, "reference") for lab, m in E1_REFERENCE] + \
             [(lab, m, "excluded") for lab, m in E1_EXCLUDED]
    out = {"models": {}, "paired": {}, "missing": [], "checks": [], "n_paired_ci_excluding_zero": 0, "n_paired": 0}
    for label, model, group in models:
        box = _load(root / E1_MAIN / model / "refexp_box" / "metrics.json")
        rec = {"label": label, "group": group, "readouts": {},
               "grounding_valid_rate": ((box or {}).get("valid") or {}).get("rate")}  # 有效 = 解析成功且不是拒答
        task = _ownbox_task(ownbox, model)
        for key, _fam, _lab, src, what, metric in E1_READOUTS:
            if src == "main":
                pt = e1_main_headline(root, model, what, metric)
            else:
                row = _ownbox_row(task, what)
                st = ((row or {}).get("strict") or {}).get(metric)
                pt = None if not st else _point(st.get("value"), st.get("ci"), st.get("n"),
                                                f"{_rel(root, ownbox_path)} :: {model}/refexp_poly/{what}/strict/{metric}")
            if pt is None or pt["value"] is None:
                if group != "excluded" and not (group == "reference" and key == "poly_structured"):
                    out["missing"].append(f"E1 {label}: {key}")
                pt = None
            rec["readouts"][key] = pt
        out["models"][model] = rec
        if group == "qwen":
            # 核对：ownbox 里的结构强制读数 = RUN-0033 该运行自己的 metrics.json
            ps = rec["readouts"].get("poly_structured")
            f = root / E1_POSTHOC2 / model / "refexp_poly__poly_structured" / "metrics.json"
            d = _load(f)
            if ps and d:
                own = next((h for h in d.get("headline") or [] if h.get("name") == "cIoU"), None)
                if own is None or abs(own["value"] - ps["value"]) > 1e-9:
                    out["checks"].append(f"不一致：{model} 结构强制多边形 cIoU，ownbox.json {ps['value']} 对 {_rel(root, f)} "
                                         f"{None if own is None else own['value']}")
                else:
                    out["checks"].append(f"一致：{model} 结构强制多边形 cIoU（ownbox.json = {_rel(root, f)}）")
            pr = {}
            for readout, _lab in E1_PAIRED:
                row = _ownbox_row(task, readout)
                pdiff = ((row or {}).get("paired_diff") or {}).get("cIoU")
                if row and row.get("minus") == "own_box" and pdiff:
                    pt = _point(pdiff.get("value"), pdiff.get("ci"), pdiff.get("n"),
                                f"{_rel(root, ownbox_path)} :: {model}/refexp_poly/{readout}/paired_diff/cIoU（减 own_box）")
                    pr[readout] = pt
                    out["n_paired"] += 1
                    if pt["low"] is not None and pt["high"] is not None and (pt["low"] > 0 or pt["high"] < 0):
                        out["n_paired_ci_excluding_zero"] += 1
                else:
                    pr[readout] = None
                    out["missing"].append(f"E1 {label}: paired {readout} − own_box")
            out["paired"][model] = pr
    return out


def collect_e2(root: Path) -> dict:
    out = {"models": {}, "missing": [], "checks": []}
    summ = _load(root / E2_SUMMARY)
    for size, model in QWEN:
        mrec = {"size": size, "tasks": {}}
        for task, *_ in E2_TASKS:
            f = root / E2_RAW / model / task / "metrics.json"
            m = _load(f)
            if m is None:
                out["missing"].append(f"E2 {size}: {task}")
                continue
            metric, better = E2S.headline_of(m)
            layers = m.get("layers")
            if not layers:
                layers = sorted(int(k.split("/")[0][1:]) for k in m["probes"] if re.fullmatch(r"L\d+/linear", k))
            n_last = max(layers)
            rec = {"metric": metric, "better": better, "n_last": n_last, "layers": [], "frac": [], "value": [], "low": [],
                   "high": [], "n_eval": m.get("n_eval"), "source": _rel(root, f)}
            for li in layers:
                p = (m.get("probes") or {}).get(f"L{li}/linear")
                if p is None:
                    continue
                ev = p.get("eval") or {}
                ci = ev.get(metric + "_ci") or {}
                rec["layers"].append(li)
                rec["frac"].append(li / n_last if n_last else 0.0)
                rec["value"].append(ev.get(metric))
                rec["low"].append(ci.get("low"))
                rec["high"].append(ci.get("high"))
            mrec["tasks"][task] = rec
            # 核对：逐层读数 = e2_summary.json 的 linear 曲线（管理员脚本的汇总）
            curve = (((((summ or {}).get("models") or {}).get(model) or {}).get(task) or {}).get("kinds") or {}).get("linear", {}).get("curve")
            if curve is not None:
                diff = [li for li, v in zip(rec["layers"], rec["value"]) if curve.get(f"L{li}") is None or abs(curve[f"L{li}"] - v) > 1e-12]
                extra = sorted(set(curve) - {f"L{li}" for li in rec["layers"]})
                out["checks"].append(("一致" if not diff and not extra else "不一致") +
                                     f"：{model}/{task} 逐层读数对 {E2_SUMMARY}" + (f"（不同的层 {diff}，多出的层 {extra}）" if diff or extra else ""))
        out["models"][model] = mrec
    return out


def _abs_dirs(root: Path, dirs) -> list[str]:
    return [str(root / d) for d in dirs]


def _merge_checks(root: Path, spec: dict) -> list[dict]:
    """评测前合并核对（D50、D67）：逐臂、逐种子、逐通道找第一个 merge_check.json（与 interference.find 同样的先到先得）。"""
    out = []
    for arm in ("c3", "e3b_c3", "e3b"):
        for s in spec["seeds"]:
            for channel, rel in (("t_text", f"t_text/{arm}/seed{s}/{spec['model']}/merge_check.json"),
                                 ("grid", f"eval/{arm}/{spec['model']}/seed{s}/merge_check.json")):
                for d in spec["dirs"]:
                    f = root / d / rel
                    if f.is_file():
                        mc = _load(f)
                        out.append({"arm": arm, "seed": s, "channel": channel, "pass": bool(mc.get("pass")),
                                    "criteria_version": mc.get("criteria_version"), "gpu": mc.get("gpu"),
                                    "failures": mc.get("failures"), "path": _rel(root, f)})
                        break
    return out


def _exact_rows_text(failures) -> str:
    """把判据 v1 的 failures 原文换成英文短句（数字从原文取）。"""
    txt = json.dumps(failures, ensure_ascii=False)
    m = re.search(r"exact_rows ([0-9.]+) < ([0-9.]+)", txt)
    if m:
        return f"restricted decoding vs. generate_batch exact-row agreement {float(m.group(1)):.3f} < {float(m.group(2)):.2f}"
    return txt


def e3_present(root: Path, spec: dict) -> bool:
    return any((root / d).is_dir() for d in spec["dirs"])


def collect_e3(root: Path) -> dict:
    out = {"sizes": [], "missing": [], "rejected": [], "gaps": [], "ttext_tasks": [t for _, t in ITF.L_ITEMS]}
    for spec in E3_SIZES:
        rec = {"size": spec["size"], "model": spec["model"], "machine": spec["machine"], "runs": spec["runs"],
               "dirs": list(spec["dirs"]), "base_dirs": list(spec["base"]), "seeds": list(spec["seeds"]),
               "present": e3_present(root, spec), "rows": {}, "gpus": []}
        if not rec["present"]:
            out["missing"].append(f"E3 {spec['size']}: {spec['runs']} 的评测目录 {', '.join(spec['dirs'])} 还没有")
            out["sizes"].append(rec)
            continue
        dirs = _abs_dirs(root, spec["dirs"])
        base = _abs_dirs(root, spec["base"])
        mcs = _merge_checks(root, spec)
        rec["merge_checks"] = mcs
        rec["gpus"] = sorted({m["gpu"] for m in mcs if m.get("gpu")})
        for m in mcs:
            if not m["pass"]:
                out["rejected"].append({"size": spec["size"], "machine": spec["machine"], **m,
                                        "reason_en": _exact_rows_text(m["failures"])})
        rejected = {(m["arm"], m["seed"], m["channel"]) for m in mcs if not m["pass"]}
        for arm, _label in E3_ARMS:
            row = {}
            for _name, task in ITF.L_ITEMS:
                if arm == "base":
                    v = ITF.headline(ITF.find(base, f"{task}/metrics.json"))
                    row[task] = {"seeds": {}, "single": v, "mean": v, "sd": None, "n": 0 if v is None else 1}
                    if v is None:
                        out["gaps"].append(f"E3 {spec['size']} base: {task}（{', '.join(spec['base'])} 里没有）")
                    continue
                vals = {s: ITF.ttext(dirs, arm, s, spec["model"], task) for s in spec["seeds"]}
                vals = {s: v for s, v in vals.items() if v is not None}
                mean, sd = ITF.summary(vals)
                row[task] = {"seeds": vals, "mean": mean, "sd": sd, "n": len(vals)}
                if arm in ("c3", "e3b_c3"):
                    for s in spec["seeds"]:
                        if s not in vals:
                            why = "rejected by merge check" if (arm, s, "t_text") in rejected else "no result file"
                            out["gaps"].append(f"E3 {spec['size']} {arm} seed {s}: {task}（{why}）")
            for task, key, *_ in E3_GRID:
                if arm in ("base", "c3"):
                    row[task] = {"seeds": {}, "mean": None, "sd": None, "n": 0, "na": True}
                    continue
                vals = {s: ITF.grid(dirs, arm, s, spec["model"], task, key) for s in spec["seeds"]}
                vals = {s: v for s, v in vals.items() if v is not None}
                mean, sd = ITF.summary(vals)
                row[task] = {"seeds": vals, "mean": mean, "sd": sd, "n": len(vals)}
                for s in spec["seeds"]:
                    if s not in vals:
                        why = "rejected by merge check" if (arm, s, "grid") in rejected else "no result file"
                        out["gaps"].append(f"E3 {spec['size']} {arm} seed {s}: {task} structured grid（{why}）")
                    elif (arm, s, "grid") in rejected:
                        out["gaps"].append(f"不一致：E3 {spec['size']} {arm} seed {s} 的网格核对没过，却有 {task} 的读数")
            rec["rows"][arm] = row
        out["sizes"].append(rec)
    return out


def _norm_block(b: dict) -> dict:
    nb = dict(b)
    for q in ("L", "P_seg", "P_depth", "R_seg", "R_depth", "grid16"):
        nb[q] = {int(k): v for k, v in (b.get(q) or {}).items()}
    for q in ("L_items", "free_items"):
        nb[q] = {name: {int(k): v for k, v in d.items()} for name, d in (b.get(q) or {}).items()}
    return nb


def _has_readings(b: dict) -> bool:
    return any(b.get(q) for q in ("L", "P_seg", "P_depth"))


def _r128_block(root: Path) -> dict | None:
    """秩 128：C3、E3b+C3 只从 RUN-0054 的目录取（不回退到 r32），E3b 只从 RUN-0050 取；用符号链接拼一个视图，
    再交给 interference.size_block（同一套算法）。"""
    spec = E3_R128
    r128 = [root / d for d in spec["dirs"] if (root / d).is_dir()]
    if not r128:
        return None
    with tempfile.TemporaryDirectory(prefix="paperfig_r128_") as tmp:
        view = Path(tmp)
        links = {"t_text/c3": [d / "t_text" / "c3" for d in r128], "t_text/e3b_c3": [d / "t_text" / "e3b_c3" for d in r128],
                 "eval/e3b_c3": [d / "eval" / "e3b_c3" for d in r128],
                 "eval/e3b": [root / d / "eval" / "e3b" for d in spec["e3b_dirs"]]}
        for rel, cands in links.items():
            src = next((c for c in cands if c.is_dir()), None)
            if src is not None:
                (view / rel).parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src, view / rel)
        b = ITF.size_block(spec["label"], spec["machine"], spec["model"], [str(view)])
    b = _norm_block(b)
    b["rank"] = 128
    b["x_size"] = spec["size"]
    b["source"] = (f"computed with scripts/mgr/interference.py size_block: C3/E3b+C3 from {', '.join(spec['dirs'])}; "
                   f"E3b from {', '.join(spec['e3b_dirs'])}")
    return b


def collect_interference(root: Path) -> dict:
    out = {"blocks": [], "missing": [], "checks": [], "json": INTERFERENCE_JSON, "trend": None}
    j = _load(root / INTERFERENCE_JSON)
    if j is None:
        out["missing"].append(f"干扰：{INTERFERENCE_JSON} 不在")
        j = {"blocks": []}
    out["trend"] = j.get("trend")
    out["relative_trend_descriptive"] = j.get("relative_trend_descriptive")
    in_json = {}
    for b in j.get("blocks") or []:
        nb = _norm_block(b)
        nb["source"] = INTERFERENCE_JSON
        nb["x_size"] = nb["size"]
        in_json[nb["size"]] = nb
    for spec in E3_SIZES:
        size = spec["size"]
        present = e3_present(root, spec)
        if size in in_json:
            b = in_json[size]
            if b.get("machine") != spec["machine"]:
                out["checks"].append(f"不一致：{size} 的机器，JSON 是 {b.get('machine')}，本脚本登记的是 {spec['machine']}")
            if present:  # 核对：用登记的目录重算一遍，应与 JSON 逐种子相同
                nb = _norm_block(ITF.size_block(size, spec["machine"], spec["model"], _abs_dirs(root, spec["dirs"])))
                bad = [q for q in ("L", "P_seg", "P_depth", "R_seg", "R_depth")
                       if nb[q].keys() != b[q].keys() or any(abs(nb[q][s] - b[q][s]) > 1e-9 for s in b[q])]
                out["checks"].append(("一致" if not bad else "不一致") + f"：{size} 用 {', '.join(spec['dirs'])} 重算的 L/P/R 对 "
                                     f"{INTERFERENCE_JSON}" + (f"（不同：{bad}）" if bad else ""))
            out["blocks"].append(b)
        elif present:
            b = _norm_block(ITF.size_block(size, spec["machine"], spec["model"], _abs_dirs(root, spec["dirs"])))
            b["x_size"] = size
            b["source"] = f"computed with scripts/mgr/interference.py size_block over {', '.join(spec['dirs'])}"
            if _has_readings(b):
                out["blocks"].append(b)
                for q in ("L", "P_seg", "P_depth"):
                    if not b[q]:
                        out["missing"].append(f"干扰 {size}: {q}（{spec['runs']} 还没有两臂配对的读数）")
            else:
                out["missing"].append(f"干扰 {size}: {spec['runs']} 的目录在，但还没有两臂配对的读数")
        else:
            out["missing"].append(f"干扰 {size}: {spec['runs']} 的评测目录 {', '.join(spec['dirs'])} 还没有")
    rb = _r128_block(root)
    if rb is None:
        out["missing"].append(f"干扰 {E3_R128['label']}: {E3_R128['runs'].split(' ')[0]} 的评测目录 {', '.join(E3_R128['dirs'])} 还没有")
    elif _has_readings(rb):
        out["blocks"].append(rb)
        for q in ("L", "P_seg", "P_depth"):
            if not rb[q]:
                out["missing"].append(f"干扰 {E3_R128['label']}: {q}（还没有两臂配对的读数）")
    else:
        out["missing"].append(f"干扰 {E3_R128['label']}: 目录在，但还没有两臂配对的读数")
    out["blocks"].sort(key=lambda b: (size_value(b["x_size"]), b.get("rank", 32)))
    for b in out["blocks"]:
        b["means"] = {}
        for q, *_ in INTERFERENCE_PANELS + (("R_depth",),):
            m, sd = ITF.summary(b.get(q) or {})
            b["means"][q] = {"mean": m, "sd": sd, "n": len(b.get(q) or {})}
    return out


def size_value(label: str) -> float:
    """名义参数量（十亿），取自尺寸标签（"0.8B" → 0.8）。"""
    m = re.match(r"([0-9.]+)B", label)
    if not m:
        raise ValueError(f"看不懂尺寸标签：{label!r}")
    return float(m.group(1))


# ---------------------------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------------------------


def _save(fig, out: Path, name: str) -> list[Path]:
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    fig.savefig(paths[0], metadata={"CreationDate": None, "Producer": None, "Creator": None})
    fig.savefig(paths[1], dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return paths


def _yerr(points):
    lo, hi = [], []
    for p in points:
        if p is None or p["value"] is None or p["low"] is None or p["high"] is None:
            lo.append(0.0)
            hi.append(0.0)
        else:
            lo.append(max(p["value"] - p["low"], 0.0))
            hi.append(max(p["high"] - p["value"], 0.0))
    return [lo, hi]


def fig_e1(e1: dict, out: Path) -> list[Path]:
    fig = plt.figure(figsize=(DOUBLE_COL_IN, 2.35), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0])
    ax, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    series = [(m, f"Qwen3.5-{lab}", SIZE_RAMP[i]) for i, (lab, m) in enumerate(QWEN)] + \
             [(m, lab, REF_RAMP[i]) for i, (lab, m) in enumerate(E1_REFERENCE)]
    n_q, n_r = len(QWEN), len(E1_REFERENCE)
    w = 0.12
    offsets = [(i - (n_q + n_r) / 2 + 0.5) * w + (0.04 if i >= n_q else -0.0) for i in range(n_q + n_r)]
    groups = list(range(len(E1_READOUTS)))
    outline = [g for g, r in zip(groups, E1_READOUTS) if r[1] == "outline"]
    ax.axvspan(min(outline) - 0.5, max(outline) + 0.5, color=BAND, zorder=0, lw=0)
    for i, (model, label, color) in enumerate(series):
        rec = e1["models"].get(model) or {"readouts": {}}
        pts = [rec["readouts"].get(r[0]) for r in E1_READOUTS]
        xs = [g + offsets[i] for g in groups]
        vals = [p["value"] if p else 0.0 for p in pts]
        ax.bar(xs, vals, width=w * 0.92, color=color, label=label, zorder=2, linewidth=0)
        ax.errorbar([x for x, p in zip(xs, pts) if p], [p["value"] for p in pts if p],
                    yerr=_yerr([p for p in pts if p]), fmt="none", ecolor=INK_2, elinewidth=0.6, capsize=1.0, capthick=0.6, zorder=3)
        for x, p in zip(xs, pts):
            if p is None:
                ax.text(x, 0.012, "n/a", rotation=90, ha="center", va="bottom", fontsize=5.5, color=MUTED, zorder=3)
    # 地板值：整组都 < 0.02 时，在组上写出来（零高度的柱子看不见）
    for g, r in zip(groups, E1_READOUTS):
        vals = [((e1["models"].get(m) or {"readouts": {}})["readouts"].get(r[0]) or {}).get("value") for m, *_ in series[:n_q]]
        vals = [v for v in vals if v is not None]
        if vals and max(vals) < 0.02:
            ax.text(g + sum(offsets[:n_q]) / n_q, 0.035, "≈0" if max(vals) > 0 else "0", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(groups)
    ax.set_xticklabels([r[2] for r in E1_READOUTS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (strict; 0–1)")
    where = [g for g, r in zip(groups, E1_READOUTS) if r[1] == "where"]
    ax.text(sum(where) / len(where), 0.985, "knows where", ha="center", va="top", fontsize=7, color=INK, fontweight="bold")
    ax.text(sum(outline) / len(outline), 0.985, "draws the outline", ha="center", va="top", fontsize=7, color=INK, fontweight="bold")
    ax.legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.92), handlelength=1.0, columnspacing=0.8, handletextpad=0.4,
              labelspacing=0.3)
    ax.set_title("(a) Zero-shot readouts", loc="left")
    # (b) 配对差
    xs = list(range(n_q))
    for j, ((readout, label), color, marker) in enumerate(zip(E1_PAIRED, (SERIES_1, SERIES_2), ("o", "s"))):
        pts = [e1["paired"].get(m, {}).get(readout) for _lab, m in QWEN]
        px = [x + (j - 0.5) * 0.22 for x, p in zip(xs, pts) if p]
        pp = [p for p in pts if p]
        if pp:
            axd.errorbar(px, [p["value"] for p in pp], yerr=_yerr(pp), fmt=marker, ms=4, color=color, ecolor=color,
                         elinewidth=0.8, capsize=1.5, capthick=0.8, label=label, zorder=3, mec="white", mew=0.5)
    axd.axhline(0, color=INK_2, lw=0.6, ls="--", zorder=1)
    axd.set_xticks(xs)
    axd.set_xticklabels([lab for lab, _ in QWEN])
    axd.set_xlim(-0.5, n_q - 0.5)
    lo = min([p["low"] for d in e1["paired"].values() for p in d.values() if p and p["low"] is not None] + [-0.05])
    axd.set_ylim(lo * 1.15, 0.05)
    axd.set_xlabel("Qwen3.5")
    axd.set_ylabel("ΔcIoU vs. own box (paired)")
    axd.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), handletextpad=0.3, borderaxespad=0.2)
    axd.set_title("(b) Structured polygon − own box", loc="left")
    return _save(fig, out, "fig_e1_gap")


def fig_e2(e2: dict, out: Path) -> list[Path]:
    fig, axes = plt.subplots(1, len(E2_TASKS), figsize=(DOUBLE_COL_IN, 2.0), layout="constrained")
    for ax, (task, title, ylabel, _ds) in zip(axes, E2_TASKS):
        for i, (size, model) in enumerate(QWEN):
            rec = (e2["models"].get(model) or {"tasks": {}})["tasks"].get(task)
            if not rec or not rec["value"]:
                continue
            c = SIZE_RAMP[i]
            x, y = rec["frac"], rec["value"]
            if all(v is not None for v in rec["low"] + rec["high"]):
                ax.fill_between(x, rec["low"], rec["high"], color=c, alpha=0.18, lw=0, zorder=1)
            ax.plot(x, y, color=c, lw=1.1, zorder=2, label=f"Qwen3.5-{size}")
            ax.plot([x[0]], [y[0]], marker="o", ms=3.6, mfc="white", mec=c, mew=0.9, zorder=4, ls="none", clip_on=False)
            ax.plot([x[-1]], [y[-1]], marker="s", ms=3.4, mfc=c, mec="white", mew=0.4, zorder=4, ls="none", clip_on=False)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        ax.grid(True, axis="both")
    handles = [Line2D([], [], color=SIZE_RAMP[i], lw=1.2, label=f"Qwen3.5-{s}") for i, (s, _m) in enumerate(QWEN)]
    handles += [Line2D([], [], marker="o", ms=3.6, mfc="white", mec=INK_2, ls="none", label="layer 0"),
                Line2D([], [], marker="s", ms=3.4, mfc=INK_2, mec="white", ls="none", label="last layer"),
                Patch(color=INK_2, alpha=0.18, lw=0, label="95% CI")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), handlelength=1.4, columnspacing=1.0)
    fig.supxlabel("Relative depth (layer index / number of layers)", fontsize=7)
    return _save(fig, out, "fig_e2_layers")


def fig_interference(itf: dict, out: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COL_IN, 2.9), layout="constrained")
    main = [b for b in itf["blocks"] if b.get("rank") is None]
    extra = [b for b in itf["blocks"] if b.get("rank") is not None]
    xs_all = [size_value(s) for s, _ in QWEN]
    for ax, (q, ylabel, title) in zip(axes.flat, INTERFERENCE_PANELS):
        mx, my = [], []
        for b in main:
            x = size_value(b["x_size"])
            vals = b.get(q) or {}
            if vals:  # 各种子的点略向左错开，不被平均值的点盖住
                ax.plot([x * SEED_OFFSET] * len(vals), list(vals.values()), ls="none", marker="o", ms=3.0, mfc="none", mec=SERIES_1,
                        mew=0.7, alpha=0.9, zorder=3)
            m = b["means"][q]["mean"]
            if m is not None:
                mx.append(x)
                my.append(m)
        if mx:
            ax.plot(mx, my, color=SERIES_1, lw=1.1, marker="o", ms=4.2, mec="white", mew=0.5, zorder=4)
        for b in extra:
            x = size_value(b["x_size"]) * EXTRA_OFFSET
            vals = b.get(q) or {}
            if vals:
                ax.plot([x * SEED_OFFSET] * len(vals), list(vals.values()), ls="none", marker="*", ms=4.0, mfc="none", mec=SERIES_2,
                        mew=0.6, zorder=3)
            m = b["means"][q]["mean"]
            if m is not None:
                ax.plot([x], [m], ls="none", marker="*", ms=7.0, color=SERIES_2, mec="white", mew=0.4, zorder=5)
        ax.axhline(0, color=INK_2, lw=0.6, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(xs_all)
        ax.set_xticklabels([s for s, _ in QWEN])
        ax.minorticks_off()
        ax.set_xlim(min(xs_all) / 1.35, max(xs_all) * 1.35)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    fig.supxlabel("Qwen3.5 parameters (log scale)", fontsize=7)
    handles = [Line2D([], [], marker="o", ms=3.0, mfc="none", mec=SERIES_1, ls="none", label="seed"),
               Line2D([], [], marker="o", ms=4.2, color=SERIES_1, mec="white", lw=1.1, label="seed mean (LoRA r32)")]
    if extra:
        handles.append(Line2D([], [], marker="*", ms=7.0, color=SERIES_2, mec="white", ls="none", label="0.8B, LoRA r128"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), handletextpad=0.3, columnspacing=0.9)
    return _save(fig, out, "fig_interference_size")


# ---------------------------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------------------------


def _fmt_cell(cell: dict, scale: float, nd: int, n_expected: int, latex: bool) -> str:
    if cell.get("na"):
        return "n/a"
    mean, sd, n = cell.get("mean"), cell.get("sd"), cell.get("n", 0)
    if mean is None:
        return "--" if latex else "—"
    s = f"{mean * scale:.{nd}f}"
    if sd is not None:
        s = f"{s}$\\pm${sd * scale:.{nd}f}" if latex else f"{s} ± {sd * scale:.{nd}f}"
    if 0 < n < n_expected and "single" not in cell:
        mark = f"$^{{n={n}}}$" if latex else f" (n={n})"
        s += mark
    return s


def _tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$",
           "−": r"$-$", "±": r"$\pm$", "↑": r"$\uparrow$", "↓": r"$\downarrow$", "°": r"$^\circ$", "Δ": r"$\Delta$", "≈": r"$\approx$",
           "<": r"$<$", ">": r"$>$"}
    return "".join(rep.get(ch, ch) for ch in s)


def e3_table_notes(e3: dict) -> list[str]:
    notes = []
    sizes = [s for s in e3["sizes"] if s["present"]]
    seed_sets = []
    for spec in E3_SIZES:
        txt = f"seeds {spec['seeds'][0]}–{spec['seeds'][-1]}" if len(spec["seeds"]) > 1 else f"seed {spec['seeds'][0]} only"
        if not seed_sets or seed_sets[-1][1] != txt:
            seed_sets.append(([spec["size"]], txt))
        else:
            seed_sets[-1][0].append(spec["size"])
    notes.append("Seed mean ± sample standard deviation over " +
                 "; ".join(f"{txt} ({', '.join(sz)})" for sz, txt in seed_sets) +
                 " (E3 preregistration §3); n=k marks cells with fewer seeds than planned. "
                 "Base rows are single deterministic evaluations.")
    notes.append(f"T-text columns are the {len(ITF.L_ITEMS)} readouts of the interference analysis (§13), ×100: strict grounding Acc@0.5, "
                 "polygon cIoU, counting accuracy and COCO detection F1@0.5; NYU depth pairs with letter swap; "
                 "position-debiased BLINK, CV-Bench and EmbSpatial. RefSeg cIoU (×100) and depth AbsRel use structure-forced "
                 "decoding of the pixel-token grid, single turn, guided upsampling.")
    mach = []
    for s in sizes:
        g = f" ({', '.join(s['gpus'])})" if s["gpus"] else ""
        mach.append(f"{s['size']}: {s['machine']}{g}")
    notes.append("Machines (all effects computed within one machine, D93-1; each base row was evaluated on the same machine "
                 "as its arms): " + "; ".join(mach) + ".")
    if e3["rejected"]:
        rj = "; ".join(f"{r['size']} {dict(E3_ARMS)[r['arm']]} seed {r['seed']} ({'grid' if r['channel'] == 'grid' else 'T-text'}"
                       f", {r['machine']}): {r['reason_en']}" for r in e3["rejected"])
        notes.append("Seeds without readings because the pre-evaluation merge check (criteria v1) rejected them (N37, kept as "
                     "v1 per D102-2): " + rj + ".")
    ttext_e3b = [(s["size"], sorted(c["seeds"]) ) for s in sizes for c in [s["rows"].get("e3b", {}).get(ITF.L_ITEMS[0][1], {})] if c.get("seeds")]
    if ttext_e3b:
        notes.append("E3b T-text readouts exist only for " + ", ".join(f"{sz} seed {','.join(map(str, sd))}" for sz, sd in ttext_e3b) +
                     " (grid-only LoRA; T-text is not part of the E3b protocol elsewhere); other E3b T-text cells are --.")
    notes.append("n/a: Base and C3 have no pixel-token channel.")
    miss = [s for s in e3["sizes"] if not s["present"]]
    if miss:
        notes.append("Missing (not yet available): " + "; ".join(f"Qwen3.5-{s['size']} ({s['runs']})" for s in miss) + ".")
    return notes


def tab_e3(e3: dict, out: Path) -> list[Path]:
    tt = [t for _, t in ITF.L_ITEMS]
    present = [s for s in e3["sizes"] if s["present"]]
    notes = e3_table_notes(e3)
    # LaTeX（booktabs）
    caption = ("\\caption{E3 comparison on Qwen3.5 (single merged weights; T-text channel and pixel-token channel). " +
               " ".join(_tex_escape(n) for n in notes) + "}")
    L = ["% Generated by scripts/figures/make_paper_figures.py; do not edit. Needs \\usepackage{booktabs,multirow,graphicx}.",
         "\\begin{table*}[t]", "\\centering", caption, "\\label{tab:e3}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{ll" + "c" * (len(tt) + len(E3_GRID)) + "}", "\\toprule",
         f" & & \\multicolumn{{{len(tt)}}}{{c}}{{T-text readouts ($\\times$100)}} & \\multicolumn{{{len(E3_GRID)}}}{{c}}{{Pixel-token grid (structured)}} \\\\",
         f"\\cmidrule(lr){{3-{2 + len(tt)}}} \\cmidrule(lr){{{3 + len(tt)}-{2 + len(tt) + len(E3_GRID)}}}",
         "Size & Arm & " + " & ".join(_tex_escape(TTEXT_LABELS[t][0]) for t in tt) + " & " +
         " & ".join(_tex_escape(g[2]) for g in E3_GRID) + " \\\\",
         " & & " + " & ".join(_tex_escape(TTEXT_LABELS[t][1]) for t in tt) + " & " +
         " & ".join(_tex_escape(g[3] + (" (×100)" if g[4] != 1.0 else "")) for g in E3_GRID) + " \\\\", "\\midrule"]
    M = ["| Size | Arm | " + " | ".join(f"{TTEXT_LABELS[t][0]} {TTEXT_LABELS[t][1]}" for t in tt) + " | " +
         " | ".join(f"{g[2]} {g[3]}" + (" (×100)" if g[4] != 1.0 else "") for g in E3_GRID) + " |",
         "|---|---|" + "---:|" * (len(tt) + len(E3_GRID))]
    for k, s in enumerate(present):
        n_exp = len(s["seeds"])
        for j, (arm, label) in enumerate(E3_ARMS):
            row = s["rows"][arm]
            cells_t = [_fmt_cell(row[t], 100.0, 1, n_exp, True) for t in tt] + \
                      [_fmt_cell(row[g[0]], g[4], 1 if g[4] != 1.0 else 3, n_exp, True) for g in E3_GRID]
            cells_m = [_fmt_cell(row[t], 100.0, 1, n_exp, False) for t in tt] + \
                      [_fmt_cell(row[g[0]], g[4], 1 if g[4] != 1.0 else 3, n_exp, False) for g in E3_GRID]
            size_cell = f"\\multirow{{{len(E3_ARMS)}}}{{*}}{{{s['size']}}}" if j == 0 else ""
            L.append(f"{size_cell} & {_tex_escape(label)} & " + " & ".join(cells_t) + " \\\\")
            M.append(f"| {s['size'] if j == 0 else ''} | {label} | " + " | ".join(cells_m) + " |")
        if k < len(present) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    md = ["# Table: E3 comparison (generated by scripts/figures/make_paper_figures.py; do not edit by hand)", ""] + M + [""] + \
         [f"- {n}" for n in notes] + [""]
    p_tex, p_md = out / "tab_e3_comparison.tex", out / "tab_e3_comparison.md"
    p_tex.write_text("\n".join(L), encoding="utf-8")
    p_md.write_text("\n".join(md), encoding="utf-8")
    return [p_tex, p_md]


# ---------------------------------------------------------------------------------------------
# 图注与 README
# ---------------------------------------------------------------------------------------------


def _n_of(e1: dict, key: str) -> str:
    ns = sorted({(rec["readouts"].get(key) or {}).get("n") for m, rec in e1["models"].items()
                 if rec["group"] != "excluded" and rec["readouts"].get(key)} - {None})
    return "/".join(f"{n:,}" for n in ns) if ns else MISSING


def _resamples(e1: dict) -> str:
    rs = sorted({p.get("n_resamples") for rec in e1["models"].values() if rec["group"] != "excluded"
                 for p in rec["readouts"].values() if p} - {None})
    return "/".join(f"{r:,}" for r in rs) if rs else MISSING


def valid_text(e1: dict, models=E1_EXCLUDED) -> str:
    """定位的有效回答率（metrics.json 的 valid.rate），按模型列出。"""
    parts = []
    for lab, m in models:
        v = (e1["models"].get(m) or {}).get("grounding_valid_rate")
        parts.append(f"{lab} {MISSING if v is None else f'{100 * v:.0f}%'}")
    return ", ".join(parts)


def caption_e1(e1: dict) -> str:
    excl = ", ".join(lab for lab, _ in E1_EXCLUDED)
    s = (f"Knowing where vs. drawing the outline (E1, zero-shot, T-text channel). (a) Qwen3.5 0.8B–9B (blue) and InternVL3.5 "
         f"2B/4B (orange, cross-family reference). Knows where: grounding Acc@0.5 on all RefCOCO/+/g evaluation splits "
         f"(n={_n_of(e1, 'grounding')}) and the model's own predicted box rasterised as a mask (cIoU, n={_n_of(e1, 'own_box')} "
         f"expressions). Draws the outline: free-form polygon cIoU (n={_n_of(e1, 'polygon')}), 16×16 grid gIoU "
         f"(n={_n_of(e1, 'grid16')}) and structure-forced polygon cIoU (≥3 vertices; n={_n_of(e1, 'poly_structured')}, the same "
         f"expressions as the own box; run for Qwen3.5 only, n/a for InternVL3.5). Strict readings: invalid answers score 0. "
         f"Error bars: 95% bootstrap CI ({_resamples(e1)} resamples). (b) Paired difference structure-forced polygon − own box "
         f"on the same expressions (≥3 and ≥8 vertices), 95% paired bootstrap CI; {e1['n_paired_ci_excluding_zero']} of "
         f"{e1['n_paired']} CIs exclude 0. {excl} are omitted: their grounding answers are mostly refusals (valid answers: "
         f"{valid_text(e1)}), so both knows-where readings measure refusal rather than localisation.")
    if e1["missing"]:
        s += " Missing: " + "; ".join(e1["missing"]) + "."
    return s


def caption_e2(e2: dict) -> str:
    parts = []
    for task, title, ylabel, ds in E2_TASKS:
        ns = sorted({(m["tasks"].get(task) or {}).get("n_eval") for m in e2["models"].values()} - {None})
        parts.append(f"{title.lower()} ({ds}, n={'/'.join(f'{n:,}' for n in ns) if ns else MISSING}; {ylabel})")
    every_other = [m["size"] for m in e2["models"].values() if any(
        len(t["layers"]) > 1 and (t["layers"][1] - t["layers"][0]) > 1 for t in m["tasks"].values())]
    s = ("Linear probes on frozen Qwen3.5 hidden states (E2). x: relative depth = layer index / number of layers. "
         "Tasks: " + "; ".join(parts) + ". Shaded: 95% bootstrap CI over evaluation samples. Open circle: layer 0 (the "
         "image tokens entering the LLM); filled square: last layer.")
    if every_other:
        s += f" {', '.join(every_other)} probed at every other layer."
    if e2["missing"]:
        s += " Missing: " + "; ".join(e2["missing"]) + "."
    return s


def caption_interference(itf: dict) -> str:
    main = [b for b in itf["blocks"] if b.get("rank") is None]
    mach = "; ".join(f"{b['size']}: {b['machine']}" for b in main)
    seeds = []
    for b in main:
        spec = next((s for s in E3_SIZES if s["size"] == b["size"]), None)
        for q in ("L", "P_seg", "P_depth"):
            n = len(b.get(q) or {})
            if spec and n < len(spec["seeds"]):
                seeds.append(f"{b['size']} {q}: {n} of {len(spec['seeds'])} seeds")
    s = (f"Interference vs. model size (E3, exploratory analysis §13). L = C3 − (E3b+C3), mean difference over the "
         f"{len(ITF.L_ITEMS)} T-text readouts of Table E3 (points). P_seg = E3b − (E3b+C3) referring-segmentation cIoU and P_depth = (E3b+C3) − E3b "
         "depth AbsRel, both with structure-forced grid decoding (positive = mixing in text hurts the pixel channel). "
         "R_seg = P_seg / cIoU(E3b), post hoc and descriptive (D96-4). Hollow markers: individual seeds (paired within seed); "
         "filled: seed mean. x: nominal parameter count, log scale. Effects are computed within one machine (D93-1): "
         f"{mach}.")
    if seeds:
        s += (" Seeds with paired readings, where fewer than planned (see Table E3 for seeds rejected by the merge check): " +
              "; ".join(seeds) + ".")
    if any(b.get("rank") for b in itf["blocks"]):
        s += (" Star: 0.8B with LoRA rank 128 for both C3 and E3b+C3 (post hoc, §13a/D99-1; P uses the rank-32 E3b), "
              "drawn slightly right of 0.8B.")
    if itf["missing"]:
        s += " Missing: " + "; ".join(_en_missing(m) for m in itf["missing"]) + "."
    return s


def _en_missing(m: str) -> str:
    """collect_interference 的缺项（中文）→ 图注里的英文短句；只缺某个量时写出是哪个量。"""
    who = "0.8B LoRA rank 128 (RUN-0054)" if "r128" in m else "Qwen3.5-9B (RUN-0056)" if "9B" in m else None
    if who is None:
        return m
    q = re.search(r": (L|P_seg|P_depth)（", m)
    return f"{who}{': ' + q.group(1) if q else ''} not yet available"


def _git_commit(root: Path) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, timeout=20).stdout.strip()
        return (sha or "unknown") + (" (working tree has uncommitted changes)" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def write_readme(root: Path, out: Path, e1, e2, e3, itf, captions: dict, files: dict) -> Path:
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = ["# 论文用图表（docs/figures/）", "",
         "<!-- 由 scripts/figures/make_paper_figures.py 生成；不要手改。数据补齐后重跑：python scripts/figures/make_paper_figures.py -->", "",
         f"- 生成时间：{now}",
         f"- 代码 commit：{_git_commit(root)}",
         f"- 结果树：`{'仓库根' if root == REPO else root}`",
         f"- matplotlib {matplotlib.__version__}；PDF 是矢量（TrueType 字体，无 Type 3），PNG 300 dpi；宽度：单栏 {SINGLE_COL_IN} in，双栏 {DOUBLE_COL_IN} in。",
         "- 配色：dataviz 参考调色板。Qwen3.5 四个尺寸用一个有序的蓝色阶（浅→深 = 0.8B→9B），InternVL3.5 用第二个有序色阶（橙）；"
         "两类对比用分类槽 1（蓝）与槽 2（橙）。都在白底上过了 validate_palette.js（有序：单色相、亮度单调、相邻 ΔL ≥ 0.06；分类两色全对 CVD ΔE 24.7）。",
         "- 每个数字都从结果文件读；逐点来源见 `data/*.json`（每个点带 source 路径）。", ""]
    # 缺的数据
    miss = e1["missing"] + e2["missing"] + e3["missing"] + itf["missing"]
    L += ["## 缺的数据（missing）", ""]
    L += [f"- {m}" for m in miss] if miss else ["- 无"]
    if e3["gaps"]:
        L += ["", "E3 表里缺的格子（按种子）：", ""] + [f"- {g}" for g in e3["gaps"]]
    L.append("")
    # 各图
    L += ["## 1. E1 差距图：`fig_e1_gap.pdf` / `fig_e1_gap.png`", "",
          "- 数据：", f"  - 定位 Acc@0.5、多边形 cIoU、grid16 gIoU：`{E1_MAIN}/<模型>/{{refexp_box,refexp_poly,grid16}}/metrics.json` 的 headline（严格读数，scope = all，bootstrap 95% CI）。",
          f"  - 自己的框 cIoU、结构强制多边形 cIoU 与配对差：`{E1_OWNBOX}`（refexp_poly 的 own_box、poly_structured、poly_structured_min8 三行，strict.cIoU 与 paired_diff.cIoU）。",
          f"  - 核对：ownbox.json 的结构强制读数与 `{E1_POSTHOC2}/<模型>/refexp_poly__poly_structured/metrics.json` 逐个相同。",
          "- 口径（RESULTS.md G1 第 1 条、G1 §2、D81-2）：",
          "  - \"知道在哪\"：定位 Acc@0.5（RefCOCO/+/g 全部评测划分，G1 §2.3 的规模斜率口径）；自己的框按 E1 多边形同一个光栅化成 mask 后的 cIoU（思考子集的那批句子，n 见图注）。",
          "  - \"画轮廓\"：主评测的多边形 cIoU、16×16 网格 gIoU、结构强制多边形 cIoU（≥3 点，与自己的框同一批句子）；各自的 n 见图注（从 metrics.json 读）。",
          "  - (b) 的配对差：结构强制多边形 − 自己的框，同一批 id 的配对 bootstrap 95% CI；两种结构强制（≥3 点、≥8 点）× 四个尺寸。",
          f"- 其他家族：InternVL3.5-2B/4B 作参照画进 (a)（都能作答，定位有效回答率 {valid_text(e1, E1_REFERENCE)}；没有跑结构强制，标 n/a）。"
          f"Gemma 4 E2B/E4B 不画：定位以拒答为主（有效回答率 {valid_text(e1)}，G1 §2.5），两个\"知道在哪\"的读数量到的是拒答而不是定位，"
          "画出来可能出现\"多边形高于定位\"的假象；数字照样写进 `data/fig_e1_gap.json`。",
          "", "**图注（英文）**", "", captions["fig_e1_gap"], ""]
    if e1["checks"]:
        L += ["核对：", ""] + [f"- {c}" for c in e1["checks"]] + [""]
    L += ["## 2. E2 逐层曲线：`fig_e2_layers.pdf` / `fig_e2_layers.png`", "",
          f"- 数据：`{E2_RAW}/<模型>/<任务>/metrics.json` 的 `probes[\"L<i>/linear\"].eval`（标题指标与它的 `_ci`）。标题指标按 `scripts/mgr/e2_summary.py` 的 headline_of 取（metrics.json 的第一个 headline）。",
          f"- 核对：逐层读数与 `{E2_SUMMARY}`（管理员汇总，G2 用的那份）的 linear 曲线逐层相同。",
          "- 口径（G2 §2.1）：线性探针（1×1 卷积）；深度 AbsRel↓（NYU）、法向平均角误差°↓（NYU）、语义分割 mIoU↑（ADE20K val）、指代分割 cIoU↑（RefCOCO val，指令在前）；n 见图注（metrics.json 的 n_eval）。"
          "横轴 = 层号 ÷ 最后一层的层号（9B 隔层取）；空心圆 = 第 0 层，实心方块 = 最后一层；阴影 = 评测样本的 bootstrap 95% CI。",
          "", "**图注（英文）**", "", captions["fig_e2_layers"], ""]
    if e2["checks"]:
        L += ["核对：", ""] + [f"- {c}" for c in e2["checks"]] + [""]
    L += ["## 3. E3 对照表：`tab_e3_comparison.tex` / `tab_e3_comparison.md`", "",
          "- 数据（每个尺寸的目录，先到先得，与 interference.py 同一套相对路径）："]
    for s in e3["sizes"]:
        L.append(f"  - {s['size']}（{s['machine']}；{s['runs']}）：臂 `{', '.join(s['dirs'])}`；基座 `{', '.join(s['base_dirs'])}`" +
                 ("" if s["present"] else "——**还没有（missing）**"))
    L += ["  - 臂的 T 文本：`<目录>/t_text/<变体>/seed<k>/<模型>/<任务>/metrics.json`；网格：`<目录>/eval/<变体>/<模型>/seed<k>/<refseg|depth>__structured/metrics.json`；"
          "基座：`<基座目录>/<任务>/metrics.json`；合并核对：同一层目录的 `merge_check.json`。",
          "- 口径：列是 §13 的 8 个 T 文本读数（`scripts/mgr/interference.py` 的 L_ITEMS，标题读数 = metrics.json 第一个 headline 的严格值，×100），"
          "加结构强制网格的指代分割 cIoU（×100）与深度 AbsRel（单轮、guided，D83-3）。种子平均 ± 样本标准差（statistics.stdev，同 interference.summary）。",
          "- 行：基座、C3、E3b+C3、E3b（基座放在每组第一行作参照）。任务书写的是\"9B 有了就加一列组\"；列已经是 10 个读数，所以尺寸做成行组，9B 有了就多一个行组。",
          "", "**表注（英文）**", ""] + [f"- {n}" for n in e3_table_notes(e3)] + [""]
    if e3["rejected"]:
        L += ["被合并核对拒绝的种子（N37；判据 v1 的 failures 原文）：", ""]
        L += [f"- {r['size']} {r['arm']} seed {r['seed']}（{r['channel']}，{r['machine']}）：{json.dumps(r['failures'], ensure_ascii=False)}；`{r['path']}`" for r in e3["rejected"]]
        L.append("")
    L += ["## 4. 干扰随尺寸的曲线：`fig_interference_size.pdf` / `fig_interference_size.png`", "",
          f"- 数据：`{INTERFERENCE_JSON}`（`scripts/mgr/interference.py` 的 JSON 输出）。JSON 里没有的尺寸（9B）在评测目录出现后，"
          "用同一个脚本的 size_block 按上表的目录现算；秩 128 的 0.8B 同样用 size_block，C3 与 E3b+C3 只取 RUN-0054 的目录，E3b 取 RUN-0050 的 r32（§13a、D99-1）。",
          "- 口径（E3 预注册 §13、D96-4）：L = C3 −（E3b+C3）的 8 项平均（点）；P_seg = E3b − (E3b+C3) 的 refseg cIoU；P_depth = (E3b+C3) − E3b 的 depth AbsRel；"
          "R_seg = P_seg ÷ E3b 的 cIoU（事后、描述性）。逐种子配对；空心 = 各种子，实心 = 种子平均。横轴 = 名义参数量（尺寸标签，对数轴）。",
          f"- §13 的趋势判定（JSON 原文，只引用不重判）：{json.dumps(itf.get('trend'), ensure_ascii=False)}；相对干扰（描述）：{json.dumps(itf.get('relative_trend_descriptive'), ensure_ascii=False)}。",
          "- 各点的来源："] + [f"  - {b['size']}：{b['source']}" for b in itf["blocks"]] + \
         ["", "**图注（英文）**", "", captions["fig_interference_size"], ""]
    if itf["checks"]:
        L += ["核对：", ""] + [f"- {c}" for c in itf["checks"]] + [""]
    L += ["## 文件", ""] + [f"- `{_rel(out, p)}`" for ps in files.values() for p in ps] + [""]
    p = out / "README.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO), help="结果树的根（默认仓库根）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <root>/docs/figures）")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.out).resolve() if a.out else root / "docs" / "figures"
    (out / "data").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    e1, e2, e3, itf = collect_e1(root), collect_e2(root), collect_e3(root), collect_interference(root)
    files = {"fig_e1_gap": fig_e1(e1, out), "fig_e2_layers": fig_e2(e2, out), "tab_e3_comparison": tab_e3(e3, out),
             "fig_interference_size": fig_interference(itf, out)}
    captions = {"fig_e1_gap": caption_e1(e1), "fig_e2_layers": caption_e2(e2), "fig_interference_size": caption_interference(itf)}
    for name, cap in captions.items():
        p = out / f"{name}_caption.tex"
        p.write_text(f"% Generated by scripts/figures/make_paper_figures.py; do not edit.\n\\caption{{{_tex_escape(cap)}}}\n", encoding="utf-8")
        files[name].append(p)
    for name, data in (("fig_e1_gap", e1), ("fig_e2_layers", e2), ("tab_e3_comparison", e3), ("fig_interference_size", itf)):
        p = out / "data" / f"{name}.json"
        p.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        files[name].append(p)
    readme = write_readme(root, out, e1, e2, e3, itf, captions, files)
    print(f"wrote {sum(len(v) for v in files.values()) + 1} files to {out} (README: {readme})")
    miss = e1["missing"] + e2["missing"] + e3["missing"] + itf["missing"]
    if miss:
        print("missing:\n  " + "\n  ".join(miss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
