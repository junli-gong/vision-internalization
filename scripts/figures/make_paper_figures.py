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
- 补评目录（DEV-0063，D102-2 = N37(b)：E3 的结构强制网格评测按判据 e5-v2 补评被拒的种子与机内配对的另一臂）：

      python scripts/figures/make_paper_figures.py \\
          --e3-prepend 2B=experiments/E3/00NN_e3_reeval_e5v2_hpcgpu07 --e3-prepend 4B=experiments/E3/00NN_e3_reeval_e5v2_hpcgpu07

  把补评目录放在该尺寸目录列表的最前面（先找到的为准）：补评过的种子从新目录读，其余照旧。干扰的 JSON
  （docs/analysis/e3_interference_by_size.json）要先用 interference.py 按同样的目录顺序重出，否则 README 的核对报"不一致"。
  网格读数的判据取它自己的 metrics.json（interference.grid_criteria）；有 e5-v2 的读数时表注与干扰图注各加一句，
  全是 v1 时输出与改动前相同。
- W2 一节（DEV-0071）：test 主表 tab_w2_test（含外部基线 LLMSeg 一块与回放一行；gIoU 的附录表 tab_w2_test_giou）、
  W2-a − W1-F 随尺寸 fig_w2_size、语言代价 fig_lang_cost、计数分档 fig_count_bins。数字都取管理员的分析 JSON
  （docs/analysis/w2_test_table.json、lang_cost_9b.json、count_errors_9b.json、LLMSeg 的 compare.json）与结果目录的
  metrics.json，不重算任何 bootstrap；回放 − W2-c 的配对差只在 docs/analysis/w2c_replay.md 的 §1 表里，从那里读。
  探索的方法（X-2、X-3、X-3L）还没判定，语言代价图默认不画，--with-explore 打开。
- 干扰图 (d) R_seg：秩 128 的读数离 r32 的范围太远时画在坐标轴之外，用箭头加数值标注（OFFSCALE_*）。
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
W2T = _load_script("_paperfig_w2_test_table", "scripts/mgr/w2_test_table.py")  # W2 test 的划分、运行目录与叶目录（SPLITS、RUNS、leaf）
LCT = _load_script("_paperfig_lang_cost_table", "scripts/mgr/lang_cost_table.py")  # 语言代价的读数键（BATTERY_ROWS、TTEXT）
CE = _load_script("_paperfig_count_errors", "scripts/mgr/count_errors.py")  # 计数的三个大档（COARSE）

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

# W2（预注册 §11–§13；DEV-0071）。划分与叶目录的约定取自 scripts/mgr/w2_test_table.py（W2T.SPLITS、W2T.RUNS、W2T.leaf）。
W2_TEST_JSON = "docs/analysis/w2_test_table.json"  # §11 的全表（RUN-0072、0073；W2-a 的配对差由 RUN-0081 补齐）
W2_TEST_RUNS = "RUN-0072 (9B), RUN-0073 (2B, 4B), RUN-0081 (W2-a paired differences)"
W2_MODEL = "qwen3.5-9b"  # 主表只有 9B
W2_REPLAY_RUN = "experiments/W2/0076_w2c_replay_test_hpcgpu06"  # 回放变体的 test（RUN-0076 补充 2）：<run>/<模型>/refseg/w2c/<划分>/
W2_REPLAY_SETTING = "w2c"
W2_REPLAY_MD = "docs/analysis/w2c_replay.md"  # 回放 − W2-c 的配对差：只在这份文档 §1 的表里（没有 JSON）
LLMSEG_RUN = "experiments/W2/0078_llmseg_hpcgpu06"  # RUN-0078：<run>/<模型>/refseg/llmseg/<划分>/metrics.json
LLMSEG_COMPARE = LLMSEG_RUN + "/compare_w2c/compare.json"  # LLMSeg − W2-c 的配对差（同样 300 句，D139）
LANG_COST_JSON = "docs/analysis/lang_cost_9b.json"  # D125-3 的并排（scripts/mgr/lang_cost_table.py）
COUNT_ERRORS_JSON = "docs/analysis/count_errors_9b.json"  # 计数分档（scripts/mgr/count_errors.py）
# 探索的候选（还没判定，--with-explore 才画）：(名字, 大题量 compare, 健康检查 compare, 基座所在节点)。没有 T 文本读数。
EXPLORE_LANG = (
    ("X-2", "experiments/X/0084_lb_explore_hpcgpu07/lb_compare_x2.json", "experiments/X/0080_x2_hpcgpu07/0080_eval/health/health_compare.json",
     "hpcgpu07"),
    ("X-3", "experiments/X/0084_lb_explore_hpcgpu07/lb_compare_x3.json", "experiments/X/0082_x3_hpcgpu07/0082_eval/health/health_compare.json",
     "hpcgpu07"),
    ("X-3L", "experiments/X/0084_lb_explore_hpcgpu06/lb_compare_x3l.json",
     "experiments/X/0083_x3l_hpcgpu06/0083_eval/health/health_compare.json", "hpcgpu06"),
)
EXPLORE_RUNS = "RUN-0080, RUN-0082, RUN-0083 (health), RUN-0084 (battery)"

# tab_w2_test 的行：(键, 英文名)。键是 w2_test_table.json per_split 的设置名；"replay" 从回放的结果目录读。
W2_ROWS = (("ownbox", "Own box"), ("w1f", "W1-F"), ("random_pick", "Random pick"), ("oracle_pick", "Oracle pick"),
           ("w2a", "W2-a"), ("w2c", "W2-c"), ("replay", "W2-c + replay"), ("baseline_e1", "E1 zero-shot polygon"))
W2_PAIRS = (("w2c-w1f", "W2-c − W1-F"), ("w2a-w1f", "W2-a − W1-F"), ("replay-w2c", "W2-c + replay − W2-c"))
LLMSEG_ROWS = (("seg", "LLMSeg, before refinement"), ("rec", "LLMSeg, after refinement (main)"), ("w2c", "W2-c, same expressions"))
LLMSEG_PAIRS = (("rec", "LLMSeg after − W2-c"), ("seg", "LLMSeg before − W2-c"))
W2_METRICS = (("cIoU", "tab_w2_test"), ("gIoU", "tab_w2_test_giou"))
DATASET_LABELS = {"refcoco": "RefCOCO", "refcoco+": "RefCOCO+", "refcocog": "RefCOCOg"}

# fig_lang_cost：(lang_cost_9b.json 的节, 键, 英文标签)。键取自 lang_cost_table 的 BATTERY_ROWS、TTEXT（前两项、两项）。
LANG_READOUTS = (
    ("battery", ".".join(LCT.BATTERY_ROWS[0]), "MMLU-Pro"),
    ("battery", ".".join(LCT.BATTERY_ROWS[1]), "IFEval prompt\nstrict"),
    ("ttext", LCT.TTEXT[0][0], "Grounding\nAcc@0.5"),
    ("ttext", LCT.TTEXT[1][0], "Counting\naccuracy"),
)
LANG_SYSTEMS = (("E3b+C3", "E3b+C3"), ("W2-c", "W2-c"), ("W2-c+回放", "W2-c + replay"))  # (lang_cost_9b.json 的系统名, 英文名)
HEALTH_LABELS = {"mmlu_pro_letter": "MMLU-Pro", "gsm8k": "GSM8K", "ifeval": "IFEval"}
# fig_count_bins：(count_errors_9b.json 的系统名, 英文名)；基座两次读数逐条相同时只画一根（base@0069）
COUNT_SYSTEMS = (("base@0069", "Base"), ("W2-c", "W2-c"), ("W2-c+回放", "W2-c + replay"), ("C3", "C3"), ("E3b+C3", "E3b+C3"))
COUNT_BASES = ("base@0069", "base@0056")

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
# 干扰图的离群读数（DEV-0071）：只在 (d) R_seg 上处理。秩 128 的一个读数（种子或平均）落在 r32 全部读数（各种子与平均）的范围
# [lo, hi] 向外再扩 OFFSCALE_SPANS × (hi − lo) 的窗口之外，就不画在坐标轴里，改在轴边画箭头并写数值；其余点照旧，纵轴只容纳窗口内的点。
OFFSCALE_PANELS = ("R_seg",)
OFFSCALE_SPANS = 1.0

# W2 图的配色：同一个系统在各张图里同一个颜色（分类槽 1–3 的三色两两过 validate_palette.js，全对 CVD ΔE ≥ 9.2）；
# 基座与 C3（对照）用中性灰。W2-a 只出现在 fig_w2_size，用槽 2（与 E3b+C3 不在同一张图里）。
SERIES_3, SERIES_4, SERIES_5, SERIES_6 = "#1baf7a", "#eda100", "#e87ba4", "#008300"  # 分类槽 3–6（--with-explore 的 X-2、X-3、X-3L）
W2_COLORS = {"W2-c": SERIES_1, "W2-a": SERIES_2, "E3b+C3": SERIES_2, "W2-c+回放": SERIES_3, "base@0069": MUTED, "C3": INK_2,
             "X-2": SERIES_4, "X-3": SERIES_5, "X-3L": SERIES_6}


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


def prepend_dirs(sizes, prepend: dict) -> tuple:
    """--e3-prepend（DEV-0063）：把补评目录放在对应尺寸的目录列表最前面（先找到的为准，于是补评过的种子覆盖原目录里的同一路径）。
    prepend = {尺寸: [目录…]}；没有给的尺寸原样。"""
    out = []
    for spec in sizes:
        extra = tuple(prepend.get(spec["size"], ()))
        out.append({**spec, "dirs": extra + tuple(spec["dirs"]), "prepended": list(extra)} if extra else spec)
    return tuple(out)


def collect_e3(root: Path, sizes=E3_SIZES) -> dict:
    out = {"sizes": [], "missing": [], "rejected": [], "gaps": [], "ttext_tasks": [t for _, t in ITF.L_ITEMS]}
    non_v1 = []  # DEV-0063：判据不是 v1 的网格读数（e5-v2 补评）
    for spec in sizes:
        rec = {"size": spec["size"], "model": spec["model"], "machine": spec["machine"], "runs": spec["runs"],
               "dirs": list(spec["dirs"]), "base_dirs": list(spec["base"]), "seeds": list(spec["seeds"]),
               "present": e3_present(root, spec), "rows": {}, "gpus": []}
        if spec.get("prepended"):  # DEV-0063：只在给了 --e3-prepend 时有这个键
            rec["prepended"] = list(spec["prepended"])
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
                for s in vals:  # DEV-0063：读数自己的 metrics.json 记的判据（interference.grid_criteria）
                    c = ITF.grid_criteria(dirs, arm, s, spec["model"], task)
                    if c != ITF.DEFAULT_CRITERIA:
                        gpu = next((m.get("gpu") for m in mcs if (m["arm"], m["seed"], m["channel"]) == (arm, s, "grid")), None)
                        non_v1.append({"size": spec["size"], "arm": arm, "seed": s, "task": task, "criteria": c, "gpu": gpu})
                for s in spec["seeds"]:
                    if s not in vals:
                        why = "rejected by merge check" if (arm, s, "grid") in rejected else "no result file"
                        out["gaps"].append(f"E3 {spec['size']} {arm} seed {s}: {task} structured grid（{why}）")
                    elif (arm, s, "grid") in rejected:
                        out["gaps"].append(f"不一致：E3 {spec['size']} {arm} seed {s} 的网格核对没过，却有 {task} 的读数")
            rec["rows"][arm] = row
        out["sizes"].append(rec)
    if non_v1:  # 全是 v1 时不加这个键（输出与改动前相同）
        out["non_v1_grid"] = non_v1
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


def collect_interference(root: Path, sizes=E3_SIZES) -> dict:
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
    for spec in sizes:
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
    for b in out["blocks"]:  # DEV-0063：两臂判据不同的一对（interference.size_block 没有算它）
        for x in b.get("mismatched_criteria") or []:
            out["checks"].append(f"不一致：{b['size']} 种子 {x['seed']} {x['task']} 的两臂判据不同（E3b {x['e3b']}、E3b+C3 {x['e3b_c3']}），"
                                 "这一对没有算 P、R；补评目录要两臂都有")
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
# 读取：W2（DEV-0071；只读分析 JSON 与结果目录的 metrics.json，不重算 bootstrap）
# ---------------------------------------------------------------------------------------------


def split_label(key: str, short: bool = False) -> str:
    """W2 划分名 → 标签："test_refcoco+_unc_testA" → "RefCOCO+ testA"；short=True 给分析文档里的写法 "refcoco+ testA"。"""
    m = re.fullmatch(r"test_(refcoco\+?|refcocog)_(?:unc|umd)_(\w+)", key)
    if not m:
        raise ValueError(f"看不懂 W2 划分名：{key!r}")
    return f"{m.group(1)} {m.group(2)}" if short else f"{DATASET_LABELS[m.group(1)]} {m.group(2)}"


def _ci_list_point(d: dict | None, level, source: str, n=None) -> dict | None:
    """分析 JSON 里 {"value"/"diff", "ci": [low, high]} 的写法 → _point。"""
    if not isinstance(d, dict):
        return None
    v = d.get("value", d.get("diff"))
    ci = d.get("ci")
    if isinstance(ci, dict):
        ci = [ci.get("low"), ci.get("high")]
    ci = ci or [None, None]
    return _point(v, {"low": ci[0], "high": ci[1], "level": level}, d.get("n", n), source)


def _leaf_headline(f: Path, name: str, root: Path, readout: str | None = None) -> dict | None:
    """结果目录 metrics.json 的标题读数（scope = all），带运行时算的 95% CI；readout 给 LLMSeg 的 rec / seg。"""
    d = _load(f)
    if d is None:
        return None
    hl = ((d.get("readouts") or {}).get(readout) or {}).get("headline") if readout else d.get("headline")
    for h in hl or []:
        if isinstance(h, dict) and h.get("name") == name and h.get("scope", "all") == "all":
            return _point(h.get("value"), h.get("ci"), h.get("n"), _rel(root, f) + (f" :: readouts.{readout}" if readout else ""))
    return None


def _num(s: str) -> float:
    return float(s.replace("−", "-"))


def replay_md_table(root: Path) -> dict:
    """docs/analysis/w2c_replay.md §1 的表：每个划分的 W2-c、回放读数（3 位小数）与回放 − W2-c 的配对差 [95% CI]，另加合并一行。
    返回 {划分键或 "pooled": {"w2c", "replay", "diff", "low", "high", "n"}}；表的格式变了就返回空（README 报缺）。"""
    f = root / W2_REPLAY_MD
    if not f.is_file():
        return {}
    txt = f.read_text(encoding="utf-8")
    m = re.search(r"^## 1\..*?$(.*?)(?=^## )", txt, re.S | re.M)
    if not m:
        return {}
    by_label = {split_label(k, short=True): k for k in W2T.SPLITS}
    num = r"[+\-−]?\d+\.\d+"
    out = {}
    for line in m.group(1).splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        pm = re.fullmatch(rf"({num}) \[({num}), ({num})\]", cells[3])
        if not pm:
            continue
        pooled = re.fullmatch(r"合并 ([\d,]+) 句", cells[0])
        key = "pooled" if pooled else by_label.get(cells[0])
        if key is None:
            continue
        out[key] = {"w2c": _num(cells[1]) if cells[1] else None, "replay": _num(cells[2]) if cells[2] else None,
                    "diff": _num(pm.group(1)), "low": _num(pm.group(2)), "high": _num(pm.group(3)),
                    "n": int(pooled.group(1).replace(",", "")) if pooled else None}
    return out


def collect_w2(root: Path) -> dict:
    """tab_w2_test（cIoU）与 tab_w2_test_giou 的全部格子。"""
    out = {"splits": list(W2T.SPLITS), "model": W2_MODEL, "rows": {}, "pairs": {}, "llmseg": {"rows": {}, "pairs": {}},
           "e1_parse": {}, "llmseg_calls": {}, "w2c_calls": {}, "missing": [], "checks": [], "per_system_resamples": [],
           "sources": {"table": W2_TEST_JSON, "replay": W2_REPLAY_RUN, "replay_pairs": W2_REPLAY_MD,
                       "llmseg": LLMSEG_RUN, "llmseg_compare": LLMSEG_COMPARE}}
    j = _load(root / W2_TEST_JSON)
    if j is None:
        out["missing"].append(f"W2 主表：{W2_TEST_JSON} 不在")
    mj = ((j or {}).get("models") or {}).get(W2_MODEL) or {}
    out["pairs_level"], out["pairs_resamples"], out["pairs_seed"] = (j or {}).get("level"), (j or {}).get("n_resamples"), (j or {}).get("seed")
    have_replay = (root / W2_REPLAY_RUN).is_dir()
    if not have_replay:
        out["missing"].append(f"W2 主表：回放的结果目录 {W2_REPLAY_RUN} 不在")
    res = set()
    for metric, _name in W2_METRICS:
        rows = out["rows"].setdefault(metric, {})
        for key, _label in W2_ROWS:
            rows[key] = {}
            for sp in W2T.SPLITS:
                if key == "replay":  # 叶目录的布局同 W2T.leaf，运行目录换成回放的
                    pt = _leaf_headline(root / W2_REPLAY_RUN / W2_MODEL / "refseg" / W2_REPLAY_SETTING / sp / "metrics.json", metric, root)
                else:
                    cell = (((mj.get("per_split") or {}).get(sp) or {}).get(key) or {}).get(metric)
                    pt = _ci_list_point(cell, (j or {}).get("level"), f"{W2_TEST_JSON} :: models.{W2_MODEL}.per_split.{sp}.{key}.{metric}",
                                        (((mj.get("per_split") or {}).get(sp) or {}).get(key) or {}).get("n"))
                    # 核对：分析 JSON 的读数 = 该运行自己的 metrics.json（运行时的 CI 也取那里的重抽次数）
                    lp = _leaf_headline(W2T.leaf(root, W2_MODEL, key, sp) / "metrics.json", metric, root)
                    if pt is not None and lp is not None:
                        if abs(lp["value"] - pt["value"]) > 1e-12 or abs((lp["low"] or 0) - (pt["low"] or 0)) > 1e-12:
                            out["checks"].append(f"不一致：{key} {sp} {metric}，{W2_TEST_JSON} {pt['value']} 对 {lp['source']} {lp['value']}")
                        if lp.get("n_resamples"):
                            res.add(lp["n_resamples"])
                if pt is None or pt["value"] is None:
                    if (have_replay if key == "replay" else j is not None):  # 整个来源不在时只报一行（上面）
                        out["missing"].append(f"W2 主表 {key} {sp} {metric}")
                    pt = None
                elif key == "replay" and pt.get("n_resamples"):
                    res.add(pt["n_resamples"])
                rows[key][sp] = pt
    out["per_system_resamples"] = sorted(res)
    if res:
        out["checks"].append(f"一致：{W2_TEST_JSON} 的 9B 各设置读数与 {W2T.RUNS[W2_MODEL]} 各叶目录的 metrics.json 逐格相同"
                             if not any(c.startswith("不一致：") for c in out["checks"]) else
                             f"见上：{W2_TEST_JSON} 与叶目录有不一致的格子")
    # 配对差（只有 cIoU）：W2-c − W1-F、W2-a − W1-F 取分析 JSON；回放 − W2-c 取 w2c_replay.md §1
    rmd = replay_md_table(root)
    if not rmd:
        out["missing"].append(f"W2 配对差 replay-w2c：{W2_REPLAY_MD} §1 的表不在或读不出")
    for key, _label in W2_PAIRS:
        rec = {}
        if key == "replay-w2c":
            for sp in W2T.SPLITS + ["pooled"]:
                r = rmd.get(sp)
                n = r["n"] if r and r["n"] else (out["rows"]["cIoU"]["replay"].get(sp) or {}).get("n")
                rec[sp] = None if r is None else _point(r["diff"], {"low": r["low"], "high": r["high"], "level": 0.95}, n,
                                                         f"{W2_REPLAY_MD} §1（{'合并' if sp == 'pooled' else split_label(sp, short=True)}）")
                if r is None and rmd:
                    out["missing"].append(f"W2 配对差 {key} {sp}（{W2_REPLAY_MD} §1 的表里没有）")
        else:
            per = (mj.get("pairs") or {}).get(key) or {}
            for sp in W2T.SPLITS:
                rec[sp] = _ci_list_point(per.get(sp), (j or {}).get("level"), f"{W2_TEST_JSON} :: models.{W2_MODEL}.pairs.{key}.{sp}")
                if rec[sp] is None and j is not None:
                    out["missing"].append(f"W2 配对差 {key} {sp}（{W2_TEST_JSON}）")
            rec["pooled"] = _ci_list_point((mj.get("pairs_pooled") or {}).get(key), (j or {}).get("level"),
                                           f"{W2_TEST_JSON} :: models.{W2_MODEL}.pairs_pooled.{key}")
            if rec["pooled"] is None and j is not None:
                out["missing"].append(f"W2 配对差 {key} 合并（{W2_TEST_JSON}）")
        out["pairs"][key] = rec
    # 核对：w2c_replay.md 里的 W2-c、回放读数（3 位小数）= 分析 JSON 与回放 metrics.json 四舍五入
    bad = []
    for sp in W2T.SPLITS:
        r = rmd.get(sp)
        for col, key in (("w2c", "w2c"), ("replay", "replay")):
            pt = out["rows"]["cIoU"][key].get(sp)
            if r is not None and pt is not None and r[col] is not None and abs(round(pt["value"], 3) - r[col]) > 1e-9:
                bad.append(f"{sp} {col}")
    if rmd:
        out["checks"].append(("一致" if not bad else "不一致") + f"：{W2_REPLAY_MD} §1 的 W2-c 与回放读数（3 位）对 {W2_TEST_JSON} 与 "
                             f"{W2_REPLAY_RUN} 的 metrics.json" + (f"（不同：{bad}）" if bad else ""))
    # E1 零样本多边形的解析失败（W2-test 的 baseline_e1 叶目录）
    for sp in W2T.SPLITS:
        d = _load(W2T.leaf(root, W2_MODEL, "baseline_e1", sp) / "metrics.json")
        p = (d or {}).get("parse") or {}
        out["e1_parse"][sp] = {"n": p.get("n"), "n_failed": p.get("n_failed")} if p else None
        w = _load(W2T.leaf(root, W2_MODEL, "w2c", sp) / "metrics.json")
        out["w2c_calls"][sp] = ((w or {}).get("n_model_calls") or {}).get("mean")
    # LLMSeg（RUN-0078）：读数与 CI 取各叶目录的 metrics.json（rec = 修正后、seg = 修正前），合并读数与配对差取 compare.json
    cmp_ = _load(root / LLMSEG_COMPARE)
    lbad = False
    if cmp_ is None:
        out["missing"].append(f"LLMSeg：{LLMSEG_COMPARE} 不在")
    for metric, _name in W2_METRICS:
        lrows = out["llmseg"]["rows"].setdefault(metric, {k: {} for k, _ in LLMSEG_ROWS})
        lpairs = out["llmseg"]["pairs"].setdefault(metric, {k: {} for k, _ in LLMSEG_PAIRS})
        for sp in W2T.SPLITS + ["pooled"]:
            blk = ((cmp_ or {}).get("splits") or {}).get(sp) if sp != "pooled" else (cmp_ or {}).get("pooled")
            src = f"{LLMSEG_COMPARE} :: {'pooled' if sp == 'pooled' else 'splits.' + sp}"
            for rd in ("rec", "seg"):
                c = ((blk or {}).get(rd) or {}).get(metric)
                lpairs[rd][sp] = _ci_list_point(c, (c or {}).get("ci", {}).get("level") if c else None, f"{src}.{rd}.{metric}")
                if lpairs[rd][sp] is not None:
                    lpairs[rd][sp]["n_resamples"] = c["ci"].get("n_resamples")
                if sp == "pooled":
                    lrows[rd][sp] = None if not c else _point(c.get("a"), None, c.get("n"), f"{src}.{rd}.{metric}.a")
                else:
                    f = root / LLMSEG_RUN / W2_MODEL / "refseg" / "llmseg" / sp / "metrics.json"
                    lrows[rd][sp] = _leaf_headline(f, metric, root, readout=rd)
                    if lrows[rd][sp] is not None and c and abs(c.get("a") - lrows[rd][sp]["value"]) > 1e-12:
                        out["checks"].append(f"不一致：LLMSeg {rd} {sp} {metric}，compare.json {c.get('a')} 对 {lrows[rd][sp]['source']}")
                        lbad = True
                if lrows[rd][sp] is None and cmp_ is not None:
                    out["missing"].append(f"LLMSeg {rd} {sp} {metric}")
                if lpairs[rd][sp] is None and cmp_ is not None:
                    out["missing"].append(f"LLMSeg − W2-c {rd} {sp} {metric}")
            cr, cs = ((blk or {}).get("rec") or {}).get(metric), ((blk or {}).get("seg") or {}).get(metric)
            lrows["w2c"][sp] = None if not cr else _point(cr.get("b"), None, cr.get("n"), f"{src}.rec.{metric}.b")
            if cr and cs and abs(cr.get("b") - cs.get("b")) > 1e-12:
                out["checks"].append(f"不一致：LLMSeg compare.json {sp} {metric} 的 W2-c 读数 rec {cr.get('b')} 与 seg {cs.get('b')} 不同")
                lbad = True
            if lrows["w2c"][sp] is None and cmp_ is not None:
                out["missing"].append(f"LLMSeg 同句的 W2-c {sp} {metric}")
    if cmp_ is not None:
        out["checks"].append(("一致" if not lbad else "不一致") +
                             f"：{LLMSEG_COMPARE} 的 LLMSeg 读数与各叶目录 metrics.json（rec、seg）逐格相同；W2-c 的 rec、seg 两份读数相同")
        for sp in W2T.SPLITS:
            d = _load(root / LLMSEG_RUN / W2_MODEL / "refseg" / "llmseg" / sp / "metrics.json") or {}
            out["llmseg_calls"][sp] = {rd: ((((d.get("readouts") or {}).get(rd) or {}).get("compute_per_image") or {}).get("n_model_calls"))
                                       for rd in ("rec", "seg")}
        out["llmseg_commit"] = next(((_load(root / LLMSEG_RUN / W2_MODEL / "refseg" / "llmseg" / sp / "metrics.json") or {})
                                     .get("llmseg_commit") for sp in W2T.SPLITS), None)
    return out


def collect_w2_size(root: Path) -> dict:
    """fig_w2_size：2B、4B、9B 的 W2-a − W1-F（合并 5,000 句），9B 另加 W2-c − W1-F；W2-a 挑选的解析失败率（叶目录 pick.parse_failure）。"""
    out = {"sizes": [], "missing": [], "json": W2_TEST_JSON}
    j = _load(root / W2_TEST_JSON)
    if j is None:
        out["missing"].append(f"W2 尺寸：{W2_TEST_JSON} 不在")
    out["level"], out["n_resamples"] = (j or {}).get("level"), (j or {}).get("n_resamples")
    for size, model in [(s, m) for s, m in QWEN if m in W2T.RUNS]:
        mj = ((j or {}).get("models") or {}).get(model) or {}
        rec = {"size": size, "model": model, "run_dir": W2T.RUNS[model], "pairs": {}, "per_split": {}, "parse_failure": {}}
        for key in ("w2a-w1f", "w2c-w1f"):
            rec["pairs"][key] = _ci_list_point((mj.get("pairs_pooled") or {}).get(key), (j or {}).get("level"),
                                               f"{W2_TEST_JSON} :: models.{model}.pairs_pooled.{key}")
            rec["per_split"][key] = {sp: _ci_list_point(((mj.get("pairs") or {}).get(key) or {}).get(sp), (j or {}).get("level"),
                                                        f"{W2_TEST_JSON} :: models.{model}.pairs.{key}.{sp}") for sp in W2T.SPLITS}
        if rec["pairs"]["w2a-w1f"] is None and j is not None:
            out["missing"].append(f"W2 尺寸 {size}: W2-a − W1-F（{W2_TEST_JSON} 的 pairs_pooled）")
        for setting in ("w2a", "w2c"):
            per, tot_n, tot_of = {}, 0, 0
            for sp in W2T.SPLITS:
                f = W2T.leaf(root, model, setting, sp) / "metrics.json"
                pf = ((_load(f) or {}).get("pick") or {}).get("parse_failure")
                per[sp] = pf
                if pf and pf.get("of"):
                    tot_n += pf["n"]
                    tot_of += pf["of"]
            if any(per.values()):
                rec["parse_failure"][setting] = {"n": tot_n, "of": tot_of, "rate": tot_n / tot_of if tot_of else None,
                                                 "complete": all(per.values()), "per_split": per,
                                                 "source": f"{W2T.RUNS[model]}/{model}/refseg/{setting}/<split>/metrics.json :: pick.parse_failure"}
        if "w2a" not in rec["parse_failure"] or not rec["parse_failure"]["w2a"]["complete"]:
            out["missing"].append(f"W2 尺寸 {size}: W2-a 挑选的解析失败率（{W2T.RUNS[model]} 的叶目录）")
        out["sizes"].append(rec)
    return out


def _flags(h: dict | None) -> list[dict]:
    """健康检查 compare 的标红行（简报 8.11 的规则，原样取 flag）。"""
    return [{"reading": r["reading"], "metric": r["metric"], "delta_pp": r.get("delta_pp"), "n": r.get("n")}
            for r in (h or {}).get("rows") or [] if r.get("flag")]


def collect_lang_cost(root: Path, with_explore: bool = False) -> dict:
    """fig_lang_cost：相对同节点基座的差（百分点）与 95% CI，全部取自 lang_cost_9b.json；--with-explore 时另读探索候选的 compare。"""
    out = {"readouts": [{"section": s, "key": k, "label": lab} for s, k, lab in LANG_READOUTS], "systems": [], "missing": [],
           "json": LANG_COST_JSON, "with_explore": with_explore}
    j = _load(root / LANG_COST_JSON)
    if j is None:
        out["missing"].append(f"语言代价：{LANG_COST_JSON} 不在")
    for name, label in LANG_SYSTEMS:
        e = ((j or {}).get("systems") or {}).get(name)
        rec = {"name": name, "label": label, "explore": False, "values": {}, "health_flags": [], "sources": (e or {}).get("sources")}
        if e is None and j is not None:
            out["missing"].append(f"语言代价 {name}（{LANG_COST_JSON} 里没有）")
        for sec, key, _lab in LANG_READOUTS:
            r = ((e or {}).get(sec) or {}).get(key)
            ci = (r or {}).get("ci95_pp") or [None, None]
            rec["values"][key] = None if not r or r.get("delta_pp") is None else \
                _point(r["delta_pp"], {"low": ci[0], "high": ci[1], "level": 0.95}, r.get("n"), f"{LANG_COST_JSON} :: systems.{name}.{sec}.{key}")
            if e is not None and rec["values"][key] is None:
                out["missing"].append(f"语言代价 {name} {key}")
            if sec == "ttext" and r:
                rec.setdefault("base_pct", {})[key] = r.get("base")
        rec["health_flags"] = [{"reading": k.split(".")[0], "metric": k.split(".", 1)[1], "delta_pp": v.get("delta_pp"), "n": v.get("n")}
                               for k, v in ((e or {}).get("health") or {}).items() if v and v.get("flag")]
        out["systems"].append(rec)
    if with_explore:
        for name, battery, health, node in EXPLORE_LANG:
            bt, h = _load(root / battery), _load(root / health)
            rec = {"name": name, "label": name, "explore": True, "node": node, "values": {}, "health_flags": _flags(h),
                   "sources": {"battery": battery, "health": health}}
            if bt is None:
                out["missing"].append(f"语言代价（探索）{name}：{battery} 不在")
            if h is None:
                out["missing"].append(f"语言代价（探索）{name}：{health} 不在")
            for sec, key, _lab in LANG_READOUTS:
                if sec != "battery":
                    rec["values"][key] = None  # 探索的候选没有 T 文本读数（图注写明）
                    continue
                reading, metric = key.split(".", 1)
                r = next((r for r in (bt or {}).get("rows") or [] if r["reading"] == reading and r["metric"] == metric), None)
                ci = (r or {}).get("ci95_pp") or [None, None]
                rec["values"][key] = None if not r else _point(r["delta_pp"], {"low": ci[0], "high": ci[1], "level": 0.95}, r.get("n"),
                                                               f"{battery} :: rows[{reading}/{metric}]")
            out["systems"].append(rec)
    return out


def collect_count_bins(root: Path) -> dict:
    """fig_count_bins：CountBench 三个大档的准确率（count_errors_9b.json），与配对基座的差及配对 bootstrap 95% CI。"""
    out = {"bins": list(CE.COARSE), "systems": [], "missing": [], "json": COUNT_ERRORS_JSON}
    j = _load(root / COUNT_ERRORS_JSON)
    if j is None:
        out["missing"].append(f"计数分档：{COUNT_ERRORS_JSON} 不在")
    out["n_boot"], out["seed"] = (j or {}).get("n_boot"), (j or {}).get("seed")
    out["same_answer_rate"] = (j or {}).get("base_0069_vs_0056_same_answer_rate")
    out["sources"] = (j or {}).get("source")
    systems = (j or {}).get("systems") or {}
    for name, label in COUNT_SYSTEMS:
        s = systems.get(name)
        rec = {"name": name, "label": label, "paired_base": (s or {}).get("paired_base"), "acc": {}, "n": {}, "delta": {}}
        if s is None and j is not None:
            out["missing"].append(f"计数分档 {name}（{COUNT_ERRORS_JSON} 里没有）")
        for b in CE.COARSE:
            r = ((s or {}).get("bins") or {}).get(b)
            rec["acc"][b] = (r or {}).get("acc")
            rec["n"][b] = (r or {}).get("n")
            vb = (r or {}).get("vs_base")
            rec["delta"][b] = None if not vb else _point(vb.get("d_acc"), {"low": vb["ci95"][0], "high": vb["ci95"][1], "level": 0.95,
                                                                          "n_resamples": out["n_boot"]}, rec["n"][b],
                                                         f"{COUNT_ERRORS_JSON} :: systems.{name}.bins.{b}.vs_base")
        out["systems"].append(rec)
    # 只画一根基座：两次基座读数逐条相同时（count_errors 的 base_0069_vs_0056_same_answer_rate = 1）
    b1, b2 = (systems.get(k) for k in COUNT_BASES)
    out["bases_identical"] = bool(b1 and b2 and out["same_answer_rate"] == 1.0 and
                                  all((b1["bins"].get(b) or {}).get("acc") == (b2["bins"].get(b) or {}).get("acc") for b in CE.COARSE))
    if j is not None and not out["bases_identical"]:
        out["missing"].append(f"计数分档：两次基座读数（{', '.join(COUNT_BASES)}）不完全相同，图里只画了 {COUNT_BASES[0]}")
    return out


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


def offscale(itf: dict, q: str) -> dict | None:
    """干扰图某个面板里要画到坐标轴之外的秩 128 读数（规则见 OFFSCALE_SPANS）；没有就返回 None（面板与改动前相同）。"""
    if q not in OFFSCALE_PANELS:
        return None
    main = [b for b in itf["blocks"] if b.get("rank") is None]
    extra = [b for b in itf["blocks"] if b.get("rank") is not None]
    v32 = [v for b in main for v in (b.get(q) or {}).values()] + \
          [b["means"][q]["mean"] for b in main if b["means"][q]["mean"] is not None]
    if not v32 or not extra:
        return None
    lo, hi = min(v32), max(v32)
    win = (lo - OFFSCALE_SPANS * (hi - lo), hi + OFFSCALE_SPANS * (hi - lo))
    pts, inside = [], list(v32)
    for b in extra:
        x = size_value(b["x_size"]) * EXTRA_OFFSET
        cand = [("seed", int(s), v, x * SEED_OFFSET) for s, v in (b.get(q) or {}).items()]
        if b["means"][q]["mean"] is not None:
            cand.append(("mean", None, b["means"][q]["mean"], x))
        for kind, seed, v, xx in cand:
            if win[0] <= v <= win[1]:
                inside.append(v)
            else:
                pts.append({"block": b["size"], "kind": kind, "seed": seed, "value": v, "x": xx, "side": "below" if v < win[0] else "above"})
    if not pts:
        return None
    lo_in, hi_in = min(inside), max(inside)
    r = hi_in - lo_in
    below, above = any(p["side"] == "below" for p in pts), any(p["side"] == "above" for p in pts)
    return {"panel": q, "window": list(win), "points": pts,
            "ylim": [lo_in - (0.45 if below else 0.08) * r, hi_in + (0.45 if above else 0.08) * r]}


def _draw_offscale(ax, off: dict) -> None:
    """轴边的箭头（指向读数所在的方向）加数值。"""
    y0, y1 = off["ylim"]
    ax.set_ylim(y0, y1)
    h = y1 - y0
    for p in off["points"]:
        # 不在假的位置画记号：只画一支从轴内指向轴边的箭头（横坐标同该读数），旁边写数值
        edge = y0 if p["side"] == "below" else y1
        tail = edge + (0.16 if p["side"] == "below" else -0.16) * h
        ax.annotate("", xy=(p["x"], edge), xytext=(p["x"], tail), zorder=5,
                    arrowprops={"arrowstyle": "-|>", "color": SERIES_2, "lw": 0.8, "mutation_scale": 6, "shrinkA": 0, "shrinkB": 0})
        ax.text(p["x"] * 1.1, (edge + tail) / 2, f"{p['value']:.2f}".replace("-", "−"), ha="left", va="center", fontsize=6,
                color=INK_2, zorder=5)


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
        off = offscale(itf, q)
        if off:  # DEV-0071：离群的秩 128 读数画在轴外（箭头 + 数值）
            _draw_offscale(ax, off)
    fig.supxlabel("Qwen3.5 parameters (log scale)", fontsize=7)
    handles = [Line2D([], [], marker="o", ms=3.0, mfc="none", mec=SERIES_1, ls="none", label="seed"),
               Line2D([], [], marker="o", ms=4.2, color=SERIES_1, mec="white", lw=1.1, label="seed mean (LoRA r32)")]
    if extra:
        handles.append(Line2D([], [], marker="*", ms=7.0, color=SERIES_2, mec="white", ls="none", label="0.8B, LoRA r128"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), handletextpad=0.3, columnspacing=0.9)
    return _save(fig, out, "fig_interference_size")


def _pct(rate) -> str:
    return MISSING if rate is None else f"{100 * rate:.0f}%" if rate >= 0.095 else f"{100 * rate:.1f}%"


def _signed(v: float, nd: int = 1) -> str:
    return f"{v:+.{nd}f}".replace("-", "−")


def fig_w2_size(sz: dict, out: Path) -> list[Path]:
    """W2-a − W1-F（合并 5,000 句，×100）随尺寸；9B 另画 W2-c − W1-F。横轴刻度下写 W2-a 挑选的解析失败率。"""
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.25), layout="constrained")
    xs = [size_value(s["size"]) for s in sz["sizes"]]
    tops = []
    for key, label, color, marker, ms in (("w2a-w1f", "W2-a − W1-F", W2_COLORS["W2-a"], "o", 4.2),
                                          ("w2c-w1f", "W2-c − W1-F", W2_COLORS["W2-c"], "D", 3.8)):
        pts = [(x, s["pairs"].get(key)) for x, s in zip(xs, sz["sizes"]) if s["pairs"].get(key)]
        if not pts:
            continue
        px = [x for x, _ in pts]
        pp = [{**p, "value": 100 * p["value"], "low": None if p["low"] is None else 100 * p["low"],
               "high": None if p["high"] is None else 100 * p["high"]} for _, p in pts]
        ax.errorbar(px, [p["value"] for p in pp], yerr=_yerr(pp), fmt=marker + ("-" if len(pp) > 1 else ""), color=color,
                    ecolor=color, ms=ms, lw=1.1, elinewidth=0.9, capsize=1.8, capthick=0.9, mec="white", mew=0.5, label=label, zorder=3)
        for x, p in zip(px, pp):
            tops.append((x, p["high"] if p["high"] is not None else p["value"], _signed(p["value"])))
    lo, hi = ax.get_ylim()
    for x, top, txt in tops:  # 数值写在误差棒上端之上
        ax.text(x, top + 0.035 * (hi - lo), txt, ha="center", va="bottom", fontsize=6, color=INK_2, zorder=4)
    ax.set_ylim(lo, hi + 0.1 * (hi - lo))
    ax.axhline(0, color=INK_2, lw=0.6, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s['size']}\n{_pct((s['parse_failure'].get('w2a') or {}).get('rate'))}" for s in sz["sizes"]])
    ax.minorticks_off()
    if xs:
        ax.set_xlim(min(xs) / 1.45, max(xs) * 1.45)
    ax.set_xlabel("Qwen3.5 size (log scale)\nbelow each size: W2-a selections that fail to parse")
    ax.set_ylabel("ΔcIoU vs. W1-F (×100, paired)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.3)
    return _save(fig, out, "fig_w2_size")


def fig_lang_cost(lc: dict, out: Path) -> list[Path]:
    """相对同节点基座的差（百分点）与 95% CI；读数自上而下，每个读数里各系统错开一行。"""
    systems = lc["systems"]
    n_s = len(systems)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 1.0 + 0.36 * len(LANG_READOUTS) * max(n_s, 3) / 3 + 0.25), layout="constrained")
    step = 0.8 / max(n_s, 1)
    lows = [0.0]
    highs = [0.0]
    for g, (sec, key, _lab) in enumerate(LANG_READOUTS):
        if sec == "ttext":
            ax.axhspan(g - 0.5, g + 0.5, color=BAND, zorder=0, lw=0)
        for j, s in enumerate(systems):
            p = s["values"].get(key)
            if p is None:
                continue
            y = g + (j - (n_s - 1) / 2) * step
            ax.errorbar([p["value"]], [y], xerr=_yerr([p]), fmt="o", ms=3.4, color=W2_COLORS.get(s["name"], INK_2),
                        ecolor=W2_COLORS.get(s["name"], INK_2), elinewidth=0.9, capsize=1.4, capthick=0.9, mec="white", mew=0.4, zorder=3)
            hi = p["high"] if p["high"] is not None else p["value"]
            ax.text(hi + 1.2, y, _signed(p["value"]), ha="left", va="center", fontsize=5.5, color=INK_2, zorder=4,
                    bbox={"fc": BAND if sec == "ttext" else "white", "ec": "none", "pad": 0.4})
            lows.append(p["low"] if p["low"] is not None else p["value"])
            highs.append(hi)
    ax.axvline(0, color=INK_2, lw=0.6, ls="--", zorder=1)
    ax.set_yticks(range(len(LANG_READOUTS)))
    ns = []
    for sec, key, lab in LANG_READOUTS:
        n = next((s["values"][key]["n"] for s in systems if s["values"].get(key) and s["values"][key].get("n")), None)
        ns.append(lab + (", T-text" if sec == "ttext" else "") + (f"\n(n={n:,})" if n else ""))
    ax.set_yticklabels(ns)
    ax.set_ylim(len(LANG_READOUTS) - 0.5, -0.5)
    ax.set_xlim(min(lows) - 3.0, max(highs) + 9.0)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.set_xlabel("Difference from the base (percentage points)")
    handles = [Line2D([], [], marker="o", ms=4.5, color=W2_COLORS.get(s["name"], INK_2), mec="white", ls="none", label=s["label"])
               for s in systems]
    fig.legend(handles=handles, loc="outside upper center", ncol=min(n_s, 3), handletextpad=0.2, columnspacing=0.9)
    return _save(fig, out, "fig_lang_cost")


def fig_count_bins(cb: dict, out: Path) -> list[Path]:
    """(a) CountBench 三档的准确率；(b) 与配对基座的差及配对 bootstrap 95% CI（基座两次读数逐条相同，只画一根基座）。"""
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(SINGLE_COL_IN, 3.0), layout="constrained", sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.0]})
    bins = cb["bins"]
    systems = cb["systems"]
    n_s = len(systems)
    w = 0.8 / max(n_s, 1)
    offs = [(j - (n_s - 1) / 2) * w for j in range(n_s)]
    for j, s in enumerate(systems):
        c = W2_COLORS.get(s["name"], INK_2)
        xs = [i + offs[j] for i in range(len(bins))]
        vals = [s["acc"].get(b) for b in bins]
        axa.bar([x for x, v in zip(xs, vals) if v is not None], [v for v in vals if v is not None], width=w * 0.86, color=c,
                label=s["label"], zorder=2, linewidth=0)
        pts = [s["delta"].get(b) for b in bins]
        if any(pts):
            pp = [p for p in pts if p]
            axb.errorbar([x for x, p in zip(xs, pts) if p], [p["value"] for p in pp], yerr=_yerr(pp), fmt="o", ms=3.4, color=c,
                         ecolor=c, elinewidth=0.9, capsize=1.4, capthick=0.9, mec="white", mew=0.4, zorder=3)
    axa.set_ylim(0, 1.0)
    axa.set_ylabel("Accuracy")
    axa.set_title("(a) Counting accuracy", loc="left")
    axb.axhline(0, color=INK_2, lw=0.6, ls="--", zorder=1)
    axb.set_ylabel("Δ accuracy vs. base")
    axb.set_title("(b) Difference from the base (paired, 95% CI)", loc="left")
    ns = [next((s["n"].get(b) for s in systems if s["n"].get(b)), None) for b in bins]
    axb.set_xticks(range(len(bins)))
    axb.set_xticklabels([f"{b}\n(n={n})" if n else b for b, n in zip(bins, ns)])
    axb.tick_params(axis="x", length=0)
    axb.set_xlabel("Ground-truth count (CountBench)")
    handles = [Patch(color=W2_COLORS.get(s["name"], INK_2), lw=0, label=s["label"]) for s in systems]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), handlelength=0.8, handletextpad=0.3, columnspacing=0.6)
    return _save(fig, out, "fig_count_bins")


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
        all_v1 = all(r.get("criteria_version") in (None, ITF.DEFAULT_CRITERIA) for r in e3["rejected"])
        rj = "; ".join(f"{r['size']} {dict(E3_ARMS)[r['arm']]} seed {r['seed']} ({'grid' if r['channel'] == 'grid' else 'T-text'}"
                       f", {r['machine']}" + ("" if all_v1 else f", criteria {r.get('criteria_version')}") + f"): {r['reason_en']}"
                       for r in e3["rejected"])
        if all_v1:  # 改动前的原文（DEV-0063 之前只有 v1）
            notes.append("Seeds without readings because the pre-evaluation merge check (criteria v1) rejected them (N37, kept as "
                         "v1 per D102-2): " + rj + ".")
        else:
            notes.append("Seeds without readings because the pre-evaluation merge check rejected them (N37): " + rj + ".")
    non_v1 = e3.get("non_v1_grid") or []
    if non_v1:  # DEV-0063（D102-2）：按 e5-v2 补评的网格读数
        seen: dict = {}
        for r in non_v1:
            seen.setdefault((r["size"], r["arm"], r["seed"], r["criteria"], r.get("gpu")), []).append(r["task"])
        items = "; ".join(f"{size} {dict(E3_ARMS)[arm]} seed {seed} ({'/'.join(tasks)}; criteria {c}" + (f"; {gpu}" if gpu else "") + ")"
                          for (size, arm, seed, c, gpu), tasks in seen.items())
        notes.append("Pre-evaluation merge check: criteria v1 (D50-3) for all readings except these structure-forced grid readings, "
                     "re-evaluated under criteria e5-v2 (the E5 criteria, which tolerate tie flips between the two decoding "
                     "implementations; unified per N34, decided before the rejected seeds' readings were seen; D102-2): " + items + ".")
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


def fmt_w2(p: dict | None, signed: bool = False, latex: bool = False) -> str:
    """W2 表的一格：×100、一位小数，有 CI 时加 [low, high]。"""
    if p is None or p.get("value") is None:
        return "--" if latex else "—"
    f = (lambda x: f"{100 * x:+.1f}") if signed else (lambda x: f"{100 * x:.1f}")
    has_ci = p.get("low") is not None and p.get("high") is not None
    if latex:
        return f"${f(p['value'])}$" + (f"\\,{{\\scriptsize $[{f(p['low'])}, {f(p['high'])}]$}}" if has_ci else "")
    return f(p["value"]) + (f" [{f(p['low'])}, {f(p['high'])}]" if has_ci else "")


def _runs_of(paths) -> str:
    """路径里的运行编号（experiments/<phase>/NNNN_…）→ "RUN-NNNN, …"。"""
    runs = sorted({m.group(1) for p in paths if p for m in [re.search(r"experiments/[^/]+/(\d{4})_", str(p))] if m})
    return ", ".join(f"RUN-{r}" for r in runs)


def w2_table_notes(w2: dict, metric: str) -> list[str]:
    res = "/".join(f"{r:,}" for r in w2["per_system_resamples"]) or MISSING
    e1 = [v for v in w2["e1_parse"].values() if v]
    all_fail = e1 and len(e1) == len(w2["splits"]) and all(v["n_failed"] == v["n"] for v in e1)
    notes = [f"Qwen3.5-9B on the five W2 test splits ({_n_split(w2, metric)} expressions each, the E1 polygon subset); "
             f"{metric} ×100 with 95% bootstrap CI over expressions ({res} resamples, as computed by each run). "
             + ("cIoU = summed intersection / summed union." if metric == "cIoU" else "gIoU = mean per-expression IoU."),
             "Own box: the model's own box (E1 protocol) as the mask; W1-F: that box plus the default GrabCut candidate (a fixed "
             "pipeline, no LLM decision); random / oracle pick: a random / the ground-truth-best of the four tool candidates; W2-a: the "
             "frozen model picks a candidate (zero-shot; candidate 1 = W1-F, also submitted when the pick fails to parse); W2-c: LoRA "
             "text supervision to propose the box and pick; W2-c + replay: W2-c trained with self-distillation replay (post hoc "
             "variant, descriptive)."]
    if all_fail:
        notes.append(f"E1 zero-shot polygon: all {e1[0]['n']:,} outputs per split fail to parse (strict scoring gives 0).")
    if metric == "cIoU":
        pr = w2.get("pairs_resamples")
        notes.append("Paired differences (points ×100): per-expression paired bootstrap of the ratio of sums, 95%"
                     + (f", {pr:,} resamples" if pr else "") + "; pooled = the five splits together (5,000 expressions). "
                     "Per-system pooled cIoU is not part of the analysis, so the pooled column has differences only.")
    else:
        notes.append("Paired gIoU differences for W2 were not computed in the analysis (see the cIoU table).")
    calls = [c for v in w2["llmseg_calls"].values() for c in [v.get("rec")] if c]
    calls_seg = [c for v in w2["llmseg_calls"].values() for c in [v.get("seg")] if c]
    w2c_calls = sorted({c for c in w2["w2c_calls"].values() if c})
    lres = sorted({p.get("n_resamples") for d in w2["llmseg"]["pairs"].get(metric, {}).values() for p in d.values() if p and p.get("n_resamples")})
    lls = ("External baseline LLMSeg" + (f" (official code, commit {w2['llmseg_commit'][:7]})" if w2.get("llmseg_commit") else "") +
           " driven by the same Qwen3.5-9B, zero-shot, on 300 hash-selected expressions per split (pooled 1,500), with W2-c on the "
           "same expressions alongside. Before refinement = after its first two stages; after refinement = its main configuration.")
    if calls and calls_seg:
        lls += (f" Model calls per image (split means): {_span(calls)} after vs. {_span(calls_seg)} before refinement" +
                (f"; W2-c {'/'.join(f'{c:g}' for c in w2c_calls)}" if w2c_calls else "") + ".")
    lls += (" LLMSeg − W2-c: paired bootstrap, 95%" + (f", {'/'.join(f'{r:,}' for r in lres)} resamples" if lres else "") +
            "; the pooled LLMSeg and W2-c readings have no CI in the analysis.")
    notes.append(lls)
    src = [W2T.RUNS[W2_MODEL], W2_REPLAY_RUN, LLMSEG_RUN]
    notes.append(f"Data: {_runs_of(src)} (LLMSeg comparison computed on CPU, D139)"
                 + ("; W2-a paired differences completed in RUN-0081" if metric == "cIoU" else "") + ".")
    if w2["missing"]:
        notes.append("Missing: " + "; ".join(w2["missing"]) + ".")
    return notes


def _span(xs, nd: int = 1) -> str:
    lo, hi = f"{min(xs):.{nd}f}", f"{max(xs):.{nd}f}"
    return lo if lo == hi else f"{lo}–{hi}"


def _n_split(w2: dict, metric: str) -> str:
    ns = sorted({p["n"] for d in w2["rows"].get(metric, {}).values() for p in d.values() if p and p.get("n")})
    return "/".join(f"{n:,}" for n in ns) if ns else MISSING


def tab_w2(w2: dict, out: Path, metric: str, name: str) -> list[Path]:
    """系统为行、划分（加合并）为列；三块：9B 各系统、配对差（只有 cIoU）、外部基线 LLMSeg。"""
    splits = w2["splits"]
    cols = splits + ["pooled"]
    heads = [split_label(sp) for sp in splits] + ["Pooled"]
    blocks = [(f"Qwen3.5-9B, {_n_split(w2, metric)} expressions per split ({metric} ×100, 95% CI)",
               [(lab, w2["rows"][metric].get(k, {}), False) for k, lab in W2_ROWS])]
    if metric == "cIoU":
        blocks.append(("Paired differences (points ×100, 95% paired bootstrap CI)",
                       [(lab, w2["pairs"].get(k, {}), True) for k, lab in W2_PAIRS]))
    blocks.append(("External baseline: LLMSeg with zero-shot Qwen3.5-9B, 300 expressions per split (pooled 1,500)",
                   [(lab, w2["llmseg"]["rows"].get(metric, {}).get(k, {}), False) for k, lab in LLMSEG_ROWS] +
                   [(lab, w2["llmseg"]["pairs"].get(metric, {}).get(k, {}), True) for k, lab in LLMSEG_PAIRS]))
    notes = w2_table_notes(w2, metric)
    title = "W2 test" + (" (gIoU, appendix)" if metric == "gIoU" else "")
    caption = (f"\\caption{{{title} on Qwen3.5-9B: referring segmentation {metric} per split. " +
               " ".join(_tex_escape(n) for n in notes) + "}")
    L = ["% Generated by scripts/figures/make_paper_figures.py; do not edit. Needs \\usepackage{booktabs,graphicx}.",
         "\\begin{table*}[t]", "\\centering", caption, f"\\label{{tab:{name.replace('tab_', '').replace('_', '-')}}}", "\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
         "System & " + " & ".join(_tex_escape(h) for h in heads) + " \\\\"]
    M = ["| System | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(cols)]
    for title_b, rows in blocks:
        L += ["\\midrule", f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\emph{{{_tex_escape(title_b)}}}}} \\\\"]
        M.append(f"| *{title_b}* |" + " |" * len(cols))
        for lab, cells, signed in rows:
            L.append(_tex_escape(lab) + " & " + " & ".join(fmt_w2(cells.get(c), signed, True) for c in cols) + " \\\\")
            M.append(f"| {lab} | " + " | ".join(fmt_w2(cells.get(c), signed, False) for c in cols) + " |")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    md = [f"# Table: {title} (generated by scripts/figures/make_paper_figures.py; do not edit by hand)", ""] + M + [""] + \
         [f"- {n}" for n in notes] + [""]
    p_tex, p_md = out / f"{name}.tex", out / f"{name}.md"
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
    e5 = sorted({(b["size"], int(k), c) for b in main for d in (b.get("pair_criteria") or {}).values() for k, c in d.items()
                 if c != ITF.DEFAULT_CRITERIA}, key=lambda x: (size_value(x[0]), x[1]))
    if e5:  # DEV-0063（D102-2）：全是 v1 时没有这一句
        s += (" Pixel pairs re-evaluated under merge-check criteria e5-v2, both arms (see Table E3): " +
              "; ".join(f"{sz} seed {k}" for sz, k, _c in e5) + ".")
    if any(b.get("rank") for b in itf["blocks"]):
        s += (" Star: 0.8B with LoRA rank 128 for both C3 and E3b+C3 (post hoc, §13a/D99-1; P uses the rank-32 E3b), "
              "drawn slightly right of 0.8B.")
    for q, _ylabel, title in INTERFERENCE_PANELS:  # DEV-0071：画到轴外的读数（没有时没有这一句）
        off = offscale(itf, q)
        if off:
            pts = "; ".join(f"{p['block'].replace('r128', 'rank 128')} {'seed ' + str(p['seed']) if p['kind'] == 'seed' else 'seed mean'}: "
                            f"{q} = {p['value']:.2f}".replace("-", "−") for p in off["points"])
            s += (f" In {title.split(' ')[0]}, {'a rank-128 reading lies' if len(off['points']) == 1 else 'rank-128 readings lie'} far "
                  f"outside the range of the rank-32 readings and {'is' if len(off['points']) == 1 else 'are'} marked by an arrow at the "
                  f"axis edge with {'its value' if len(off['points']) == 1 else 'their values'} ({pts}); the axis covers all other points.")
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


def caption_w2_size(sz: dict) -> str:
    res = sz.get("n_resamples")
    pf = []
    for s in sz["sizes"]:
        r = (s["parse_failure"].get("w2a") or {})
        pf.append(f"{s['size']} {_pct(r.get('rate'))} ({r['n']:,}/{r['of']:,})" if r.get("rate") is not None and r.get("complete")
                  else f"{s['size']} see text")
    w2c = [s for s in sz["sizes"] if s["pairs"].get("w2c-w1f")]
    w2c_pf = [(s["size"], (s["parse_failure"].get("w2c") or {}).get("rate")) for s in w2c]
    s = ("W2-a − W1-F across model size (Qwen3.5 2B, 4B, 9B; W2 test, five splits pooled, 5,000 expressions; cIoU ×100; "
         f"95% paired bootstrap CI{f', {res:,} resamples' if res else ''}). Diamond: W2-c − W1-F, run at "
         f"{', '.join(x['size'] for x in w2c) or 'no size'} only. Below each size: the share of W2-a's selection answers that fail to "
         f"parse ({'; '.join(pf)}); a selection that fails to parse falls back to W1-F, which pulls W2-a toward W1-F at small sizes.")
    if w2c_pf and all(r is not None for _sz, r in w2c_pf):
        s += " W2-c selection parse failures: " + "; ".join(f"{sz_} {_pct(r)}" for sz_, r in w2c_pf) + "."
    s += f" Data: {W2_TEST_RUNS}."
    if sz["missing"]:
        s += " Missing: " + "; ".join(sz["missing"]) + "."
    return s


def caption_lang_cost(lc: dict) -> str:
    main = [s for s in lc["systems"] if not s["explore"]]
    expl = [s for s in lc["systems"] if s["explore"]]
    base = next((s.get("base_pct") for s in main if s.get("base_pct")), None) or {}
    bt = " / ".join(f"{base[k]:.1f}%" for _sec, k, _lab in LANG_READOUTS if k in base)
    flags = []
    for s in lc["systems"]:
        fl = ", ".join(f"{HEALTH_LABELS.get(f['reading'], f['reading'])} {_signed(f['delta_pp'])} (n={f['n']})" for f in s["health_flags"])
        flags.append(f"{s['label']}: {fl or 'none'}")
    srcs = [v for s in main for v in (s.get("sources") or {}).values() if isinstance(v, str)]
    s = ("Language cost on Qwen3.5-9B: difference from the base model evaluated on the same node (percentage points; 95% paired "
         "bootstrap CI). MMLU-Pro (letter answers) and IFEval (prompt-level strict) from the large language battery; grounding "
         "Acc@0.5 and counting accuracy (CountBench) on the T-text channel (shaded"
         + (f"; base {bt}" if bt else "") + "). Systems: E3b+C3 (E3), W2-c, and W2-c + replay (post hoc self-distillation replay). "
         "Not drawn: the small health check (brief §8.11: MMLU-Pro 100, GSM8K 50, IFEval 50 items) flags "
         + "; ".join(flags[:len(main)]) + ".")
    if expl:
        s += (" Exploration candidates (not yet judged): " + ", ".join(f"{x['label']} (base on {x['node']})" for x in expl) +
              "; T-text readouts were not measured for them; health-check flags " + "; ".join(flags[len(main):]) + ".")
    s += f" Data: {LANG_COST_JSON.split('/')[-1]} ({_runs_of(srcs)})" + (f"; {EXPLORE_RUNS}" if expl else "") + "."
    if lc["missing"]:
        s += " Missing: " + "; ".join(lc["missing"]) + "."
    return s


def caption_count_bins(cb: dict) -> str:
    ns = [next((x["n"].get(b) for x in cb["systems"] if x["n"].get(b)), None) for b in cb["bins"]]
    pairs = {}
    for x in cb["systems"]:
        if x["paired_base"]:
            pairs.setdefault(x["paired_base"], []).append(x["label"])
    srcs = (cb.get("sources") or {})
    run_of = {k: _runs_of([v]) for k, v in srcs.items()}
    nb = cb.get("n_boot")
    s = (f"Counting accuracy by ground-truth count on CountBench ({sum(n for n in ns if n) or MISSING} items; bins "
         + ", ".join(f"{b} (n={n})" for b, n in zip(cb["bins"], ns)) + "), Qwen3.5-9B, T-text channel. (a) Accuracy. (b) Difference "
         "from the base, 95% paired bootstrap CI" + (f" ({nb:,} resamples)" if nb else "") + ". "
         + "; ".join(f"{', '.join(v)} paired with the base run of {run_of.get(k) or k}" for k, v in pairs.items()) + ".")
    if cb["bases_identical"]:
        s += (f" The two base runs give identical answers (agreement {cb['same_answer_rate']:.3f}), so one base bar is drawn.")
    s += f" Data: {COUNT_ERRORS_JSON.split('/')[-1]} ({_runs_of(srcs.values())})."
    if cb["missing"]:
        s += " Missing: " + "; ".join(cb["missing"]) + "."
    return s


def _git_commit(root: Path) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, timeout=20).stdout.strip()
        return (sha or "unknown") + (" (working tree has uncommitted changes)" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def readme_w2(w2: dict, sz: dict, lc: dict, cb: dict, captions: dict) -> list[str]:
    """README 的 W2 各节（DEV-0071）。"""
    leaf = f"{W2T.RUNS[W2_MODEL]}/{W2_MODEL}/refseg/<设置>/<划分>/metrics.json"
    res = "/".join(f"{r:,}" for r in w2["per_system_resamples"]) or MISSING
    L = ["## 5. W2 test 主表：`tab_w2_test.tex` / `tab_w2_test.md`（gIoU 附录表 `tab_w2_test_giou.*`）", "",
         "- 数据（都是已有的读数，不重算 bootstrap）：",
         f"  - 9B 各设置（自己的框、W1-F、随机挑选、最优候选、W2-a、W2-c、E1 零样本多边形）的读数与 95% CI：`{W2_TEST_JSON}` 的 "
         f"models.{W2_MODEL}.per_split（RUN-0072；CI 是各运行 metrics.json 里运行时算的，{res} 次重抽）。核对：与 `{leaf}` 逐格相同。",
         f"  - W2-c + 回放：`{W2_REPLAY_RUN}/{W2_MODEL}/refseg/{W2_REPLAY_SETTING}/<划分>/metrics.json` 的 headline（RUN-0076 补充 2）。",
         f"  - 配对差 W2-c − W1-F、W2-a − W1-F：同一份 JSON 的 pairs 与 pairs_pooled（配对 bootstrap，ratio-of-sums，"
         f"{w2.get('pairs_resamples') or 0:,} 次，种子 {w2.get('pairs_seed') or MISSING}；W2-a 由 RUN-0081 在 HPC 上补齐）。",
         f"  - 回放 − W2-c：**没有 JSON**，只在 `{W2_REPLAY_MD}` §1 的表里（3 位小数；重抽次数文档里没写），从那张表解析；核对：表里的 W2-c、"
         "回放读数与上两项的值四舍五入相同。",
         f"  - E1 零样本多边形的解析失败：`{W2T.RUNS[W2_MODEL]}/{W2_MODEL}/refseg/baseline_e1/<划分>/metrics.json` 的 parse。",
         f"  - 外部基线 LLMSeg（RUN-0078）：修正前（seg）、修正后（rec，主配置）的读数与 CI 取 `{LLMSEG_RUN}/{W2_MODEL}/refseg/llmseg/<划分>/metrics.json` "
         f"的 readouts.<seg|rec>.headline；合并读数、同样 300 句上的 W2-c 读数（RUN-0072）与 LLMSeg − W2-c 的配对差取 `{LLMSEG_COMPARE}`（D139，CPU 补跑）；"
         "每图调用次数取 readouts.<seg|rec>.compute_per_image.n_model_calls。",
         "- 版式：系统为行、划分（加合并）为列——论文里指代分割表的惯例；任务书\"列：自己的框、W1-F…\"按\"列出这些系统\"理解。"
         "三块：9B 各系统；配对差（W2-c − W1-F、W2-a − W1-F、回放 − W2-c）；外部基线 LLMSeg（修正前、修正后两行，同句 W2-c 并列，下接两行配对差）。"
         "数值 ×100、一位小数，CI 用 [low, high]。",
         "- 合并列：分析里没有逐系统的合并读数（w2_test_table.json 只有合并的配对差；W2-a 的逐句数据超过同步上限，不在仓库），"
         "所以 9B 各系统一块的合并列是\"—\"，合并只出现在配对差与 LLMSeg 两块（LLMSeg 的合并读数没有 CI）。",
         "- gIoU 附录表：同一版式；W2 的配对差一块没有（分析只算了 cIoU 的配对差），LLMSeg − W2-c 的 gIoU 差取 compare.json。",
         "", "**表注（英文，cIoU）**", ""] + [f"- {n}" for n in w2_table_notes(w2, "cIoU")] + [""]
    if w2["checks"]:
        L += ["核对：", ""] + [f"- {c}" for c in w2["checks"]] + [""]
    pf_src = "; ".join(f"{s['size']}：`{s['parse_failure']['w2a']['source']}`" for s in sz["sizes"] if s["parse_failure"].get("w2a"))
    L += ["## 6. W2-a 随尺寸：`fig_w2_size.pdf` / `fig_w2_size.png`", "",
          f"- 数据：`{W2_TEST_JSON}` 的 models.<模型>.pairs_pooled（w2a-w1f；9B 另有 w2c-w1f），合并 5,000 句；{W2_TEST_RUNS}。",
          f"- 挑选的解析失败率（横轴刻度下）：W2-a 叶目录 metrics.json 的 pick.parse_failure，5 个划分合计 Σn ÷ Σof（分母 = 有候选的句子）：{pf_src or MISSING}。"
          "9B 的数在 RUN-0072 的结果目录里有（w2_test.md 第 2 节写的\"见 RUN-0072 报告\"），直接取。",
          "- 口径：纵轴 = 配对差 ×100；误差棒 = 95% 配对 bootstrap CI；横轴 = 名义参数量（对数轴）。W2-c 只在 9B 跑过（菱形）。",
          "", "**图注（英文）**", "", captions["fig_w2_size"], ""]
    explore = "（本次生成时 --with-explore 打开，画了探索的候选）" if lc["with_explore"] else "（本次生成时 --with-explore 关着，缺省）"
    L += ["## 7. 语言代价：`fig_lang_cost.pdf` / `fig_lang_cost.png`", "",
          f"- 数据：`{LANG_COST_JSON}`（`scripts/mgr/lang_cost_table.py`，D125-3）的 battery（大题量 MMLU-Pro 2,000 题、IFEval 541 条）与 ttext（定位、计数）"
          "四个读数，每格是相对同节点基座的差（百分点）与 95% CI；健康检查的标红（health.*.flag）只写进图注，不进图。各系统的来源路径（JSON 原文）：",
          ] + [f"  - {s['name']}：" + "；".join(f"{k} `{v}`" for k, v in (s.get("sources") or {}).items() if isinstance(v, str) and "/" in v)
               for s in lc["systems"] if not s["explore"]] + [
          f"- 探索的候选（X-2、X-3、X-3L）：判定还没做，缺省不画；`--with-explore` 打开后，大题量取 "
          + "、".join(f"`{b}`" for _n, b, _h, _nd in EXPLORE_LANG) + "，健康检查的标红取 " + "、".join(f"`{h}`" for _n, _b, h, _nd in EXPLORE_LANG)
          + f"（{EXPLORE_RUNS}）；它们没有 T 文本读数，图注写明。{explore}",
          "- 版式：读数自上而下（底色 = T 文本通道），每个读数里三个系统各一行，点 = 差，横线 = 95% CI，右侧写数值。",
          "", "**图注（英文）**", "", captions["fig_lang_cost"], "",
          "## 8. 计数分档：`fig_count_bins.pdf` / `fig_count_bins.png`", "",
          f"- 数据：`{COUNT_ERRORS_JSON}`（`scripts/mgr/count_errors.py`）的三个大档 {', '.join(cb['bins'])}："
          "(a) 各系统的准确率（bins.<档>.acc）；(b) 与配对基座的差及配对 bootstrap 95% CI（bins.<档>.vs_base）。预测文件（JSON 原文）：",
          ] + [f"  - {k}：`{v}`" for k, v in (cb.get("sources") or {}).items()] + [
          "- 为什么分两个面板：分析 JSON 里只有\"与基座之差\"的 CI，没有各档准确率本身的 CI（不重算），所以 CI 画在 (b) 上，(a) 只画准确率。",
          f"- 基座：W2-c、W2-c + 回放配 base@0069，C3、E3b+C3 配 base@0056；两次基座读数逐条相同（{cb.get('same_answer_rate')}）时只画一根（base@0069）。",
          "", "**图注（英文）**", "", captions["fig_count_bins"], ""]
    return L


def write_readme(root: Path, out: Path, e1, e2, e3, itf, captions: dict, files: dict, w2=None, sz=None, lc=None, cb=None) -> Path:
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = ["# 论文用图表（docs/figures/）", "",
         "<!-- 由 scripts/figures/make_paper_figures.py 生成；不要手改。数据补齐后重跑：python scripts/figures/make_paper_figures.py -->", "",
         f"- 生成时间：{now}",
         f"- 代码 commit：{_git_commit(root)}",
         f"- 结果树：`{'仓库根' if root == REPO else root}`",
         f"- matplotlib {matplotlib.__version__}；PDF 是矢量（TrueType 字体，无 Type 3），PNG 300 dpi；宽度：单栏 {SINGLE_COL_IN} in，双栏 {DOUBLE_COL_IN} in。",
         "- 配色：dataviz 参考调色板。Qwen3.5 四个尺寸用一个有序的蓝色阶（浅→深 = 0.8B→9B），InternVL3.5 用第二个有序色阶（橙）；"
         "两类对比用分类槽 1（蓝）与槽 2（橙）。都在白底上过了 validate_palette.js（有序：单色相、亮度单调、相邻 ΔL ≥ 0.06；分类两色全对 CVD ΔE 24.7）。",
         "- W2 的图（DEV-0071）：同一个系统在各张图里同一个颜色——W2-c 槽 1（蓝）、E3b+C3 槽 2（橙）、W2-c + 回放 槽 3（青）；基座与 C3 "
         "（对照）用中性灰（浅、深）；W2-a 只在 fig_w2_size 出现，用槽 2（与 E3b+C3 不同图）。槽 1–3 三色全对过 validate_palette.js（CVD ΔE ≥ 9.2）；"
         "--with-explore 的 X-2、X-3、X-3L 用槽 4–6，按画出的相邻顺序过了检查。青色对白底不足 3:1，图上带数值标签或有表可查（lang_cost_9b.md、count_errors_9b.md）。",
         "- 每个数字都从结果文件读；逐点来源见 `data/*.json`（每个点带 source 路径）。", ""]
    # 缺的数据
    w2s = [x for x in (w2, sz, lc, cb) if x is not None]
    miss = e1["missing"] + e2["missing"] + e3["missing"] + itf["missing"] + [m for x in w2s for m in x["missing"]]
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
                 ("" if s["present"] else "——**还没有（missing）**") +
                 (f"；补评目录（--e3-prepend，排在最前、先到先得）`{', '.join(s['prepended'])}`" if s.get("prepended") else ""))
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
    offs = [o for q, *_ in INTERFERENCE_PANELS for o in [offscale(itf, q)] if o]
    L[L.index("**图注（英文）**", L.index("## 4. 干扰随尺寸的曲线：`fig_interference_size.pdf` / `fig_interference_size.png`")) - 1:
      L.index("**图注（英文）**", L.index("## 4. 干扰随尺寸的曲线：`fig_interference_size.pdf` / `fig_interference_size.png`")) - 1] = [
        f"- (d) R_seg 的离群读数（DEV-0071，任务书第 5 条，二选一选了\"箭头 + 数值\"，没有用断轴）：秩 128 的读数（种子或平均）落在 r32 全部 R_seg 读数"
        f"（各种子与平均）的范围 [lo, hi] 向外再扩 {OFFSCALE_SPANS:g} 倍跨度的窗口之外时，不画进坐标轴，改在轴边画一个指向它的箭头并写数值；"
        "纵轴只容纳窗口内的点（含秩 128 的其余点与平均值）。其余三个面板不变。本次：" +
        ("；".join(f"{p['block']} {'种子 ' + str(p['seed']) if p['kind'] == 'seed' else '平均'} = {p['value']:.3f}"
                  f"（窗口 [{o['window'][0]:.3f}, {o['window'][1]:.3f}]，纵轴 [{o['ylim'][0]:.3f}, {o['ylim'][1]:.3f}]）"
                  for o in offs for p in o["points"]) or "没有落在窗口之外的点，面板与改动前相同") + "。"]
    if w2 is not None:
        L += readme_w2(w2, sz, lc, cb, captions)
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
    ap.add_argument("--e3-prepend", action="append", default=[], metavar="SIZE=DIR[,DIR…]",
                    help="DEV-0063：把补评目录（相对 --root）放在该尺寸 E3 目录列表的最前面（先找到的为准；可重复）。"
                         "干扰的 JSON 要用 interference.py 按同样的目录顺序重出，否则核对报不一致")
    ap.add_argument("--with-explore", action="store_true",
                    help="DEV-0071：语言代价图另画探索的候选 X-2、X-3、X-3L（判定还没做，缺省不画）")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    known = [s["size"] for s in E3_SIZES]
    prepend: dict = {}
    for item in a.e3_prepend:
        size, sep, dirs = item.partition("=")
        if not sep or size not in known or not dirs:
            ap.error(f"--e3-prepend 要写成 SIZE=DIR[,DIR…]，SIZE 取 {known}，收到 {item!r}")
        for d in dirs.split(","):
            if not (root / d).is_dir():
                ap.error(f"--e3-prepend 的目录 {d!r} 不在 {root} 下")
            prepend.setdefault(size, []).append(d)
    sizes = prepend_dirs(E3_SIZES, prepend)
    out = Path(a.out).resolve() if a.out else root / "docs" / "figures"
    (out / "data").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    e1, e2, e3, itf = collect_e1(root), collect_e2(root), collect_e3(root, sizes), collect_interference(root, sizes)
    files = {"fig_e1_gap": fig_e1(e1, out), "fig_e2_layers": fig_e2(e2, out), "tab_e3_comparison": tab_e3(e3, out),
             "fig_interference_size": fig_interference(itf, out)}
    captions = {"fig_e1_gap": caption_e1(e1), "fig_e2_layers": caption_e2(e2), "fig_interference_size": caption_interference(itf)}
    for name, cap in captions.items():
        p = out / f"{name}_caption.tex"
        p.write_text(f"% Generated by scripts/figures/make_paper_figures.py; do not edit.\n\\caption{{{_tex_escape(cap)}}}\n", encoding="utf-8")
        files[name].append(p)
    offs = {q: o for q, *_ in INTERFERENCE_PANELS for o in [offscale(itf, q)] if o}
    if offs:  # DEV-0071：画到轴外的读数（没有时不加这个键，data JSON 与改动前相同）
        itf["offscale"] = offs
    for name, data in (("fig_e1_gap", e1), ("fig_e2_layers", e2), ("tab_e3_comparison", e3), ("fig_interference_size", itf)):
        p = out / "data" / f"{name}.json"
        p.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        files[name].append(p)
    # W2（DEV-0071）：在原有四项之后出，原有输出不受影响
    w2, sz, lc, cb = collect_w2(root), collect_w2_size(root), collect_lang_cost(root, a.with_explore), collect_count_bins(root)
    for metric, name in W2_METRICS:
        files[name] = tab_w2(w2, out, metric, name)
    files["fig_w2_size"] = fig_w2_size(sz, out)
    files["fig_lang_cost"] = fig_lang_cost(lc, out)
    files["fig_count_bins"] = fig_count_bins(cb, out)
    w2caps = {"fig_w2_size": caption_w2_size(sz), "fig_lang_cost": caption_lang_cost(lc), "fig_count_bins": caption_count_bins(cb)}
    for name, cap in w2caps.items():
        p = out / f"{name}_caption.tex"
        p.write_text(f"% Generated by scripts/figures/make_paper_figures.py; do not edit.\n\\caption{{{_tex_escape(cap)}}}\n", encoding="utf-8")
        files[name].append(p)
    captions.update(w2caps)
    for name, data in (("tab_w2_test", w2), ("fig_w2_size", sz), ("fig_lang_cost", lc), ("fig_count_bins", cb)):
        p = out / "data" / f"{name}.json"
        p.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        files[name].append(p)
    readme = write_readme(root, out, e1, e2, e3, itf, captions, files, w2, sz, lc, cb)
    print(f"wrote {sum(len(v) for v in files.values()) + 1} files to {out} (README: {readme})")
    miss = e1["missing"] + e2["missing"] + e3["missing"] + itf["missing"] + w2["missing"] + sz["missing"] + lc["missing"] + cb["missing"]
    if miss:
        print("missing:\n  " + "\n  ".join(miss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
