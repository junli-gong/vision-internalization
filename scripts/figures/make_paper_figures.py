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
- 通过 G-EX 的主方法（DEV-0073；D147、D149）：C1 = X-2 是论文的主方法，C2 = X-3 是"去掉修正流"的消融，另有能力归属的
  椭圆消融（预注册 X-5a、X-6）。tab_w2_test 与 gIoU 表加这三行与配对差，合并列取 docs/analysis/w2_test_pooled.json（D152）；
  新增正文主表 tab_main、语言代价四方表 tab_lang_cost（附录版 tab_lang_cost_full 加 X-3）、计算表 tab_compute。配对差、LLMSeg 同句与
  计算记账取 docs/analysis/x_test_table.json。语言代价图与计数分档缺省画 X-2；--with-explore 再加 X-3（C2）与 X-3L。
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
XT = _load_script("_paperfig_x_test_table", "scripts/mgr/x_test_table.py")  # 候选的运行目录与设置名（CANDS、MAIN、ABL、M）
WTP = _load_script("_paperfig_w2_test_pooled", "scripts/mgr/w2_test_pooled.py")  # 合并读数的系统名（SYSTEMS：名字、运行目录、设置；REF）

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
# --with-explore 另画的探索候选（没有进 G-EX，预注册 X-5"不选"）：(名字, 大题量 compare, 健康检查 compare, 基座所在节点)。没有 T 文本读数。
EXPLORE_LANG = (
    ("X-3L", "experiments/X/0084_lb_explore_hpcgpu06/lb_compare_x3l.json",
     "experiments/X/0083_x3l_hpcgpu06/0083_eval/health/health_compare.json", "hpcgpu06"),
)
EXPLORE_RUNS = "RUN-0083 (health), RUN-0084 (battery)"

# tab_w2_test 的行：(键, 英文名)。键是 w2_test_table.json per_split 的设置名；"replay" 从回放的结果目录读。
W2_ROWS = (("ownbox", "Own box"), ("w1f", "W1-F"), ("random_pick", "Random pick"), ("oracle_pick", "Oracle pick"),
           ("w2a", "W2-a"), ("w2c", "W2-c"), ("replay", "W2-c + replay"), ("baseline_e1", "E1 zero-shot polygon"))
W2_PAIRS = (("w2c-w1f", "W2-c − W1-F"), ("w2a-w1f", "W2-a − W1-F"), ("replay-w2c", "W2-c + replay − W2-c"))
LLMSEG_ROWS = (("seg", "LLMSeg, before refinement"), ("rec", "LLMSeg, after refinement (main)"), ("w2c", "W2-c, same expressions"))
LLMSEG_PAIRS = (("rec", "LLMSeg after − W2-c"), ("seg", "LLMSeg before − W2-c"))
W2_METRICS = (("cIoU", "tab_w2_test"), ("gIoU", "tab_w2_test_giou"))
DATASET_LABELS = {"refcoco": "RefCOCO", "refcoco+": "RefCOCO+", "refcocog": "RefCOCOg"}

# 通过 G-EX 的候选（DEV-0073；D147、D149；预注册 X-5a、X-6）：C1 = X-2 主方法，C2 = X-3 消融（去掉修正流），C1 的椭圆消融（能力归属）。
# 运行目录、设置名取自 x_test_table.py（XT.CANDS、XT.MAIN、XT.ABL、XT.M），合并读数的系统名取自 w2_test_pooled.py（WTP.SYSTEMS）。
X_TEST_JSON = "docs/analysis/x_test_table.json"  # 候选的配对差、LLMSeg 同句、计算记账（RUN-0092 对 RUN-0072、RUN-0078）
W2_POOLED_JSON = "docs/analysis/w2_test_pooled.json"  # 9B 各系统合并 5,000 句的读数与 CI（D152；W2-a 由 RUN-0095 抽回逐句交并）
C1, C2 = "C1 (X-2)", "C2 (X-3)"  # x_test_table.json 的候选名
assert C1 in XT.CANDS and C2 in XT.CANDS, f"x_test_table.py 的候选名变了：{list(XT.CANDS)}"
# (键, 候选, 设置, x_test_table.json 里的读数键与计算键, 英文名)
X_ROWS = (("c1", C1, XT.MAIN, "main", "C1 = X-2 (ours)"),
          ("c1_ellipse", C1, XT.ABL, "ablation", "C1, in-box ellipse instead of spline"),
          ("c2", C2, XT.MAIN, "main", "C2 = X-3 (no correction stream)"))
# 配对差（cIoU 的键；gIoU 加前缀 g_）：(键, 候选, x_test_table.json 的键, 英文名)
X_PAIRS = (("c1-w2c", C1, "main_minus_w2c", "C1 − W2-c"), ("c1-w1f", C1, "main_minus_w1f", "C1 − W1-F"),
           ("c1-ellipse", C1, "main_minus_ablation", "C1 − in-box ellipse ablation"), ("c2-w2c", C2, "main_minus_w2c", "C2 − W2-c"))
# LLMSeg 一块（同样 1,500 句）：x_test_table.json 的 llmseg 子块
X_LLMSEG_ROWS = (("c1", "main_on_subset", "C1 = X-2, same expressions"),)
X_LLMSEG_PAIRS = (("c1-rec", "main_minus_llmseg_rec", "C1 − LLMSeg after"), ("c1-seg", "main_minus_llmseg_seg", "C1 − LLMSeg before"))
ROW_LABELS = dict(W2_ROWS) | {k: lab for k, _c, _s, _j, lab in X_ROWS}
SHORT_LABELS = {"ownbox": "own box", "w1f": "W1-F", "oracle_pick": "oracle pick", "w2a": "W2-a", "w2c": "W2-c", "c1": "C1",
                "c1_ellipse": "ellipse ablation", "c2": "C2"}  # 表注 "Data:" 一句里用的短名
# tab_main（正文主表）：两组行（工具与 LLM 挑选、我们的方法与消融）
MAIN_ROWS = (("ownbox", "w1f", "oracle_pick", "w2a", "w2c"), ("c1", "c1_ellipse", "c2"))
# tab_compute 的行；计算记账取 x_test_table.json：候选取 candidates.<候选>.compute.<键>，其余取 compute_reference.<名>
COMPUTE_ROWS = (("w1f", "W1-F"), ("w2c", "W2-c"), ("c1", ROW_LABELS["c1"]), ("c1_ellipse", ROW_LABELS["c1_ellipse"]),
                ("c2", ROW_LABELS["c2"]), ("llmseg", "LLMSeg (full pipeline)"))
COMPUTE_REF = {"w1f": "W1-F", "w2c": "W2-c", "llmseg": "LLMSeg"}  # compute_reference 的名字（LLMSeg 按前缀找：名字里带句数）
COMPUTE_WALL_NA = ("llmseg",)  # 墙钟与其他行不同口径，不列（x_test_table.md 的说明）
COMPUTE_COLS = (("n_calls", "Model calls", 2), ("generated_tokens", "Generated tokens", 1), ("prompt_tokens", "Prompt tokens", 0),
                ("call_time_s", "Model-call time (s)", 2), ("wall_time_s", "Wall clock (s)", 2))  # (键, 英文表头, 小数位；同 x_test_table.md)

# fig_lang_cost：(lang_cost_9b.json 的节, 键, 英文标签)。键取自 lang_cost_table 的 BATTERY_ROWS、TTEXT（前两项、两项）。
LANG_READOUTS = (
    ("battery", ".".join(LCT.BATTERY_ROWS[0]), "MMLU-Pro"),
    ("battery", ".".join(LCT.BATTERY_ROWS[1]), "IFEval prompt\nstrict"),
    ("ttext", LCT.TTEXT[0][0], "Grounding\nAcc@0.5"),
    ("ttext", LCT.TTEXT[1][0], "Counting\naccuracy"),
)
# lang_cost_9b.json 的系统名（= lang_cost_table.SYSTEMS 的键）→ 英文名
LANG_LABELS = {"E3b+C3": "E3b+C3", "W2-c": "W2-c", "W2-c+回放": "W2-c + replay", "X-2（C1）": "C1 = X-2 (ours)",
               "X-3（C2，消融）": "C2 = X-3 (ablation)"}
LANG_SYSTEMS = tuple((n, LANG_LABELS[n]) for n in ("E3b+C3", "W2-c", "W2-c+回放", "X-2（C1）"))  # 四方对照（D147-4、预注册 X-6）
LANG_ABLATION = (("X-3（C2，消融）", LANG_LABELS["X-3（C2，消融）"]),)  # 附录版与 --with-explore 另加
HEALTH_LABELS = {"mmlu_pro_letter": "MMLU-Pro", "gsm8k": "GSM8K", "ifeval": "IFEval"}
# tab_lang_cost 的英文行名（键同 lang_cost_9b.json）
LANG_TABLE_LABELS = {"battery": {".".join(LCT.BATTERY_ROWS[0]): "MMLU-Pro, letter answers", ".".join(LCT.BATTERY_ROWS[1]): "IFEval, prompt-level strict"},
                     "ttext": {LCT.TTEXT[0][0]: "Grounding Acc@0.5", LCT.TTEXT[1][0]: "CountBench counting accuracy"}}
LANG_SECTIONS = (("health", "Health check"), ("battery", "Language battery"), ("ttext", "T-text"))
# fig_count_bins：(count_errors_9b.json 的系统名, 英文名)；基座两次读数逐条相同时只画一根（base@0069）
COUNT_SYSTEMS = (("base@0069", "Base"), ("W2-c", "W2-c"), ("W2-c+回放", "W2-c + replay"), ("X-2（C1）", "C1 = X-2 (ours)"),
                 ("C3", "C3"), ("E3b+C3", "E3b+C3"))
COUNT_EXPLORE = (("X-3（C2）", "C2 = X-3 (ablation)"),)  # --with-explore 时排在 X-2 之后
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
# DEV-0073：C1 = X-2（主方法）固定用槽 7（紫，对白底 ≥ 3:1，C1 出现的每张图都是它）；与槽 1–3 全对过 validate_palette.js
# （CVD ΔE ≥ 9.2，常规 ≥ 16.3）；--with-explore 的 X-3（C2）用槽 5、X-3L 用槽 6，按画出的相邻顺序过检查。槽 4（黄）不再用。
SERIES_3, SERIES_5, SERIES_6, SERIES_7 = "#1baf7a", "#e87ba4", "#008300", "#4a3aa7"  # 分类槽 3、5、6、7
W2_COLORS = {"W2-c": SERIES_1, "W2-a": SERIES_2, "E3b+C3": SERIES_2, "W2-c+回放": SERIES_3, "base@0069": MUTED, "C3": INK_2,
             "X-2（C1）": SERIES_7, "X-3（C2，消融）": SERIES_5, "X-3（C2）": SERIES_5, "X-3L": SERIES_6}


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
    """分析 JSON 里 {"value"/"diff"/"ciou"/"giou", "ci": [low, high]} 的写法 → _point（后两种是 x_test_table、w2_test_pooled 的写法）。"""
    if not isinstance(d, dict):
        return None
    v = next((d[k] for k in ("value", "diff", "ciou", "giou") if k in d), None)
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


def row_run_setting(key: str) -> tuple[str, str]:
    """tab_w2_test 一行的（运行目录, 设置）：9B 各设置在 RUN-0072，回放在 RUN-0076 补充 2，候选在 RUN-0092。"""
    x = next((r for r in X_ROWS if r[0] == key), None)
    if x is not None:
        return XT.CANDS[x[1]], x[2]
    if key == "replay":
        return W2_REPLAY_RUN, W2_REPLAY_SETTING
    return W2T.RUNS[W2_MODEL], key


def pooled_name(key: str) -> str | None:
    """w2_test_pooled.json 里这一行的系统名：按（运行目录, 设置）对上 w2_test_pooled.SYSTEMS（W2-a 的运行目录记 None，即 RUN-0072 的 w2a）。"""
    run, setting = row_run_setting(key)
    return next((n for n, r, s in WTP.SYSTEMS if (r or WTP.REF) == run and s == setting), None)


def _x_leaf(root: Path, cand: str, setting: str, sp: str) -> Path:
    """候选的叶目录：<RUN-0092 的运行目录>/<模型>/refseg/<设置>/<划分>/（同 x_test_table.leaf，根换成 --root）。"""
    return root / XT.CANDS[cand] / XT.M / "refseg" / setting / sp


def _x_cell(xj, cand: str, sp: str, key: str, out: dict, metric: str, row: str, sub: str | None = None) -> dict | None:
    """x_test_table.json 的一格 → _point。候选整块缺（文件不在、没跑完）时返回 None、不另报（collect_w2 开头报过一行）；
    候选在、这一格的键没有时记进 gaps（分析没有算这一格，表里是"—"，不是待补的数据）。"""
    c = ((xj or {}).get("candidates") or {}).get(cand) or {}
    if c.get("status") != "完成":
        return None
    where = "pooled" if sp == "pooled" else f"per_split.{sp}"
    blk = c.get("pooled") if sp == "pooled" else (c.get("per_split") or {}).get(sp)
    if sub is not None:
        blk = (blk or {}).get(sub)
        where += f".{sub}"
    path = f"{X_TEST_JSON} :: candidates.{cand}.{where}.{key}"
    if blk is None:
        out["missing"].append(f"W2 {row} {sp} {metric}（{X_TEST_JSON} 没有 candidates.{cand}.{where}）")
        return None
    if blk.get(key) is None:
        out["gaps"].append({"metric": metric, "row": row, "col": sp, "path": path})
        return None
    pt = _ci_list_point(blk[key], xj.get("level"), path)
    pt["n_resamples"] = xj.get("n_resamples")
    return pt


def _xj_value(xj, path: tuple, metric: str):
    """核对用：x_test_table.json 里的读数块（不记 gaps）。path = ("reference", 设置) 或 ("candidates", 候选, 划分或 pooled, 键)。"""
    if path[0] == "reference":
        return ((((xj or {}).get("reference") or {}).get(path[1] + ("" if metric == "cIoU" else "_giou"))) or {}).get(path[2])
    c = (((xj or {}).get("candidates") or {}).get(path[1])) or {}
    blk = c.get("pooled") if path[2] == "pooled" else (c.get("per_split") or {}).get(path[2])
    return (blk or {}).get(("" if metric == "cIoU" else "g_") + path[3])


def collect_w2(root: Path) -> dict:
    """tab_w2_test（cIoU）与 tab_w2_test_giou 的全部格子；DEV-0073 起也是 tab_main、tab_compute 的数据（候选三行、合并列、计算记账）。"""
    out = {"splits": list(W2T.SPLITS), "model": W2_MODEL, "rows": {}, "pairs": {}, "x_pairs": {}, "llmseg": {"rows": {}, "pairs": {}},
           "e1_parse": {}, "llmseg_calls": {}, "w2c_calls": {}, "compute": {}, "missing": [], "gaps": [], "checks": [],
           "per_system_resamples": [], "x_status": {},
           "sources": {"table": W2_TEST_JSON, "replay": W2_REPLAY_RUN, "replay_pairs": W2_REPLAY_MD,
                       "llmseg": LLMSEG_RUN, "llmseg_compare": LLMSEG_COMPARE, "x_test": X_TEST_JSON, "pooled": W2_POOLED_JSON,
                       "candidates": dict(XT.CANDS)}}
    j = _load(root / W2_TEST_JSON)
    if j is None:
        out["missing"].append(f"W2 主表：{W2_TEST_JSON} 不在")
    mj = ((j or {}).get("models") or {}).get(W2_MODEL) or {}
    out["pairs_level"], out["pairs_resamples"], out["pairs_seed"] = (j or {}).get("level"), (j or {}).get("n_resamples"), (j or {}).get("seed")
    have_replay = (root / W2_REPLAY_RUN).is_dir()
    if not have_replay:
        out["missing"].append(f"W2 主表：回放的结果目录 {W2_REPLAY_RUN} 不在")
    # DEV-0073：候选（x_test_table.json 与 RUN-0092 的叶目录）、合并列（w2_test_pooled.json）
    xj, pj = _load(root / X_TEST_JSON), _load(root / W2_POOLED_JSON)
    if xj is None:
        out["missing"].append(f"W2 探索候选：{X_TEST_JSON} 不在")
    for cand in (C1, C2):
        st = ((((xj or {}).get("candidates") or {}).get(cand)) or {}).get("status")
        out["x_status"][cand] = st
        if xj is not None and st != "完成":
            out["missing"].append(f"W2 探索候选 {cand}：{X_TEST_JSON} 里 status 是 {st or '（没有这个候选）'}")
    out["x_level"], out["x_resamples"], out["x_seed"] = (xj or {}).get("level"), (xj or {}).get("n_resamples"), (xj or {}).get("seed")
    if pj is None:
        out["missing"].append(f"W2 合并列：{W2_POOLED_JSON} 不在")
    out["pooled_level"], out["pooled_resamples"], out["pooled_seed"] = (pj or {}).get("level"), (pj or {}).get("n_resamples"), (pj or {}).get("seed")
    out["pooled_sources"] = {n: e.get("source") for n, e in ((pj or {}).get("systems") or {}).items() if e.get("status") == "完成"}
    have_x = {cand: (root / XT.CANDS[cand]).is_dir() for cand in (C1, C2)}
    for cand, ok in have_x.items():
        if not ok:
            out["missing"].append(f"W2 主表 {cand}：结果目录 {XT.CANDS[cand]} 不在")
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
    # DEV-0073：候选三行，每个划分取 RUN-0092 叶目录 metrics.json 的 headline（与上面各行同一口径：运行时算的 CI）
    xbad, n_x = [], 0
    for metric, _name in W2_METRICS:
        rows = out["rows"][metric]
        for key, cand, setting, jkey, _label in X_ROWS:
            rows[key] = {}
            for sp in W2T.SPLITS:
                pt = _leaf_headline(_x_leaf(root, cand, setting, sp) / "metrics.json", metric, root)
                if pt is None or pt["value"] is None:
                    if have_x[cand]:
                        out["missing"].append(f"W2 主表 {key} {sp} {metric}")
                    pt = None
                else:
                    if pt.get("n_resamples"):
                        res.add(pt["n_resamples"])
                    # 核对：叶目录的读数 = x_test_table.json 的读数（只比读数：分析的 CI 是 10,000 次重抽，叶目录是运行时的）
                    xv = _xj_value(xj, ("candidates", cand, sp, jkey), metric)
                    if xv is not None:
                        n_x += 1
                        if abs(xv.get(metric.lower()) - pt["value"]) > 1e-12:
                            xbad.append(f"{key} {sp} {metric}")
                rows[key][sp] = pt
    out["per_system_resamples"] = sorted(res)
    if n_x:
        out["checks"].append(("一致" if not xbad else "不一致") + f"：{X_TEST_JSON} 的候选逐划分读数与 RUN-0092 各叶目录 metrics.json 的 headline"
                             f"（{n_x} 格）" + (f"（不同：{xbad}）" if xbad else ""))
    # 合并列：所有行都取 w2_test_pooled.json（D152，5 个划分逐 id 合并，10,000 次重抽）
    for metric, _name in W2_METRICS:
        for key in list(out["rows"][metric]):
            name = pooled_name(key)
            e = ((pj or {}).get("systems") or {}).get(name) if name else None
            c = (e or {}).get(metric) if (e or {}).get("status") == "完成" else None
            pt = _ci_list_point(c, (pj or {}).get("level"), f"{W2_POOLED_JSON} :: systems.{name}.{metric}")
            if pt is not None:
                pt["n_resamples"] = pj.get("n_resamples")
            elif pj is not None and metric == W2_METRICS[0][0]:  # cIoU、gIoU 一起缺，只报一行
                why = ("没有对应的系统" if e is None else f"status 是 {e.get('status')}" +
                       (f"，缺 {', '.join(e['missing'])}" if e.get("missing") else ""))
                out["missing"].append(f"W2 合并列 {name or key}（{W2_POOLED_JSON}：{why}）")
            out["rows"][metric][key]["pooled"] = pt
    # 核对：w2_test_pooled.json 与 x_test_table.json 的合并读数（W2-c、W1-F、候选三行；同一套重抽，读数与 CI 都比）
    pbad, n_p = [], 0
    for metric, _name in W2_METRICS:
        for key, path in [("w2c", ("reference", "w2c", "pooled")), ("w1f", ("reference", "w1f", "pooled"))] + \
                         [(k, ("candidates", cand, "pooled", jk)) for k, cand, _s, jk, _l in X_ROWS]:
            p, xv = out["rows"][metric][key].get("pooled"), _xj_value(xj, path, metric)
            if p is None or xv is None:
                continue
            n_p += 1
            if any(abs(a - b) > 1e-12 for a, b in zip((xv[metric.lower()], *xv["ci"]), (p["value"], p["low"], p["high"]))):
                pbad.append(f"{key} {metric}")
    if n_p:
        out["checks"].append(("一致" if not pbad else "不一致") + f"：{W2_POOLED_JSON} 的合并读数与 {X_TEST_JSON}（W2-c、W1-F 的 reference，"
                             f"候选的 pooled）逐个相同，读数与 CI（{n_p} 格）" + (f"（不同：{pbad}）" if pbad else ""))
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
    # DEV-0073：候选的配对差取 x_test_table.json（逐句配对 bootstrap，ratio-of-sums；gIoU 取 g_ 键）
    for metric, _name in W2_METRICS:
        pre = "" if metric == "cIoU" else "g_"
        out["x_pairs"][metric] = {pk: {sp: _x_cell(xj, cand, sp, pre + jk, out, metric, lab) for sp in W2T.SPLITS + ["pooled"]}
                                  for pk, cand, jk, lab in X_PAIRS}
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
    # DEV-0073：LLMSeg 一块加 C1 的同句读数与 C1 − LLMSeg（x_test_table.json 的 llmseg 子块）；tab_main 的外部基线另取
    # 同样 1,500 句上 LLMSeg 的合并读数（带 CI，x_test_table.json；compare.json 的合并读数没有 CI）
    lxbad, n_lx = [], 0
    for metric, _name in W2_METRICS:
        pre = "" if metric == "cIoU" else "g_"
        for k, jk, lab in X_LLMSEG_ROWS:
            out["llmseg"]["rows"][metric][k] = {sp: _x_cell(xj, C1, sp, pre + jk, out, metric, lab, sub="llmseg") for sp in W2T.SPLITS + ["pooled"]}
        for k, jk, lab in X_LLMSEG_PAIRS:
            out["llmseg"]["pairs"][metric][k] = {sp: _x_cell(xj, C1, sp, pre + jk, out, metric, lab, sub="llmseg") for sp in W2T.SPLITS + ["pooled"]}
        xp = {rd: _x_cell(xj, C1, "pooled", f"{pre}llmseg_{rd}", out, metric, f"LLMSeg {rd}, pooled with CI", sub="llmseg") for rd in ("rec", "seg")}
        out["llmseg"].setdefault("x_pooled", {})[metric] = xp
        for rd, p in xp.items():  # 核对：与 compare.json 的合并读数相同
            q = out["llmseg"]["rows"][metric][rd].get("pooled")
            if p is not None and q is not None:
                n_lx += 1
                if abs(p["value"] - q["value"]) > 1e-12:
                    lxbad.append(f"{rd} {metric}")
    if n_lx:
        out["checks"].append(("一致" if not lxbad else "不一致") + f"：{X_TEST_JSON} 的 LLMSeg 合并读数（同样 1,500 句）与 {LLMSEG_COMPARE} 的"
                             f"（{n_lx} 格）" + (f"（不同：{lxbad}）" if lxbad else ""))
    # 计算记账（x_test_table.json：候选取 candidates.<候选>.compute，其余取 compute_reference；每句均值，5 个划分按句数加权）
    cref = (xj or {}).get("compute_reference") or {}
    for key, _lab in COMPUTE_ROWS:
        x = next((r for r in X_ROWS if r[0] == key), None)
        if x is not None:
            c = ((((xj or {}).get("candidates") or {}).get(x[1]) or {}).get("compute") or {}).get(x[3])
            src = f"{X_TEST_JSON} :: candidates.{x[1]}.compute.{x[3]}"
            expected = out["x_status"].get(x[1]) == "完成"
        else:
            want = COMPUTE_REF[key]
            name = want if want in cref else next((k for k in cref if k.startswith(want)), None)
            c, src, expected = cref.get(name) if name else None, f"{X_TEST_JSON} :: compute_reference.{name or want}", xj is not None
        out["compute"][key] = None if not c else {**c, "source": src}
        if c is None and expected:
            out["missing"].append(f"计算记账 {key}（{src} 没有）")
    # 样条的控制点数：C1 运行的 config.yaml（x2.n，检查点训练时定的 n）
    cfg = _x_leaf(root, C1, XT.MAIN, W2T.SPLITS[0]) / "config.yaml"
    m = re.search(r"^x2:\s*\n\s+n:\s*(\d+)\s*$", cfg.read_text(encoding="utf-8"), re.M) if cfg.is_file() else None
    out["spline_n"], out["spline_n_source"] = (int(m.group(1)) if m else None), f"{_rel(root, cfg)} :: x2.n"
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


def _node(path: str | None) -> str | None:
    """来源路径里的节点名（运行目录名的 _hpcgpuNN 后缀）。"""
    m = re.search(r"_(hpcgpu\d+)", path or "")
    return m.group(1) if m else None


def _lang_nodes(src: dict) -> dict:
    """lang_cost_9b.json 一个系统的各项检查配的基座在哪个节点：健康检查、大题量取 compare 文件的运行目录，T 文本取基座的目录。"""
    return {"health": _node(src.get("health")), "battery": _node(src.get("battery")), "ttext": _node(src.get("ttext_base"))}


def collect_lang_cost(root: Path, with_explore: bool = False) -> dict:
    """fig_lang_cost：相对同节点基座的差（百分点）与 95% CI，取自 lang_cost_9b.json（缺省四方：E3b+C3、W2-c、回放、X-2）；
    --with-explore 时再加 X-3（C2，同一份 JSON）与 X-3L（探索候选，读它自己的 compare，没有 T 文本）。"""
    out = {"readouts": [{"section": s, "key": k, "label": lab} for s, k, lab in LANG_READOUTS], "systems": [], "missing": [],
           "json": LANG_COST_JSON, "with_explore": with_explore}
    j = _load(root / LANG_COST_JSON)
    if j is None:
        out["missing"].append(f"语言代价：{LANG_COST_JSON} 不在")
    for name, label in LANG_SYSTEMS + (LANG_ABLATION if with_explore else ()):
        e = ((j or {}).get("systems") or {}).get(name)
        rec = {"name": name, "label": label, "explore": False, "role": "ablation" if (name, label) in LANG_ABLATION else "main",
               "values": {}, "health_flags": [], "sources": (e or {}).get("sources"), "nodes": _lang_nodes((e or {}).get("sources") or {})}
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
            rec = {"name": name, "label": name, "explore": True, "role": "explore", "node": node, "nodes": {"health": node, "battery": node},
                   "values": {}, "health_flags": _flags(h),
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


def collect_count_bins(root: Path, with_explore: bool = False) -> dict:
    """fig_count_bins：CountBench 三个大档的准确率（count_errors_9b.json），与配对基座的差及配对 bootstrap 95% CI。
    缺省画 X-2（C1）；--with-explore 时在它后面再加 X-3（C2）。"""
    out = {"bins": list(CE.COARSE), "systems": [], "missing": [], "json": COUNT_ERRORS_JSON, "with_explore": with_explore}
    j = _load(root / COUNT_ERRORS_JSON)
    if j is None:
        out["missing"].append(f"计数分档：{COUNT_ERRORS_JSON} 不在")
    out["n_boot"], out["seed"] = (j or {}).get("n_boot"), (j or {}).get("seed")
    out["same_answer_rate"] = (j or {}).get("base_0069_vs_0056_same_answer_rate")
    out["sources"] = (j or {}).get("source")
    systems = (j or {}).get("systems") or {}
    order = list(COUNT_SYSTEMS)
    if with_explore:
        i = [n for n, _l in order].index("X-2（C1）") + 1
        order[i:i] = list(COUNT_EXPLORE)
    for name, label in order:
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


def collect_lang_table(root: Path) -> dict:
    """tab_lang_cost（四方）与 tab_lang_cost_full（加 X-3）：lang_cost_9b.json 的健康检查三项（差与标红）、大题量两项与 T 文本两项
    （差与 95% CI），都是相对同节点基座的百分点。读数（差、CI、基座）只取 lang_cost_9b.json 这一个来源：健康检查的基座取 base_pct，
    大题量与 T 文本取 base（大题量的 base 由 origin/main 5b3bb70ee 补上）。与来源 lb_compare 文件的核对在测试里。表注里的重抽次数与
    标红门槛不是读数，取 JSON 记的来源 compare 文件的 statistics 与 threshold_pp。"""
    out = {"json": LANG_COST_JSON, "systems": [], "missing": [], "resamples": {}, "threshold_pp": None,
           "rows": [{"section": "health", "key": ".".join(r)} for r in LCT.HEALTH_ROWS] +
                   [{"section": s, "key": k} for s, k, _lab in LANG_READOUTS]}
    j = _load(root / LANG_COST_JSON)
    if j is None:
        out["missing"].append(f"语言代价表：{LANG_COST_JSON} 不在")
    res: dict = {"health": set(), "battery": set(), "ttext": {LCT.N_RES}}
    for name, label in LANG_SYSTEMS + LANG_ABLATION:
        e = ((j or {}).get("systems") or {}).get(name)
        src = (e or {}).get("sources") or {}
        rec = {"name": name, "label": label, "role": "ablation" if (name, label) in LANG_ABLATION else "main", "sources": src,
               "nodes": _lang_nodes(src), "cells": {}}
        if e is None and j is not None:
            out["missing"].append(f"语言代价表 {name}（{LANG_COST_JSON} 里没有）")
        hc, bc = (_load(root / src["health"]) if src.get("health") else None), (_load(root / src["battery"]) if src.get("battery") else None)
        if hc:
            res["health"].add((((hc.get("statistics") or {}).get("bootstrap")) or {}).get("n_resamples"))
            out["threshold_pp"] = out["threshold_pp"] or hc.get("threshold_pp")
        if bc:
            res["battery"].add((((bc.get("statistics") or {}).get("bootstrap")) or {}).get("n_resamples"))
        for row in out["rows"]:
            sec, key = row["section"], row["key"]
            r = ((e or {}).get(sec) or {}).get(key)
            path = f"{LANG_COST_JSON} :: systems.{name}.{sec}.{key}"
            if not r or r.get("delta_pp") is None:
                rec["cells"][f"{sec}/{key}"] = None
                if e is not None:
                    out["missing"].append(f"语言代价表 {name} {sec} {key}")
                continue
            ci = r.get("ci95_pp") or [None, None]
            base = r.get("base_pct") if sec == "health" else r.get("base")
            if base is None:
                out["missing"].append(f"语言代价表 {name} {sec} {key} 的基座读数（{path} 里是空的）")
            rec["cells"][f"{sec}/{key}"] = {"delta": r["delta_pp"], "low": ci[0], "high": ci[1], "n": r.get("n"),
                                            "flag": bool(r.get("flag")) if sec == "health" else None, "base": base,
                                            "node": rec["nodes"][sec], "source": path}
        out["systems"].append(rec)
    out["resamples"] = {k: sorted(x for x in v if x) for k, v in res.items()}
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


def _legend_cols(n: int, per_row: int) -> int:
    """图例的列数：每行至多 per_row 项，行数最少，各行尽量一样长（4 项 → 2 × 2，6 项 → 3 × 2）。"""
    rows = max(1, math.ceil(n / per_row))
    return max(1, math.ceil(n / rows))


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
    fig.legend(handles=handles, loc="outside upper center", ncol=_legend_cols(n_s, 3), handletextpad=0.2, columnspacing=0.9)
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
    fig.legend(handles=handles, loc="outside upper center", ncol=_legend_cols(len(handles), 5), handlelength=0.8, handletextpad=0.3,
               columnspacing=0.6)
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


def _x_def_note(w2: dict) -> str:
    """C1、椭圆消融、C2 的定义（预注册 X-5、X-5a、X-6；G-EX §1）。样条的控制点数取 C1 运行的 config.yaml。"""
    n = w2.get("spline_n")
    spline = f"a closed B-spline with {n} control points" if n else "a closed B-spline"
    return (f"C1 = X-2 (ours): a single LoRA on the same Qwen3.5-9B; the model proposes a box, then traces the object's outline as {spline}; "
            "the rasterised spline seeds GrabCut as a trimap (definite bands inside and outside the spline, unknown in between). GrabCut "
            "(OpenCV graph cut) has no learned parameters, and no external learned vision model is used in training or inference. "
            "C1, in-box ellipse instead of spline: the same run with the spline replaced by the ellipse inscribed in the proposed box, "
            "everything else unchanged (capability attribution). C2 = X-3: C1's training recipe without the signed box-correction "
            "stream (trained for C1 but not used at its inference).")


W2_DEF_NOTE = ("Own box: the model's own box (E1 protocol) as the mask; W1-F: that box plus the default GrabCut candidate (a fixed "
               "pipeline, no LLM decision); random / oracle pick: a random / the ground-truth-best of the four tool candidates; W2-a: the "
               "frozen model picks a candidate (zero-shot; candidate 1 = W1-F, also submitted when the pick fails to parse); W2-c: LoRA "
               "text supervision to propose the box and pick; W2-c + replay: W2-c trained with self-distillation replay (post hoc "
               "variant, descriptive).")


def _pooled_n(w2: dict, metric: str = "cIoU") -> int | None:
    return next((p["n"] for d in w2["rows"].get(metric, {}).values() for p in [d.get("pooled")] if p and p.get("n")), None)


def _gap_text(w2: dict, metric: str) -> str:
    """分析没有算的格子（gaps），按行归并：五个划分都缺写 "per split"，否则列出划分。"""
    by: dict = {}
    for g in w2.get("gaps") or []:
        if g["metric"] == metric:
            by.setdefault(g["row"], []).append(g["col"])
    parts = []
    for row, cols in by.items():
        if set(cols) == set(w2["splits"]):
            parts.append(f"{row} per split")
        else:
            parts.append(f"{row} ({', '.join('pooled' if c == 'pooled' else split_label(c) for c in cols)})")
    return "; ".join(parts)


def _nfmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else MISSING


def _res_text(rs) -> str:
    rs = sorted({r for r in rs if r})
    return "/".join(f"{r:,}" for r in rs) if rs else MISSING


def w2_table_notes(w2: dict, metric: str) -> list[str]:
    res = "/".join(f"{r:,}" for r in w2["per_system_resamples"]) or MISSING
    e1 = [v for v in w2["e1_parse"].values() if v]
    all_fail = e1 and len(e1) == len(w2["splits"]) and all(v["n_failed"] == v["n"] for v in e1)
    npool = _pooled_n(w2, metric)
    notes = [f"Qwen3.5-9B on the five W2 test splits ({_n_split(w2, metric)} expressions each, the E1 polygon subset); "
             f"{metric} ×100 with 95% bootstrap CI over expressions: per split as computed by each run ({res} resamples); pooled = the "
             f"five splits together ({f'{npool:,}' if npool else MISSING} expressions) from {W2_POOLED_JSON.split('/')[-1]} "
             f"({_res_text([w2.get('pooled_resamples')])} resamples). "
             + ("cIoU = summed intersection / summed union." if metric == "cIoU" else "gIoU = mean per-expression IoU."),
             W2_DEF_NOTE + " " + _x_def_note(w2)]
    if all_fail:
        notes.append(f"E1 zero-shot polygon: all {e1[0]['n']:,} outputs per split fail to parse (strict scoring gives 0).")
    xres = _res_text([w2.get("x_resamples")])
    if metric == "cIoU":
        wres = _res_text([w2.get("pairs_resamples")])
        srcs = f"W2 rows: {W2_TEST_JSON.split('/')[-1]}; C1/C2 rows: {X_TEST_JSON.split('/')[-1]}"
        notes.append("Paired differences (points ×100): per-expression paired bootstrap of the ratio of sums, 95%, " +
                     (f"{wres} resamples ({srcs})" if wres == xres else f"{wres} resamples for the W2 rows and {xres} for the C1/C2 rows ({srcs})")
                     + "; pooled = the five splits together.")
    else:
        notes.append(f"Paired gIoU differences (mean per-expression IoU; per-expression paired bootstrap, 95%, {xres} resamples) are in the "
                     f"analysis only for C1 and C2 ({X_TEST_JSON.split('/')[-1]}); the W2 rows have paired differences in cIoU only "
                     "(see the cIoU table).")
    calls = [c for v in w2["llmseg_calls"].values() for c in [v.get("rec")] if c]
    calls_seg = [c for v in w2["llmseg_calls"].values() for c in [v.get("seg")] if c]
    w2c_calls = sorted({c for c in w2["w2c_calls"].values() if c})
    lres = sorted({p.get("n_resamples") for k, d in w2["llmseg"]["pairs"].get(metric, {}).items() if k in dict(LLMSEG_PAIRS)
                   for p in d.values() if p and p.get("n_resamples")})
    lls = ("External baseline LLMSeg" + (f" (official code, commit {w2['llmseg_commit'][:7]})" if w2.get("llmseg_commit") else "") +
           " driven by the same Qwen3.5-9B, zero-shot, on 300 hash-selected expressions per split (pooled 1,500), with W2-c and C1 on "
           "the same expressions alongside. Before refinement = after its first two stages; after refinement = its main configuration.")
    if calls and calls_seg:
        lls += (f" Model calls per image (split means): {_span(calls)} after vs. {_span(calls_seg)} before refinement" +
                (f"; W2-c {'/'.join(f'{c:g}' for c in w2c_calls)}" if w2c_calls else "") + ".")
    lls += (" LLMSeg − W2-c: paired bootstrap, 95%" + (f", {'/'.join(f'{r:,}' for r in lres)} resamples" if lres else "") +
            f"; the pooled LLMSeg and W2-c readings have no CI in that comparison. C1 rows: {X_TEST_JSON.split('/')[-1]}, "
            f"{xres} resamples.")
    notes.append(lls)
    gaps = _gap_text(w2, metric)
    if gaps:
        notes.append(f"—: not computed in the analysis ({X_TEST_JSON.split('/')[-1]}): {gaps}.")
    src = [W2T.RUNS[W2_MODEL], W2_REPLAY_RUN, LLMSEG_RUN] + list(XT.CANDS.values()) + list(w2.get("pooled_sources", {}).values())
    notes.append(f"Data: {_runs_of(src)} (LLMSeg comparison computed on CPU, D139)"
                 + ("; W2-a paired differences completed in RUN-0081" if metric == "cIoU" else "") +
                 "; " + _pooled_extra_text(w2) + ".")
    if w2["missing"]:
        notes.append("Missing: " + "; ".join(w2["missing"]) + ".")
    return notes


def _pooled_extra_text(w2: dict) -> str:
    """合并列里来源不是该行自己运行目录的系统（W2-a 的逐句交并由 RUN-0095 从 RUN-0072 抽回）。"""
    extra = []
    for key in ROW_LABELS:
        name = pooled_name(key)
        src = (w2.get("pooled_sources") or {}).get(name)
        run = row_run_setting(key)[0]
        if src and _runs_of([src]) and _runs_of([src]) != _runs_of([run]):
            extra.append(f"{ROW_LABELS[key]} per-expression scores for the pooled column extracted in {_runs_of([src])} from {_runs_of([run])}")
    return "; ".join(extra) if extra else "pooled column from the runs above"


def _span(xs, nd: int = 1) -> str:
    lo, hi = f"{min(xs):.{nd}f}", f"{max(xs):.{nd}f}"
    return lo if lo == hi else f"{lo}–{hi}"


def _n_split(w2: dict, metric: str) -> str:
    ns = sorted({p["n"] for d in w2["rows"].get(metric, {}).values() for c, p in d.items() if c != "pooled" and p and p.get("n")})
    return "/".join(f"{n:,}" for n in ns) if ns else MISSING


def _write_table(out: Path, name: str, caption: str, label: str, colspec: str, header_tex: list[str], body_tex: list[str],
                 md_title: str, md_rows: list[str], notes: list[str]) -> list[Path]:
    """同 tab_w2 的版式：table*、booktabs、resizebox；Markdown 同一份表加表注。"""
    L = ["% Generated by scripts/figures/make_paper_figures.py; do not edit. Needs \\usepackage{booktabs,graphicx" +
         (",multirow" if any("\\multirow" in x for x in body_tex) else "") + "}.",
         "\\begin{table*}[t]", "\\centering", caption.replace("—", "--"), f"\\label{{tab:{label}}}", "\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         *header_tex, *body_tex, "\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    md = [f"# Table: {md_title} (generated by scripts/figures/make_paper_figures.py; do not edit by hand)", ""] + md_rows + [""] + \
         [f"- {n}" for n in notes] + [""]
    p_tex, p_md = out / f"{name}.tex", out / f"{name}.md"
    p_tex.write_text("\n".join(L), encoding="utf-8")
    p_md.write_text("\n".join(md), encoding="utf-8")
    return [p_tex, p_md]


def tab_w2(w2: dict, out: Path, metric: str, name: str) -> list[Path]:
    """系统为行、划分（加合并）为列；三块：9B 各系统（含 C1、椭圆消融、C2）、配对差、外部基线 LLMSeg（含 C1 的同句读数与配对差）。"""
    splits = w2["splits"]
    cols = splits + ["pooled"]
    heads = [split_label(sp) for sp in splits] + ["Pooled"]
    blocks = [(f"Qwen3.5-9B, {_n_split(w2, metric)} expressions per split ({metric} ×100, 95% CI)",
               [(ROW_LABELS[k], w2["rows"][metric].get(k, {}), False) for k in [k for k, _l in W2_ROWS] + [r[0] for r in X_ROWS]])]
    pairs = [(lab, w2["pairs"].get(k, {}), True) for k, lab in W2_PAIRS] if metric == "cIoU" else []
    pairs += [(lab, w2["x_pairs"].get(metric, {}).get(k, {}), True) for k, _c, _j, lab in X_PAIRS]
    blocks.append(("Paired differences (points ×100, 95% paired bootstrap CI)", pairs))
    blocks.append(("External baseline: LLMSeg with zero-shot Qwen3.5-9B, 300 expressions per split (pooled 1,500)",
                   [(lab, w2["llmseg"]["rows"].get(metric, {}).get(k, {}), False) for k, lab in LLMSEG_ROWS] +
                   [(lab, w2["llmseg"]["rows"].get(metric, {}).get(k, {}), False) for k, _j, lab in X_LLMSEG_ROWS] +
                   [(lab, w2["llmseg"]["pairs"].get(metric, {}).get(k, {}), True) for k, lab in LLMSEG_PAIRS] +
                   [(lab, w2["llmseg"]["pairs"].get(metric, {}).get(k, {}), True) for k, _j, lab in X_LLMSEG_PAIRS]))
    notes = w2_table_notes(w2, metric)
    title = "W2 test" + (" (gIoU, appendix)" if metric == "gIoU" else "")
    caption = (f"\\caption{{{title} on Qwen3.5-9B: referring segmentation {metric} per split. " +
               " ".join(_tex_escape(n) for n in notes) + "}")
    body, M = [], ["| System | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(cols)]
    for title_b, rows in blocks:
        body += ["\\midrule", f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\emph{{{_tex_escape(title_b)}}}}} \\\\"]
        M.append(f"| *{title_b}* |" + " |" * len(cols))
        for lab, cells, signed in rows:
            body.append(_tex_escape(lab) + " & " + " & ".join(fmt_w2(cells.get(c), signed, True) for c in cols) + " \\\\")
            M.append(f"| {lab} | " + " | ".join(fmt_w2(cells.get(c), signed, False) for c in cols) + " |")
    return _write_table(out, name, caption, name.replace("tab_", "").replace("_", "-"), "l" + "c" * len(cols),
                        ["System & " + " & ".join(_tex_escape(h) for h in heads) + " \\\\"], body, title, M, notes)


# ---- tab_main（DEV-0073：论文正文主表）------------------------------------------------------------


def _calls_text(c: dict | None) -> str:
    """每句的模型调用 / 生成 token（x_test_table.json 的计算记账）；没有记录写"—"。"""
    if not c:
        return "—"
    calls = f"{c['n_calls']:.1f}"
    return f"{calls[:-2] if calls.endswith('.0') else calls} / {c['generated_tokens']:,.0f}"


# 预注册 X-6 的预测（写于任何候选的 test 运行之前）："C1 与 C2 在 test 上相差 ±0.02 以内"（cIoU）；表注拿合并读数之差对照它
X6_C1_C2_MARGIN = 0.02


def c1_c2_note(w2: dict) -> str:
    """表注一句（DEV-0073 审查第 1 条，不加粗）：C1 在 test 之前就定为主方法（X-5a）、C2 是去掉修正流的简化版、合并读数之差对照 X-6 的预测。"""
    rows = w2["rows"].get("cIoU", {})
    p1, p2 = (rows.get("c1") or {}).get("pooled"), (rows.get("c2") or {}).get("pooled")
    s = ("C1 was designated the main method before any test run (preregistration X-5a, the project lead's decision); C2 is C1's "
         "simplified version with the box-correction stream removed")
    if not p1 or not p2 or p1.get("value") is None or p2.get("value") is None:
        return s + "."
    d = p2["value"] - p1["value"]
    return (s + f"; pooled, C2 is {100 * abs(d):.1f} points {'above' if d >= 0 else 'below'} C1, "
            f"{'within' if abs(d) <= X6_C1_C2_MARGIN else 'outside'} the ±{100 * X6_C1_C2_MARGIN:g} points predicted in preregistration X-6.")


def main_table_notes(w2: dict) -> list[str]:
    npool = _pooled_n(w2)
    xp = (w2["llmseg"].get("x_pooled") or {}).get("cIoU") or {}
    n_ext = (xp.get("rec") or {}).get("n")
    keys = [k for g in MAIN_ROWS for k in g]
    by_run: dict = {}
    for k in keys:
        by_run.setdefault(_runs_of([row_run_setting(k)[0]]), []).append(SHORT_LABELS[k])
    by_run.setdefault(_runs_of([LLMSEG_RUN]), []).append("LLMSeg")
    nodes = sorted({n for k in keys for n in [_node(row_run_setting(k)[0])] if n} | ({_node(LLMSEG_RUN)} - {None}))
    notes = [f"Referring segmentation on the W2 test with Qwen3.5-9B: RefCOCO testA/testB, RefCOCO+ testA/testB and RefCOCOg test, "
             f"{_n_split(w2, 'cIoU')} expressions per split (the E1 polygon subset); cIoU ×100 (summed intersection / summed union). "
             f"Pooled: the five splits together ({f'{npool:,}' if npool else MISSING} expressions) with a 95% bootstrap CI over expressions "
             f"({_res_text([w2.get('pooled_resamples')])} resamples); per-split CIs and gIoU are in the appendix tables.",
             _x_def_note(w2),
             c1_c2_note(w2),
             "Own box: the model's own box (E1 protocol) as the mask; W1-F: that box plus the default GrabCut candidate (a fixed pipeline, "
             "no LLM decision); oracle pick: the ground-truth-best of the four tool candidates; W2-a: the frozen model picks a candidate "
             "(zero-shot); W2-c: LoRA text supervision to propose the box and pick a candidate.",
             "Calls / gen. tokens: model calls and generated tokens per expression (means over the test expressions; LLMSeg over its "
             f"{f'{n_ext:,}' if n_ext else MISSING}), from the compute accounting of {X_TEST_JSON.split('/')[-1]} (see the compute table); "
             "— = no compute record for that row in the analysis.",
             "External baseline: LLMSeg" + (f" (official code, commit {w2['llmseg_commit'][:7]})" if w2.get("llmseg_commit") else "") +
             f" driven by the same zero-shot Qwen3.5-9B, main configuration (after refinement), on "
             f"{f'{n_ext:,}' if n_ext else MISSING} hash-selected expressions (300 per split), with C1 on the same expressions. "
             f"C1 − LLMSeg: per-expression paired bootstrap of the ratio of sums, 95%, {_res_text([w2.get('x_resamples')])} resamples "
             f"(seed {w2.get('x_seed') or MISSING}); the two readings' CIs from the same resampling.",
             "Data: " + "; ".join(f"{r} ({', '.join(labs)})" for r, labs in by_run.items()) + "; " + _pooled_extra_text(w2) +
             (f"; all on {', '.join(nodes)} (D93-1)" if len(nodes) == 1 else "") + "."]
    if w2["missing"]:
        notes.append("Missing: " + "; ".join(w2["missing"]) + ".")
    return notes


def tab_main(w2: dict, out: Path) -> list[Path]:
    """正文主表：行 = 自己的框、W1-F、最优候选、W2-a、W2-c | C1、椭圆消融、C2；列 = 5 个划分（只读数）与合并（带 CI）、每句调用 / 生成 token；
    下接外部基线一块（同样 1,500 句的合并读数与配对差）。不加粗（DEV-0073 审查：最高值落在消融 C2 上，加粗 C1 又会误导）。"""
    splits = w2["splits"]
    heads = [split_label(sp) for sp in splits] + ["Pooled", "Calls / gen. tokens"]
    rows = w2["rows"].get("cIoU", {})
    body, M = [], ["| System | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    for group in MAIN_ROWS:
        body.append("\\midrule")
        for k in group:
            r = rows.get(k, {})
            cells_t = [fmt_w2({"value": (r.get(sp) or {}).get("value")}, False, True) for sp in splits]
            cells_m = [fmt_w2({"value": (r.get(sp) or {}).get("value")}, False, False) for sp in splits]
            cells_t += [fmt_w2(r.get("pooled"), False, True), _tex_escape(_calls_text(w2["compute"].get(k))).replace("—", "--")]
            cells_m += [fmt_w2(r.get("pooled"), False, False), _calls_text(w2["compute"].get(k))]
            body.append(_tex_escape(ROW_LABELS[k]) + " & " + " & ".join(cells_t) + " \\\\")
            M.append(f"| {ROW_LABELS[k]} | " + " | ".join(cells_m) + " |")
    xp = (w2["llmseg"].get("x_pooled") or {}).get("cIoU") or {}
    n_ext = (xp.get("rec") or {}).get("n")
    ext_title = (f"External baseline on the same {f'{n_ext:,}' if n_ext else MISSING} expressions (300 per split), pooled")
    ext = [("LLMSeg, after refinement (main)", xp.get("rec"), False, w2["compute"].get("llmseg")),
           (ROW_LABELS["c1"] + ", same expressions", (w2["llmseg"]["rows"].get("cIoU", {}).get("c1") or {}).get("pooled"), False,
            w2["compute"].get("c1")),
           ("C1 − LLMSeg, after refinement (points)", (w2["llmseg"]["pairs"].get("cIoU", {}).get("c1-rec") or {}).get("pooled"), True, None)]
    body += ["\\midrule", f"\\multicolumn{{{len(heads) + 1}}}{{l}}{{\\emph{{{_tex_escape(ext_title)}}}}} \\\\"]
    M.append(f"| *{ext_title}* |" + " |" * len(heads))
    for lab, p, signed, comp in ext:
        calls = "" if signed else _calls_text(comp)
        body.append(_tex_escape(lab) + " & " * (len(splits) + 1) + fmt_w2(p, signed, True) + " & " +
                    _tex_escape(calls).replace("—", "--") + " \\\\")
        M.append(f"| {lab} |" + " |" * len(splits) + f" {fmt_w2(p, signed, False)} | {calls} |")
    notes = main_table_notes(w2)
    caption = ("\\caption{Main result: referring segmentation on the W2 test (Qwen3.5-9B, cIoU). " +
               " ".join(_tex_escape(n) for n in notes) + "}")
    header = ["System & " + " & ".join(_tex_escape(h) for h in heads) + " \\\\"]
    return _write_table(out, "tab_main", caption, "main", "l" + "c" * len(heads), header, body, "main result (W2 test, cIoU)", M,
                        notes)


# ---- tab_compute（DEV-0073：计算记账）------------------------------------------------------------


def compute_table_notes(w2: dict) -> list[str]:
    ns = {k: (c or {}).get("n") for k, c in w2["compute"].items()}
    n_main = sorted({n for k, n in ns.items() if n and k != "llmseg"})
    runs = [row_run_setting(k)[0] for k, _l in COMPUTE_ROWS if k != "llmseg"] + [LLMSEG_RUN]
    nodes = sorted({n for r in runs for n in [_node(r)] if n})
    return [f"Compute per expression on the W2 test, Qwen3.5-9B: means over {'/'.join(f'{n:,}' for n in n_main) or MISSING} expressions "
            f"(split means weighted by expressions; LLMSeg over its {_nfmt(ns.get('llmseg'))}), from "
            f"the compute accounting of {X_TEST_JSON.split('/')[-1]} (preregistration X-6).",
            "Model-call time: time spent in model calls; wall clock: each run's own per-expression record, including the non-learned "
            "tools (GrabCut).",
            "LLMSeg's wall clock is not listed (—): its compute.json records wall time accumulated under concurrent requests, a different "
            "measure from the per-expression wall clock of the other rows.",
            f"Data: {_runs_of(runs)}" + (f", all on {nodes[0]}" if len(nodes) == 1 else "") + "."] + \
           (["Missing: " + "; ".join(m for m in w2["missing"] if m.startswith("计算记账")) + "."]
            if any(m.startswith("计算记账") for m in w2["missing"]) else [])


def tab_compute(w2: dict, out: Path) -> list[Path]:
    heads = ["Expressions"] + [h for _k, h, _nd in COMPUTE_COLS]
    body, M = ["\\midrule"], ["| System | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    for key, lab in COMPUTE_ROWS:
        c = w2["compute"].get(key)
        cells = ["—" if not c else f"{c['n']:,}"]
        for k, _h, nd in COMPUTE_COLS:
            na = c is None or c.get(k) is None or (k == "wall_time_s" and key in COMPUTE_WALL_NA)
            cells.append("—" if na else f"{c[k]:,.{nd}f}")
        body.append(_tex_escape(lab) + " & " + " & ".join("--" if x == "—" else x for x in cells) + " \\\\")
        M.append(f"| {lab} | " + " | ".join(cells) + " |")
    notes = compute_table_notes(w2)
    caption = ("\\caption{Compute accounting on the W2 test (per expression). " + " ".join(_tex_escape(n) for n in notes) + "}")
    header = ["System & " + " & ".join(_tex_escape(h) for h in heads) + " \\\\"]
    return _write_table(out, "tab_compute", caption, "compute", "l" + "r" * len(heads), header, body, "compute accounting (W2 test)",
                        M, notes)


# ---- tab_lang_cost（DEV-0073：语言代价四方对照；附录版加 X-3）-------------------------------------


def _lang_cell(c: dict | None, sec: str, latex: bool) -> str:
    if c is None:
        return "--" if latex else "—"
    d = f"{c['delta']:+.1f}"
    if sec == "health":
        return (f"${d}$" + ("\\textsuperscript{\\dag}" if c["flag"] else "")) if latex else d + ("†" if c["flag"] else "")
    if c.get("low") is None or c.get("high") is None:
        return f"${d}$" if latex else d
    ci = f"[{c['low']:+.1f}, {c['high']:+.1f}]"
    return f"${d}$\\,{{\\scriptsize ${ci}$}}" if latex else f"{d} {ci}"


def _lang_base(systems: list[dict], row: dict) -> str:
    """表头下的基座读数：各系统配对的基座读数相同就写一个；不同就按节点分别写（同一节点有两个读数时按系统写）。"""
    vals = [(c["base"], c["node"], s["label"]) for s in systems for c in [s["cells"].get(f"{row['section']}/{row['key']}")]
            if c and c.get("base") is not None]
    distinct = list(dict.fromkeys(round(v, 1) for v, _n, _l in vals))
    if not distinct:
        return "—"
    if len(distinct) == 1:
        return f"{distinct[0]:.1f}"
    groups = [(v, sorted({n for vv, n, _l in vals if round(vv, 1) == v}), [l for vv, _n, l in vals if round(vv, 1) == v]) for v in distinct]
    by_node = all(len(set(ns)) == 1 for _v, ns, _l in groups) and len({ns[0] for _v, ns, _l in groups}) == len(groups)
    return " / ".join(f"{v:.1f} ({ns[0] if by_node else ', '.join(ls)})" for v, ns, ls in groups)


def _lang_row_label(systems: list[dict], row: dict) -> str:
    sec, key = row["section"], row["key"]
    n = next((c["n"] for s in systems for c in [s["cells"].get(f"{sec}/{key}")] if c and c.get("n")), None)
    lab = HEALTH_LABELS.get(key.split(".")[0], key) if sec == "health" else LANG_TABLE_LABELS[sec].get(key, key)
    return lab + (f" (n={n:,})" if n else "")


SECTION_WORDS = {"health": "health check", "battery": "language battery", "ttext": "T-text"}


def _pairing_text(systems: list[dict]) -> str:
    """每个系统各项检查配的基座在哪个节点（来源路径的 _hpcgpuNN）；同样配法的系统合成一组。"""
    groups: dict = {}
    for s in systems:
        nodes = {sec: n for sec, n in (s.get("nodes") or {}).items() if sec in SECTION_WORDS and n}
        if not nodes:
            continue
        if len(set(nodes.values())) == 1:
            desc = f"every check against the base on {next(iter(nodes.values()))}"
        else:
            by: dict = {}
            for sec in SECTION_WORDS:
                if sec in nodes:
                    by.setdefault(nodes[sec], []).append(SECTION_WORDS[sec])
            desc = ", ".join(f"{' and '.join(secs)} against the base on {n}" for n, secs in by.items())
        groups.setdefault(desc, []).append(s["label"])
    return "; ".join(f"{', '.join(labs)}: {desc}" for desc, labs in groups.items())


def lang_table_notes(lt: dict, systems: list[dict]) -> list[str]:
    def n_of(sec, key):
        return next((c["n"] for s in systems for c in [s["cells"].get(f"{sec}/{key}")] if c and c.get("n")), None)
    hn = [f"{HEALTH_LABELS.get(r['key'].split('.')[0])} {n_of('health', r['key']) or MISSING}" for r in lt["rows"] if r["section"] == "health"]
    bk = [r["key"] for r in lt["rows"] if r["section"] == "battery"]
    tk = [r["key"] for r in lt["rows"] if r["section"] == "ttext"]
    srcs = [v for s in systems for v in (s.get("sources") or {}).values() if isinstance(v, str)]
    thr = lt.get("threshold_pp")
    notes = [f"Language cost on Qwen3.5-9B under one set of checks (D147-4, preregistration X-6): each cell is the difference from the base "
             "model evaluated on the same node, in percentage points. Base: the base model's reading (%); where the bases on two nodes "
             "differ, each is given with its node.",
             f"Health check (brief §8.11; fixed items: {', '.join(hn)}): difference only; † = flagged by its rule (the headline metric "
             f"drops by more than {thr if thr is not None else MISSING} points).",
             f"Language battery (MMLU-Pro {_nfmt(n_of('battery', bk[0]))} questions, letter answers; IFEval "
             f"{_nfmt(n_of('battery', bk[1]))} prompts, prompt-level strict) and T-text (grounding Acc@0.5 on "
             f"{_nfmt(n_of('ttext', tk[0]))} expressions; counting accuracy on CountBench, {_nfmt(n_of('ttext', tk[1]))} items): "
             f"difference [95% paired bootstrap CI over items; battery {_res_text(lt['resamples'].get('battery', []))} resamples, T-text "
             f"{_res_text(lt['resamples'].get('ttext', []))} resamples].",
             "Pairing (D93-1): " + _pairing_text(systems) + ".",
             "Systems: E3b+C3 (E3, pixel-token channel mixed with text), W2-c, W2-c + replay (W2-c with post hoc self-distillation replay) "
             "and C1 = X-2 (ours; its LoRA also trains on self-distillation replay of the base model's own answers, general instructions "
             "and counting)" + ("; C2 = X-3 (ablation: C1's recipe without the box-correction stream)"
                                if any(s["role"] == "ablation" for s in systems) else "") + ".",
             f"Data: {LANG_COST_JSON.split('/')[-1]} ({_runs_of(srcs)})."]
    shown = {s["name"] for s in systems}  # 只列这张表里的系统的缺项（四方表不列 X-3 的）
    miss = [m for m in lt["missing"] if not any(n in m for n, _l in LANG_SYSTEMS + LANG_ABLATION if n not in shown)]
    if miss:
        notes.append("Missing: " + "; ".join(miss) + ".")
    return notes


def tab_lang_cost(lt: dict, out: Path, full: bool = False) -> list[Path]:
    """行 = 健康检查三项、大题量两项、T 文本两项（按节分组）；列 = 基座读数与各系统（四方，附录版加 X-3）。"""
    systems = [s for s in lt["systems"] if full or s["role"] == "main"]
    heads = ["Check", "Reading", "Base"] + [s["label"] for s in systems]
    body, M = [], ["| " + " | ".join(heads) + " |", "|---|---|---:|" + "---:|" * len(systems)]
    for sec, sec_lab in LANG_SECTIONS:
        rows = [r for r in lt["rows"] if r["section"] == sec]
        body.append("\\midrule")
        for i, row in enumerate(rows):
            lab, base = _lang_row_label(systems, row), _lang_base(systems, row)
            cells_t = [_lang_cell(s["cells"].get(f"{sec}/{row['key']}"), sec, True) for s in systems]
            cells_m = [_lang_cell(s["cells"].get(f"{sec}/{row['key']}"), sec, False) for s in systems]
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{_tex_escape(sec_lab)}}}" if i == 0 else ""
            body.append(f"{first} & {_tex_escape(lab)} & {_tex_escape(base).replace('—', '--')} & " + " & ".join(cells_t) + " \\\\")
            M.append(f"| {sec_lab if i == 0 else ''} | {lab} | {base} | " + " | ".join(cells_m) + " |")
    notes = lang_table_notes(lt, systems)
    name = "tab_lang_cost_full" if full else "tab_lang_cost"
    title = "Language cost" + (" (with the C2 ablation, appendix)" if full else "")
    caption = (f"\\caption{{{title}: difference from the base on the same node (Qwen3.5-9B, percentage points). " +
               " ".join(_tex_escape(n) for n in notes) + "}")
    header = [" & ".join(_tex_escape(h) for h in heads) + " \\\\"]
    return _write_table(out, name, caption, name.replace("tab_", "").replace("_", "-"), "lll" + "c" * len(systems), header, body,
                        title[0].lower() + title[1:], M, notes)


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


LANG_DESCRIPTIONS = {"E3b+C3": "E3b+C3 (E3)", "W2-c+回放": "W2-c + replay (post hoc self-distillation replay)"}  # 图注里的系统说明


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def caption_lang_cost(lc: dict) -> str:
    main = [s for s in lc["systems"] if s["role"] == "main"]
    extra = [s for s in lc["systems"] if s["role"] != "main"]
    base = next((s.get("base_pct") for s in main if s.get("base_pct")), None) or {}
    bt = " / ".join(f"{base[k]:.1f}%" for _sec, k, _lab in LANG_READOUTS if k in base)
    flags = []
    for s in lc["systems"]:
        fl = ", ".join(f"{HEALTH_LABELS.get(f['reading'], f['reading'])} {_signed(f['delta_pp'])} (n={f['n']})" for f in s["health_flags"])
        flags.append(f"{s['label']}: {fl or 'none'}")
    srcs = [v for s in lc["systems"] if not s["explore"] for v in (s.get("sources") or {}).values() if isinstance(v, str)]
    expl = [s for s in lc["systems"] if s["explore"]]
    s = ("Language cost on Qwen3.5-9B: difference from the base model evaluated on the same node (percentage points; 95% paired "
         "bootstrap CI). MMLU-Pro (letter answers) and IFEval (prompt-level strict) from the large language battery; grounding "
         "Acc@0.5 and counting accuracy (CountBench) on the T-text channel (shaded"
         + (f"; base {bt}" if bt else "") + "). Systems: " + _and([LANG_DESCRIPTIONS.get(x["name"], x["label"]) for x in main]) + ".")
    if extra:
        s += (" Also shown: " + _and([f"{x['label']} (C1's recipe without the box-correction stream)" if x["role"] == "ablation" else
                                      f"{x['label']} (an exploration candidate not taken to G-EX; T-text not measured)" for x in extra]) + ".")
    s += (" Pairing (D93-1): " + _pairing_text(lc["systems"]) + ". Not drawn: the small health check (brief §8.11: MMLU-Pro 100, GSM8K 50, "
          "IFEval 50 items) flags " + "; ".join(flags) + ".")
    s += f" Data: {LANG_COST_JSON.split('/')[-1]} ({_runs_of(srcs)})" + (f"; X-3L: {EXPLORE_RUNS}" if expl else "") + "."
    if lc["missing"]:
        s += " Missing: " + "; ".join(lc["missing"]) + "."
    return s


def caption_count_bins(cb: dict) -> str:
    ns = [next((x["n"].get(b) for x in cb["systems"] if x["n"].get(b)), None) for b in cb["bins"]]
    pairs = {}
    for x in cb["systems"]:
        if x["paired_base"]:
            pairs.setdefault(x["paired_base"], []).append(x["label"])
    drawn = {x["name"] for x in cb["systems"]} | {x["paired_base"] for x in cb["systems"] if x["paired_base"]}
    srcs = {k: v for k, v in (cb.get("sources") or {}).items() if k in drawn}  # 只算画出来的系统与它们的基座
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


def readme_w2(w2: dict, sz: dict, lc: dict, cb: dict, lt: dict, captions: dict) -> list[str]:
    """README 的 W2 各节（DEV-0071；DEV-0073 加候选、合并列与三张新表）。"""
    leaf = f"{W2T.RUNS[W2_MODEL]}/{W2_MODEL}/refseg/<设置>/<划分>/metrics.json"
    res = "/".join(f"{r:,}" for r in w2["per_system_resamples"]) or MISSING
    pres, xres = _res_text([w2.get("pooled_resamples")]), _res_text([w2.get("x_resamples")])
    xleaf = f"`{XT.CANDS[C1]}/{XT.M}/refseg/<{XT.MAIN}|{XT.ABL}>/<划分>/metrics.json`（C2 在 `{XT.CANDS[C2]}`，设置 {XT.MAIN}）"
    L = ["## 5. W2 test 主表：`tab_w2_test.tex` / `tab_w2_test.md`（gIoU 附录表 `tab_w2_test_giou.*`）", "",
         "- 数据（都是已有的读数，不重算 bootstrap）：",
         f"  - 9B 各设置（自己的框、W1-F、随机挑选、最优候选、W2-a、W2-c、E1 零样本多边形）的读数与 95% CI：`{W2_TEST_JSON}` 的 "
         f"models.{W2_MODEL}.per_split（RUN-0072；CI 是各运行 metrics.json 里运行时算的，{res} 次重抽）。核对：与 `{leaf}` 逐格相同。",
         f"  - W2-c + 回放：`{W2_REPLAY_RUN}/{W2_MODEL}/refseg/{W2_REPLAY_SETTING}/<划分>/metrics.json` 的 headline（RUN-0076 补充 2）。",
         f"  - C1 = X-2（主方法）、C1 的椭圆消融、C2 = X-3（DEV-0073）：每个划分取 {xleaf} 的 headline（RUN-0092；与上面各行同一口径，"
         f"运行时的 CI）。核对：读数与 `{X_TEST_JSON}` 的 per_split 相同（CI 不比：分析是 {xres} 次重抽）。",
         f"  - 合并列（所有行）：`{W2_POOLED_JSON}`（`scripts/mgr/w2_test_pooled.py`，D152）：5 个划分逐 id 合并，cIoU = Σinter / Σunion、"
         f"gIoU = 逐句 IoU 的均值，95% CI {pres} 次重抽，种子 {w2.get('pooled_seed') or MISSING}；各系统的名字按（运行目录, 设置）对上 "
         "w2_test_pooled.SYSTEMS。W2-a 的逐句交并由 RUN-0095 从 HPC 抽回（"
         + (f"`{(w2.get('pooled_sources') or {}).get(pooled_name('w2a'))}`" if (w2.get("pooled_sources") or {}).get(pooled_name("w2a")) else "还没有")
         + "）。缺的系统写\"—\"并进上面的 missing。核对：W2-c、W1-F 与候选三行的合并读数与 CI 和 x_test_table.json 相同。",
         f"  - 配对差 W2-c − W1-F、W2-a − W1-F：`{W2_TEST_JSON}` 的 pairs 与 pairs_pooled（配对 bootstrap，ratio-of-sums，"
         f"{w2.get('pairs_resamples') or 0:,} 次，种子 {w2.get('pairs_seed') or MISSING}；W2-a 由 RUN-0081 在 HPC 上补齐）。",
         f"  - 配对差 C1 − W2-c、C1 − W1-F、C1 − 椭圆消融、C2 − W2-c：`{X_TEST_JSON}`（`scripts/mgr/x_test_table.py`）的 candidates.<候选>."
         f"per_split.<划分> 与 pooled 的 main_minus_w2c、main_minus_w1f、main_minus_ablation（逐句配对 bootstrap，{xres} 次，种子 "
         f"{w2.get('x_seed') or MISSING}）；gIoU 表取对应的 g_ 键。",
         f"  - 回放 − W2-c：**没有 JSON**，只在 `{W2_REPLAY_MD}` §1 的表里（3 位小数；重抽次数文档里没写），从那张表解析；核对：表里的 W2-c、"
         "回放读数与上两项的值四舍五入相同。",
         f"  - E1 零样本多边形的解析失败：`{W2T.RUNS[W2_MODEL]}/{W2_MODEL}/refseg/baseline_e1/<划分>/metrics.json` 的 parse。",
         f"  - 外部基线 LLMSeg（RUN-0078）：修正前（seg）、修正后（rec，主配置）的读数与 CI 取 `{LLMSEG_RUN}/{W2_MODEL}/refseg/llmseg/<划分>/metrics.json` "
         f"的 readouts.<seg|rec>.headline；合并读数、同样 300 句上的 W2-c 读数（RUN-0072）与 LLMSeg − W2-c 的配对差取 `{LLMSEG_COMPARE}`（D139，CPU 补跑）；"
         "每图调用次数取 readouts.<seg|rec>.compute_per_image.n_model_calls。C1 的同句读数与 C1 − LLMSeg（修正后、修正前）取 "
         f"`{X_TEST_JSON}` 的 per_split.<划分>.llmseg 与 pooled.llmseg（main_on_subset、main_minus_llmseg_rec、main_minus_llmseg_seg）。",
         "- 版式：系统为行、划分（加合并）为列——论文里指代分割表的惯例。三块：9B 各系统（DEV-0073 起末尾加 C1、C1 的椭圆消融、C2 三行）；"
         "配对差（W2-c − W1-F、W2-a − W1-F、回放 − W2-c，加 C1 − W2-c、C1 − W1-F、C1 − 椭圆消融、C2 − W2-c）；外部基线 LLMSeg（修正前、"
         "修正后两行，同句 W2-c 与 C1 并列，下接 LLMSeg − W2-c 两行与 C1 − LLMSeg 两行）。数值 ×100、一位小数，CI 用 [low, high]。",
         "- 合并列：DEV-0073 起各系统一块的合并列都取 w2_test_pooled.json（D152；表注写明来源与重抽次数，删掉了 DEV-0071 的"
         "\"Per-system pooled cIoU is not part of the analysis\"）。配对差的合并取各自的分析 JSON；LLMSeg 的合并读数照旧取 compare.json（没有 CI）。",
         "- gIoU 附录表：同一版式；配对差一块只有 C1、C2 的四行（W2 的配对差分析只算了 cIoU）；C1 − W1-F 与 LLMSeg 一块 C1 两行的逐划分 gIoU "
         "取 x_test_table.json 的 per_split.<划分>.g_main_minus_w1f 与 per_split.<划分>.llmseg.g_*（origin/main 5b3bb70ee 补上）。"
         + ("分析里仍没有的格子写\"—\"，见上面\"分析里没有算的格子\"。" if w2.get("gaps") else "") + "LLMSeg − W2-c 的 gIoU 差取 compare.json。",
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
    explore = "（本次生成时 --with-explore 打开）" if lc["with_explore"] else "（本次生成时 --with-explore 关着，缺省）"
    L += ["## 7. 语言代价：`fig_lang_cost.pdf` / `fig_lang_cost.png`", "",
          f"- 数据：`{LANG_COST_JSON}`（`scripts/mgr/lang_cost_table.py`，D125-3、D147-4）的 battery（大题量 MMLU-Pro 2,000 题、IFEval 541 条）与 ttext（定位、计数）"
          "四个读数，每格是相对同节点基座的差（百分点）与 95% CI；健康检查的标红（health.*.flag）只写进图注，不进图。"
          "缺省画四方对照的四个系统（E3b+C3、W2-c、W2-c+回放、X-2（C1）；DEV-0073 起 X-2 的 T 文本由 RUN-0091 补齐）。本次画出的系统的来源路径（JSON 原文）：",
          ] + [f"  - {s['name']}：" + "；".join(f"{k} `{v}`" for k, v in (s.get("sources") or {}).items() if isinstance(v, str) and "/" in v)
               for s in lc["systems"] if not s["explore"]] + [
          f"- `--with-explore`：再加 X-3（C2，消融；同一份 JSON 的 \"{LANG_ABLATION[0][0]}\"）与 X-3L（探索候选，没有进 G-EX；大题量取 "
          + "、".join(f"`{b}`" for _n, b, _h, _nd in EXPLORE_LANG) + "，健康检查的标红取 " + "、".join(f"`{h}`" for _n, _b, h, _nd in EXPLORE_LANG)
          + f"，{EXPLORE_RUNS}；没有 T 文本读数，图注写明）。{explore}",
          "- 版式：读数自上而下（底色 = T 文本通道），每个读数里各系统各一行，点 = 差，横线 = 95% CI，右侧写数值。"
          "健康检查的标红只写在图注（本次：" + "；".join(f"{s['name']} " + ("、".join(f"{f['reading']} {f['delta_pp']:+.1f}" for f in s["health_flags"]) or "没有标红")
                                           for s in lc["systems"]) + "）。配对的基座节点从来源路径的 _hpcgpuNN 读：" + _pairing_text(lc["systems"]) + "。",
          "", "**图注（英文）**", "", captions["fig_lang_cost"], "",
          "## 8. 计数分档：`fig_count_bins.pdf` / `fig_count_bins.png`", "",
          f"- 数据：`{COUNT_ERRORS_JSON}`（`scripts/mgr/count_errors.py`）的三个大档 {', '.join(cb['bins'])}："
          "(a) 各系统的准确率（bins.<档>.acc）；(b) 与配对基座的差及配对 bootstrap 95% CI（bins.<档>.vs_base）。缺省画基座、W2-c、W2-c+回放、"
          "X-2（C1）、C3、E3b+C3；--with-explore 时在 X-2 后面再加 X-3（C2）" + ("（本次打开）" if cb.get("with_explore") else "（本次关着）") +
          "。预测文件（JSON 原文）：",
          ] + [f"  - {k}：`{v}`" for k, v in (cb.get("sources") or {}).items()] + [
          "- 为什么分两个面板：分析 JSON 里只有\"与基座之差\"的 CI，没有各档准确率本身的 CI（不重算），所以 CI 画在 (b) 上，(a) 只画准确率。",
          f"- 基座：W2-c、W2-c + 回放、X-2（C1）配 base@0069，C3、E3b+C3 配 base@0056；两次基座读数逐条相同（{cb.get('same_answer_rate')}）时只画一根（base@0069）。",
          "", "**图注（英文）**", "", captions["fig_count_bins"], ""]
    L += ["## 9. 正文主表：`tab_main.tex` / `tab_main.md`（DEV-0073）", "",
          "- 行：自己的框、W1-F、最优候选（4 个工具候选里的真值最优）、W2-a、W2-c | C1 = X-2（ours）、C1 的椭圆消融、C2 = X-3。"
          "列：5 个划分（只有读数）、合并（读数与 95% CI）、每句调用 / 生成 token。cIoU ×100，一位小数。",
          f"- 数据：每个划分同 `tab_w2_test`（RUN-0072 的 w2_test_table.json 与 RUN-0092 的叶目录）；合并列取 `{W2_POOLED_JSON}`（{pres} 次重抽）；"
          f"末列取 `{X_TEST_JSON}` 的 compute（候选）与 compute_reference（W1-F、W2-c、LLMSeg），自己的框、最优候选、W2-a 在分析里没有计算记录，写\"—\"并在表注说明。",
          f"- 外部基线一块：同样 1,500 句上 LLMSeg 修正后（主配置）的合并读数与 CI、C1 的同句合并读数与 CI、C1 − LLMSeg 修正后的配对差与 CI，都取 "
          f"`{X_TEST_JSON}` 的 candidates.{C1}.pooled.llmseg（llmseg_rec、main_on_subset、main_minus_llmseg_rec；{xres} 次重抽）。核对：LLMSeg 的合并读数与 "
          "compare.json 相同。",
          "- 不加粗（DEV-0073 审查第 1 条：合并列的最高值落在消融 C2 上，加粗 C1 又会误导）。表注另加一句：C1 在任何 test 运行之前就定为主方法"
          f"（预注册 X-5a，负责人裁定），C2 是去掉修正流的简化版，合并读数 C2 与 C1 之差对照预注册 X-6 的预测（±{100 * X6_C1_C2_MARGIN:g} 点）。",
          "", "**表注（英文）**", ""] + [f"- {n}" for n in main_table_notes(w2)] + [""]
    full_extra = [s["label"] for s in lt["systems"] if s["role"] != "main"]
    main_lt = [s for s in lt["systems"] if s["role"] == "main"]
    L += ["## 10. 语言代价四方对照表：`tab_lang_cost.tex` / `tab_lang_cost.md`（附录版 `tab_lang_cost_full.*`；DEV-0073）", "",
          f"- 数据：`{LANG_COST_JSON}`：健康检查三项（health.<读数>：差与 flag）、大题量 MMLU-Pro 与 IFEval 严格（battery：差与 95% CI）、T 文本的定位与计数"
          "（ttext：差与 95% CI）。列：E3b+C3、W2-c、W2-c+回放、X-2（C1）；附录版再加 " + ("、".join(full_extra) or MISSING) + "。",
          "- 基座读数（Base 列，表头下）：都取 lang_cost_9b.json 这一个来源——健康检查取 health.*.base_pct、大题量取 battery.*.base"
          "（origin/main 5b3bb70ee 修了 lang_cost_table.py 的键名之后才有数）、T 文本取 ttext.*.base。与来源 lb_compare 文件的核对（基座读数、差与 CI）"
          "写在测试里（tests/test_paper_figures.py）。同一读数各系统配的基座若不同，就按节点分别写；本次：" +
          ("；".join(f"{_lang_row_label(main_lt, r)} = {_lang_base(main_lt, r)}" for r in lt["rows"] if " / " in _lang_base(main_lt, r))
           or "各读数的基座都相同") + "。",
          "- 健康检查只给差与是否标红（† = 照简报 8.11 的规则标红）；大题量与 T 文本给差 [95% CI]。",
          "", "**表注（英文，四方表）**", ""] + [f"- {n}" for n in lang_table_notes(lt, main_lt)] + [""]
    L += ["## 11. 计算记账表：`tab_compute.tex` / `tab_compute.md`（DEV-0073）", "",
          f"- 数据：`{X_TEST_JSON}` 的 candidates.<候选>.compute（C1 主设置与椭圆消融、C2 主设置）与 compute_reference（W1-F、W2-c、LLMSeg）：每句的模型调用、"
          "生成 token、输入 token、模型调用时间、墙钟（5 个划分按句数加权的均值）。小数位同 x_test_table.md。",
          "- LLMSeg 的墙钟写\"—\"：x_test_table.md 的说明——LLMSeg 的 compute.json 记的墙钟是并发下的累计，与其他行的每句墙钟不同口径。",
          "", "**表注（英文）**", ""] + [f"- {n}" for n in compute_table_notes(w2)] + [""]
    return L


def write_readme(root: Path, out: Path, e1, e2, e3, itf, captions: dict, files: dict, w2=None, sz=None, lc=None, cb=None,
                 lt=None) -> Path:
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
         "青色对白底不足 3:1，图上带数值标签或有表可查（lang_cost_9b.md、count_errors_9b.md）。",
         "- C1 = X-2（主方法，DEV-0073）固定用分类槽 7（紫 #4a3aa7，对白底 ≥ 3:1），出现 C1 的每张图（fig_lang_cost、fig_count_bins）都是它；"
         "--with-explore 的 X-3（C2）用槽 5（粉）、X-3L 用槽 6（绿）；槽 4（黄，DEV-0071 给 X-2 的）不再用。validate_palette.js（白底 #ffffff）："
         "fig_lang_cost 缺省四色（橙、蓝、青、紫）全对 PASS（CVD ΔE 最小 9.2，常规最小 16.3）；--with-explore 六色按画出的相邻顺序 PASS（相邻 CVD ΔE 最小 17.6、"
         "常规 24.0）；fig_count_bins 缺省（灰、蓝、青、紫、深灰、橙）与 --with-explore（再加粉）按画出的相邻顺序 PASS（CVD ΔE 最小 15.9、常规最小 17.2；"
         "两个灰色是有意的中性色，色度检查照例不适用）。粉与橙全对的常规 ΔE 12.9 不到 15，但两者在两张图里都不相邻，图例与数值标签可查。",
         "- 每个数字都从结果文件读；逐点来源见 `data/*.json`（每个点带 source 路径）。", ""]
    # 缺的数据
    w2s = [x for x in (w2, sz, lc, cb, lt) if x is not None]
    miss = e1["missing"] + e2["missing"] + e3["missing"] + itf["missing"] + [m for x in w2s for m in x["missing"]]
    L += ["## 缺的数据（missing）", ""]
    L += [f"- {m}" for m in miss] if miss else ["- 无"]
    if e3["gaps"]:
        L += ["", "E3 表里缺的格子（按种子）：", ""] + [f"- {g}" for g in e3["gaps"]]
    L.append("")
    if w2 is not None and w2.get("gaps"):  # DEV-0073：分析 JSON 在、候选在，但这一格的键没有（分析没有算），表里写"—"
        L += ["## 分析里没有算的格子（表里是\"—\"，不是待补的数据）", ""]
        grouped: dict = {}
        for g in w2["gaps"]:
            col = "pooled" if g["col"] == "pooled" else "<划分>"
            grouped.setdefault((g["metric"], g["row"], g["path"].replace(g["col"], col) if col != "pooled" else g["path"]), []).append(g["col"])
        for (metric, row, path), cols in grouped.items():
            where = "合并" if cols == ["pooled"] else ("5 个划分" if set(cols) == set(w2["splits"]) else "、".join(cols))
            L.append(f"- {metric} {row}（{where}）：`{path}` 没有")
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
        L += readme_w2(w2, sz, lc, cb, lt, captions)
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
                    help="DEV-0073：语言代价图与计数分档在缺省（含 X-2 = C1）之外再画 X-3（C2，消融）；语言代价图另加探索候选 X-3L")
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
    w2, sz, lc, cb = collect_w2(root), collect_w2_size(root), collect_lang_cost(root, a.with_explore), collect_count_bins(root, a.with_explore)
    lt = collect_lang_table(root)  # DEV-0073
    for metric, name in W2_METRICS:
        files[name] = tab_w2(w2, out, metric, name)
    files["tab_main"] = tab_main(w2, out)
    files["tab_lang_cost"] = tab_lang_cost(lt, out)
    files["tab_lang_cost_full"] = tab_lang_cost(lt, out, full=True)
    files["tab_compute"] = tab_compute(w2, out)
    files["fig_w2_size"] = fig_w2_size(sz, out)
    files["fig_lang_cost"] = fig_lang_cost(lc, out)
    files["fig_count_bins"] = fig_count_bins(cb, out)
    w2caps = {"fig_w2_size": caption_w2_size(sz), "fig_lang_cost": caption_lang_cost(lc), "fig_count_bins": caption_count_bins(cb)}
    for name, cap in w2caps.items():
        p = out / f"{name}_caption.tex"
        p.write_text(f"% Generated by scripts/figures/make_paper_figures.py; do not edit.\n\\caption{{{_tex_escape(cap)}}}\n", encoding="utf-8")
        files[name].append(p)
    captions.update(w2caps)
    main_data = {"rows": {k: {"label": ROW_LABELS[k], **w2["rows"].get("cIoU", {}).get(k, {})} for g in MAIN_ROWS for k in g},
                 "compute": w2["compute"],
                 "external": {"llmseg_rec": (w2["llmseg"].get("x_pooled") or {}).get("cIoU", {}).get("rec"),
                              "c1": (w2["llmseg"]["rows"].get("cIoU", {}).get("c1") or {}).get("pooled"),
                              "c1-rec": (w2["llmseg"]["pairs"].get("cIoU", {}).get("c1-rec") or {}).get("pooled")},
                 "missing": w2["missing"]}
    compute_data = {"rows": {k: w2["compute"].get(k) for k, _l in COMPUTE_ROWS}, "wall_not_listed": list(COMPUTE_WALL_NA)}
    for name, data in (("tab_w2_test", w2), ("fig_w2_size", sz), ("fig_lang_cost", lc), ("fig_count_bins", cb), ("tab_main", main_data),
                       ("tab_lang_cost", lt), ("tab_compute", compute_data)):
        p = out / "data" / f"{name}.json"
        p.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        files[name].append(p)
    readme = write_readme(root, out, e1, e2, e3, itf, captions, files, w2, sz, lc, cb, lt)
    print(f"wrote {sum(len(v) for v in files.values()) + 1} files to {out} (README: {readme})")
    miss = e1["missing"] + e2["missing"] + e3["missing"] + itf["missing"] + w2["missing"] + sz["missing"] + lc["missing"] + cb["missing"] + \
        lt["missing"]
    if miss:
        print("missing:\n  " + "\n  ".join(miss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
