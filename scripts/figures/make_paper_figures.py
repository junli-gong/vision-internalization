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
- 统一主表（DEV-0077；D154，预注册 U）：论文正文主表 tab_unified（12 个系统在同样的 5 个划分 × 300 句上，灰色的文献 SOTA 块放在最后）与
  它的图 fig_unified（5 个划分面板加合并面板，文献线只画在划分面板）；分档图 fig_bins（人与非人、形状复杂度四分位）；附录的种子波动表
  tab_seeds。数字取 docs/analysis/unified_table.json、x_bins.json、seed_variance.json 与 docs/references/refseg_sota.json。tab_main 改作附录的
  全 5,000 句版，只改 caption。
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

# DEV-0077（D154；预注册 U）：统一主表 tab_unified 与图 fig_unified、分档图 fig_bins、种子波动附录表 tab_seeds。
# 数字都取管理员的分析 JSON（unified_table.py、x_bins.py、seed_variance.py 的输出）与文献读数 refseg_sota.json，不重算 bootstrap。
UNIFIED_JSON = "docs/analysis/unified_table.json"  # 12 个系统 × 5 个划分 × 300 句（RUN-0072、0078、0092、0095、0097、0100）
SOTA_JSON = "docs/references/refseg_sota.json"  # 专家 SOTA 的文献数字（预注册 U-3a）：完整 test 划分，×100，只作灰色参考
X_BINS_JSON = "docs/analysis/x_bins.json"  # 按人与非人、形状复杂度分档（预注册 U-6；元数据 RUN-0099）
SEED_JSON = "docs/analysis/seed_variance.json"  # C1 配方三个种子（开发数据与语言检查，不是 test）
SUBSET_CONFIG = "configs/w2_llmseg_subsets.json"  # 统一主表的句子（hash_rank，与 LLMSeg 相同）；只读它的 salt 与规则写表注
UT = _load_script("_paperfig_unified_table", "scripts/mgr/unified_table.py")  # 统一主表的系统清单（主方法名 MAIN、运行目录）
XB = _load_script("_paperfig_x_bins", "scripts/mgr/x_bins.py")  # 分档的约定（PERSON = COCO 类别 1、MIN_SUPER）
SV = _load_script("_paperfig_seed_variance", "scripts/mgr/seed_variance.py")  # 种子的运行目录与健康检查路径（SEEDS）
# unified_table.json 的系统名 → (表里的英文名, 图里的短名)。已有的英文名取 ROW_LABELS、LLMSEG_ROWS（同一个系统在各表同名）。
UNIFIED_LABELS = {
    UT.MAIN: (ROW_LABELS["c1"], "C1 = X-2 (ours)"),
    "X-3（C2，消融）": (ROW_LABELS["c2"], "C2 = X-3"),
    "W2-c": (ROW_LABELS["w2c"], "W2-c"),
    "W2-a": (ROW_LABELS["w2a"], "W2-a"),
    "模型自己的框": (ROW_LABELS["ownbox"], "Own box"),
    "W1-F": (ROW_LABELS["w1f"], "W1-F"),
    "E1 结构强制多边形（9B 零样本）": ("E1 structure-forced polygon (zero-shot)", "E1 structured polygon"),
    "9B E3b 像素 token 网格": ("E3b pixel-token grid (structure-forced decoding)", "E3b pixel-token grid"),
    "LLMSeg 修正后（主配置）": (dict(LLMSEG_ROWS)["rec"], "LLMSeg after refinement"),
    "LLMSeg 修正前": (dict(LLMSEG_ROWS)["seg"], "LLMSeg before refinement"),
    "LISA-7B（xinlai/LISA-7B-v1）": ("LISA-7B, released v1 weights (run by us)", "LISA-7B v1 (run by us)"),
    "C1 的椭圆消融": (ROW_LABELS["c1_ellipse"], "C1 with ellipse for spline"),
}
# unified_table.json 的组（预注册 U-2 的四组）→ (表里的组标题, 图例), 颜色。C1 另用槽 7（W2_COLORS 的 X-2）；W2-c 所在的组用 W2-c 的槽 1；
# 外部方法槽 2、"各种手"槽 3、附带的椭圆消融用"对照"的中性深灰（同 C3）。按画出的相邻顺序（紫、蓝、青、橙、深灰）过 validate_palette.js。
UNIFIED_GROUPS = {"本文的方法": ("Our methods (Qwen3.5-9B)", "Our other methods"),
                  "各种手": ("Other hands: the same Qwen3.5-9B outputs the mask another way", "Other hands (same Qwen3.5-9B)"),
                  "外部方法": ("External methods", "External methods"),
                  "附带": ("Capability attribution (ablation of C1)", "Attribution ablation")}
UNIFIED_SOTA_LABEL = "Expert models from the literature (full test splits; reference only, not paired)"
# 文献模型用的视觉模型与训练数据：JSON 的 note 已写了骨干与训练数据的（UNINEXT-H、DeRIS-L）直接用 note；note 写的是别的事的两行在这里补上
# （照 refseg_sota_notes.md 的"选作灰线的 4 行"），note 照录在它后面。
SOTA_TRAINING = {"LISA-7B (ft)": "LLaVA-7B + SAM, fine-tuned on RefCOCO/+/g",
                 "GLaMM (FT)": "Vicuna-7B + SAM-H, GranD pre-training, then fine-tuned on RefCOCO/+/g"}
# 新跑三项（叶目录只有这 300 句）：核对 unified_table.json 的逐划分读数 = 叶目录 metrics.json 的 headline。(系统名, 模型目录, 设置)
UNIFIED_SUBSET_LEAVES = (("E1 结构强制多边形（9B 零样本）", W2_MODEL, "u_e1_poly_structured"),
                         ("9B E3b 像素 token 网格", W2_MODEL, "u_e3b_grid"),
                         ("LISA-7B（xinlai/LISA-7B-v1）", "lisa-7b-v1", "lisa_7b"))
UNIFIED_LLMSEG = {"LLMSeg 修正后（主配置）": "rec", "LLMSeg 修正前": "seg"}  # LLMSeg 叶目录的 readouts
UNIFIED_LISA = "LISA-7B（xinlai/LISA-7B-v1）"
# fig_bins：x_bins.json 的两种分档（按前缀找）与两个配对差；配对差的颜色取被减的那个系统在 fig_unified 里的颜色
BINS_SCHEMES = (("(i)", "(a) Person vs. non-person"), ("(iii)", "(b) Shape complexity c = P²/A, quartiles"))
BINS_PERSON_LABELS = {"人": "Person", "非人": "Non-person"}
# (x_bins.json 的键, 图例, 被减系统的组（定颜色）, 图注里的说法)
BINS_DIFFS = (("x2_minus_w2c", "C1 − W2-c", "本文的方法", "against W2-c"),
              ("x2_minus_ellipse", "C1 − in-box ellipse ablation", "附带", "against the in-box ellipse ablation"))
# tab_seeds：种子波动的两个设置（seed_variance.json 的 settings 键）
SEED_SETTINGS = ((XT.MAIN, "C1 recipe (spline)"), (XT.ABL, "In-box ellipse ablation"))
DEV_SPLIT_LABELS = {"explore_val_refcoco_unc_val": "explore_val RefCOCO val", "explore_val_refcoco+_unc_val": "explore_val RefCOCO+ val",
                    "explore_val_refcocog_umd_val": "explore_val RefCOCOg val"}

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
# DEV-0077：fig_unified 按组分色（见 UNIFIED_GROUPS），C1 固定槽 7；文献的灰色竖虚线用 MUTED（只是参考，不是系统）
UNIFIED_GROUP_COLORS = {"本文的方法": W2_COLORS["W2-c"], "各种手": SERIES_3, "外部方法": SERIES_2, "附带": W2_COLORS["C3"]}
UNIFIED_MAIN_COLOR = W2_COLORS["X-2（C1）"]
SOTA_COLOR = MUTED


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
    out["spline_n"], out["spline_n_source"] = _spline_n(root)  # 样条的控制点数（C1 运行的 config.yaml）
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
# 读取：统一主表、分档、种子波动（DEV-0077；只读分析 JSON 与文献读数，不重算 bootstrap）
# ---------------------------------------------------------------------------------------------


def _ut_point(c: dict | None, level, n_res, source: str) -> dict | None:
    """分析 JSON 的 {"ciou"/"giou"/"diff", "ci": [low, high], "n"} → _point，记上重抽次数（JSON 顶层的 n_resamples）。"""
    pt = _ci_list_point(c, level, source)
    if pt is not None:
        pt["n_resamples"] = n_res
    return pt


def _value_only(p: dict | None) -> dict | None:
    return None if p is None else {"value": p.get("value")}


def run_ids(texts) -> list[str]:
    """来源字符串里的运行编号，按出现的顺序："RUN-0072（逐句交并取 RUN-0095）" → ["0072", "0095"]；
    "experiments/W2/0097_lisa_hpcgpu06" → ["0097"]（同 _runs_of 的目录写法）。"""
    out: list[str] = []
    for t in texts:
        for m in re.finditer(r"RUN-(\d{4})|experiments/[^/\s]+/(\d{4})_", t or ""):
            r = m.group(1) or m.group(2)
            if r not in out:
                out.append(r)
    return out


def _spline_n(root: Path) -> tuple[int | None, str]:
    """样条的控制点数：C1 运行的 config.yaml（x2.n，检查点训练时定的 n）。"""
    cfg = _x_leaf(root, C1, XT.MAIN, W2T.SPLITS[0]) / "config.yaml"
    m = re.search(r"^x2:\s*\n\s+n:\s*(\d+)\s*$", cfg.read_text(encoding="utf-8"), re.M) if cfg.is_file() else None
    return (int(m.group(1)) if m else None), f"{_rel(root, cfg)} :: x2.n"


def _close(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-12


def collect_unified(root: Path) -> dict:
    """tab_unified 与 fig_unified：unified_table.json 的 12 个系统（照 JSON 的顺序与组，一个不少）与 refseg_sota.json 的文献读数。
    读数与 CI 都取 JSON（逐划分与合并都是 unified_table.py 在这 300 / 1,500 句上算的 10,000 次重抽）；另与叶目录、x_test_table.json、
    LLMSeg 的 compare.json 逐格核对。"""
    out = {"json": UNIFIED_JSON, "sota_json": SOTA_JSON, "splits": list(W2T.SPLITS), "systems": [], "sota": [], "missing": [],
           "checks": [], "lisa": {}}
    uj, sj = _load(root / UNIFIED_JSON), _load(root / SOTA_JSON)
    if uj is None:
        out["missing"].append(f"统一主表：{UNIFIED_JSON} 不在")
    if sj is None:
        out["missing"].append(f"统一主表的文献读数：{SOTA_JSON} 不在")
    uj = uj or {}
    for k in ("subset", "n_per_split", "n_resamples", "level", "seed", "main"):
        out[k] = uj.get(k)
    sub = _load(root / (uj.get("subset") or SUBSET_CONFIG)) or {}
    out["subset_salt"] = sub.get("salt")
    res, lev = uj.get("n_resamples"), uj.get("level")
    for e in uj.get("systems") or []:
        name, group = e.get("name"), e.get("group")
        lab = UNIFIED_LABELS.get(name)
        if lab is None:
            out["missing"].append(f"统一主表：系统 {name} 没有英文名（UNIFIED_LABELS），表里用原名")
        if group not in UNIFIED_GROUPS:
            out["missing"].append(f"统一主表：组 {group} 没有英文名（UNIFIED_GROUPS），表里用原名")
        done = e.get("status") == "完成"
        rec = {"name": name, "label": (lab or (name, name))[0], "short": (lab or (name, name))[1], "group": group,
               "main": name == uj.get("main"), "source": e.get("source"), "status": e.get("status"), "runs": run_ids([e.get("source")]),
               "per_split": {}, "pooled": {}, "minus": {}, "minus_per_split": {}}
        if not done:  # 跑坏的、没跑完的照样列出（预注册 U-1：照实报告），格子写"—"
            out["missing"].append(f"统一主表 {name}：status 是 {e.get('status')}（{e.get('source')}）")
        src = f"{UNIFIED_JSON} :: systems[{name}]"
        for metric in ("cIoU", "gIoU"):
            for sp in W2T.SPLITS:
                c = ((((e.get("per_split") or {}).get(sp)) or {}).get(metric)) if done else None
                rec["per_split"].setdefault(sp, {})[metric] = _ut_point(c, lev, res, f"{src}.per_split.{sp}.{metric}")
                if done and c is None:
                    out["missing"].append(f"统一主表 {name} {sp} {metric}（{UNIFIED_JSON} 里没有）")
            c = ((e.get("pooled") or {}).get(metric)) if done else None
            rec["pooled"][metric] = _ut_point(c, lev, res, f"{src}.pooled.{metric}")
            if done and c is None:
                out["missing"].append(f"统一主表 {name} 合并 {metric}（{UNIFIED_JSON} 里没有）")
            mm = ((((e.get("main_minus") or {}).get("pooled")) or {}).get(metric)) if done and not rec["main"] else None
            rec["minus"][metric] = _ut_point(mm, lev, res, f"{src}.main_minus.pooled.{metric}")
            if done and not rec["main"] and mm is None:
                out["missing"].append(f"统一主表 主方法 − {name} {metric}（{UNIFIED_JSON} 里没有）")
        for sp in W2T.SPLITS:  # 逐划分的配对差（只有 cIoU）：只进 data JSON
            mm = (((e.get("main_minus") or {}).get("per_split")) or {}).get(sp) if done and not rec["main"] else None
            rec["minus_per_split"][sp] = _ut_point(mm, lev, res, f"{src}.main_minus.per_split.{sp}")
        out["systems"].append(rec)
    if uj and sum(s["main"] for s in out["systems"]) != 1:
        out["missing"].append(f"统一主表：{UNIFIED_JSON} 的 main（{uj.get('main')}）不是恰好一个系统")
    if uj.get("main") and uj.get("main") != UT.MAIN:
        out["missing"].append(f"统一主表：{UNIFIED_JSON} 的 main 是 {uj.get('main')}，unified_table.py 的 MAIN 是 {UT.MAIN}")
    # 文献读数（×100，完整 test 划分）：按划分名对上 W2 的划分
    if sj:
        if list(sj.get("splits") or []) != list(W2T.SPLITS):
            out["missing"].append(f"统一主表的文献读数：{SOTA_JSON} 的划分（{sj.get('splits')}）与 W2 test 的不同，按划分名对齐")
        for r in sj.get("rows") or []:
            vals = dict(zip(sj.get("splits") or [], r.get("values") or []))
            out["sota"].append({"name": r.get("name"), "kind": r.get("kind"), "metric": r.get("metric"),
                                "values": {sp: vals.get(sp) for sp in W2T.SPLITS}, "source": r.get("source"), "note": r.get("note"),
                                "training": SOTA_TRAINING.get(r.get("name")), "ref": f"{SOTA_JSON} :: rows[{r.get('name')}]"})
            if r.get("name") not in SOTA_TRAINING and not r.get("note"):
                out["missing"].append(f"统一主表的文献读数 {r.get('name')}：没有视觉模型与训练数据的说明（SOTA_TRAINING 或 JSON 的 note）")
            if any(vals.get(sp) is None for sp in W2T.SPLITS):
                out["missing"].append(f"统一主表的文献读数 {r.get('name')}：缺划分")
        out["sota_unit"], out["sota_caveat"] = sj.get("unit"), sj.get("caveat")
    by = {s["name"]: s for s in out["systems"] if s["status"] == "完成"}
    # 核对 1：新跑三项的叶目录只有这 300 句，逐划分读数 = 叶目录 metrics.json 的 headline（CI 不比：叶目录是 1,000 次重抽）
    bad, n = [], 0
    for name, mdir, setting in UNIFIED_SUBSET_LEAVES:
        s = by.get(name)
        run = next((m.group(0) for m in [re.search(r"experiments/\S+", (s or {}).get("source") or "")] if m), None)
        if s is None or run is None:
            continue
        for sp in W2T.SPLITS:
            f = root / run / mdir / "refseg" / setting / sp / "metrics.json"
            d = _load(f) or {}
            if name == UNIFIED_LISA and d and not out["lisa"]:
                out["lisa"] = {"revision": d.get("revision"), "lisa_commit": d.get("lisa_commit"), "protocol": d.get("protocol"),
                               "model": d.get("model"), "source": _rel(root, f)}
            for metric in ("cIoU", "gIoU"):
                lp, p = _leaf_headline(f, metric, root), s["per_split"][sp][metric]
                n += 1
                if lp is None or p is None or not _close(lp["value"], p["value"]) or lp.get("n") != p.get("n"):
                    bad.append(f"{name} {sp} {metric}")
    if n:
        out["checks"].append(("一致" if not bad else "不一致") + f"：{UNIFIED_JSON} 的新跑三项（E1 多边形、E3b 网格、LISA）逐划分读数与句数 = "
                             f"叶目录 metrics.json 的 headline（{n} 格）" + (f"（不同或缺：{bad}）" if bad else ""))
    # 核对 2：LLMSeg 修正前后 = RUN-0078 叶目录 readouts.<seg|rec>.headline（同样 300 句）
    bad, n = [], 0
    for name, rd in UNIFIED_LLMSEG.items():
        s = by.get(name)
        if s is None:
            continue
        for sp in W2T.SPLITS:
            f = root / LLMSEG_RUN / W2_MODEL / "refseg" / "llmseg" / sp / "metrics.json"
            if not out.get("llmseg_commit"):
                out["llmseg_commit"] = (_load(f) or {}).get("llmseg_commit")
            for metric in ("cIoU", "gIoU"):
                lp, p = _leaf_headline(f, metric, root, readout=rd), s["per_split"][sp][metric]
                n += 1
                if lp is None or p is None or not _close(lp["value"], p["value"]):
                    bad.append(f"{name} {sp} {metric}")
    if n:
        out["checks"].append(("一致" if not bad else "不一致") + f"：{UNIFIED_JSON} 的 LLMSeg 修正前后逐划分读数 = {LLMSEG_RUN} 叶目录的 "
                             f"readouts.<seg|rec>.headline（{n} 格）" + (f"（不同或缺：{bad}）" if bad else ""))
    # 核对 3：C1 的同句读数（逐划分与合并，读数与 CI）与 C1 − LLMSeg = x_test_table.json 的 llmseg 子块（同一套句子、同样的重抽）
    xj = _load(root / X_TEST_JSON)
    c1 = (((xj or {}).get("candidates") or {}).get(C1)) or {}
    m = by.get(uj.get("main"))
    bad, n = [], 0
    if m and c1:
        for sp in W2T.SPLITS + ["pooled"]:
            blk = ((c1.get("pooled") if sp == "pooled" else (c1.get("per_split") or {}).get(sp)) or {}).get("llmseg") or {}
            for metric, jk in (("cIoU", "main_on_subset"), ("gIoU", "g_main_on_subset")):
                p = m["pooled"][metric] if sp == "pooled" else m["per_split"][sp][metric]
                c = blk.get(jk) or {}
                n += 1
                if p is None or not all(_close(a, b) for a, b in zip((c.get(metric.lower()), *(c.get("ci") or [None, None])),
                                                                       (p["value"], p["low"], p["high"]))):
                    bad.append(f"C1 {sp} {metric}")
        for name, rd in UNIFIED_LLMSEG.items():
            s, c = by.get(name), (((c1.get("pooled") or {}).get("llmseg")) or {}).get(f"main_minus_llmseg_{rd}") or {}
            if s is None:
                continue
            p = s["minus"]["cIoU"]
            n += 1
            if p is None or not all(_close(a, b) for a, b in zip((c.get("diff"), *(c.get("ci") or [None, None])), (p["value"], p["low"], p["high"]))):
                bad.append(f"C1 − {name}")
    if n:
        out["checks"].append(("一致" if not bad else "不一致") + f"：{UNIFIED_JSON} 的 C1 同句读数（逐划分与合并，读数与 CI）与 C1 − LLMSeg = "
                             f"{X_TEST_JSON} 的 llmseg 子块（{n} 格）" + (f"（不同：{bad}）" if bad else ""))
    # 核对 4：W2-c 的同句读数 = LLMSeg 对照的 compare.json 里的 W2-c（b）
    cmp_ = _load(root / LLMSEG_COMPARE)
    s = by.get("W2-c")
    bad, n = [], 0
    if s and cmp_:
        for sp in W2T.SPLITS + ["pooled"]:
            blk = (cmp_.get("pooled") if sp == "pooled" else (cmp_.get("splits") or {}).get(sp)) or {}
            for metric in ("cIoU", "gIoU"):
                p = s["pooled"][metric] if sp == "pooled" else s["per_split"][sp][metric]
                n += 1
                if p is None or not _close(((blk.get("rec") or {}).get(metric) or {}).get("b"), p["value"]):
                    bad.append(f"W2-c {sp} {metric}")
    if n:
        out["checks"].append(("一致" if not bad else "不一致") + f"：{UNIFIED_JSON} 的 W2-c 同句读数 = {LLMSEG_COMPARE} 的 W2-c（{n} 格）"
                             + (f"（不同：{bad}）" if bad else ""))
    out["spline_n"], out["spline_n_source"] = _spline_n(root)
    # 运行编号与节点（表注 "Data:"）：系统来源里的编号；只写编号的（RUN-0072、0078）按已知的运行目录找节点
    known = {r: d for d in (W2T.RUNS[W2_MODEL], LLMSEG_RUN, *XT.CANDS.values()) for r in run_ids([d])}
    dirs = {r: (re.search(r"experiments/\S+", s["source"] or "") or [None])[0] for s in out["systems"] for r in s["runs"][:1]}
    out["runs"] = sorted({r for s in out["systems"] for r in s["runs"]})
    out["nodes"] = sorted({n_ for r in out["runs"] for n_ in [_node(dirs.get(r) or known.get(r))] if n_})
    return out


def _bin_label(key: str, prefix: str, cuts: list) -> str | None:
    """x_bins.json 的档名 → 英文：人 / 非人；"c Q1 (≤22)" → "Q1: c ≤ 22.0"（切点取 JSON 的 cuts.complexity，一位小数）。"""
    if prefix == "(i)":
        return BINS_PERSON_LABELS.get(key)
    if "A = 0" in key:
        return "A = 0 (not in the quartiles)"
    m = re.match(r"c Q([1-4])\b", key)
    if not m or len(cuts) != 3:
        return None
    k, q = int(m.group(1)), [f"{x:.1f}" for x in cuts]
    return {1: f"Q1: c ≤ {q[0]}", 2: f"Q2: {q[0]} < c ≤ {q[1]}", 3: f"Q3: {q[1]} < c ≤ {q[2]}", 4: f"Q4: c > {q[2]}"}[k]


def collect_bins(root: Path) -> dict:
    """fig_bins：x_bins.json 的 (i) 人与非人、(iii) 形状复杂度四分位；每档的句数与 X-2 − W2-c、X-2 − 椭圆的配对差与 95% CI；每个划分里人的占比。"""
    out = {"json": X_BINS_JSON, "schemes": [], "missing": [], "person_share": {}}
    j = _load(root / X_BINS_JSON)
    if j is None:
        out["missing"].append(f"分档：{X_BINS_JSON} 不在")
    j = j or {}
    for k in ("n", "n_resamples", "level", "seed", "meta"):
        out[k] = j.get(k)
    out["cuts"] = j.get("cuts") or {}
    out["person_share"] = {sp: (j.get("person_share_by_split") or {}).get(sp) for sp in W2T.SPLITS}
    if j and not all(out["person_share"].values()):
        out["missing"].append(f"分档：{X_BINS_JSON} 的 person_share_by_split 缺划分")
    out["runs"] = run_ids([XT.REF, XB.X2_RUN, j.get("meta")])
    schemes = j.get("schemes") or {}
    for prefix, title in BINS_SCHEMES:
        key = next((k for k in schemes if k.startswith(prefix + " ")), None)
        if key is None and j:
            out["missing"].append(f"分档：{X_BINS_JSON} 没有 {prefix} 的分档")
        bins = []
        for b, r in (schemes.get(key) or {}).items():
            lab = _bin_label(b, prefix, out["cuts"].get("complexity") or [])
            if lab is None:
                out["missing"].append(f"分档 {key} 的档 {b}：没有英文名，图里用原名")
            rec = {"key": b, "label": lab or b, "n": r.get("n"), "x2": r.get("x2"), "w2c": r.get("w2c"), "ellipse": r.get("ellipse"), "diffs": {}}
            for dk, _lab, _g, _p in BINS_DIFFS:
                rec["diffs"][dk] = _ut_point(r.get(dk), j.get("level"), j.get("n_resamples"), f"{X_BINS_JSON} :: schemes.{key}.{b}.{dk}")
                if rec["diffs"][dk] is None:
                    out["missing"].append(f"分档 {key} {b} {dk}（{X_BINS_JSON} 里没有）")
            bins.append(rec)
        out["schemes"].append({"prefix": prefix, "key": key, "title": title, "bins": bins})
    return out


def collect_seeds(root: Path) -> dict:
    """tab_seeds：seed_variance.json 的三个种子——X-2 主设置与椭圆消融的合并 cIoU 与 CI、− W2-c 参照、− seed 0、跨种子均值与样本标准差；
    语言读数（大题量 MMLU-Pro、IFEval 严格，T 文本定位、计数，相对同节点基座的差与 CI）；健康检查的差与标红（题数与门槛取健康检查的 compare 文件）。"""
    out = {"json": SEED_JSON, "seeds": [], "missing": [], "across": {}, "resamples": {}}
    j = _load(root / SEED_JSON)
    if j is None:
        out["missing"].append(f"种子波动：{SEED_JSON} 不在")
    j = j or {}
    n_res, lev = j.get("n_resamples"), j.get("level")
    out.update({"n_resamples": n_res, "level": lev, "seed_bootstrap": j.get("seed"), "splits": j.get("splits") or []})
    ref = j.get("w2c_ref") or {}
    out["w2c_ref"] = {"source": ref.get("source"), "pooled": _ut_point(ref.get("pooled"), lev, n_res, f"{SEED_JSON} :: w2c_ref.pooled"),
                      "split_n": {sp: ((ref.get("per_split") or {}).get(sp) or {}).get("n") for sp in out["splits"]}}
    res: dict = {"battery": set(), "ttext": {LCT.N_RES}}
    for s, e in (j.get("seeds") or {}).items():
        rec = {"seed": int(s), "status": e.get("status"), "source": e.get("source"), "node": _node(e.get("source")), "settings": {},
               "language": {}, "health": {}, "lang_sources": {}, "lang_nodes": {}}
        out["seeds"].append(rec)
        if e.get("status") != "完成":
            out["missing"].append(f"种子波动 seed {s}：status 是 {e.get('status')}（{e.get('source')}）")
            continue
        src = f"{SEED_JSON} :: seeds.{s}"
        for st, _lab in SEED_SETTINGS:
            x = (e.get("settings") or {}).get(st) or {}
            rec["settings"][st] = {k: _ut_point(x.get(jk), lev, n_res, f"{src}.settings.{st}.{jk}")
                                   for k, jk in (("pooled", "pooled"), ("minus_w2c", "minus_w2c_pooled"), ("minus_seed0", "minus_seed0_pooled"))}
            for k in ("pooled", "minus_w2c") + (("minus_seed0",) if int(s) != 0 else ()):
                if rec["settings"][st][k] is None:
                    out["missing"].append(f"种子波动 seed {s} {st} {k}（{SEED_JSON} 里没有）")
        lang = e.get("language") or {}
        rec["lang_sources"] = lang.get("sources") or {}
        rec["lang_nodes"] = {"battery": _node(rec["lang_sources"].get("battery")), "ttext": _node(rec["lang_sources"].get("ttext_base"))}
        for sec, key, _lab in LANG_READOUTS:
            r = (lang.get(sec) or {}).get(key)
            ci = (r or {}).get("ci95_pp") or [None, None]
            rec["language"][key] = None if not r or r.get("delta_pp") is None else {
                "delta": r["delta_pp"], "low": ci[0], "high": ci[1], "n": r.get("n"), "base": r.get("base"), "section": sec,
                "source": f"{src}.language.{sec}.{key}"}
            if rec["language"][key] is None:
                out["missing"].append(f"种子波动 seed {s} 语言读数 {sec} {key}（{SEED_JSON} 里没有）")
        bc = _load(root / rec["lang_sources"]["battery"]) if rec["lang_sources"].get("battery") else None
        if bc:
            res["battery"].add((((bc.get("statistics") or {}).get("bootstrap")) or {}).get("n_resamples"))
        # 健康检查：差与标红取 seed_variance.json；题数与标红门槛取该种子的 health_compare.json（seed_variance.py 的 SEEDS）
        h = e.get("health") or {}
        hpath = (SV.SEEDS.get(int(s)) or (None, None))[1]
        hc = _load(root / hpath) if hpath else None
        rec["health_source"], rec["health_node"] = hpath, _node(hpath)
        rec["health_threshold"] = (hc or {}).get("threshold_pp")
        for reading, metric in LCT.HEALTH_ROWS:
            k = f"{reading}.{metric}"
            r = (h.get("rows") or {}).get(k)
            n = next((x.get("n") for x in (hc or {}).get("rows") or [] if x.get("reading") == reading and x.get("metric") == metric), None)
            rec["health"][k] = None if r is None or r.get("delta_pp") is None else {
                "delta": r["delta_pp"], "flag": bool(r.get("flag")), "n": n, "source": f"{src}.health.rows.{k}"}
            if rec["health"][k] is None:
                out["missing"].append(f"种子波动 seed {s} 健康检查 {k}（{SEED_JSON} 里没有）")
    for st, _lab in SEED_SETTINGS:
        a = (j.get("across_seeds") or {}).get(st) or {}
        out["across"][st] = {"seeds": a.get("seeds"), "mean": a.get("mean"), "sd": a.get("sd"), "source": f"{SEED_JSON} :: across_seeds.{st}"}
    out["resamples"] = {k: sorted(x for x in v if x) for k, v in res.items()}
    out["runs"] = {"seeds": {r["seed"]: run_ids([r["source"]]) for r in out["seeds"]}, "w2c_ref": run_ids([out["w2c_ref"]["source"]])}
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


# ---- fig_unified、fig_bins（DEV-0077）------------------------------------------------------------

UNIFIED_GROUP_GAP = 0.5  # 组与组之间多空的行高
UNIFIED_XMAX = 90.0  # 横轴 0–90（cIoU ×100）：最高的文献读数 86.5 也在轴内
SOTA_LABEL_STEP = 0.85  # 文献线的名字一行一个，放在第一根横条之上


def unified_color(s: dict) -> str:
    """fig_unified 的颜色：C1 固定槽 7，其余按组（UNIFIED_GROUP_COLORS）。"""
    return UNIFIED_MAIN_COLOR if s["main"] else UNIFIED_GROUP_COLORS.get(s["group"], INK_2)


def unified_rows(systems: list[dict]) -> list[float]:
    """每个系统的纵坐标（自上而下，JSON 的顺序）；组与组之间空 UNIFIED_GROUP_GAP。"""
    ys, y, prev = [], 0.0, None
    for s in systems:
        if prev is not None and s["group"] != prev:
            y += UNIFIED_GROUP_GAP
        ys.append(y)
        y += 1.0
        prev = s["group"]
    return ys


def sota_layout(u: dict, sp: str) -> list[dict]:
    """一个划分面板里的文献线：按读数从小到大，名字自下而上一行一个（名字写在线的左边，线只画到自己那一行，名字不会压到别的线）。"""
    rows = sorted((r for r in u["sota"] if r["values"].get(sp) is not None), key=lambda r: r["values"][sp])
    return [{"name": r["name"], "value": r["values"][sp], "label_y": -0.95 - k * SOTA_LABEL_STEP} for k, r in enumerate(rows)]


def fig_unified(u: dict, out: Path) -> list[Path]:
    """5 个划分各一个面板加"合并"面板：横条 = 各系统的 cIoU（×100），误差棒 = unified_table.json 的 95% CI；划分面板里画文献的灰色竖虚线
    （完整 test 划分，只作参考），合并面板不画（文献没有合并读数）。"""
    systems = u["systems"]
    ys = unified_rows(systems)
    panels = u["splits"] + ["pooled"]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_IN, 4.6), sharey=True, layout="constrained")
    n_sota = max((len(sota_layout(u, sp)) for sp in u["splits"]), default=0)
    y_bottom = (ys[-1] if ys else 0.0) + 0.7
    y_top = -0.95 - max(n_sota - 1, 0) * SOTA_LABEL_STEP - 0.6
    for ax, col in zip(axes.flat, panels):
        pts = [(s["pooled"] if col == "pooled" else s["per_split"][col])["cIoU"] for s in systems]
        pp = [None if p is None else {**p, "value": 100 * p["value"], "low": None if p["low"] is None else 100 * p["low"],
                                      "high": None if p["high"] is None else 100 * p["high"]} for p in pts]
        ax.barh(ys, [p["value"] if p else 0.0 for p in pp], height=0.72, color=[unified_color(s) for s in systems], linewidth=0, zorder=2)
        have = [(y, p) for y, p in zip(ys, pp) if p]
        if have:
            ax.errorbar([p["value"] for _y, p in have], [y for y, _p in have], xerr=_yerr([p for _y, p in have]), fmt="none", ecolor=INK,
                        elinewidth=0.6, capsize=1.0, capthick=0.6, zorder=3)
        for y, p in zip(ys, pp):
            if p is None:  # 没跑完或跑坏的系统：照样占一行，写 n/a
                ax.text(1.0, y, "n/a", ha="left", va="center", fontsize=5.5, color=MUTED, zorder=3)
            elif col == "pooled":  # 合并面板写数值（划分面板的数值见 tab_unified）
                ax.text((p["high"] if p["high"] is not None else p["value"]) + 1.0, y, f"{p['value']:.1f}", ha="left", va="center",
                        fontsize=5.5, color=INK_2, zorder=4)
        if col == "pooled":
            ax.text(1.0, (-0.95 + y_top + 0.6) / 2, "No literature lines: the literature\nreports no pooled reading.", ha="left",
                    va="center", fontsize=5.5, color=INK_2, style="italic")
            n = next((p["n"] for p in pts if p and p.get("n")), None)
            ax.set_title(f"Pooled (n={_nfmt(n)})", loc="left")
        else:
            for r in sota_layout(u, col):
                ax.plot([r["value"], r["value"]], [y_bottom, r["label_y"]], color=SOTA_COLOR, lw=0.7, ls=(0, (3.0, 1.6)), zorder=4)
                ax.text(r["value"] - 0.8, r["label_y"], r["name"], ha="right", va="center", fontsize=5.3, color=INK_2, zorder=5)
            n = next((p["n"] for p in pts if p and p.get("n")), None)
            ax.set_title(f"{split_label(col)} (n={_nfmt(n)})", loc="left")
        ax.set_xlim(0, UNIFIED_XMAX)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.grid(False)
        ax.grid(True, axis="x")
        ax.tick_params(axis="y", length=0)
    axes[0, 0].set_yticks(ys)
    axes[0, 0].set_yticklabels([s["short"] for s in systems])
    axes[0, 0].set_ylim(y_bottom, y_top)  # 自上而下
    fig.supxlabel("cIoU (×100); error bars: 95% bootstrap CI", fontsize=7)
    handles = [Patch(color=UNIFIED_MAIN_COLOR, lw=0, label=next((s["short"] for s in systems if s["main"]), "main method"))]
    for g in dict.fromkeys(s["group"] for s in systems):
        handles.append(Patch(color=UNIFIED_GROUP_COLORS.get(g, INK_2), lw=0, label=UNIFIED_GROUPS.get(g, (g, g))[1]))
    if u["sota"]:
        handles.append(Line2D([], [], color=SOTA_COLOR, lw=0.7, ls=(0, (3.0, 1.6)), label="Literature, full test split (reference)"))
    fig.legend(handles=handles, loc="outside upper center", ncol=_legend_cols(len(handles), 3), handlelength=1.2, handletextpad=0.4,
               columnspacing=1.0)
    return _save(fig, out, "fig_unified")


def fig_unified_data(u: dict) -> dict:
    """data/fig_unified.json：每个面板画出的每根横条（读数、CI、来源、颜色）与文献线（读数、名字所在的行）。"""
    panels = {}
    for col in u["splits"] + ["pooled"]:
        panels[col] = {"bars": [{"system": s["name"], "label": s["short"], "group": s["group"], "color": unified_color(s),
                                 **((s["pooled"] if col == "pooled" else s["per_split"][col]).get("cIoU") or {"value": None})}
                                for s in u["systems"]],
                       "sota": [] if col == "pooled" else sota_layout(u, col)}
    return {"panels": panels, "json": u["json"], "sota_json": u["sota_json"], "n_resamples": u.get("n_resamples"), "missing": u["missing"]}


def fig_bins(bn: dict, out: Path) -> list[Path]:
    """(a) 人与非人、(b) 形状复杂度四分位：每档两根横条——C1 − W2-c 与 C1 − 椭圆消融（×100，95% 配对 CI），颜色取被减系统在 fig_unified 里的颜色；
    每档的句数写在刻度名下。"""
    schemes = bn["schemes"]
    heights = [max(len(s["bins"]), 1) for s in schemes]
    fig, axes = plt.subplots(len(schemes), 1, figsize=(SINGLE_COL_IN, 1.05 + 0.5 * sum(heights)), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    nd = len(BINS_DIFFS)
    h = 0.8 / nd
    offs = [(k - (nd - 1) / 2) * h for k in range(nd)]
    allv = [0.0]
    for ax, sch in zip(axes[:, 0], schemes):
        for k, (dk, _lab, g, _p) in enumerate(BINS_DIFFS):
            c = UNIFIED_GROUP_COLORS[g]
            for i, b in enumerate(sch["bins"]):
                p = b["diffs"].get(dk)
                if p is None:
                    continue
                v = 100 * p["value"]
                lo, hi = (100 * p["low"] if p["low"] is not None else v), (100 * p["high"] if p["high"] is not None else v)
                y = i + offs[k]
                ax.barh([y], [v], height=h * 0.9, color=c, linewidth=0, zorder=2)
                ax.errorbar([v], [y], xerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.0,
                            capthick=0.6, zorder=3)
                ax.text(max(hi, v, 0) + 0.6, y, _signed(v), ha="left", va="center", fontsize=5.5, color=INK_2, zorder=4)
                allv += [lo, hi]
        ax.set_yticks(range(len(sch["bins"])))
        ax.set_yticklabels([f"{b['label']}\n(n={_nfmt(b['n'])})" for b in sch["bins"]])
        ax.set_ylim(max(len(sch["bins"]), 1) - 0.5, -0.5)  # 分档 JSON 不在时面板留空（README 报缺）
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color=INK_2, lw=0.6, zorder=1)
        ax.grid(False)
        ax.grid(True, axis="x")
        ax.set_title(sch["title"], loc="left")
    lo, hi = min(allv), max(allv)
    axes[-1, 0].set_xlim(min(lo - 1.0, 0.0), hi + 0.22 * (hi - min(lo, 0.0)) + 1.0)
    axes[-1, 0].set_xlabel("Paired ΔcIoU (×100), 95% CI")
    handles = [Patch(color=UNIFIED_GROUP_COLORS[g], lw=0, label=lab) for _dk, lab, g, _p in BINS_DIFFS]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    return _save(fig, out, "fig_bins")


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
           "<": r"$<$", ">": r"$>$", "²": r"$^2$"}
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
                 md_title: str, md_rows: list[str], notes: list[str], after_tex: list[str] | None = None) -> list[Path]:
    """同 tab_w2 的版式：table*、booktabs、resizebox；Markdown 同一份表加表注。after_tex（DEV-0077）：表下面、table* 之内的几行
    （放 caption 装不下的表注；caption 太长时 LaTeX 量单行宽度会超出最大尺寸而报错）。"""
    L = ["% Generated by scripts/figures/make_paper_figures.py; do not edit. Needs \\usepackage{booktabs,graphicx" +
         (",multirow" if any("\\multirow" in x for x in body_tex) else "") +
         (",xcolor" if any("\\textcolor" in x for x in body_tex) else "") + "}.",
         "\\begin{table*}[t]", "\\centering", caption.replace("—", "--"), f"\\label{{tab:{label}}}", "\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         *header_tex, *body_tex, "\\bottomrule", "\\end{tabular}}", *(after_tex or []), "\\end{table*}", ""]
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


def tab_main(w2: dict, out: Path, unified: dict | None = None) -> list[Path]:
    """全 5,000 句表（DEV-0073 时是正文主表；DEV-0077 起改作附录，正文主表是 tab_unified，只改了 caption）：
    行 = 自己的框、W1-F、最优候选、W2-a、W2-c | C1、椭圆消融、C2；列 = 5 个划分（只读数）与合并（带 CI）、每句调用 / 生成 token；
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
    npool = _pooled_n(w2)
    # DEV-0077：正文主表换成 tab_unified（统一主表），这张改作附录的全 5,000 句版；只改 caption（标题与表注第一句），数值不变
    caption = (f"\\caption{{Full W2 test, all {f'{npool:,}' if npool else MISSING} expressions (appendix; Qwen3.5-9B, cIoU). " +
               main_appendix_note(w2, unified, latex=True) + " " + " ".join(_tex_escape(n) for n in notes) + "}")
    header = ["System & " + " & ".join(_tex_escape(h) for h in heads) + " \\\\"]
    return _write_table(out, "tab_main", caption, "main", "l" + "c" * len(heads), header, body,
                        f"full W2 test, all {f'{npool:,}' if npool else MISSING} expressions (appendix; cIoU)", M,
                        [main_appendix_note(w2, unified, latex=False)] + notes)


def main_appendix_note(w2: dict, unified: dict | None, latex: bool) -> str:
    """tab_main 表注的第一句（DEV-0077）：这是附录的全 5,000 句版，正文主表是 tab_unified（所有系统同样的句子）。"""
    npool, n_u = _pooled_n(w2), (unified or {}).get("n_per_split")
    ref = "Table~\\ref{tab:unified}" if latex else "tab_unified"
    esc = _tex_escape if latex else (lambda x: x)
    return (esc(f"Appendix version on all {f'{npool:,}' if npool else MISSING} W2 test expressions ({_n_split(w2, 'cIoU')} per split); "
                "the main-text results table is ") + ref +
            esc(f", where every system, including the external methods, is evaluated on the same "
                f"{_nfmt(n_u)} expressions per split."))


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


# ---- tab_unified（DEV-0077：论文正文主表）、tab_seeds（附录）--------------------------------------

U_DEF_NOTE = ("Own box: the model's own box (E1 protocol) as the mask; W1-F: that box plus the default GrabCut candidate (a fixed "
              "pipeline, no LLM decision); W2-a: the frozen model picks one of four tool candidates (zero-shot); W2-c: LoRA text "
              "supervision to propose the box and pick a candidate. E1 structure-forced polygon: the zero-shot Qwen3.5-9B writes the "
              "outline as a polygon, decoding forced to a valid polygon of at least 3 vertices, thinking off. E3b pixel-token grid: the "
              "E3b LoRA on Qwen3.5-9B (seed-0 checkpoint of the E3 runs) emits the mask as a grid of pixel tokens; structure-forced "
              "decoding, single turn, guided upsampling.")


def _sota_cell(v, latex: bool) -> str:
    """文献读数（JSON 里已是 ×100）：一位小数；表里灰色。"""
    if v is None:
        return "\\textcolor{gray}{--}" if latex else "—"
    return f"\\textcolor{{gray}}{{${v:.1f}$}}" if latex else f"{v:.1f}"


def lisa_hf_id(name: str) -> str | None:
    """unified_table.json 里 LISA 的系统名带着发布权重的 HF 仓库名："LISA-7B（xinlai/LISA-7B-v1）" → "xinlai/LISA-7B-v1"。"""
    m = re.search(r"（([^（）]+/[^（）]+)）", name or "")
    return m.group(1) if m else None


def unified_table_notes(u: dict) -> list[str]:
    """tab_unified 的全部表注（.md 与 README 照这个顺序列出）。"""
    return [t for _where, t in unified_notes_tagged(u)]


def unified_notes_tagged(u: dict) -> list[tuple[str, str]]:
    """(位置, 表注)：位置 "cap" 进 .tex 的 caption，"below" 放在表下面的注里（系统定义与文献出处太长，全放进 caption 时 LaTeX 量单行宽度
    超出最大尺寸而报错）。.md 不分位置，按这个顺序全部列出。"""
    systems = u["systems"]
    main = next((s for s in systems if s["main"]), None)
    npool = next((s["pooled"]["cIoU"]["n"] for s in systems if s["pooled"].get("cIoU") and s["pooled"]["cIoU"].get("n")), None)
    res = _res_text([u.get("n_resamples")])
    lisa = u.get("lisa") or {}
    lisa_row = next((s for s in systems if s["name"] == UNIFIED_LISA), None)
    notes = [f"Every system on the same {_nfmt(npool)} expressions: the five W2 test splits (RefCOCO testA/testB, RefCOCO+ testA/testB, "
             f"RefCOCOg test), {_nfmt(u.get('n_per_split'))} per split, chosen by hash rank (the first {_nfmt(u.get('n_per_split'))} of "
             f"each 1,000-expression split by sha256 rank with salt {u.get('subset_salt') or MISSING}; {u.get('subset') or SUBSET_CONFIG}), "
             "the same expressions as LLMSeg's run (preregistration U-1). cIoU ×100 = summed intersection / summed union; gIoU ×100 = "
             f"mean per-expression IoU. Pooled: the five splits together with a 95% bootstrap CI over expressions ({res} resamples, seed "
             f"{u.get('seed') or MISSING}); per-split CIs are in the figure of this table and in unified_table.json. "
             f"C1 − system: per-expression paired bootstrap of the ratio of sums, 95%, {res} resamples. Failed pipelines and unparsable "
             "outputs score as empty masks.",
             "Reported as run, nothing selected: every system on the list preregistered before the three new baselines were run "
             "(preregistration U-2) is in the table whatever its result, and the list was not changed after their readings were seen. "
             "No entry is set in bold. " + (f"{main['label']} was designated the main method before any test run (preregistration "
                                             "X-5a); C2 = X-3 is its simplified version without the box-correction stream."
                                             if main else ""),
             _x_def_note(u) + " " + U_DEF_NOTE + " LLMSeg" + (f" (official code, commit {u['llmseg_commit'][:7]})" if u.get("llmseg_commit")
                                                               else "") +
             ": driven by the same zero-shot Qwen3.5-9B; before refinement = after its first two stages; after refinement = its main "
             "configuration."]
    notes = [("cap", notes[0]), ("cap", notes[1]), ("below", notes[2])]
    if lisa_row is not None:
        notes.append(("cap", "LISA-7B appears twice, and the two are not the same weights. The solid row is the released "
                     f"{lisa_hf_id(lisa_row['name']) or lisa.get('model') or MISSING}"
                     + (f" (revision {lisa['revision'][:7]})" if lisa.get("revision") else "") +
                     ", run by us with the official code" + (f" (commit {lisa['lisa_commit'][:7]})" if lisa.get("lisa_commit") else "") +
                     " and its refer_seg evaluation protocol (the protocol of the paper's tables), in a separate environment; its outputs "
                     "are used only for scoring, never in our data, labels or rewards. The released v1 weights have no published readings. "
                     "The gray row LISA-7B (ft) is the paper's model fine-tuned on RefCOCO/+/g, on the full test splits. LISA's training "
                     "data include COCO images, which overlap with RefCOCO test images at the image level (images, not referring "
                     "expressions)."))
    if u["sota"]:
        notes.append(("cap", "Gray rows: expert models as reported in the literature on the full test splits, not on our "
                      f"{_nfmt(u.get('n_per_split'))}-expression subsets; cIoU (oIoU, the same quantity) ×100, rounded to one decimal; "
                      "reference only, never paired, no pooled reading. Each uses learned vision models and is not bound by our "
                      "constraint that no learned vision model is used; their vision models, training data and sources are listed "
                      "under the table."))
        notes.append(("below", "Literature rows: " +
                      " ".join(f"{r['name']} ({r['kind']}; {r['training'] or r.get('note') or MISSING}): {r['source']}" +
                               (f"; {r['note']}" if r["training"] and r.get("note") else "") + "." for r in u["sota"])))
    by_run: dict = {}
    for s in systems:
        for k, r in enumerate(s["runs"]):
            by_run.setdefault(r, []).append(re.sub(r" \(.*\)$", "", s["short"]) + ("" if k == 0 else " per-expression scores"))
    nodes = u.get("nodes") or []
    notes.append(("cap", "Data: " + "; ".join(f"RUN-{r} ({', '.join(by_run[r])})" for r in sorted(by_run)) +
                  (f"; all on {nodes[0]} (D93-1)" if len(nodes) == 1 else f"; nodes {', '.join(nodes)}" if nodes else "") +
                  f"; readings from {UNIFIED_JSON.split('/')[-1]}, literature from {SOTA_JSON.split('/')[-1]}."))
    if u["missing"]:
        notes.append(("cap", "Missing: " + "; ".join(u["missing"]) + "."))
    return notes


def tab_unified(u: dict, out: Path) -> list[Path]:
    """正文主表（D154；预注册 U）：行 = unified_table.json 的系统（照它的顺序，按组分块，一个不少），最后是灰色的文献块；
    列 = 5 个划分的 cIoU、合并 cIoU [95% CI]、合并 gIoU、C1 − 该系统的 cIoU 配对差 [CI]。×100，一位小数；不加粗。"""
    splits = u["splits"]
    heads = [split_label(sp) for sp in splits] + ["Pooled cIoU [95% CI]", "Pooled gIoU", "C1 − system, cIoU [95% CI]"]
    ncol = len(heads) + 1
    body, M = [], ["| System | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    prev = object()
    for s in u["systems"]:
        if s["group"] != prev:
            gt = UNIFIED_GROUPS.get(s["group"], (s["group"],))[0]
            body += ["\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{_tex_escape(gt)}}}}} \\\\"]
            M.append(f"| *{gt}* |" + " |" * len(heads))
            prev = s["group"]
        per = [_value_only(s["per_split"][sp]["cIoU"]) for sp in splits]
        pooled, g, mm = s["pooled"].get("cIoU"), _value_only(s["pooled"].get("gIoU")), s["minus"].get("cIoU")
        cells_t = [fmt_w2(p, False, True) for p in per] + [fmt_w2(pooled, False, True), fmt_w2(g, False, True), fmt_w2(mm, True, True)]
        cells_m = [fmt_w2(p, False, False) for p in per] + [fmt_w2(pooled, False, False), fmt_w2(g, False, False), fmt_w2(mm, True, False)]
        body.append(_tex_escape(s["label"]) + " & " + " & ".join(cells_t) + " \\\\")
        M.append(f"| {s['label']} | " + " | ".join(cells_m) + " |")
    if u["sota"]:
        body += ["\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\textcolor{{gray}}{{\\emph{{{_tex_escape(UNIFIED_SOTA_LABEL)}}}}}}} \\\\"]
        M.append(f"| *{UNIFIED_SOTA_LABEL}; gray in the paper* |" + " |" * len(heads))
        for r in u["sota"]:
            lab = f"{r['name']}, {r['kind']}"
            body.append(f"\\textcolor{{gray}}{{{_tex_escape(lab)}}} & " + " & ".join(_sota_cell(r["values"].get(sp), True) for sp in splits) +
                        " & " + " & ".join(["\\textcolor{gray}{--}"] * 3) + " \\\\")
            M.append(f"| {lab} | " + " | ".join(_sota_cell(r["values"].get(sp), False) for sp in splits) + " | — | — | — |")
    tagged = unified_notes_tagged(u)
    caption = ("\\caption{Unified main table: referring segmentation with every system on the same W2 test expressions (cIoU $\\times$100). " +
               " ".join(_tex_escape(n) for w, n in tagged if w == "cap") + " Definitions of the systems are under the table.}")
    below = ["\\par\\smallskip", "\\begin{minipage}{\\textwidth}\\footnotesize",
             " ".join(_tex_escape(n) for w, n in tagged if w == "below"), "\\end{minipage}"]
    header = ["System & " + " & ".join(_tex_escape(h) for h in heads) + " \\\\"]
    return _write_table(out, "tab_unified", caption, "unified", "l" + "c" * len(heads), header, body,
                        "unified main table (same 1,500 W2 test expressions for every system, cIoU)", M, [n for _w, n in tagged],
                        after_tex=below)


def seeds_table_notes(sd: dict) -> list[str]:
    ref = sd.get("w2c_ref") or {}
    ns = ref.get("split_n") or {}
    parts = [f"{DEV_SPLIT_LABELS.get(sp, sp)} {_nfmt(n)}" for sp, n in ns.items()]
    npool = (ref.get("pooled") or {}).get("n")
    res = _res_text([sd.get("n_resamples")])
    seeds = [s for s in sd["seeds"] if s["status"] == "完成"]
    thr = next((s["health_threshold"] for s in seeds if s.get("health_threshold") is not None), None)
    hn = [f"{HEALTH_LABELS.get(k.split('.')[0], k)} {_nfmt(next((s['health'][k]['n'] for s in seeds if s['health'].get(k) and s['health'][k].get('n')), None))}"
          for k in (f"{r}.{m}" for r, m in LCT.HEALTH_ROWS)]
    tbase = sorted({r for s in seeds for r in run_ids([s["lang_sources"].get("ttext_base")])})
    tnode = sorted({s["lang_nodes"].get("ttext") for s in seeds if s["lang_nodes"].get("ttext")})
    notes = [f"Seed variation of C1's training recipe (X-2) on Qwen3.5-9B, three training seeds. Everything here is on development data "
             "and language checks, not on the test splits. Segmentation: the development splits pooled (" + ", ".join(parts) +
             f"; {_nfmt(npool)} expressions), cIoU ×100 with a 95% bootstrap CI; − W2-c reference: paired difference against a W2-c run "
             "on the same expressions; − seed 0: paired difference against seed 0; per-expression paired bootstrap of the ratio of sums, "
             f"95%, {res} resamples (seed {sd.get('seed_bootstrap') or MISSING}). Across seeds: mean ± sample standard deviation of the "
             "pooled cIoU over the three seeds (×100, two decimals). The in-box ellipse ablation replaces the spline with the ellipse "
             "inscribed in the proposed box, everything else unchanged.",
             "Language: difference from the base model in percentage points [95% paired bootstrap CI over items]. Language battery "
             "(MMLU-Pro letter answers, IFEval prompt-level strict; "
             f"{_res_text(sd['resamples'].get('battery', []))} resamples) against the base on the seed's node; T-text (grounding Acc@0.5, "
             f"CountBench counting accuracy; {_res_text(sd['resamples'].get('ttext', []))} resamples) against the base T-text run of "
             f"{', '.join(f'RUN-{r}' for r in tbase) or MISSING}" + (f" on {', '.join(tnode)}" if tnode else "") + ".",
             f"Health check (brief §8.11; fixed items: {', '.join(hn)}): difference only, against the base on the seed's node; † = flagged "
             f"by its rule (the headline metric drops by more than {thr if thr is not None else MISSING} points).",
             "Runs: " + "; ".join(
                 f"seed {s['seed']}: training and development evaluation {', '.join(f'RUN-{r}' for r in run_ids([s['source']]))} "
                 f"({s['node'] or MISSING}), language battery {', '.join(f'RUN-{r}' for r in run_ids([s['lang_sources'].get('battery')]))} "
                 f"({s['lang_nodes'].get('battery') or MISSING}), T-text {', '.join(f'RUN-{r}' for r in run_ids([s['lang_sources'].get('ttext')]))} "
                 f"({_node(s['lang_sources'].get('ttext')) or MISSING})"
                 for s in seeds) +
             f"; W2-c reference {', '.join(f'RUN-{r}' for r in run_ids([ref.get('source')])) or MISSING}. Data: {SEED_JSON.split('/')[-1]}."]
    if sd["missing"]:
        notes.append("Missing: " + "; ".join(sd["missing"]) + ".")
    return notes


def tab_seeds(sd: dict, out: Path) -> list[Path]:
    """附录：C1 配方三个种子的波动。行 = 分割（主设置、椭圆消融各三行：合并、− W2-c 参照、− seed 0）、大题量两项、T 文本两项、健康检查三项；
    列 = 各种子与跨种子的均值 ± 样本标准差（只有合并读数有）。"""
    seeds = sd["seeds"]
    heads = ["Reading"] + [f"Seed {s['seed']}" for s in seeds] + ["Across seeds"]
    body, M = [], ["| " + " | ".join(heads) + " |", "|---|" + "---:|" * (len(seeds) + 1)]
    npool = ((sd.get("w2c_ref") or {}).get("pooled") or {}).get("n")
    sections: list[tuple[str, list]] = []
    seg = []
    for st, lab in SEED_SETTINGS:
        a = sd["across"].get(st) or {}
        across = (f"{100 * a['mean']:.2f} ± {100 * a['sd']:.2f}" if a.get("mean") is not None and a.get("sd") is not None else "—")
        seg.append((f"{lab}, pooled (n={_nfmt(npool)})", [((s["settings"].get(st) or {}).get("pooled"), False) for s in seeds], across))
        seg.append((f"{lab} − W2-c reference", [((s["settings"].get(st) or {}).get("minus_w2c"), True) for s in seeds], "—"))
        seg.append((f"{lab} − seed 0", [((s["settings"].get(st) or {}).get("minus_seed0"), True) for s in seeds], "—"))
    sections.append(("Segmentation, development data (cIoU ×100)", seg))
    for sec_name, sec in (("Language battery (points)", "battery"), ("T-text (points)", "ttext")):
        rows = []
        for s_, key, _lab in LANG_READOUTS:
            if s_ != sec:
                continue
            n = next((s["language"][key]["n"] for s in seeds if s["language"].get(key) and s["language"][key].get("n")), None)
            rows.append((f"{LANG_TABLE_LABELS[sec].get(key, key)} (n={_nfmt(n)})", [(s["language"].get(key), "lang") for s in seeds], "—"))
        sections.append((sec_name, rows))
    hrows = []
    for r, m in LCT.HEALTH_ROWS:
        k = f"{r}.{m}"
        n = next((s["health"][k]["n"] for s in seeds if s["health"].get(k) and s["health"][k].get("n")), None)
        hrows.append((f"{HEALTH_LABELS.get(r, r)} (n={_nfmt(n)})", [(s["health"].get(k), "health") for s in seeds], "—"))
    sections.append(("Health check (points; † flagged)", hrows))

    def cell(c, kind, latex):
        if kind == "lang":
            return _lang_cell(c, "battery", latex)
        if kind == "health":
            return _lang_cell(c, "health", latex)
        return fmt_w2(c, kind, latex)
    for sec_lab, rows in sections:
        body += ["\\midrule", f"\\multicolumn{{{len(heads)}}}{{l}}{{\\emph{{{_tex_escape(sec_lab)}}}}} \\\\"]
        M.append(f"| *{sec_lab}* |" + " |" * (len(heads) - 1))
        for lab, cells, across in rows:
            body.append(f"{_tex_escape(lab)} & " + " & ".join(cell(c, k, True) for c, k in cells) + " & " +
                        _tex_escape(across).replace("—", "--") + " \\\\")
            M.append(f"| {lab} | " + " | ".join(cell(c, k, False) for c, k in cells) + f" | {across} |")
    notes = seeds_table_notes(sd)
    caption = ("\\caption{Seed variation of C1's training recipe (appendix; development data and language checks, not test). " +
               " ".join(_tex_escape(n) for n in notes) + "}")
    header = [" & ".join(_tex_escape(h) for h in heads) + " \\\\"]
    return _write_table(out, "tab_seeds", caption, "seeds", "l" + "c" * (len(seeds) + 1), header, body,
                        "seed variation of C1's recipe (appendix; development data, not test)", M, notes)


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


COLOR_NAMES = {SERIES_1: "blue", SERIES_2: "orange", SERIES_3: "aqua", SERIES_7: "violet", INK_2: "dark gray", MUTED: "gray"}
REF_UNIFIED = "⟦tab:unified⟧"  # 图注里指向 tab_unified 的占位：.tex 换成 Table~\ref{tab:unified}，README 换成 `tab_unified`


def caption_tex(cap: str) -> str:
    return _tex_escape(cap).replace(REF_UNIFIED, "Table~\\ref{tab:unified}")


def caption_md(cap: str) -> str:
    return cap.replace(REF_UNIFIED, "`tab_unified`")


def caption_unified(u: dict) -> str:
    systems = u["systems"]
    main = next((s for s in systems if s["main"]), None)
    npool = next((s["pooled"]["cIoU"]["n"] for s in systems if s["pooled"].get("cIoU") and s["pooled"]["cIoU"].get("n")), None)
    parts = [f"{main['short']} {COLOR_NAMES.get(UNIFIED_MAIN_COLOR, '')}"] if main else []
    for g in dict.fromkeys(s["group"] for s in systems):
        members = [s["short"] for s in systems if s["group"] == g and not s["main"]]
        gl = re.sub(r" \(.*\)$", "", UNIFIED_GROUPS.get(g, (g, g))[1])
        parts.append(f"{gl[0].lower() + gl[1:]} {COLOR_NAMES.get(UNIFIED_GROUP_COLORS.get(g), '')} ({', '.join(members)})")
    lisa = next((s for s in systems if s["name"] == UNIFIED_LISA), None)
    runs = ", ".join(f"RUN-{r}" for r in u.get("runs") or [])
    s = (f"Every system on the same {_nfmt(npool)} W2 test expressions ({_nfmt(u.get('n_per_split'))} per split, the expressions of "
         f"{REF_UNIFIED}), Qwen3.5-9B unless named otherwise. Bars: cIoU ×100 in each split (first five panels) and pooled (last panel, "
         f"with values); error bars: 95% bootstrap CI over the expressions of the panel ({_res_text([u.get('n_resamples')])} resamples). "
         "Colors by group: " + "; ".join(parts) + ".")
    if u["sota"]:
        s += (" Gray dashed lines: expert models as reported in the literature on the full test splits (" +
              _and([r["name"] for r in u["sota"]]) + f"; sources in {REF_UNIFIED}), not on these subsets; reference only, never paired. "
              "The pooled panel has none: the literature reports no pooled reading.")
        if lisa is not None and any(r["name"].startswith("LISA") for r in u["sota"]):
            s += (" LISA-7B (ft) is the paper's model fine-tuned on RefCOCO/+/g, not the released v1 weights run by us "
                  f"({COLOR_NAMES.get(UNIFIED_GROUP_COLORS.get(lisa['group']), '')} bar).")
    s += f" Data: {UNIFIED_JSON.split('/')[-1]} ({runs}); {SOTA_JSON.split('/')[-1]}."
    if u["missing"]:
        s += " Missing: " + "; ".join(u["missing"]) + "."
    return s


def caption_bins(bn: dict) -> str:
    cuts = (bn.get("cuts") or {}).get("complexity") or []
    share = []
    for sp in W2T.SPLITS:
        v = bn["person_share"].get(sp)
        share.append(f"{split_label(sp)} " + (MISSING if not v else f"{100 * v['person']:.0f}%"))
    diffs = _and([f"{phrase} ({COLOR_NAMES.get(UNIFIED_GROUP_COLORS[g], '')})" for _dk, _lab, g, phrase in BINS_DIFFS])
    titles = {s["prefix"]: s["title"].split(" ", 1)[0] for s in bn["schemes"]}
    a0 = any("A = 0" in b["key"] for s in bn["schemes"] for b in s["bins"])
    s = (f"Where {UNIFIED_LABELS[UT.MAIN][1]} gains, on all {_nfmt(bn.get('n'))} W2 test expressions (Qwen3.5-9B): paired cIoU "
         f"difference ×100 {diffs}; 95% paired bootstrap CI over expressions ({_res_text([bn.get('n_resamples')])} resamples, seed "
         f"{bn.get('seed') or MISSING}); n per bin below each label. {titles.get('(i)', '(a)')} Person (COCO category {XB.PERSON}) vs. "
         f"non-person. {titles.get('(iii)', '(b)')} Quartiles of the shape complexity c = P²/A of the ground-truth mask (P: length of "
         "the outer contour in pixels; A: area in pixels); cut points " + (_and([f"{x:.1f}" for x in cuts]) if cuts else MISSING) +
         f", computed from the {_nfmt(bn.get('n'))} ground-truth masks alone and fixed before any system's bin readings were computed "
         "(preregistration U-6)" + ("; masks with A = 0 form their own bin" if a0 else "") +
         ". Share of expressions that refer to a person: " + "; ".join(share) + ". Descriptive only. Data: "
         f"{X_BINS_JSON.split('/')[-1]} ({', '.join(f'RUN-{r}' for r in bn.get('runs') or [])}; the last is the per-expression "
         "metadata).")
    if bn["missing"]:
        s += " Missing: " + "; ".join(bn["missing"]) + "."
    return s


def readme_unified(u: dict, bn: dict, sd: dict, captions: dict) -> list[str]:
    """README 的 DEV-0077 各节：统一主表、它的图、分档、种子波动。"""
    L = ["## 12. 统一主表（正文主表）：`tab_unified.tex` / `tab_unified.md`（DEV-0077）", "",
         f"- 数据：`{UNIFIED_JSON}`（`scripts/mgr/unified_table.py`；预注册 U；D154）：{len(u['systems'])} 个系统 × 5 个划分 × "
         f"{_nfmt(u.get('n_per_split'))} 句（`{u.get('subset') or SUBSET_CONFIG}`，hash_rank，salt {u.get('subset_salt') or MISSING}，与 LLMSeg 同一份）。"
         f"逐划分与合并的 cIoU、gIoU 与 95% CI、主方法 − 各系统的配对差都是 JSON 里的（{_res_text([u.get('n_resamples')])} 次重抽，种子 "
         f"{u.get('seed') or MISSING}），不重算。",
         "- 行：照 JSON 的 systems 顺序，按 group 分块（本文的方法、各种手、外部方法、附带），一个不少；跑坏或没跑完的照样占一行，格子写\"—\"并进 missing。"
         "英文名：已有的系统用 tab_main / tab_w2_test 的同一个名字（ROW_LABELS、LLMSEG_ROWS），新的三项见 UNIFIED_LABELS。",
         "- 列：5 个划分的 cIoU（只读数）、合并 cIoU [95% CI]、合并 gIoU（只读数）、C1 − 该系统的 cIoU 配对差 [95% CI]（主方法一行写\"—\"）。×100，一位小数；不加粗。",
         f"- 灰色文献块（表的最后）：`{SOTA_JSON}`（预注册 U-3a；说明在 `docs/references/refseg_sota_notes.md`）的 {len(u['sota'])} 行，只填 5 个划分"
         "（JSON 里已是 ×100，表里取一位小数），合并与配对差写\"—\"；.tex 里用 xcolor 的 gray。表注里每行的视觉模型与训练数据：JSON 的 note 已写了的"
         "（UNINEXT-H、DeRIS-L）直接用 note，note 写的是别的事的（LISA-7B (ft)、GLaMM (FT)）取 SOTA_TRAINING（照 refseg_sota_notes.md 的表）并照录 note；"
         "出处照录 JSON 的 source。",
         "- 表注：子集的取法与句数、照实报告不挑、各系统的定义、LISA 两处不是同一个权重（发布的 v1 与论文的微调版；COCO 图像层面的重叠，D156）、"
         "文献数字在完整 test 划分上算且不参与配对、运行编号与节点。",
         "- .tex 的表注分两处：各系统的定义与文献各行的视觉模型、训练数据与出处放在表下面（table* 之内的 minipage），其余在 caption。"
         "全放进 caption 时约 5,000 字符，pdflatex 量 caption 单行宽度时报 Dimension too large（article 类与 caption 宏包都一样）。.md 不分，全部列为表注。",
         "", "**表注（英文）**", ""] + [f"- {n}" for n in unified_table_notes(u)] + [""]
    if u["checks"]:
        L += ["核对：", ""] + [f"- {c}" for c in u["checks"]] + [""]
    L += ["## 13. 统一主表的图：`fig_unified.pdf` / `fig_unified.png`（DEV-0077）", "",
          f"- 数据：同 tab_unified（`{UNIFIED_JSON}` 与 `{SOTA_JSON}`）。5 个划分各一个面板，最后一个面板是合并的 "
          f"{_nfmt(next((s['pooled']['cIoU']['n'] for s in u['systems'] if s['pooled'].get('cIoU')), None))} 句。",
          "- 误差棒：每个划分取 JSON 的 per_split CI（unified_table.py 在这 300 句上算的，10,000 次重抽）；合并面板取 pooled CI（10,000 次）。"
          "任务书写的是\"取各叶目录 metrics.json 的 CI\"：RUN-0072、0092 的叶目录是每个划分 1,000 句（不是这 300 句），新跑三项与 LLMSeg 的叶目录是"
          "这 300 句但只有 1,000 次重抽，所以统一取 JSON 的（同一口径、同一套重抽）；叶目录的读数另做核对（见上）。",
          "- 文献线：只在划分面板画，每条一根灰色竖虚线（MUTED），名字写在线的左边，按读数从小到大自下而上一行一个，线只画到自己那一行（名字不压别的线）。"
          "合并面板不画（文献没有合并读数），写一句说明。",
          "- 配色（按组分色；同一个系统在各图同色的约定只对 C1 与 W2-c 成立，其余系统按组上色）：C1 = X-2 槽 7（紫）；本文的方法其余（C2、W2-c、W2-a）"
          "槽 1（蓝，W2-c 的颜色）；各种手 槽 3（青）；外部方法 槽 2（橙）；附带的椭圆消融 中性深灰（同 C3 的\"对照\"）。"
          "validate_palette.js（白底 #ffffff）：按画出的相邻顺序（紫、蓝、青、橙、深灰）CVD ΔE 最小 9.2、常规最小 16.3，全对也是这两个数；"
          "深灰是有意的中性色，色度检查照例不适用；青色对白底不足 3:1，合并面板写了数值、划分面板的数值见 tab_unified（表格视图）。",
          "", "**图注（英文）**", "", caption_md(captions["fig_unified"]), ""]
    L += ["## 14. 分档：`fig_bins.pdf` / `fig_bins.png`（DEV-0077）", "",
          f"- 数据：`{X_BINS_JSON}`（`scripts/mgr/x_bins.py`；预注册 U-6）的 (i) 人与非人、(iii) 形状复杂度 c = P²/A 的四分位档：每档的句数、"
          "x2_minus_w2c 与 x2_minus_ellipse（逐句配对 bootstrap，ratio-of-sums，95%，10,000 次）。(ii) COCO 大类与 (iv) 相对面积不画（任务书只要 (a)(b)）。",
          "- 每个划分里人的占比取 person_share_by_split，写进图注。切点取 JSON 的 cuts.complexity（一位小数）。",
          "- 版式：单栏，两个面板上下排；每档两根横条（C1 − W2-c、C1 − 椭圆消融），横线 = 95% CI，右侧写数值，句数写在刻度名下。颜色取被减系统在 "
          "fig_unified 里的颜色：W2-c 蓝（槽 1）、椭圆消融深灰（validate_palette.js：两色 CVD ΔE 21.3、常规 21.8）。",
          "", "**图注（英文）**", "", caption_md(captions["fig_bins"]), ""]
    L += ["## 15. 种子波动（附录）：`tab_seeds.tex` / `tab_seeds.md`（DEV-0077）", "",
          f"- 数据：`{SEED_JSON}`（`scripts/mgr/seed_variance.py`；D150、D151、D153）：三个种子的 {XT.MAIN}（C1 的配方）与 {XT.ABL}（椭圆消融）"
          "的合并 cIoU 与 95% CI、− W2-c 参照、− seed 0（逐句配对 bootstrap，10,000 次）、跨种子的均值与样本标准差（across_seeds）；"
          "语言读数（大题量 MMLU-Pro、IFEval 严格，T 文本定位、计数；相对基座的差与 95% CI）；健康检查的差与标红（†）。"
          "x2_select、x2_select5 两个设置与 IFEval instruction-level 不进表（任务书没有要）。",
          "- 健康检查的题数与标红门槛取各种子的 health_compare.json（路径取 seed_variance.py 的 SEEDS）；大题量的重抽次数取各自 lb_compare 的 statistics。",
          "- 这些都只在开发数据（explore_val 三层、devtune800、devcheck2）与语言检查上，不是 test；表注写明，并列出各自的运行编号与节点。",
          "", "**表注（英文）**", ""] + [f"- {n}" for n in seeds_table_notes(sd)] + [""]
    return L


def _git_commit(root: Path) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, timeout=20).stdout.strip()
        return (sha or "unknown") + (" (working tree has uncommitted changes)" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def readme_w2(w2: dict, sz: dict, lc: dict, cb: dict, lt: dict, captions: dict, u: dict | None = None) -> list[str]:
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
    L += ["## 9. 附录的全 5,000 句表：`tab_main.tex` / `tab_main.md`（DEV-0073；DEV-0077 起正文主表换成 `tab_unified`，见第 12 节）", "",
          "- DEV-0077：只改了 caption（标题与表注第一句：附录的全 5,000 句版，正文主表是 tab_unified），数值、版式与 label（tab:main）不变。",
          "- 行：自己的框、W1-F、最优候选（4 个工具候选里的真值最优）、W2-a、W2-c | C1 = X-2（ours）、C1 的椭圆消融、C2 = X-3。"
          "列：5 个划分（只有读数）、合并（读数与 95% CI）、每句调用 / 生成 token。cIoU ×100，一位小数。",
          f"- 数据：每个划分同 `tab_w2_test`（RUN-0072 的 w2_test_table.json 与 RUN-0092 的叶目录）；合并列取 `{W2_POOLED_JSON}`（{pres} 次重抽）；"
          f"末列取 `{X_TEST_JSON}` 的 compute（候选）与 compute_reference（W1-F、W2-c、LLMSeg），自己的框、最优候选、W2-a 在分析里没有计算记录，写\"—\"并在表注说明。",
          f"- 外部基线一块：同样 1,500 句上 LLMSeg 修正后（主配置）的合并读数与 CI、C1 的同句合并读数与 CI、C1 − LLMSeg 修正后的配对差与 CI，都取 "
          f"`{X_TEST_JSON}` 的 candidates.{C1}.pooled.llmseg（llmseg_rec、main_on_subset、main_minus_llmseg_rec；{xres} 次重抽）。核对：LLMSeg 的合并读数与 "
          "compare.json 相同。",
          "- 不加粗（DEV-0073 审查第 1 条：合并列的最高值落在消融 C2 上，加粗 C1 又会误导）。表注另加一句：C1 在任何 test 运行之前就定为主方法"
          f"（预注册 X-5a，负责人裁定），C2 是去掉修正流的简化版，合并读数 C2 与 C1 之差对照预注册 X-6 的预测（±{100 * X6_C1_C2_MARGIN:g} 点）。",
          "", "**表注（英文）**", ""] + [f"- {n}" for n in [main_appendix_note(w2, u, latex=False)] + main_table_notes(w2)] + [""]
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
                 lt=None, u=None, bn=None, sd=None) -> Path:
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
         "- 统一主表的图与分档（DEV-0077）：fig_unified 按组分色，C1 = X-2 仍是槽 7（紫），本文的方法其余槽 1（蓝，W2-c 的颜色）、各种手槽 3（青）、"
         "外部方法槽 2（橙）、附带的椭圆消融中性深灰（同 C3 的\"对照\"）；文献的竖虚线用中性灰（MUTED）。validate_palette.js（白底 #ffffff）："
         "按画出的相邻顺序（紫、蓝、青、橙、深灰）与全对都是 CVD ΔE 最小 9.2、常规最小 16.3（深灰是中性色，色度检查照例不适用）。"
         "fig_bins 的两色（蓝、深灰）CVD ΔE 21.3、常规 21.8（深灰的色度检查同上不适用）。按组上色时 C2、W2-a、E3b、LISA 等与它们在别的图里的颜色不同（那些图里没有组）。",
         "- 每个数字都从结果文件读；逐点来源见 `data/*.json`（每个点带 source 路径）。", ""]
    # 缺的数据
    w2s = [x for x in (w2, sz, lc, cb, lt, u, bn, sd) if x is not None]
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
        L += readme_w2(w2, sz, lc, cb, lt, captions, u)
    if u is not None and bn is not None and sd is not None:
        L += readme_unified(u, bn, sd, captions)
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
    u, bn, sd = collect_unified(root), collect_bins(root), collect_seeds(root)  # DEV-0077
    for metric, name in W2_METRICS:
        files[name] = tab_w2(w2, out, metric, name)
    files["tab_main"] = tab_main(w2, out, u)
    files["tab_lang_cost"] = tab_lang_cost(lt, out)
    files["tab_lang_cost_full"] = tab_lang_cost(lt, out, full=True)
    files["tab_compute"] = tab_compute(w2, out)
    files["fig_w2_size"] = fig_w2_size(sz, out)
    files["fig_lang_cost"] = fig_lang_cost(lc, out)
    files["fig_count_bins"] = fig_count_bins(cb, out)
    # DEV-0077：在原有各项之后出，原有输出（除 tab_main 的 caption）不受影响
    files["tab_unified"] = tab_unified(u, out)
    files["fig_unified"] = fig_unified(u, out)
    files["fig_bins"] = fig_bins(bn, out)
    files["tab_seeds"] = tab_seeds(sd, out)
    w2caps = {"fig_w2_size": caption_w2_size(sz), "fig_lang_cost": caption_lang_cost(lc), "fig_count_bins": caption_count_bins(cb)}
    for name, cap in w2caps.items():
        p = out / f"{name}_caption.tex"
        p.write_text(f"% Generated by scripts/figures/make_paper_figures.py; do not edit.\n\\caption{{{_tex_escape(cap)}}}\n", encoding="utf-8")
        files[name].append(p)
    captions.update(w2caps)
    ucaps = {"fig_unified": caption_unified(u), "fig_bins": caption_bins(bn)}
    for name, cap in ucaps.items():
        p = out / f"{name}_caption.tex"
        p.write_text(f"% Generated by scripts/figures/make_paper_figures.py; do not edit.\n\\caption{{{caption_tex(cap)}}}\n", encoding="utf-8")
        files[name].append(p)
    captions.update(ucaps)
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
    for name, data in (("tab_unified", u), ("fig_unified", fig_unified_data(u)), ("fig_bins", bn), ("tab_seeds", sd)):
        p = out / "data" / f"{name}.json"
        p.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        files[name].append(p)
    readme = write_readme(root, out, e1, e2, e3, itf, captions, files, w2, sz, lc, cb, lt, u, bn, sd)
    print(f"wrote {sum(len(v) for v in files.values()) + 1} files to {out} (README: {readme})")
    miss = e1["missing"] + e2["missing"] + e3["missing"] + itf["missing"] + w2["missing"] + sz["missing"] + lc["missing"] + cb["missing"] + \
        lt["missing"] + u["missing"] + bn["missing"] + sd["missing"]
    if miss:
        print("missing:\n  " + "\n  ".join(miss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
