"""E5 训练后系统的评测（DEV-0037 §4）：一个 E5 检查点 × 一个任务 → <out>/trained/<起点>/<模型>/<n 标记>/seed<k>/<任务>[__dev]/。

**加载**（scripts/eval_e5_trained.py；src/e5/stack.load_stacked）：按 E5 检查点 manifest 的构造方式建 base → 合并起点 LoRA → 核对起点
manifest 的 sha256 与权重指纹 → load_trained(E5 检查点, merge=False)。

**评测前的合并核对**（stacked_merge_check；读数用 scripts/check_training_interfaces.py 的累计器 GenAgg、TfAgg；判据 e5-v2 见下）：
起点 LoRA 合并之后，E5 LoRA 合并前（A）对合并后（B），在原地合并（同一个适配器，不加载两份）：
- 样本：E5 val 划分每类 n_per_task 条（configs/e5_train.json 的 eval.merge_check；序号规则同评测集，src/e5/protocol.eval_items 的 dev），
  E5 训练后的提示与允许集合；按格式（二值 c、d、e / 起伏 a、b）分组，组内按 merge_check.batch_size 成批（同一批同一个允许集合），关思考。
- A（未合并）：受限解码 generate_restricted（评测实际走的路径）+ 以它为目标的教师强制；merge_lora；B（合并后）：
  training.restricted_consistency（generate_restricted 对 generate_batch(allowed) 对以 B 的受限解码为目标的教师强制）+ 以 A 的序列为目标的
  教师强制。每批 3 次整段生成。比较（checkpoint.compare_merge_sides，两侧的生成都是 generate_restricted、教师强制的目标都是 A 的序列）：
  * merge：受限解码逐位一致率（只报不判；CPU fp32 时要逐个相同）与教师强制在全词表上的 argmax 一致率、不一致处的 logit 差（v1 的 merge 一节）；
  * restricted.across_merge：同一批隐藏状态只在允许集合上比较（v1 的 merge 一节，同 E3b 带网格 token 的检查点）；
  * restricted.by_thinking：B 上 restricted_consistency 的读数（v1 的 restricted 一节）。
  与 E3b 的检查点核对（脚本）的差别：across_merge 的公共目标取合并前的序列（脚本取合并后受限解码的序列；原地合并之后合并前的模型已不在）。
- **判据 e5-v2**（DEV-0046；D96-1 = N34 的 (b′)；只用于这里，E1、E3 的核对仍是 v1 = D50-3，scripts/check_training_interfaces.py 不变）：
  纯函数 judge_e5v2(checks, exact=…, error=…) → 各节（merge、restricted；出错时另有 error）的失败原因；阈值只在 E5V2_THRESHOLDS 写一处。
  * 判定（GPU / bf16）：restricted.across_merge 的教师强制 argmax_agreement ≥ 0.98（只在允许集合上）；restricted.by_thinking.<模式> 的
    teacher_forcing.agreement ≥ 0.999；所有比较（E5V2_GAP_COMPARISONS：merge 的 generate 与 teacher_forcing、restricted 的
    decode_vs_generate_batch 与 teacher_forcing、across_merge 的 teacher_forcing）分歧处的 gap 最大值 ≤ 2.0、中位数 ≤ 0.5（没有分歧时自然
    通过；只看 gap，top2_gap 只报告）；有分歧而算不出 gap（n_divergence_without_gap > 0，或分歧数多于 gap.n）、核对出错、缺节，都判不过。
  * 只报告：merge 的全词表 teacher_forcing.argmax_agreement；decode_vs_generate_batch 的 exact_rows、token_agreement；merge 的 generate.*
    的其余读数（它的 gap 照上一条判）。
  * 精确模式（CPU fp32，exact = True，只在 tiny 测试里出现）：沿用 v1 的精确规则（处处逐个相同），不放宽。
  * v1 的判定照算、只报告：merge_check.json 的 v1_report_only = {criteria_version: v1, failures, pass, criteria}（judge + criteria_block +
    grid_criteria_block，与改动前同一套）。criteria_version = e5-v2，criteria 写判定项与阈值、只报告的项、精确模式的规则与依据；pass 只由
    e5-v2 决定，checks.<节>.pass 也是 e5-v2 的。已有文件可以用 judge_document_e5v2 重判。
- **指标层面的合并前后比较**（DEV-0046；D96-2；只报告，不影响 pass）：同一次加载、同一批样本与成批方式，A 侧（E5 LoRA 未合并，在 merge_lora
  之前）与 B 侧（合并后）各跑一次 src/e5/token_grid.decode_grids(decode=structured)——与正式评测 run_task 同一个调用；两侧都用 record()
  （E5Tokens.pred + scoring.score，relief_expected 同正式评测）解析评分，逐行配对。A 侧进出都核对 LoRA 状态是 injected，解码期间逐个 LoRA
  层数层与 lora_B 的前向次数（LoraAudit），被调用的层没走 LoRA 分支就报错（绕开了 LoRA 或偷偷合并）；B 侧核对 merged。写进 metric_level：逐行（id、任务、
  两侧的 iou_grid、iou_full、起伏误差，差，预测网格与起伏是否逐格相同、不同的格子数）；逐类（by_task）、逐格式（by_format）与总体（overall），
  各给全部行（all）与只算标题口径行（headline）两份：两侧 iou_grid 平均、差（B − A）、逐行差绝对值的最大值，起伏误差（只取两侧都有效的行）
  同一组读数，预测完全相同的行数，无效预测数；flags：总体 iou_grid 平均差的绝对值 > 0.01 → iou，总体起伏误差平均差的绝对值 > 2 × 0.01 → relief
  （METRIC_LEVEL_THRESHOLDS，D96-2）；A、B 两侧的耗时。出错写进 metric_level.error（核对的 pass 不受影响；评测脚本在 stderr 打一行警告）。
  精确模式下两侧读数逐个相同、差恰好为 0（tests 钉住）。
- 输出 <模型目录>/merge_check.json；不通过（或出错）抛 src/e3/merge_check.MergeCheckFailed，调用方拒绝评测（不写任何任务目录）。同一个
  E5 检查点、同一起点、同一组设置按 e5-v2 已通过的结果直接复用（只做 merge_lora）；旧版本（v1）的文件不论过没过都不复用，重新核对。写新文件
  之前，已有文件的 criteria_version 与当前不同（含读不出来的）就改名留存为 merge_check.<旧版本>.json（重名加 .1、.2…；新文件的 previous
  记下改名），证据不删；同一版本的重试照旧覆盖。
- 只做核对：scripts/eval_e5_trained.py --check-only（加载、核对、写 merge_check.json、打印摘要，不评任何任务、不建任务目录）。

**单轮**（唯一的流程；多轮在 E5 不做）：样本 = src/e5/protocol.eval_items(任务, 子集, 每档条数)（与零样本完全相同的清单）；提示 = 指令原文 +
E3b 网格指令的结尾句（src/e5/token_grid.PromptFormat），指令在前；src/e5/token_grid.decode_grids(允许集合 = E5Tokens.allowed_ids(任务)，
max_new_tokens = n(n + 1) + 1 = 1057，停止 token = 本家族的回合结束 token，关思考，贪心)；按样本顺序连续成批（批大小进运行身份）。
**解码方式**（DEV-0043；EvalSpec.decode，scripts/eval_e5_trained.py 默认取 configs/e5_train.json 的 decode 一节，v3 是 structured，D83-3；
--decode restricted 作诊断）：
- structured（结构强制，规则见 src/e5/token_grid 的模块说明）：叶目录 <任务>[__dev]__structured（同 E3b 的约定，D84）；运行身份的 decoding
  另记 mode、logits_processor、structure（GridStructure.as_dict，含规则）与依据，metrics.json 另记 decode，README 写明；
- restricted（generate_restricted）：叶目录、运行身份、metrics.json、README 与改动前逐字节相同（tests 用 36bcbbf 的快照钉住）。
续跑时解码方式不同就拒绝混写（运行身份不同）。上面的合并核对与解码方式无关（它核对的是 E5 LoRA 合并前后；结构强制只是在合并后同一份
logits 上逐位置收窄，合并前后的 logits 由教师强制在全词表与允许集合上比较），两种解码共用同一份 merge_check.json；指标层面的比较（只报告）
固定用结构强制（正式口径），不随 --decode 变。
解析与评分：token 序列 → E5 的格子表示 → src/e5/scoring.score（与零样本同一套评分）；汇总 scoring.aggregate、门槛 scoring.gate（三条基线的
逐样本配对 bootstrap CI，D59-4）；标题口径与判定的统计范围就是 src/e5/protocol.in_headline（与零样本同一个函数；D72-1 / D73：b 统计
环绕的 [20,180] 三档），这里不另写。

**输出**：predictions.jsonl（每条：样本字段、提示、停止、记账字段、预测的网格 / 起伏文本、解析失败原因（失败时另存 token id）、逐条得分；不含
墙钟）、metrics.json（与零样本同一结构：headline / overall / by_bin / by_group / parse / gate / baselines_headline）、compute.json、
config.yaml（运行身份）、meta.json、README.md、logs/timing.jsonl、logs/run_<k>.json。
**续跑**（--resume）：config.yaml 必须相同；已完成的 id 必须是样本顺序的前缀，退回到整批的边界接着跑；metrics 只由 predictions 与生成器算出，
续跑后逐字节相同。

**只评起点**（DEV-0040，D81-1；E5 预注册 §8 第 2 项：N = 0 的读数，区分截距与斜率，不单独参与 H5 判定；scripts/eval_e5_trained.py
--start-only）：起点检查点不经 E5 训练，直接在同一个评测集上评。
- 起点只收 START_ONLY_KINDS（网格 token 行训练过的起点：v3 的 base_warmup、e3b_c3、c3_warmup，DEV-0043；另保留 e3b，不属于 v3 主比较）：
  e3a 的网格 token 行是新加的、没训练过，base 没有起点检查点。
- 加载：src/e5/stack.load_start_only（make_base → setup_start：合并起点 LoRA、恢复起点的网格 token 行），与上面的加载只差没有 E5 LoRA。
- 评测前的核对：起点检查点自己的合并核对（src/e5/stack.start_merge_check，即 scripts/train_e5.py 开训前做的那一项；结果放在
  <out>/train/<起点>/<模型>/start_checks/，与训练同一个位置，已通过的复用）与权重指纹（合并起点 LoRA 之后 = factory.load_trained(起点,
  merge=True) 的，src/e5/stack.reference_fingerprint，同训练开头）。上面的"E5 LoRA 合并前后"一项不适用（没有 E5 LoRA），跳过，理由写进
  运行身份（START_ONLY_SKIPPED）、README 与运行记录；不写 merge_check.json。
- 评测本身（样本清单、提示、允许集合、max_new_tokens、停止 token、批组成、解析、评分、汇总与门槛）就是 run_task，一行不另写。
- 输出：<out>/trained/<起点>/<模型>/n0/seed<k>/<任务>[__dev]/（START_ONLY_TAG：数据量档的标记，与 n<数据量> 同一层），文件同上；运行身份另记
  start_only: true、n_train: 0，checkpoint 一节记起点（manifest 的 sha256 等）、权重指纹、配种子与核对（start_only_checkpoint）；
  metrics.json 另记 start_only、n_train。不用 --start-only 时输出与改动前逐字节相同（测试钉住）。

**不合并的评测**（DEV-0048，D99-2 的敏感性分析；E5 预注册 §8c 补充；scripts/eval_e5_trained.py --unmerged，只用于 --checkpoint）：
同一个 E5 检查点，E5 LoRA 挂着、不合并，在 test 上用正式评测的同一条路径再评一次。
- 加载同上（base → 合并起点 LoRA → 核对起点 → load_trained(E5 检查点, merge=False)），之后**不合并**，E5 LoRA 保持 injected。
- 评测前的合并核对跳过，理由（UNMERGED_SKIPPED）写进运行身份、README 与运行记录：评测的就是未合并的模型，合并核对见同一检查点的
  merge_check.json（脚本不改这个文件，只读出它的 sha256、判据版本与判定记进运行记录，供溯源）。
- 评测就是 run_task + decode_grids(decode=structured)，与正式评测一行不另写：样本、提示、批大小、解析、评分、汇总与门槛都不变。只收结构强制
  （EvalSpec 校验；D99-2 写明结构强制解码）。run_task 的 LoRA 状态检查只在 spec.unmerged 时改为必须是 injected，其余照旧（merged / none）。
- LoRA 审计（UnmergedAudit）：逐批用 LoraAudit 包住解码（规则同 DEV-0046：被调用的 LoRA 层与它的 lora_B 前向次数相等，至少一层被调用，
  模型上找得到 LoRA 层），解码前后都核对 LoRA 状态是 injected。审计不过就报错：这一批的预测不写，metrics 不写；审计结果（含出错）写进
  logs/run_<k>.json 的 lora_audit。
- 输出：同一个模型目录下另起叶目录 <任务>[__dev]__structured__unmerged（UNMERGED_SUFFIX）；运行身份另记 lora_merge = unmerged、
  lora_merge_basis（D99-2）、lora_merge_rule 与 skipped_checks；metrics.json 另记 merge = unmerged；README 写明。正式评测的叶目录与
  merge_check.json 都不碰。续跑同正式评测（运行身份必须相同）。
- 不用 --unmerged 时，predictions、metrics、config.yaml、README、merge_check.json 与打印行都与改动前（2db1055）逐字节相同（tests 钉住）。
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.e3.merge_check import MergeCheckFailed, load_cti
from src.e5 import scoring as SC
from src.e5.protocol import TASK_LETTER, eval_items, headline_scope_text, num_bins
from src.e5.runner import check_platform, make_sample, read_jsonl_tolerant
from src.e5.token_grid import (
    RESTRICTED,
    STRUCTURED,
    E5Tokens,
    PromptFormat,
    check_decode,
    decode_grids,
    decode_record,
    decoding_identity,
    max_new_tokens,
    sample_images,
    task_format,
)
from src.e5.train_config import E5TrainConfig
from src.runlog import write_run_metadata
from src.synth.seeding import canonical_task

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SUB = "trained"
PREDICTIONS = "predictions.jsonl"
METRICS = "metrics.json"
COMPUTE = "compute.json"
CONFIG = "config.yaml"
README = "README.md"
TIMING = "logs/timing.jsonl"
MERGE_CHECK = "merge_check.json"
MERGE_CHECK_FORMAT = "visint-e5-stacked-merge-check/1"
SYSTEM = "trained"
# 只评起点的起点（模块说明）：D81-1 的 base_warmup、e3b，加 v3 的 e3b_c3、c3_warmup（D83-5，DEV-0043）。= 配置里带检查点、网格 token 行
# 来自检查点（训练过）的起点（tests 钉住）；e3b 不属于 v3 主比较，保留作消融的 N = 0 读数（D83-2：原有各臂照实报告）
START_ONLY_KINDS: tuple[str, ...] = ("base_warmup", "e3b_c3", "c3_warmup", "e3b")
STRUCTURED_SUFFIX = "__structured"  # DEV-0043：结构强制的叶目录后缀（同 src/e3/evaluate.STRUCTURED_SUFFIX 的约定，D84）
START_ONLY_N_TRAIN = 0
START_ONLY_TAG = f"n{START_ONLY_N_TRAIN}"  # 数据量档的标记（同 src/e5/train_data.TrainSet.tag 的 n<数据量>）
START_ONLY_RULE = ("DEV-0040（D81-1；E5 预注册 §8 第 2 项）：只评起点，N = 0——起点检查点不经 E5 训练，加载 = src/e5/stack.load_start_only"
                   "（与训练后评测只差没有 E5 LoRA），评测与训练后评测同一套函数；用于区分截距与斜率，不单独参与 H5 判定")
START_ONLY_SKIPPED = {"e5_lora_merge": "训练后评测的“E5 LoRA 合并前后”核对（stacked_merge_check）不适用：只评起点没有 E5 LoRA，"
                                       "起点 LoRA 已合并，评测的模型就是起点合并评测时的权重（由起点检查点的合并核对与权重指纹核对）"}
# 合并核对的判据 e5-v2（DEV-0046，D96-1 = N34 的 (b′)；规则见模块说明与 judge_e5v2）。阈值只在这里写一处。数值与 v1（D50-3）的同名阈值
# 相同，但这是单独冻结的一版（gap 最大值的边界是 ≤，v1 的 restricted 是 <），不 import v1 的常量：v1 归 E1、E3，两版各自冻结
CRITERIA_VERSION = "e5-v2"
CRITERIA_BASIS = "D96-1（N34 的 (b′)）"
E5V2_THRESHOLDS: Mapping[str, float] = MappingProxyType({
    "across_merge_min_tf_argmax_agreement": 0.98,  # restricted.across_merge 的教师强制 argmax 一致率（只在允许集合上）
    "restricted_min_tf_agreement": 0.999,  # restricted.by_thinking.<模式>.teacher_forcing.agreement
    "max_gap": 2.0,  # 所有比较：分歧处 gap 的最大值 ≤ 它
    "max_median_gap": 0.5,  # 所有比较：分歧处 gap 的中位数 ≤ 它
})
# 判 gap 的比较（D96-1"所有比较"）：(节, by_thinking 所在的路径, 比较名, 是不是生成序列的逐行比较)
E5V2_GAP_COMPARISONS: tuple[tuple[str, tuple[str, ...], str, bool], ...] = (
    ("merge", ("merge",), "generate", True),
    ("merge", ("merge",), "teacher_forcing", False),
    ("restricted", ("restricted",), "decode_vs_generate_batch", True),
    ("restricted", ("restricted",), "teacher_forcing", False),
    ("restricted", ("restricted", "across_merge"), "teacher_forcing", False),
)
# 指标层面的合并前后比较（DEV-0046，D96-2；只报告，不影响 pass）：运行级（全部 40 行）读数差的绝对值超过它就记 flags
_ML_IOU_FLAG = 0.01  # D96-2：运行级 iou_grid 平均差的绝对值 > 它 → flags.iou
METRIC_LEVEL_THRESHOLDS: Mapping[str, float] = MappingProxyType({
    "iou_grid_mean_diff_abs": _ML_IOU_FLAG,
    # 起伏误差的噪声底（管理员定，用户可改）：mean_headline_v1 把 a 类的标题分记作 1 − 起伏误差/2，所以与 IoU 同一尺度的噪声底是它的 2 倍
    # （D96-2；tests 钉住数值）
    "relief_error_mean_diff_abs": 2 * _ML_IOU_FLAG,
})
METRIC_LEVEL_BASIS = ("D96-2：只报告、不判定；同一次加载、同一批 val 样本与成批方式，E5 LoRA 未合并（A）与合并后（B）各跑一次结构强制解码"
                      "（src/e5/token_grid.decode_grids，与正式评测同一个函数），E5 的正式解析与评分（record：E5Tokens.pred + scoring.score）；"
                      "起伏误差的噪声底取 IoU 阈值的 2 倍，同一尺度（mean_headline_v1 把 a 类的标题分记作 1 − 起伏误差/2）")
# 不合并的评测（DEV-0048，D99-2；模块说明）
UNMERGED_SUFFIX = "__unmerged"  # 叶目录后缀，接在 STRUCTURED_SUFFIX 之后
LORA_MERGE_UNMERGED = "unmerged"  # 运行身份的 lora_merge 与 metrics.json 的 merge
UNMERGED_BASIS = ("D99-2（负责人，2026-10-17；E5 预注册 §8c 补充，写于读到任何 E5 训练后 test 分数之前）：敏感性分析——H5 判定用到的 12 个"
                  "运行（E3b+C3 与 C3+预热，1K、4K，各 3 个种子）用未合并的 E5 LoRA 在 test 上再评一次，结构强制解码，其余同正式评测；门报告"
                  "同时报告合并与未合并两种口径下的 H5 结论")
UNMERGED_RULE = ("DEV-0048：加载同正式评测（base → 合并起点 LoRA → 核对起点 → load_trained(E5 检查点, merge=False)），E5 LoRA 保持 injected、"
                 "不合并；评测走 run_task 与 decode_grids(decode=structured)，与正式评测同一条路径（样本、提示、批大小、解析、评分、汇总、门槛"
                 "都不变）；解码期间逐批做 LoRA 审计（LoraAudit，规则同 DEV-0046），审计不过就报错、这一批不写、不写 metrics")
UNMERGED_SKIPPED = {"e5_lora_merge": "D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json"}


def model_dir(out: str | Path, start: str, model_key: str, tag: str, seed: int, data_seed: int | None = None) -> Path:
    """<out>/trained/<起点>/<模型>/<n 标记>/seed<k>（数据种子不等于种子时 seed<k>_d<数据种子>，同训练的叶目录，src/e5/train.seed_dir_name）。"""
    from src.e5.train import seed_dir_name

    return Path(out) / EVAL_SUB / start / model_key / tag / seed_dir_name(seed, seed if data_seed is None else data_seed)


@dataclass(frozen=True)
class EvalSpec:
    start: str
    model_key: str
    tag: str
    seed: int
    task: str
    subset: str
    n_per_bin: int
    n_resamples: int
    batch_size: int
    tiny: bool
    dtype: str
    limit: int | None = None
    data_seed: int | None = None
    start_only: bool = False  # DEV-0040：只评起点（tag 必须是 START_ONLY_TAG）
    decode: str = RESTRICTED  # DEV-0043：解码方式（模块说明）
    unmerged: bool = False  # DEV-0048：E5 LoRA 不合并（D99-2 的敏感性分析；只收 E5 检查点、结构强制）

    def __post_init__(self) -> None:
        check_decode(self.decode)
        if self.start_only and (self.tag != START_ONLY_TAG or self.start not in START_ONLY_KINDS or self.data_seed is not None):
            raise ValueError(f"只评起点：tag 必须是 {START_ONLY_TAG}、起点是 {START_ONLY_KINDS} 之一、没有数据种子，收到 "
                             f"{self.tag!r}、{self.start!r}、{self.data_seed!r}")
        if not self.start_only and self.tag == START_ONLY_TAG:
            raise ValueError(f"{START_ONLY_TAG} 是只评起点的标记，E5 检查点的评测不能用")
        if self.unmerged and (self.start_only or self.decode != STRUCTURED):
            raise ValueError(f"不合并的评测（DEV-0048，D99-2）只用于 E5 检查点、结构强制解码，收到 start_only={self.start_only}、"
                             f"decode={self.decode!r}")

    def items(self) -> list[Any]:
        its = eval_items(self.task, self.subset, self.n_per_bin)
        return its if self.limit is None else its[: self.limit]

    def leaf(self, out: str | Path) -> Path:
        name = canonical_task(self.task) + ("" if self.subset == "eval" else f"__{self.subset}") + \
            (STRUCTURED_SUFFIX if self.decode == STRUCTURED else "") + (UNMERGED_SUFFIX if self.unmerged else "")
        return model_dir(out, self.start, self.model_key, self.tag, self.seed, self.data_seed) / name


# ---------------------------------------------------------------------------
# 评测前的合并核对（模块说明）
# ---------------------------------------------------------------------------


def merge_check_items(n_per_task: int, tasks: Sequence[str]) -> list[Any]:
    """val 划分每类前 n_per_task 条（eval_items 的 dev 清单，档轮转）。"""
    out = []
    for t in tasks:
        k = math.ceil(n_per_task / num_bins(t))
        out += eval_items(t, "dev", k)[:n_per_task]
    return out


def _reusable(doc: Mapping[str, Any], key: Mapping[str, Any], criteria_version: str = CRITERIA_VERSION) -> bool:
    """同一个 E5 检查点、同一组设置、按当前判据（e5-v2）已通过的结果才复用；旧版本（v1）的文件不论过没过都不复用。"""
    try:
        return bool(doc.get("format") == MERGE_CHECK_FORMAT and doc.get("pass") is True and not doc.get("error")
                    and doc["key"] == dict(key) and doc["criteria_version"] == criteria_version)
    except (KeyError, TypeError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# 判据 e5-v2（D96-1；纯函数，可以直接拿已有的 merge_check.json 判）
# ---------------------------------------------------------------------------


def _by_thinking(checks: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any] | None:
    cur: Any = checks
    for p in path:
        cur = cur.get(p) if isinstance(cur, Mapping) else None
    bt = cur.get("by_thinking") if isinstance(cur, Mapping) else None
    return bt if isinstance(bt, Mapping) and bt else None


def e5v2_criteria(exact: bool) -> dict[str, Any]:
    """写进 merge_check.json 的 criteria（判据 e5-v2，规则见模块说明与 judge_e5v2）。"""
    t = E5V2_THRESHOLDS
    return {
        "version": CRITERIA_VERSION,
        "status": f"冻结：{CRITERIA_BASIS}，2026-10-15，定于看到任何 E5 训练后的 test 分数之前",
        "basis": CRITERIA_BASIS,
        "applies_to": "只用于 E5 训练后评测前的合并核对（src/e5/trained_eval.stacked_merge_check），所有 E5 运行统一适用；E1、E3 的"
                      "核对仍是 v1（D50-3，scripts/check_training_interfaces.py，不变）",
        "exact": exact,
        "exact_reason": "CPU fp32：沿用 v1 的精确规则（处处逐个相同），不放宽" if exact else
                        "GPU / bf16：判允许集合上的合并前后一致与受限教师强制，及所有比较里分歧处的 gap",
        "per_thinking_mode": True,
        "thresholds": dict(t),
        "gap_definition": "同 v1（scripts/check_training_interfaces.py 的模块说明）：分歧处两侧所选 token 在参照 logits 上之差；只判 gap，"
                          "top2_gap 只报告",
        "gap_comparisons": [".".join(path) + f".by_thinking.<模式>.{name}" for _, path, name, _ in E5V2_GAP_COMPARISONS],
        "rules": [
            f"restricted.across_merge.by_thinking.<模式>.teacher_forcing.argmax_agreement ≥ "
            f"{t['across_merge_min_tf_argmax_agreement']}（合并前后，只在允许集合上比较）",
            f"restricted.by_thinking.<模式>.teacher_forcing.agreement ≥ {t['restricted_min_tf_agreement']}",
            f"gap_comparisons 的每个比较：分歧处 gap.max ≤ {t['max_gap']}，gap.median ≤ {t['max_median_gap']}（gap.n = 0 即没有分歧时自然通过）",
            "任何比较里有分歧而算不出 gap（n_divergence_without_gap > 0，或分歧数多于 gap.n）判不过",
            "核对出错（error 不空）判不过；merge、restricted、restricted.across_merge 缺 by_thinking 一节判不过",
        ],
        "report_only": [
            "merge.by_thinking.<模式>.teacher_forcing.argmax_agreement（全词表）",
            "restricted.by_thinking.<模式>.decode_vs_generate_batch.exact_rows、token_agreement",
            "merge.by_thinking.<模式>.generate 的 exact_rows、token_agreement 等其余读数（它的 gap 照上面的 gap 规则判）",
            "top2_gap、max_abs_logit、max_abs_hidden",
            "v1 的判定（v1_report_only：scripts/check_training_interfaces.judge + criteria_block + grid_criteria_block）",
            "指标层面的合并前后比较（metric_level，D96-2）",
        ],
        "exact_rules": ["CPU fp32（exact）：沿用 v1 的精确规则——merge 与 restricted 的生成 exact_rows = token_agreement = 1，教师强制没有"
                        "不一致的位置，across_merge 的教师强制没有不一致的位置（scripts/check_training_interfaces.judge，criteria_block(True) "
                        "+ grid_criteria_block(True)）"],
    }


def _judge_gaps(where: str, s: Any, is_generation: bool) -> list[str]:
    t = E5V2_THRESHOLDS
    if not isinstance(s, Mapping):
        return [f"{where}：缺这一项读数"]
    g = s.get("gap") or {}
    n_gap = int(g.get("n") or 0)
    n_div = int(s.get("n_divergent_rows" if is_generation else "n_disagreements") or 0)
    missing = max(n_div - n_gap, int(s.get("n_divergence_without_gap") or 0))
    bad = []
    if missing > 0:
        bad.append(f"{where}：{missing} 处分歧算不出 logit 差（gap）")
    if n_gap:
        if g.get("max") is None or g["max"] > t["max_gap"]:
            bad.append(f"{where}：分歧处 gap 最大 {g.get('max')} > {t['max_gap']}")
        if g.get("median") is None or g["median"] > t["max_median_gap"]:
            bad.append(f"{where}：分歧处 gap 中位数 {g.get('median')} > {t['max_median_gap']}")
    return bad


def _below(v: Any, threshold: float) -> bool:
    return v is None or float(v) < threshold


def judge_e5v2(checks: Mapping[str, Any], *, exact: bool, error: Any = None) -> dict[str, list[str]]:
    """判据 e5-v2（D96-1；纯函数）：merge_check.json 的 checks → 各节（merge、restricted；出错时另有 error）的失败原因，空表 = 这一节通过。
    exact（CPU fp32）时沿用 v1 的精确规则（scripts/check_training_interfaces.judge，不放宽）。阈值见 E5V2_THRESHOLDS。"""
    t = E5V2_THRESHOLDS
    out: dict[str, list[str]] = {"merge": [], "restricted": []}
    if error:
        out["error"] = [f"核对出错：{error.get('repr') if isinstance(error, Mapping) else error}"]
    checks = checks if isinstance(checks, Mapping) else {}
    paths = {("merge",): "merge", ("restricted",): "restricted", ("restricted", "across_merge"): "restricted"}
    present = {p: _by_thinking(checks, p) for p in paths}
    for p, bt in present.items():
        if bt is None:
            out[paths[p]].append(f"{'.'.join(p)}：缺 by_thinking 一节，没有可判的读数")
    if any(bt is None for bt in present.values()):
        return out
    if exact:
        cti = load_cti()
        crit = cti.criteria_block(True)
        crit["restricted_grid"] = cti.grid_criteria_block(True)
        for name, bad in cti.judge(dict(checks), crit).items():
            out.setdefault(name, []).extend(bad)
        return out
    for mode, r in present[("restricted",)].items():
        tf = (r or {}).get("teacher_forcing") or {}
        if not tf.get("n_tokens"):
            out["restricted"].append(f"restricted/{mode}：教师强制没有可比较的位置")
        elif _below(tf.get("agreement"), t["restricted_min_tf_agreement"]):
            out["restricted"].append(f"restricted/{mode}：教师强制的 agreement {tf.get('agreement')} < {t['restricted_min_tf_agreement']}")
    for mode, r in present[("restricted", "across_merge")].items():
        tf = (r or {}).get("teacher_forcing") or {}
        if not tf.get("n_tokens"):
            out["restricted"].append(f"restricted/across_merge/{mode}：教师强制没有可比较的位置")
        elif _below(tf.get("argmax_agreement"), t["across_merge_min_tf_argmax_agreement"]):
            out["restricted"].append(f"restricted/across_merge/{mode}：合并前后受限教师强制的 argmax_agreement {tf.get('argmax_agreement')} "
                                     f"< {t['across_merge_min_tf_argmax_agreement']}")
    for section, path, name, is_gen in E5V2_GAP_COMPARISONS:
        for mode, r in present[path].items():
            out[section] += _judge_gaps("/".join(path) + f"/{mode}/{name}", (r or {}).get(name), is_gen)
    return out


def judge_document_e5v2(doc: Mapping[str, Any]) -> dict[str, Any]:
    """已有的 merge_check.json（任何版本）按 e5-v2 重判：exact 取文件里 criteria.exact，error 取文件里的 error。不改 doc。"""
    exact = bool((doc.get("criteria") or {}).get("exact", False))
    failures = judge_e5v2(doc.get("checks") or {}, exact=exact, error=doc.get("error"))
    return {"criteria_version": CRITERIA_VERSION, "failures": failures, "pass": all(not bad for bad in failures.values())}


def v1_report(checks: Mapping[str, Any], exact: bool) -> dict[str, Any]:
    """v1 的判定照算、只报告（D96-1）：scripts/check_training_interfaces 的 judge + criteria_block + grid_criteria_block，与改动前相同。"""
    cti = load_cti()
    crit = cti.criteria_block(exact)
    crit["restricted_grid"] = cti.grid_criteria_block(exact)
    failures = cti.judge(dict(checks), crit)
    return {"criteria_version": cti.CRITERIA_VERSION, "report_only": True, "failures": failures,
            "pass": all(not bad for bad in failures.values()), "criteria": crit}


# ---------------------------------------------------------------------------
# 指标层面的合并前后比较（D96-2；只报告）
# ---------------------------------------------------------------------------


class LoraAudit:
    """A 侧（E5 LoRA 未合并）结构强制解码期间的运行期审计：逐个 LoRA 层数它自己（peft 包上的 LoRA 线性层）与它的 lora_B 的前向次数。
    peft 的 LoRA 层在已合并或禁用 adapter 时只算 base_layer、不调 lora_B，所以"层被调用了、lora_B 没有同样多次" = 解码绕开了 LoRA（偷偷
    合并或禁用）；一个 LoRA 层都没被调用 = 解码没经过挂 LoRA 的模型；模型上找不到 LoRA 层 = 没挂着。三种都报错，并给出第一个模块名。
    模型本来就不调用的层（例如共享 KV 的层的 k、v 投影）只报个数，不算违规。label 是报错信息的前缀（默认指标层面的 A 侧；不合并的评测
    UnmergedAudit 另给，DEV-0048）。"""

    def __init__(self, ad: Any, label: str = "指标层面 A 侧") -> None:
        self._mods = [(name, m) for name, m in ad.model.named_modules() if hasattr(m, "lora_B") and "default" in m.lora_B]
        self.layer_calls = {name: 0 for name, _ in self._mods}
        self.lora_calls = {name: 0 for name, _ in self._mods}
        self._handles: list[Any] = []
        self._label = label

    def __enter__(self) -> "LoraAudit":
        for name, m in self._mods:
            self._handles.append(m.register_forward_hook(self._hook(self.layer_calls, name)))
            self._handles.append(m.lora_B["default"].register_forward_hook(self._hook(self.lora_calls, name)))
        return self

    @staticmethod
    def _hook(table: dict[str, int], name: str) -> Callable[..., None]:
        def count(_m: Any, _i: Any, _o: Any) -> None:
            table[name] += 1

        return count

    def __exit__(self, *exc: Any) -> bool:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def result(self) -> dict[str, Any]:
        if not self._mods:
            raise RuntimeError(f"{self._label}：模型上找不到任何 LoRA 层（lora_B），E5 LoRA 没有挂着")
        called = [n for n, c in self.layer_calls.items() if c > 0]
        if not called:
            raise RuntimeError(f"{self._label}：结构强制解码期间 {len(self._mods)} 个 LoRA 层一个都没被调用，解码没经过挂 LoRA 的模型")
        bad = [n for n in called if self.lora_calls[n] != self.layer_calls[n]]
        if bad:
            n0 = bad[0]
            raise RuntimeError(f"{self._label}：{len(bad)} 个 LoRA 层被调用时没走 LoRA 分支（第一个：{n0}，层 {self.layer_calls[n0]} 次、"
                               f"lora_B {self.lora_calls[n0]} 次），解码绕开了 LoRA（已合并或禁用）")
        v = [self.layer_calls[n] for n in called]
        idle = [n for n, c in self.layer_calls.items() if c == 0]
        return {"n_modules": len(self._mods), "n_called": len(called), "n_not_called": len(idle), "not_called": idle[:20],
                "min_calls": min(v), "max_calls": max(v),
                "rule": "A 侧解码期间逐个 LoRA 层数层与 lora_B 的前向次数：被调用的层两者必须相等（peft 的 LoRA 层已合并或禁用时不调 lora_B），"
                        "且至少有一层被调用；没被调用的层只报个数"}


def _ml_decode(ad: Any, e5tok: E5Tokens, runs: Sequence[tuple[Any, Any, str]], n: int, *, side: str, expect_state: str,
               audit: bool) -> tuple[list[list[Any]], dict[str, Any]]:
    """一侧的结构强制解码：与正式评测（run_task）同一个 decode_grids(decode=STRUCTURED)，每批一次。进出都核对 LoRA 状态。"""
    if ad.lora_state != expect_state:
        raise RuntimeError(f"指标层面 {side} 侧：LoRA 状态应为 {expect_state}，实际 {ad.lora_state}")
    t0 = time.perf_counter()
    au = LoraAudit(ad) if audit else None
    res = []
    with au if au is not None else contextlib.nullcontext():
        for k, (inp, _, fmt) in enumerate(runs):
            res.append(list(decode_grids(ad, e5tok, inp, fmt, n, decode=STRUCTURED, where=f"[合并核对·指标层面 {side} 侧第 {k} 批]")))
    if ad.lora_state != expect_state:
        raise RuntimeError(f"指标层面 {side} 侧：解码之后 LoRA 状态变成了 {ad.lora_state}（应为 {expect_state}）")
    info = {"lora_state": expect_state, "seconds": time.perf_counter() - t0}
    if au is not None:
        info["lora_audit"] = au.result()
    return res, info


def _ml_side(rec: Mapping[str, Any]) -> dict[str, Any]:
    rel = rec["score"].get("relief")
    return {"valid": bool(rec["valid"]), "failure": rec["failure"], "iou_grid": rec["score"]["iou_grid"],
            "iou_full": rec["score"]["iou_full"], "relief_error": None if rel is None else rel["error"],
            "relief_reason": None if rel is None else rel["reason"]}


def _cells_differ(a: Sequence[str] | None, b: Sequence[str] | None) -> int | None:
    if a is None or b is None or len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return None
    return sum(cx != cy for x, y in zip(a, b) for cx, cy in zip(x, y))


def metric_level_row(rec_a: Mapping[str, Any], rec_b: Mapping[str, Any]) -> dict[str, Any]:
    """同一条样本 A、B 两侧的 record() → 逐行读数：两侧得分、差（B − A），预测网格（起伏格式另比起伏）是否逐格相同。"""
    if rec_a["id"] != rec_b["id"]:
        raise RuntimeError(f"指标层面：两侧的行对不上（{rec_a['id']} ≠ {rec_b['id']}）")
    a, b = _ml_side(rec_a), _ml_side(rec_b)
    relief = task_format(rec_a["task"]) == "relief"
    pa, pb = rec_a["pred"], rec_b["pred"]
    same_grid = pa["grid"] == pb["grid"]
    same_relief = (pa["relief"] == pb["relief"]) if relief else None
    ra, rb = a["relief_error"], b["relief_error"]
    return {"id": rec_a["id"], "task": rec_a["task"], "format": task_format(rec_a["task"]), "bin": rec_a["bin"],
            "headline": bool(rec_a["headline"]), "a": a, "b": b,
            "diff": {"iou_grid": b["iou_grid"] - a["iou_grid"], "iou_full": b["iou_full"] - a["iou_full"],
                     "relief_error": None if ra is None or rb is None else rb - ra},
            "same_grid": same_grid, "same_relief": same_relief,
            "identical": bool(same_grid and same_relief is not False and a["failure"] == b["failure"]),
            "n_grid_cells_differ": _cells_differ(pa["grid"], pb["grid"]),
            "n_relief_cells_differ": _cells_differ(pa["relief"], pb["relief"]) if relief else None}


def _mean_of(v: Sequence[float]) -> float | None:
    return sum(v) / len(v) if v else None


def _pair_block(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    am, bm = _mean_of([p[0] for p in pairs]), _mean_of([p[1] for p in pairs])
    return {"a_mean": am, "b_mean": bm, "diff": None if am is None else bm - am,
            "max_abs_row_diff": max((abs(y - x) for x, y in pairs), default=None)}


def metric_level_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """一组行的读数：iou_grid 两侧平均、差（B − A）、逐行差绝对值的最大值；起伏误差（只取两侧都有效的行）同一组读数；预测完全相同的
    行数；两侧各自的无效预测数。没有起伏格式的行时 relief_error 为 None。"""
    out = {"n": len(rows), "iou_grid": _pair_block([(r["a"]["iou_grid"], r["b"]["iou_grid"]) for r in rows]), "relief_error": None,
           "n_identical": sum(bool(r["identical"]) for r in rows),
           "n_invalid": {"a": sum(not r["a"]["valid"] for r in rows), "b": sum(not r["b"]["valid"] for r in rows)}}
    rel = [r for r in rows if r["format"] == "relief"]
    if rel:
        both = [(r["a"]["relief_error"], r["b"]["relief_error"]) for r in rel
                if r["a"]["relief_error"] is not None and r["b"]["relief_error"] is not None]
        out["relief_error"] = {"n_rows": len(rel), "n_both_valid": len(both), **_pair_block(both)}
    return out


def _both_scopes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"all": metric_level_block(rows), "headline": metric_level_block([r for r in rows if r["headline"]])}


def metric_level_summary(rows: Sequence[Mapping[str, Any]], tasks: Sequence[str]) -> dict[str, Any]:
    """逐类、逐格式与总体（各给全部行与只算标题口径行两份），及运行级的两个标记（阈值 METRIC_LEVEL_THRESHOLDS，D96-2）。"""
    th = METRIC_LEVEL_THRESHOLDS
    overall = _both_scopes(rows)
    d_iou = overall["all"]["iou_grid"]["diff"]
    d_rel = (overall["all"]["relief_error"] or {}).get("diff")
    return {"overall": overall,
            "by_task": {canonical_task(t): _both_scopes([r for r in rows if r["task"] == canonical_task(t)]) for t in tasks},
            "by_format": {f: _both_scopes([r for r in rows if r["format"] == f]) for f in ("binary", "relief")},
            "flags": {"iou": None if d_iou is None else abs(d_iou) > th["iou_grid_mean_diff_abs"],
                      "relief": None if d_rel is None else abs(d_rel) > th["relief_error_mean_diff_abs"],
                      "rule": f"总体（全部行）iou_grid 平均差的绝对值 > {th['iou_grid_mean_diff_abs']} → iou；总体起伏误差（两侧都有效的行）"
                              f"平均差的绝对值 > {th['relief_error_mean_diff_abs']} → relief（D96-2）"}}


def _metric_level_init(exact: bool) -> dict[str, Any]:
    return {"report_only": True, "affects_pass": False, "status": "not_run", "basis": METRIC_LEVEL_BASIS,
            "thresholds": dict(METRIC_LEVEL_THRESHOLDS), "exact": exact,
            "decode": {"mode": STRUCTURED, "function": "src/e5/token_grid.decode_grids(decode=structured)，与正式评测 run_task 同一个调用"},
            "samples": "同 data：合并核对的 val 样本与成批（每批同一种格式），A、B 两侧逐行配对",
            "scoring": "trained_eval.record（E5Tokens.pred + scoring.score，relief_expected = 起伏格式），与正式评测的 predictions 同一套",
            "sides": {"a": {"model": "E5 LoRA 未合并（起点 LoRA 已合并）"}, "b": {"model": "E5 LoRA 合并后"}},
            "flags": None, "overall": None, "by_task": None, "by_format": None, "rows": None, "timing_s": {}, "error": None}


def _metric_level_fill(ml: dict[str, Any], run_samples: Sequence[tuple[Sequence[Any], Sequence[str]]],
                       res_a: Sequence[Sequence[Any]], res_b: Sequence[Sequence[Any]], e5tok: E5Tokens, n: int,
                       tasks: Sequence[str], order: Sequence[str]) -> None:
    """逐行配对两侧的 record()，按 order（data.sample_ids 的样本顺序；成批是按格式分组的）排好，写汇总。"""
    rows = {}
    for (chunk, texts), ra, rb in zip(run_samples, res_a, res_b, strict=True):
        for s, txt, x, y in zip(chunk, texts, ra, rb, strict=True):
            r = metric_level_row(record(s, txt, x, e5tok, n), record(s, txt, y, e5tok, n))
            rows[r["id"]] = r
    if sorted(rows) != sorted(order):
        raise RuntimeError("指标层面：逐行读数与合并核对的样本清单不同")
    ordered = [rows[i] for i in order]
    ml.update(metric_level_summary(ordered, tasks))
    ml["rows"] = ordered
    ml["status"] = "done"


def metric_level_brief(ml: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """摘要与打印用：总体差、两个标记、预测相同的行数、出错。"""
    if not ml:
        return None
    ov = ((ml.get("overall") or {}).get("all")) or {}
    err = ml.get("error")
    return {"status": ml.get("status"), "n": ov.get("n"), "n_identical": ov.get("n_identical"),
            "iou_grid_diff": (ov.get("iou_grid") or {}).get("diff"),
            "relief_error_diff": (ov.get("relief_error") or {}).get("diff"),
            "flags": {k: v for k, v in (ml.get("flags") or {}).items() if k in ("iou", "relief")} or None,
            "error": (err.get("repr") if isinstance(err, Mapping) else err) if err else None}


def metric_level_line(brief: Mapping[str, Any] | None) -> str:
    """stderr 的一行：指标层面的合并前后比较（只报告）。"""
    if not brief:
        return "[合并核对·指标层面] 没有读数（文件里没有 metric_level）"
    if brief.get("error"):
        return f"警告：[合并核对·指标层面] 出错（只报告，不影响合并核对的判定）：{brief['error']}"

    def f(x: Any) -> str:
        return "None" if x is None else f"{x:+.4f}"

    fl = brief.get("flags") or {}
    return (f"[合并核对·指标层面，只报告] 总体 iou_grid 差（B − A）{f(brief.get('iou_grid_diff'))}，起伏误差差 "
            f"{f(brief.get('relief_error_diff'))}；预测相同 {brief.get('n_identical')}/{brief.get('n')} 行；flags.iou={fl.get('iou')} "
            f"flags.relief={fl.get('relief')}")


# ---------------------------------------------------------------------------
# 核对文件
# ---------------------------------------------------------------------------


def preserve_previous(out: Path, criteria_version: str) -> dict[str, Any] | None:
    """已有的 merge_check.json 的判据版本与当前不同（含读不出来的）：改名留存为 merge_check.<旧版本>.json，重名加 .1、.2…；证据不删。
    同一版本不动（重试照旧覆盖）。返回 {criteria_version, renamed_to} 或 None。"""
    if not out.exists():
        return None
    try:
        old = json.loads(out.read_text(encoding="utf-8"))
        old_v = old.get("criteria_version") if isinstance(old, Mapping) else None
    except (OSError, ValueError):
        old_v = None
    if old_v == criteria_version:
        return None
    import re

    label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(old_v)) if old_v else "unknown"
    stem = out.name[: -len(out.suffix)] if out.suffix else out.name
    dst, k = out.with_name(f"{stem}.{label}.json"), 0
    while dst.exists():
        k += 1
        dst = out.with_name(f"{stem}.{label}.{k}.json")
    out.rename(dst)
    return {"criteria_version": old_v, "renamed_to": dst.name}


def _summary(doc: Mapping[str, Any], path: Path, *, reused: bool) -> dict[str, Any]:
    import hashlib

    return {"file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pass": bool(doc.get("pass")),
            "criteria_version": doc.get("criteria_version"), "n": (doc.get("data") or {}).get("n"),
            "checks": sorted(doc.get("checks") or {}), "reused": reused, "failures": doc.get("failures"),
            "error": (doc.get("error") or {}).get("repr") if doc.get("error") else None,
            "v1_pass": (doc.get("v1_report_only") or {}).get("pass"), "metric_level": metric_level_brief(doc.get("metric_level")),
            "previous": doc.get("previous")}


def stacked_merge_check(ad: Any, e5tok: E5Tokens, prompt: PromptFormat, *, n: int, n_per_task: int, tasks: Sequence[str],
                        batch_size: int, tiny: bool, exact: bool, key: Mapping[str, Any], out_json: str | Path,
                        enable_thinking: bool = False, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """起点合并之后的 E5 LoRA 合并核对（判据 e5-v2，另报 v1 与指标层面的比较；模块说明）。ad 必须挂着未合并的 E5 LoRA；返回时已合并。
    通过返回摘要，否则抛 MergeCheckFailed（摘要在它的 summary 里）。"""
    import traceback

    import torch

    from src.adapters import checkpoint as CK
    from src.adapters import training

    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    cti = load_cti()
    out = Path(out_json)
    if out.exists():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        if _reusable(old, key):
            log(f"[合并核对] 复用已通过的 {out}（判据 {CRITERIA_VERSION}）")
            ad.merge_lora()
            return _summary(old, out, reused=True)
    if ad.lora_state != "injected":
        raise RuntimeError(f"合并核对要挂着未合并的 E5 LoRA（当前 {ad.lora_state}）")
    mode = cti.MODE_KEYS[bool(enable_thinking)]
    mnt = max_new_tokens(n)
    ml = _metric_level_init(exact)
    doc: dict[str, Any] = {"format": MERGE_CHECK_FORMAT, "written_utc": None, "key": dict(key),
                           "criteria_version": CRITERIA_VERSION, "criteria": e5v2_criteria(exact), "checks": {}, "failures": {},
                           "pass": None, "v1_report_only": None, "metric_level": ml, "previous": None, "data": None,
                           "generation": {"max_new_tokens": mnt, "enable_thinking": bool(enable_thinking), "batch_size": int(batch_size)},
                           "timing_s": {}, "error": None,
                           "rule": "src/e5/trained_eval.py 的模块说明：起点 LoRA 合并之后，E5 LoRA 合并前（A）对合并后（B），原地合并；"
                                   f"判据 {CRITERIA_VERSION}（{CRITERIA_BASIS}）= judge_e5v2，pass 只由它决定；v1（D50-3）的判定照算、只报告"
                                   "（v1_report_only）；指标层面的合并前后比较只报告（metric_level，D96-2）"}
    t0 = time.perf_counter()
    try:
        for t in tasks:
            check_platform(t, tiny)
        items = merge_check_items(n_per_task, tasks)
        samples = [make_sample(it) for it in items]
        runs = []  # (inputs, ids, 格式)
        run_samples = []  # (这批的样本, 提示文本)：指标层面的 record 用
        for fmt in ("binary", "relief"):
            group = [s for s in samples if task_format(s.task) == fmt]
            for lo in range(0, len(group), int(batch_size)):
                chunk = group[lo: lo + int(batch_size)]
                texts = [prompt.render(s, n) for s in chunk]
                runs.append(([ad.build_inputs(sample_images(s), txt, prompt.order) for s, txt in zip(chunk, texts)],
                             [f"{s.task}/{s.split}/{s.index}" for s in chunk], fmt))
                run_samples.append((chunk, texts))
        doc["data"] = {"subset": "dev", "split": "val", "n_per_task": int(n_per_task), "tasks": list(tasks), "n": len(samples),
                       "sample_ids": [f"{s.task}/{s.split}/{s.index}" for s in samples], "n_runs": len(runs)}
        th = bool(enable_thinking)
        sides_a = []
        for inp, _, fmt in runs:  # A：未合并，受限解码 + 教师强制
            res = ad.generate_restricted(list(inp), e5tok.allowed_ids_of(fmt), max_new_tokens=mnt, stop_ids=[e5tok.stop_id],
                                         enable_thinking=th)
            tgt = [list(r.token_ids) for r in res]
            with torch.no_grad():
                tf = ad.training_forward(list(inp), tgt, enable_thinking=th)
            sides_a.append(CK.MergeSide(generations=[list(res)], targets=[tgt], hidden=[tf.target_hidden.detach()],
                                        offsets=[tuple(tf.target_offsets)]))
            del tf
        doc["timing_s"]["unmerged_s"] = time.perf_counter() - t0
        res_a = None
        try:  # 指标层面 A 侧（只报告：出错只记进 metric_level.error，不影响核对）
            res_a, info = _ml_decode(ad, e5tok, runs, n, side="A", expect_state="injected", audit=True)
            ml["sides"]["a"].update(info)
            ml["timing_s"]["a_s"] = info["seconds"]
        except Exception as e:  # noqa: BLE001
            ml["status"], ml["error"] = "error", {"repr": repr(e), "side": "A", "traceback": traceback.format_exc()[-4000:]}
        t1 = time.perf_counter()
        ad.merge_lora()
        gen = cti.GenAgg(mnt, ("before", "after"))
        tf_full = cti.TfAgg(("before", "after"), "argmax_agreement")
        tf_allowed = cti.TfAgg(("unmerged", "merged"), "argmax_agreement")
        rgen = cti.GenAgg(mnt, ("generate_batch", "restricted"))
        rtf = cti.TfAgg(("decoded", "argmax"), "agreement")
        for (inp, ids, fmt), a in zip(runs, sides_a):  # B：合并后
            allowed = e5tok.allowed_ids_of(fmt)
            rec = cti.RecordingAdapter(ad)
            rep = training.restricted_consistency(rec, inp, allowed, max_new_tokens=mnt, stop_ids=[e5tok.stop_id],
                                                  enable_thinking=th)
            rgen.add(rep["decode_vs_generate_batch"], ids, th)
            tf = rep["teacher_forcing"]
            dis = [(b_, p, x, y, g, t2_) for (b_, p, x, y, g), t2_ in zip(tf["disagreements"], tf["disagreement_top2_gaps"])]
            rtf.add(tf["n_tokens"], dis, ids, th)
            if len(rec.restricted) != 1:
                raise RuntimeError(f"restricted_consistency 应调一次 generate_restricted，实际 {len(rec.restricted)} 次")
            on_a = CK.merge_side(ad, [(inp, th)], max_new_tokens=mnt, targets=a.targets, generate=False)
            b = CK.MergeSide(generations=[list(rec.restricted[0])], targets=a.targets, hidden=on_a.hidden, offsets=on_a.offsets)
            (rep,) = CK.compare_merge_sides(ad, a, b, allowed_ids=allowed, chunk_size=cti.TF_CHUNK)
            gen.add(rep["generate"], ids, th)
            tf = rep["teacher_forcing"]
            tf_allowed.add(tf["n_tokens"], tf["disagreements"], ids, th, tf["max_abs_logit"], tf["max_abs_hidden"])
            (repf,) = CK.compare_merge_sides(ad, a, b, allowed_ids=None, chunk_size=cti.TF_CHUNK)
            tf = repf["teacher_forcing"]
            tf_full.add(tf["n_tokens"], tf["disagreements"], ids, th, tf["max_abs_logit"], tf["max_abs_hidden"])
            del a, b, on_a, rec
        doc["timing_s"]["merged_s"] = time.perf_counter() - t1
        if res_a is not None:
            try:  # 指标层面 B 侧与比较（只报告）
                res_b, info = _ml_decode(ad, e5tok, runs, n, side="B", expect_state="merged", audit=False)
                ml["sides"]["b"].update(info)
                ml["timing_s"]["b_s"] = info["seconds"]
                _metric_level_fill(ml, run_samples, res_a, res_b, e5tok, n, tasks, doc["data"]["sample_ids"])
            except Exception as e:  # noqa: BLE001
                ml["status"], ml["error"] = "error", {"repr": repr(e), "side": "B", "traceback": traceback.format_exc()[-4000:]}
        n_rows = len(samples)
        doc["checks"] = {
            "merge": {"mode": "e5_stacked", "compared": "E5 LoRA 未合并（起点已合并）vs 合并后；生成 = 两侧各自的 generate_restricted，"
                                                        "教师强制的目标 = 合并前受限解码的序列，logits 在全词表上比较",
                      "by_thinking": {mode: {"n_rows": n_rows, "generate": gen.summary(), "teacher_forcing": tf_full.summary()}}},
            "restricted": {"on_model": "合并后（起点 LoRA 与 E5 LoRA 都已合并）",
                           "by_thinking": {mode: {"n_rows": n_rows, "decode_vs_generate_batch": rgen.summary(),
                                                  "teacher_forcing": rtf.summary()}},
                           "across_merge": {"compared": "同 merge 的两侧隐藏状态，只在允许集合上比较 logits",
                                            "by_thinking": {mode: {"n_rows": n_rows, "teacher_forcing": tf_allowed.summary()}}}},
        }
        doc["v1_report_only"] = v1_report(doc["checks"], exact)
        doc["failures"] = judge_e5v2(doc["checks"], exact=exact)
        for name, bad in doc["failures"].items():
            doc["checks"][name]["pass"] = not bad
        doc["pass"] = all(not bad for bad in doc["failures"].values())
        doc["allowed"] = e5tok.layout()
    except Exception as e:  # noqa: BLE001 — 证据先落盘
        doc["error"] = {"repr": repr(e), "traceback": traceback.format_exc()[-6000:]}
        doc["failures"] = {"error": [f"核对出错：{e!r}"]}
        doc["pass"] = False
        if doc["v1_report_only"] is None:
            doc["v1_report_only"] = {"criteria_version": cti.CRITERIA_VERSION, "report_only": True, "failures": None, "pass": False,
                                     "note": "核对出错，没有可判的读数（见 error）"}
        if ml["status"] != "done" and ml["error"] is None:
            ml["status"], ml["error"] = "error", {"repr": "合并核对本身出错，指标层面的比较没有做完（见上一层的 error）"}
    doc["timing_s"]["total_s"] = time.perf_counter() - t0
    doc["written_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc["previous"] = preserve_previous(out, CRITERIA_VERSION)
    if doc["previous"] is not None:
        log(f"[合并核对] 旧的核对文件（判据 {doc['previous']['criteria_version']}）改名留存为 {out.parent / doc['previous']['renamed_to']}")
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summ = _summary(doc, out, reused=False)
    if not doc["pass"]:
        raise MergeCheckFailed(f"评测前的合并核对没有通过（判据 {CRITERIA_VERSION}）：{summ['failures'] or summ['error']}；拒绝评测（{out}）",
                               summ)
    return summ


# ---------------------------------------------------------------------------
# 单轮评测
# ---------------------------------------------------------------------------


def ids_digest(ids: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def run_identity(spec: EvalSpec, items: Sequence[Any], e5tok: E5Tokens, prompt: PromptFormat, cfg: E5TrainConfig,
                 checkpoint: Mapping[str, Any], n: int) -> dict[str, Any]:
    t = canonical_task(spec.task)
    ids = [it.id for it in items]
    from src.synth.constants import SYNTH_GENERATOR_VERSION, TASK_GENERATOR_VERSIONS

    head: dict[str, Any] = {"experiment": "E5", "framework": "src/e5（DEV-0037：训练后系统的评测）", "system": SYSTEM}
    if spec.start_only:  # DEV-0040（模块说明）；不是只评起点时运行身份与改动前逐字节相同
        head.update(framework="src/e5（DEV-0040：只评起点，N = 0）", start_only=True, n_train=START_ONLY_N_TRAIN,
                    start_only_rule=START_ONLY_RULE)
    if spec.unmerged:  # DEV-0048（模块说明）；不用 --unmerged 时运行身份与改动前逐字节相同
        head.update(lora_merge=LORA_MERGE_UNMERGED, lora_merge_basis=UNMERGED_BASIS, lora_merge_rule=UNMERGED_RULE,
                    skipped_checks=dict(UNMERGED_SKIPPED))
    return {
        **head,
        "task": t,
        "task_letter": TASK_LETTER[t],
        "subset": spec.subset,
        "data": {"generator": "src.synth.generate", "split": "test" if spec.subset == "eval" else "val",
                 "n_per_bin": spec.n_per_bin, "n_samples": len(ids), "sample_ids_sha256": ids_digest(ids), "limit": spec.limit,
                 "index_rule": "序号 0..B·n−1，档 = 序号 mod B（src/e5/protocol.py；与零样本相同）",
                 "task_generator_version": TASK_GENERATOR_VERSIONS[t], "synth_generator_version": SYNTH_GENERATOR_VERSION},
        "n_resamples": spec.n_resamples,
        "tiny": spec.tiny,
        "dtype": spec.dtype,
        "headline_scope": headline_scope_text(t),
        "gate": {"rule": "D59-4：配对 bootstrap 95% CI，判定条件同时胜过 " + "、".join(SC.JUDGMENT_BASELINES)},
        "checkpoint": dict(checkpoint),
        "decoding": decoding_identity(e5tok, t, n, spec.decode),  # restricted 与改动前逐字节相同（DEV-0043）
        "batching": {"batch_size": spec.batch_size, "order": "样本顺序，连续成批，左填充（适配器）"},
        "prompt": prompt.as_dict(),
        "config": cfg.identity(),
    }


def start_only_checkpoint(start: Any, *, seed: int, seed_pairing: Mapping[str, Any], fingerprint: str,
                          device_key: Mapping[str, Any], start_check: Mapping[str, Any]) -> dict[str, Any]:
    """只评起点的运行身份里 checkpoint 一节（模块说明）：起点（src/e5/stack.StartRef.identity：种类、manifest 的 sha256、LoRA、构造方式、
    叶目录的变体与种子；不含路径）、合并起点 LoRA 之后的权重指纹（与 factory.load_trained(起点, merge=True) 的相同才评测）、配种子、
    做了与跳过的核对。只放确定的字段（续跑时整份比较）；核对的文件与是否复用进运行记录 logs/run_<k>.json。"""
    return {"start_only": True, "n_train": START_ONLY_N_TRAIN, "start": dict(start.identity()),
            "fingerprint": {"value": fingerprint, "device_key": dict(device_key),
                            "rule": "src/e5/stack.stack_fingerprint：合并起点 LoRA 之后、加网格 token 之前；必须等于 "
                                    "factory.load_trained(起点, merge=True) 的（src/e5/stack.reference_fingerprint）"},
            "seed": int(seed), "seed_pairing": dict(seed_pairing),
            "checks": {"done": {"start_merge_check": {k: start_check.get(k) for k in ("purpose", "checks", "n", "criteria_version")},
                                "start_fingerprint": "合并起点 LoRA 之后的权重指纹 = factory.load_trained(起点, merge=True) 的"},
                       "skipped": dict(START_ONLY_SKIPPED)}}


def _prepare(leaf: Path, ident: Mapping[str, Any], resume: bool, cfg: E5TrainConfig) -> None:
    def norm(x: Mapping[str, Any]) -> dict[str, Any]:
        return yaml.safe_load(yaml.safe_dump(dict(x), sort_keys=False, allow_unicode=True))

    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old, new = norm(yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))), norm(ident)
        if old != new:
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"{leaf}：续跑的配置与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        files = [REPO_ROOT / cfg.path if not Path(cfg.path).is_absolute() else Path(cfg.path)]
        write_run_metadata(leaf, config=dict(ident), seed=BOOTSTRAP_SEED, data_files=files, repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)


def _append(path: Path, lines: Sequence[str]) -> None:
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def _truncate_lines(path: Path, k: int) -> None:
    raw = path.read_bytes()
    pos = 0
    for _ in range(k):
        pos = raw.index(b"\n", pos) + 1
    with open(path, "r+b") as f:
        f.truncate(pos)


def _jsonl(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False) + "\n"


def record(sample: Any, prompt_text: str, res: Any, e5tok: E5Tokens, n: int) -> dict[str, Any]:
    """一条生成 → predictions.jsonl 的一行（与零样本的记录同一套字段：样本字段、记账、解析、得分）。"""
    h, w = sample.target_mask.shape
    pred = e5tok.pred(sample.task, res.token_ids, n, h, w)
    rec = SC.sample_fields(sample)
    rec.update({"prompt": prompt_text, "stopped": res.stopped, "truncated": res.stopped is False, "thinking_unclosed": False,
                "n_generated_tokens": int(res.n_generated_tokens), "n_prompt_tokens": int(res.n_prompt_tokens),
                "n_visual_tokens": int(res.n_visual_tokens), "n_image_pixels": int(res.n_image_pixels or 0),
                "valid": pred.valid, "failure": pred.failure, "pred": SC.pred_record(pred, store_rle=False),
                "score": SC.score(sample, pred, relief_expected=task_format(sample.task) == "relief")})
    if pred.failure is not None:
        rec["token_ids"] = [int(t) for t in res.token_ids]
    return rec


def decode_line(spec: EvalSpec, ident: Mapping[str, Any]) -> str:
    """README 的解码一行：restricted 与改动前逐字节相同；structured 写明结构强制（DEV-0043）。"""
    d = ident["decoding"]
    if spec.decode == RESTRICTED:
        return (f"- 解码：受限（{d['format']}，{len(d['allowed_ids'])} 个允许 token），单轮，"
                f"max_new_tokens = {d['max_new_tokens']}，关思考，贪心，批大小 {spec.batch_size}")
    st = d["structure"]
    n_val = len(st["value_ids"]) if not st["value_ids_per_row"] else len(st["value_ids"][0])
    return (f"- 解码：结构强制（D83-3，v3 的正式口径；{d['format']}，每个格子位置只允许 {n_val} 个值 token，行末只允许行分隔，最后只允许"
            f"停止 token，长度恒为 {st['length']}；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 {spec.batch_size}")


def readme_text(spec: EvalSpec, ident: Mapping[str, Any], m: Mapping[str, Any]) -> str:
    from src.e5.runner import _block_lines, _ci_text, _fmt

    ck = ident["checkpoint"]
    if spec.start_only:  # DEV-0040（模块说明）
        sp, ch = ck["seed_pairing"], ck["checks"]
        mc = ch["done"]["start_merge_check"]
        title = f"# E5 · 只评起点（{spec.start}，N = 0，不经 E5 训练）"
        ck_lines = [f"- 起点：{ck['start']['kind']}（manifest {str(ck['start']['manifest_sha256'])[:16]}…；合并起点 LoRA 之后的权重指纹 "
                    f"{str(ck['fingerprint']['value'])[:16]}…）；配种子 {sp.get('status')}（起点的种子 {sp.get('start_seed')}，"
                    f"--allow-seed-mismatch {sp.get('allow_mismatch')}）",
                    f"- 评测前的核对：起点检查点的合并核对（{mc.get('purpose')}：{'、'.join(mc.get('checks') or [])}，判据 "
                    f"{mc.get('criteria_version')}）与权重指纹（= factory.load_trained(起点, merge=True) 的）；跳过："
                    + "；".join(f"{k}（{v}）" for k, v in ch["skipped"].items()),
                    "- 用途：N = 0 的读数，区分截距与斜率（E5 预注册 §8 第 2 项），不单独参与 H5 判定"]
    else:
        title = f"# E5 · 训练后（{spec.start} → E5）"
        ck_lines = [f"- 检查点：E5 manifest {str(ck.get('manifest_sha256'))[:16]}…；起点 {ck.get('start', {}).get('kind')}（manifest "
                    f"{str(ck.get('start', {}).get('manifest_sha256'))[:16]}…）；训练运行身份 {str(ck.get('train_identity_sha256'))[:16]}…"]
    if spec.unmerged:  # DEV-0048（模块说明）；不用 --unmerged 时 README 与改动前逐字节相同
        title = f"# E5 · 训练后（{spec.start} → E5，E5 LoRA 不合并）"
        merged_leaf = spec.leaf(".").name[: -len(UNMERGED_SUFFIX)]
        ck_lines += [f"- E5 LoRA：**不合并**（lora_merge = {ident['lora_merge']}，LoRA 状态 injected）。D99-2 的敏感性分析：同一个检查点、同一条"
                     f"评测路径（run_task + decode_grids(decode=structured)），样本、提示、批大小、解析、评分、汇总与门槛都与正式评测相同；"
                     f"正式评测（合并后）的结果在同一模型目录的 {merged_leaf}/",
                     "- 评测前的核对：起点 manifest 的 sha256 与权重指纹照常；跳过："
                     + "；".join(f"{k}（{v}）" for k, v in ident["skipped_checks"].items()),
                     "- LoRA 审计：解码期间逐批核对每个被调用的 LoRA 层都走了 lora_B（LoraAudit，规则同 DEV-0046），结果在 logs/run_<k>.json 的 "
                     "lora_audit；审计不过就报错，不写 metrics"]
    lines = [f"{title}· {spec.model_key} · {spec.tag} · seed {spec.seed} · {m['task']}"
             f"（{TASK_LETTER[m['task']]}）· {spec.subset}", "",
             f"- 样本：{ident['data']['split']} 划分，每档 {spec.n_per_bin} 条，n = {m['n']}（id 指纹 "
             f"{ident['data']['sample_ids_sha256'][:16]}…，与零样本相同的清单）；生成器任务版本 {ident['data']['task_generator_version']}",
             *ck_lines,
             decode_line(spec, ident),
             f"- 标题口径：{m['headline_scope']}", "", "## 标题（严格 / 有效率 / 条件读数）", ""]
    lines += _block_lines(m["headline"])
    g = m.get("gate") or {}
    pre, jud = g.get("prerequisite"), g.get("judgment")
    lines += ["", "## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）", ""]
    if pre:
        lines.append(f"- 先决条件（{g['iou_key']}）：系统 {_fmt(pre['system'])} vs 最强非真值基线 {pre['strongest']} "
                     f"{_fmt(pre['value'])}；差 {_fmt(pre['diff'])}，CI {_ci_text(pre['diff_ci'])} → pass={pre['pass']}")
    if jud:
        lines.append(f"- 判定条件：{jud['status']}" + (f"，系统起伏误差 {_fmt(jud.get('system'))}，n = {jud.get('n')} → "
                                                     f"pass={jud.get('pass')}" if jud["status"] == "evaluated" else ""))
        for name, c in (jud.get("baselines") or {}).items():
            lines.append(f"  - vs {name}：基线 {_fmt(c['value'])}；差 {_fmt(c['diff'])}，CI {_ci_text(c['diff_ci'])} → 胜过={c['beats']}")
    lines.append(f"- imagining：{g.get('imagining')}")
    p = m["parse"]
    lines += ["", "## 解析", "", f"- 解析失败率：{_fmt(p['failure_rate'])}（{p['n_failed']}/{p['n']}）；分原因：{p['failures']}",
              f"- 截断率：{_fmt((m.get('truncation') or {}).get('rate'))}", "",
              "## 文件", "", "- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/", ""]
    return "\n".join(lines)


def finalize(leaf: Path, spec: EvalSpec, items: Sequence[Any], ident: Mapping[str, Any]) -> dict[str, Any]:
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    ids = [it.id for it in items]
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    samples = [make_sample(it) for it in items[: len(recs)]]
    body = SC.aggregate(spec.task, recs, n_resamples=spec.n_resamples, enable_thinking=False)
    base_recs, roles = SC.baseline_records(samples)
    body["gate"] = SC.gate(spec.task, recs, base_recs, roles, n_resamples=spec.n_resamples)
    body["baselines_headline"] = {nm: SC.blocks([r for r in v if r["headline"]], 0) for nm, v in sorted(base_recs.items())}
    body["baseline_roles"] = dict(sorted(roles.items()))
    body["complete"] = len(recs) == len(ids)
    body["n_expected"] = len(ids)
    head: dict[str, Any] = {"system": SYSTEM, "start": spec.start, "model": spec.model_key, "tag": spec.tag}
    if spec.start_only:  # DEV-0040；不是只评起点时 metrics.json 与改动前逐字节相同
        head.update(start_only=True, n_train=START_ONLY_N_TRAIN)
    dec = decode_record(spec.decode)
    if dec is not None:  # DEV-0043：只在结构强制时加（restricted 的 metrics.json 与改动前逐字节相同）
        head["decode"] = dec
    if spec.unmerged:  # DEV-0048：只在不合并时加（正式评测的 metrics.json 与改动前逐字节相同）
        head["merge"] = LORA_MERGE_UNMERGED
    metrics = {**head, "seed": spec.seed, "subset": spec.subset, **body}
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing = {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}
    comp = []
    for r in recs:
        wall = float(timing.get(r["id"], {}).get("wall_time_s", 0.0))
        c = Compute.call(generated_tokens=r["n_generated_tokens"], prompt_tokens=r["n_prompt_tokens"],
                         visual_tokens=r["n_visual_tokens"], image_pixels=r["n_image_pixels"], call_time_s=wall, new_round=True)
        comp.append(c.as_sample(wall_time_s=wall))
    write_compute_json(leaf / COMPUTE, merge(comp))
    (leaf / README).write_text(readme_text(spec, ident, metrics), encoding="utf-8")
    return metrics


class UnmergedAudit:
    """不合并的评测（DEV-0048）的 LoRA 审计：每批一个 LoraAudit 包住 decode_grids（规则同 DEV-0046：模型上找得到 LoRA 层；至少一层被调用；
    被调用的层与它的 lora_B 前向次数相等），解码前后都核对 LoRA 状态是 injected。不过就抛 RuntimeError（调用方不写这一批、不写 metrics），
    原因记在 error 里。summary() 进 logs/run_<k>.json 的 lora_audit：逐批读数汇总（层数、被调用过的层数、从没被调用的层、每批每层的最少
    与最多调用次数）。"""

    LABEL = "不合并的评测（--unmerged，D99-2）"
    RULE = ("DEV-0048（D99-2）：每批一个 LoraAudit（规则同 DEV-0046：被调用的 LoRA 层与它的 lora_B 前向次数必须相等，至少一层被调用，模型上"
            "找得到 LoRA 层），解码前后 LoRA 状态都必须是 injected；审计不过就报错，这一批的预测不写，不写 metrics")

    def __init__(self) -> None:
        self.n_batches = 0
        self.n_modules: int | None = None
        self.called: set[str] = set()
        self.all_names: list[str] = []
        self.min_calls: int | None = None
        self.max_calls: int | None = None
        self.error: str | None = None

    def _fail(self, msg: str) -> None:
        self.error = msg
        raise RuntimeError(msg)

    def decode(self, ad: Any, e5tok: E5Tokens, inputs: Sequence[Any], fmt: str, n: int, *, decode: str, where: str) -> list[Any]:
        if ad.lora_state != "injected":
            self._fail(f"{self.LABEL}{where}：解码之前 LoRA 状态是 {ad.lora_state}（应为 injected）")
        au = LoraAudit(ad, label=f"{self.LABEL}{where}")
        with au:
            res = decode_grids(ad, e5tok, inputs, fmt, n, decode=decode, where=where)
        if ad.lora_state != "injected":
            self._fail(f"{self.LABEL}{where}：解码之后 LoRA 状态变成了 {ad.lora_state}（应为 injected）")
        try:
            r = au.result()
        except RuntimeError as e:
            self._fail(str(e))
        self.n_batches += 1
        self.n_modules = r["n_modules"]
        self.all_names = list(au.layer_calls)
        self.called |= {k for k, c in au.layer_calls.items() if c > 0}
        self.min_calls = r["min_calls"] if self.min_calls is None else min(self.min_calls, r["min_calls"])
        self.max_calls = r["max_calls"] if self.max_calls is None else max(self.max_calls, r["max_calls"])
        return res

    def summary(self) -> dict[str, Any]:
        never = [k for k in self.all_names if k not in self.called]
        status = "fail" if self.error else ("pass" if self.n_batches else "no_batches")
        return {"status": status, "error": self.error, "lora_state": "injected", "n_batches": self.n_batches,
                "n_modules": self.n_modules, "n_called": len(self.called), "n_not_called": len(never) if self.all_names else None,
                "not_called": never[:20], "min_calls": self.min_calls, "max_calls": self.max_calls, "rule": self.RULE}


def run_task(ad: Any, e5tok: E5Tokens, prompt: PromptFormat, cfg: E5TrainConfig, spec: EvalSpec, out: str | Path, *,
             checkpoint: Mapping[str, Any], n: int, resume: bool = False, argv: Sequence[str] = (),
             run_extra: Mapping[str, Any] | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """一个任务的单轮评测（模块说明）。ad 必须是合并后的模型；只有 spec.unmerged（DEV-0048）时必须挂着未合并的 E5 LoRA（injected），
    解码逐批过 LoRA 审计（UnmergedAudit），结果写进运行记录的 lora_audit。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if spec.unmerged:  # DEV-0048
        if ad.lora_state != "injected":
            raise RuntimeError(f"不合并的评测（D99-2）要挂着未合并的 E5 LoRA（当前 LoRA 状态 {ad.lora_state}）")
    elif ad.lora_state not in ("merged", "none"):
        raise RuntimeError(f"评测要在合并之后（当前 LoRA 状态 {ad.lora_state}）")
    t = canonical_task(spec.task)
    check_platform(t, spec.tiny)
    items = spec.items()
    leaf = spec.leaf(out)
    ident = run_identity(spec, items, e5tok, prompt, cfg, checkpoint, n)
    _prepare(leaf, ident, resume, cfg)
    path = leaf / PREDICTIONS
    done = read_jsonl_tolerant(path, repair=True)
    ids = [it.id for it in items]
    if [r["id"] for r in done] != ids[: len(done)]:
        raise RuntimeError(f"{path} 里已完成的 id 不是样本顺序的前缀，不能续跑")
    bs = int(spec.batch_size)
    keep = len(done) if len(done) == len(ids) else (len(done) // bs) * bs
    if keep < len(done):
        _truncate_lines(path, keep)
        log(f"[{t}] 已完成 {len(done)} 条不在批的边界上，退回到 {keep} 条整批重跑")
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "n_done_before": keep, "n_remaining": len(ids) - keep, "batch_size": bs, "finished_utc": None, "run": run_k,
               **dict(run_extra or {})}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_all = time.perf_counter()
    fmt = task_format(t)
    n_batches = 0
    audit = UnmergedAudit() if spec.unmerged else None  # DEV-0048
    for lo in range(keep, len(items), bs):
        batch = [make_sample(it) for it in items[lo: lo + bs]]
        t0 = time.perf_counter()
        texts = [prompt.render(s, n) for s in batch]
        inputs = [ad.build_inputs(sample_images(s), txt, prompt.order) for s, txt in zip(batch, texts)]
        prep = time.perf_counter() - t0
        if audit is None:
            results = decode_grids(ad, e5tok, inputs, fmt, n, decode=spec.decode, where=f"[{t} 第 {lo // bs} 批]")
        else:  # DEV-0048：同一个 decode_grids，外面包 LoRA 审计；不过就报错，这一批不写（证据先写进运行记录）
            try:
                results = audit.decode(ad, e5tok, inputs, fmt, n, decode=spec.decode, where=f"[{t} 第 {lo // bs} 批]")
            except Exception as e:
                run_log.update(lora_audit=audit.summary(), error=repr(e), n_batches=n_batches)
                run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                raise
        lines, tlines = [], []
        for s, txt, res in zip(batch, texts, results):
            rec = record(s, txt, res, e5tok, n)
            lines.append(_jsonl(rec))
            tlines.append(_jsonl({"id": rec["id"], "run": run_k, "batch_index": n_batches, "batch_rows": len(batch),
                                  "wall_time_s": float(res.wall_time_s), "batch_wall_time_s": float(res.batch_wall_time_s or 0.0),
                                  "prep_s": prep / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(path, lines)
        n_batches += 1
        hi = lo + len(batch)
        if n_batches == 1 or n_batches % 20 == 0 or hi == len(ids):
            log(f"[E5 trained {spec.start} {spec.model_key} {spec.tag} seed{spec.seed} {t}/{spec.subset}"
                f"{' unmerged' if spec.unmerged else ''}] {hi}/{len(ids)}")
    if audit is not None:  # DEV-0048：审计结果先进运行记录，再算 metrics
        run_log["lora_audit"] = audit.summary()
        run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = finalize(leaf, spec, items, ident)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   wall_s=round(time.perf_counter() - t_all, 3), n_batches=n_batches)
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


__all__ = ["CRITERIA_BASIS", "CRITERIA_VERSION", "E5V2_GAP_COMPARISONS", "E5V2_THRESHOLDS", "EVAL_SUB", "EvalSpec", "LORA_MERGE_UNMERGED",
           "LoraAudit", "MERGE_CHECK", "MERGE_CHECK_FORMAT", "METRIC_LEVEL_BASIS", "METRIC_LEVEL_THRESHOLDS", "START_ONLY_KINDS",
           "START_ONLY_N_TRAIN", "START_ONLY_RULE", "START_ONLY_SKIPPED", "START_ONLY_TAG", "STRUCTURED_SUFFIX", "UNMERGED_BASIS",
           "UNMERGED_RULE", "UNMERGED_SKIPPED", "UNMERGED_SUFFIX", "UnmergedAudit", "decode_line", "e5v2_criteria", "finalize",
           "judge_document_e5v2", "judge_e5v2", "merge_check_items", "metric_level_block", "metric_level_brief", "metric_level_line",
           "metric_level_row", "metric_level_summary", "model_dir", "preserve_previous", "record", "run_task", "stacked_merge_check",
           "start_only_checkpoint", "v1_report"]
