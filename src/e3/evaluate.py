"""E3b 的评测（DEV-0030 §1；简报 §8.5、§4 第 11 条、§12）：一个检查点 × 一个任务 → <out>/eval/<变体>/<模型>/seed<k>/<任务>/。

**加载**（调用方：scripts/eval_e3b.py）：先跑评测前的合并核对（src/e3/merge_check：check_training_interfaces --checkpoint 的同一套
函数，判据 v1 = D50-3；检查项取配置的 eval.merge_check.checks.grid_eval：merge 加 restricted，restricted 用检查点自己的网格 token 与
本模块的单轮设置，DEV-0032；不通过就不评测），再 factory.load_trained(检查点, merge=True)，attach_vocab 取回网格 token 的布局。
**判据 e5-v2**（DEV-0063，D102-2；resolve_eval(merge_criteria = "e5-v2")，scripts/eval_e3b.py --merge-criteria e5-v2，只配
decode = structured）：合并核对的读数照旧，判法换成 E5 的 e5-v2（src/e3/merge_check 的模块说明）。设置与运行身份（config.yaml 的
eval.merge_criteria）、metrics.json 的 merge_criteria、README 各记一笔；不给（或给 v1）时没有这个键，输出与原来逐字节相同（tests 用
8207cec 的快照钉住）。叶目录名不变：换了判据的评测写进新的 --out（续跑时判据不同就是运行身份不同，拒绝混写；指令检查的正常一侧
判据不同不配对）。

**单轮**（主要数字）：图 + 网格指令（tasks.<任务>.grid_template / grid_instruction，指令在前）→ adapter.generate_restricted，
允许集合 = tokens.allowed_ids(任务, allowed_scope)（与训练损失同一个函数、同一个集合），max_new_tokens = 目标序列长度 n(n + 1) + 1，
停止 token = 本家族的回合结束 token → targets.parse_target（严格，不补齐、不猜）→ 网格上采样到原图后算指标（src/e3/eval_scoring）。

**叠加反馈多轮**（K = 总轮数，configs/e3b_train.json 的 eval.multi_turn；多轮流程在看测试结果之前固定，硬约束 11）：
- 第 1 轮就是单轮（同一次调用），所以 K = 1 与单轮逐个相同（不是另跑一遍）；
- 第 r ≥ 2 轮：上一轮之后的状态网格经 feedback_image（训练修正对的同一个渲染入口）画回原图，输入 [原图, 叠加图] + 修正模板
  （与训练的修正对同一模板）；
- 状态 = 最近一个解析成功的网格：本轮解析失败时沿用上一轮的状态；还没有任何有效网格的样本不再调用（记 skipped，计算量 0）；
- 固定轮数，不早停；一次跑到 max(k_values) 轮，每轮单独记录（predictions_round<r>.jsonl）与评分；K 轮流程的输出 = 第 K 轮之后的状态。
- 记账（每轮、每条）：prompt token、生成 token、视觉 token、像素、GPU 墙钟（generate 的整批墙钟按行等分，CUDA 前后同步）与
  准备输入的 CPU 墙钟（读图、渲染叠加图、处理器）；compute_multi_K<K>.json = 第 1..K 轮的调用之和（src/accounting）。
- **只跑一部分 K**（DEV-0039，D72-4 (c)：多轮与匹配计算量的评测每个尺寸只做 1 个种子，单轮的主读数保留全部种子）：resolve_eval 的
  k_values（scripts/eval_e3b.py --k-values）覆盖配置的 eval.multi_turn.k_values，必须是那一列从 1 开始的前缀（k_values_prefix），
  如 [1] 或 [1, 2]；匹配计算量的单轮随之只做这些 K，[1] 时只有第 1 轮、没有任何采样。第 1 轮（= 单轮）的调用与批组成与 K 无关，
  所以 predictions.jsonl 与单轮指标与跑全部 K 时逐字节相同（tests 钉住）。生效的 K 进运行身份（config.yaml 的 eval.k_values、
  n_rounds、n_samples），覆盖时另记 eval.k_values_override（配置的那一列与依据）；不给或给的就是配置那一列时没有这个键，设置与运行身份
  与原来逐字节相同。续跑时 K 不同就是运行身份不同，拒绝混写（同下面的续跑规则）。metrics.json 的结构不变，只是 by_K 里的 K 变少。

**匹配计算量的单轮**（eval.matched_compute；简报 §12"多轮的增益必须和匹配计算量的单轮对比"）。可行的方案：
1. 多次采样取多数（采用）：与 K 轮同样的 K 次单轮调用（同样的调用次数、约同样的生成 token 数）。第 1 次就是贪心的单轮，其余 K − 1 次
   在受限分布上采样（adapter.generate_batch(allowed_token_ids, do_sample, temperature, top_k = 0, top_p = 1, repetition_penalty = 1)：
   采样也只在网格 token 上算 logits）；每批之前按 (seed, 采样序号, 批序号) 的哈希设 torch 的种子，同机器同批配置下确定。合成：分割逐格
   多数票（平票取第一张有效网格的值，即贪心有效时取贪心）；深度逐格档号平均后 np.rint（对数深度的平均）；没有有效网格记失败。
   采样嵌套复用（K = 2 用第 1 次采样，K = 4 用前 3 次）。视觉 token：K 轮流程 (2K − 1)·V，这里 K·V，照实报告（偏向多轮一侧）。
2. 更大的图像预算：不可行。检查点的图像口径写死在 manifest 里（load_trained 核对，口径不同拒绝加载），换口径还是训练外的分布。
3. 光度增强的测试时集成（多张亮度 / 对比度变体，逐格投票）：可行、与任务无关，但多样性来源是我们挑的增强，不是模型自己的不确定性；
   作为备选，没有实现。
4. 开思考或加思考预算：不可行。网格的允许集合里没有思考段的 token（受限解码从第一个 token 起就只能写网格）。
5. 同一输入的贪心重复 K 次：确定性解码下逐个相同，等于单轮，没有意义。

**评测集**（eval.sets）：分割 = E1 的 RefCOCO（unc）val 评测总体（load_refcoco(subset = eval)，剔除 E1 开发子集与 D20 探针图），按被指代
物体是不是 person 分列；深度 = NYUv2 官方 654（src/e2/data.load_nyu_test），Eigen crop。--tiny：分割用 tests/fixtures/e3/eval_root，
深度用 tests/fixtures/e1/root。

**指标**（metrics.json；eval_scoring 的逐样本值 → bootstrap CI，src/stats，种子与次数取 src/constants）：
- 分割：每个流程 × 每种上采样一组 cIoU（累计交 / 累计并，ratio_of_sums 的 CI）、gIoU、Boundary IoU（逐样本平均）；按 person /
  non_person / unknown 分列；解析失败按 metrics.seg 的约定记 0。网格层面的诊断：格子 IoU、逐格相同的比例。
- 深度：每个流程 × 每种上采样：AbsRel、δ1（条件读数：只在解析成功的图上），δ1_strict（解析失败的图记 0），尺度-平移对齐的读数
  （诊断）；网格层面：可监督格子上的平均档差。
- 配对比较（默认上采样）：对每个 K ≥ 2，多轮 − 单轮、多轮 − 匹配的单轮、匹配的单轮 − 单轮的配对 bootstrap CI（分割 gIoU、cIoU、
  Boundary IoU；深度 δ1_strict 与两边都有效的图上的 AbsRel）。

**输出**（<叶目录> = <out>/eval/<变体>/<模型>/seed<k>/<任务>/）：
- predictions.jsonl（第 1 轮 = 单轮）、predictions_round<r>.jsonl（r ≥ 2）、predictions_sample<s>.jsonl（匹配计算量的采样）：每条
  样本一行（id、指令、是否调用、网格的紧凑编码、解析失败原因（失败时另存 token id）、状态、记账字段）；不含墙钟，逐字节可复现。
- scores.jsonl：逐样本、逐流程、逐上采样的指标；metrics.json：全部汇总（只由 predictions 与 GT 算出）；
- compute.json（单轮）、compute_multi_K<K>.json、compute_matched_K<K>.json（src/accounting 的 schema 3，含 GPU 与端到端墙钟）；
  compute_by_stage.json：每轮（每个阶段）单独的记账（{阶段: schema 3 的字典}）；
  metrics.json 的 compute 一节只列由输出决定的计数（调用、轮数、各类 token、像素）与相对单轮的倍数，所以 metrics.json、scores.jsonl、
  README.md 与续跑无关、逐字节可复现（墙钟只在 compute*.json 与 logs/）；
- config.yaml（运行身份：检查点 manifest 的 sha256、模型、任务、评测集 id 指纹、指令、允许集合、批配置、多轮与匹配的设置、上采样
  规则、E3b 配置的 sha256）与 meta.json；README.md；logs/timing.jsonl（每条每阶段的墙钟）、logs/run_<k>.json。
**续跑**（--resume）：config.yaml 必须与本次完全相同；各阶段按顺序跑（第 r 轮要第 r − 1 轮完整），每个阶段的文件只追加整批，
中断在一批中间时退回到这一批的开头整批重跑（批组成与一次跑完相同）。没有 --resume 而目录非空：拒绝覆盖。

**结构强制解码**（DEV-0041 §2，D83；resolve_eval(decode = "structured")，scripts/eval_e3b.py --decode structured）：每一次生成（单轮、
多轮的每一轮、匹配计算量的每次采样）都改走 adapter.generate_batch(allowed_token_ids = 同一允许集合, logits_processor =
[src/grid/structured_decode.StructuredGridProcessor])：第 r 行的格子只允许本任务的值 token，行末只允许行分隔，最后一个位置只允许停止
token，长度恒为 n(n + 1) + 1；贪心（do_sample = False），采样同上面的受限采样（同一处理器）；批组成、续跑、评分与 restricted 相同。
每批之后逐行核对结构（check_structured_ids，处理器没生效就报错）。叶目录名加后缀 __structured（<任务>__structured），设置与运行身份
（config.yaml 的 eval.decode）、metrics.json 的 decode、README 各记一笔；不给（或给 restricted）时没有这个键，输出与原来逐字节相同
（tests 用 53aefd2 的快照钉住）。与 generate_restricted 的关系（CPU fp32 逐 token 相同；bf16 上的差异见合并核对）见 structured_decode
的模块说明。

**指令检查**（DEV-0041 §1，只用于 refseg、只跑单轮：k_values = [1]；scripts/eval_e3b.py --instruction-check {none,swap}）：
- none：去掉指令。定义与 E3a 的 no_instruction 相同：指令换成 configs/e3a_train.json 的 eval.leak_checks.no_instruction.instruction
  （空串，由 src/readout/evaluate.resolve_eval 读出，不另写），user 轮里只剩图像；允许集合、解码与正常评测相同。
- swap：指代句换成同一张图里另一个物体的指代句（候选与取法见 SWAP_RULE；没有候选的样本跳过并计数）。每条预测对两个目标评分：
  换上去的物体（swapped）与原来的物体（original）。
- 叶目录 <任务>__instr_<种类>[__structured]，与正常评测并列；正常评测那一侧取已有的结果（--normal-leaf，默认同一 --out 里同一解码的
  <任务>/），按 id 配对，不重跑：它的运行身份（模型、检查点、网格上下文、token 布局、配置、批大小、上采样、解码）必须与本次相同，它的
  样本顺序的前 N 条必须就是本次的 N 条（--limit N）；它的单轮网格在本次的评分规则下重算，原目标上的读数必须与它的 scores.jsonl 逐项
  相同（核对配对没有错位）。只用默认上采样（同 E3a 的泄漏检查）。
- 批组成：N 取批大小的整数倍时，none 的每一批与正常评测的对应批是同一组样本（只是提示不同）；swap 跳过样本后按剩下的样本连续成批，
  与正常评测的批不再一一对应——配对按 id 做，跳过的样本不进配对，条数照记。
- 输出：predictions.jsonl（同单轮）、swap_pairs.jsonl（swap：每条样本的候选或跳过）、scores.jsonl（每条：两侧在各目标上的统计）、
  metrics.json（各目标上本次与正常评测的 cIoU、gIoU、BIoU，配对差 = 正常 − 检查的配对 bootstrap CI，跳过条数）、compute.json、
  config.yaml（运行身份另记 instruction_check 与 normal 的指纹）、README.md、logs/。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.accounting import Compute, merge, write_compute_json
from src.constants import GUIDED_FILTER_EPS, GUIDED_FILTER_RADIUS
from src.data import loaders as L
from src.data.records import LoadResult, Sample
from src.data.subsets import DEV_SUBSETS_PATH
from src.e1.scoring import PERSON_CATEGORY, ci_dict
from src.e3 import eval_scoring as ES
from src.e3.config import E3Config
from src.e3.data import load_rgb
from src.e3.feedback import DEPTH_COLORMAP, feedback_image
from src.e3.merge_check import CRITERIA as MERGE_CRITERIA
from src.e3.merge_check import DEFAULT_CRITERIA as DEFAULT_MERGE_CRITERIA
from src.e3.merge_check import E5V2_E3_BASIS, settings_from_config
from src.e3.targets import parse_target, seq_len
from src.e3.tokens import GRID_TASKS, GridTokens
from src.grid.structured_decode import STRUCTURED_RULE, GridStructure, StructuredGridProcessor, check_structured_ids
from src.runlog import write_run_metadata
from src.stats import bootstrap_ci, paired_bootstrap_ci, ratio_of_sums

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_REFSEG_ROOT = REPO_ROOT / "tests" / "fixtures" / "e3" / "eval_root"
FIXTURE_REFSEG_DEV = REPO_ROOT / "tests" / "fixtures" / "e3" / "eval_dev_subsets.json"
FIXTURE_DEPTH_ROOT = REPO_ROOT / "tests" / "fixtures" / "e1" / "root"
REFSEG_DATASET = "refcoco"  # RefCOCO（unc）

PREDICTIONS = "predictions.jsonl"
SCORES = "scores.jsonl"
METRICS = "metrics.json"
COMPUTE = "compute.json"
COMPUTE_BY_STAGE = "compute_by_stage.json"  # 每轮（每个阶段）单独的记账，含墙钟
CONFIG = "config.yaml"
README = "README.md"
TIMING = "logs/timing.jsonl"
MERGE_CHECK = "merge_check.json"
EVAL_SUB = "eval"
SAMPLING_KW = {"top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}  # 采样不截断、不惩罚：就是受限分布本身（模块说明）
GROUPS = ("person", "non_person", "unknown")
DECODE_MODES = ("restricted", "structured")  # DEV-0041 §2：restricted = generate_restricted（默认，不进运行身份）
STRUCTURED_SUFFIX = "__structured"
# 评测前合并核对的判据（DEV-0063，D102-2；模块说明）：v1 是默认、不进运行身份；e5-v2 只配结构强制解码
MERGE_CRITERIA_RULE = ("评测前的合并核对按 e5-v2 判（src/e3/merge_check.rejudge_e5v2 → src/e5/trained_eval.judge_e5v2，同一个函数）；读数"
                       "照旧来自 scripts/check_training_interfaces.py --checkpoint；v1 的判定只报告（merge_check.json 的 v1_report_only）")
INSTRUCTION_CHECKS = ("none", "swap")  # DEV-0041 §1
SWAP_PAIRS = "swap_pairs.jsonl"
SWAP_SEED = 20261008  # swap 候选的取法的种子（事先固定，DEV-0041 §1）
SWAP_RULE = ("候选 = 评测总体（load_eval_set 的全部样本，不受 --limit 影响）里同一张图（fields.image_id）、不同标注 id（fields.ann_id）的"
             "全部指代句，按评测总体的顺序排列；取第 k 个，k = sha256('visint-e3b-instr-swap|<SWAP_SEED>|<样本 id>') 的前 8 字节"
             "（大端整数）mod 候选数；没有候选的样本跳过并计数。指令 = 同一模板（tasks.refseg.grid_template）填候选的指代句")
NONE_RULE = ("去掉指令：指令换成 E3a 的 no_instruction（configs/e3a_train.json 的 eval.leak_checks.no_instruction.instruction，"
             "由 src/readout/evaluate.resolve_eval 读出），user 轮里只剩图像；允许集合与解码与正常评测相同")


# ---------------------------------------------------------------------------
# 路径、评测集、设置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalPaths:
    """refseg_root：含 refcoco/ 的 E1 落地根；depth_root：含 nyuv2/ 的 E1 落地根（真实运行两者相同，--data-root）；dev_subsets：
    E1 的开发子集清单（RefCOCO val 去掉开发子集用）。"""

    refseg_root: Path
    depth_root: Path
    dev_subsets: Path = DEV_SUBSETS_PATH
    strict: bool = True

    @classmethod
    def single_root(cls, root: str | Path, dev_subsets: str | Path | None = None) -> "EvalPaths":
        return cls(refseg_root=Path(root), depth_root=Path(root), dev_subsets=Path(dev_subsets or DEV_SUBSETS_PATH))

    @classmethod
    def fixtures(cls) -> "EvalPaths":
        return cls(refseg_root=FIXTURE_REFSEG_ROOT, depth_root=FIXTURE_DEPTH_ROOT, dev_subsets=FIXTURE_REFSEG_DEV)

    def as_dict(self) -> dict[str, Any]:
        return {"refseg_root": str(self.refseg_root), "depth_root": str(self.depth_root), "dev_subsets": str(self.dev_subsets),
                "strict": self.strict}


def load_eval_set(task: str, paths: EvalPaths) -> LoadResult:
    """评测集（规则见模块说明与配置的 eval.sets）。"""
    if task == "refseg":
        return L.load_refcoco(paths.refseg_root, REFSEG_DATASET, "val", subset="eval", dev_config=paths.dev_subsets,
                              strict=paths.strict)
    if task == "depth":
        from src.e2.data import load_nyu_test

        return load_nyu_test(paths.depth_root, strict=paths.strict)
    raise ValueError(f"任务 {task!r} 没有网格评测（有：{GRID_TASKS}）")


K_VALUES_OVERRIDE_BASIS = ("D72-4 (c)：多轮与匹配计算量的评测每个尺寸只做 1 个种子，单轮的主读数保留全部种子；其余种子用 "
                           "scripts/eval_e3b.py --k-values 只跑配置那一列的前缀（第 1 轮 = 单轮与跑全部 K 时逐字节相同）")


def config_k_values(cfg: E3Config) -> list[int]:
    """配置的 eval.multi_turn.k_values（必须从 1 开始、严格递增）。"""
    ks = [int(k) for k in cfg.doc["eval"]["multi_turn"]["k_values"]]
    if not ks or ks[0] != 1 or any(b <= a for a, b in zip(ks, ks[1:])):
        raise ValueError(f"eval.multi_turn.k_values 必须从 1 开始、严格递增，收到 {ks}")
    return ks


def k_values_prefix(config_ks: Sequence[int], requested: Sequence[int]) -> list[int]:
    """--k-values 的取值规则（DEV-0039）：必须是配置那一列从 1 开始的前缀（[1]、[1, 2]……），否则 ValueError。返回生效的 K。"""
    req = [int(k) for k in requested]
    cks = [int(k) for k in config_ks]
    if not req or req != cks[: len(req)]:
        raise ValueError(f"--k-values 必须是配置的 eval.multi_turn.k_values {cks} 从 1 开始的前缀（如 {cks[:1]} 或 {cks[:2]}），"
                         f"收到 {req}")
    return req


def resolve_eval(cfg: E3Config, *, tiny: bool, batch_size: int | None = None, score_workers: int | None = None,
                 merge_check_n: int | None = None, k_values: Sequence[int] | None = None,
                 decode: str | None = None, merge_criteria: str | None = None) -> dict[str, Any]:
    """configs/e3b_train.json 的 eval 节 → 本次生效的设置（--tiny 时按 eval.tiny 覆盖规模；命令行可再覆盖批大小、评分进程数、
    合并核对的 n，以及只跑配置 K 那一列的前缀：k_values，规则见模块说明）。decode（DEV-0041 §2）：None 或 restricted = 原来的
    受限解码，设置里没有 decode 这个键（逐字节不变）；structured = 结构强制，设置里加 decode。merge_criteria（DEV-0063）：None 或 v1 =
    原来的判据，设置里没有这个键（逐字节不变）；e5-v2 = 设置里加 merge_criteria（进运行身份），只配 structured。结构不合规就报错。"""
    e = cfg.doc["eval"]
    config_ks = config_k_values(cfg)
    ks = config_ks if k_values is None else k_values_prefix(config_ks, k_values)
    up = e["upsample"]
    methods = [str(m) for m in up["methods"]]
    if up["primary"] not in methods or methods[0] != up["primary"] or set(methods) - set(ES.UPSAMPLE_METHODS):
        raise ValueError(f"eval.upsample：primary 必须是 methods 的第一个，methods 只能取 {ES.UPSAMPLE_METHODS}，收到 {up}")
    mc = e["matched_compute"]
    if mc["method"] != "greedy_plus_samples_vote" or not float(mc["temperature"]) > 0:
        raise ValueError(f"eval.matched_compute 只实现了 greedy_plus_samples_vote（temperature > 0），收到 {mc}")
    t = e["tiny"] if tiny else {}
    out = {
        "version": e["version"],
        "batch_size": int(batch_size if batch_size is not None else t.get("batch_size", e["batch_size"])),
        "batch_size_config": int(e["batch_size"]),
        "k_values": ks,
        "n_rounds": max(ks),
        "n_samples": max(ks) - 1,
        "upsample": {"primary": up["primary"], "methods": methods, "threshold": float(up["threshold"]),
                     "guided_filter": {"radius": GUIDED_FILTER_RADIUS, "eps": GUIDED_FILTER_EPS}},
        "matched_compute": {"method": mc["method"], "temperature": float(mc["temperature"]), "seed": int(mc["seed"]),
                            "sampling_kw": dict(SAMPLING_KW)},
        "merge_check": settings_from_config(cfg.doc, tiny=tiny, n=merge_check_n, purpose="grid_eval"),
        "score_workers": int(score_workers if score_workers is not None else t.get("score_workers", e["score_workers"])),
    }
    if out["batch_size"] < 1 or out["score_workers"] < 0:
        raise ValueError(f"批大小、合并核对的 n 必须 ≥ 1，评分进程数 ≥ 0：{out}")
    if ks != config_ks:  # 只在真的覆盖时加这个键：不给 --k-values（或给的就是配置那一列）时设置与运行身份逐字节不变
        out["k_values_override"] = {"config_k_values": list(config_ks), "k_values": list(ks), "basis": K_VALUES_OVERRIDE_BASIS}
    if decode is not None and decode not in DECODE_MODES:
        raise ValueError(f"decode 必须是 {DECODE_MODES} 之一，收到 {decode!r}")
    if decode == "structured":  # 只在结构强制时加这个键（DEV-0041 §2）：restricted 的设置与运行身份逐字节不变
        out["decode"] = {"mode": "structured", "rule": STRUCTURED_RULE,
                         "greedy": "adapter.generate_batch(allowed_token_ids = 任务的允许集合, logits_processor = "
                                   "[StructuredGridProcessor], do_sample = False)",
                         "sampling": "匹配计算量的采样：同上，do_sample = True，temperature 与 sampling_kw 同 matched_compute",
                         "basis": "D83：E3b+C3 与 E5 全部网格任务的正式口径；现有各臂另报作诊断"}
    if merge_criteria is not None and merge_criteria not in MERGE_CRITERIA:
        raise ValueError(f"merge_criteria 必须是 {MERGE_CRITERIA} 之一，收到 {merge_criteria!r}")
    if merge_criteria not in (None, DEFAULT_MERGE_CRITERIA):  # 只在 e5-v2 时加这个键（DEV-0063）：v1 的设置与运行身份逐字节不变
        if decode != "structured":
            raise ValueError(f"merge_criteria {merge_criteria} 只用于结构强制的网格评测（decode = structured，D102-2），收到 decode {decode!r}")
        out["merge_criteria"] = {"version": merge_criteria, "rule": MERGE_CRITERIA_RULE, "basis": E5V2_E3_BASIS}
    return out


def decode_mode(settings: Mapping[str, Any]) -> str:
    """本次的解码方式（DEV-0041 §2）：设置里没有 decode 就是 restricted。"""
    d = settings.get("decode")
    return "restricted" if d is None else str(d["mode"])


def merge_criteria_version(settings: Mapping[str, Any]) -> str:
    """本次评测前合并核对的判据（DEV-0063）：设置里没有 merge_criteria 就是 v1。"""
    c = settings.get("merge_criteria")
    return DEFAULT_MERGE_CRITERIA if c is None else str(c["version"])


def leaf_name(task: str, settings: Mapping[str, Any] | None = None, check: str | None = None) -> str:
    """叶目录名：<任务>，指令检查加 __instr_<种类>，结构强制再加 __structured（restricted、没有检查时就是任务名，与原来相同）。"""
    if check is not None and check not in INSTRUCTION_CHECKS:
        raise ValueError(f"指令检查只有 {INSTRUCTION_CHECKS}，收到 {check!r}")
    name = task if check is None else f"{task}__instr_{check}"
    return name + (STRUCTURED_SUFFIX if settings is not None and decode_mode(settings) == "structured" else "")


def identity_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """进运行身份的设置：评分进程数不改变任何结果，不进；合并核对另记在 merge_check.json。"""
    return {k: v for k, v in settings.items() if k not in ("score_workers", "merge_check")}


# ---------------------------------------------------------------------------
# 任务上下文：指令、允许集合、长度
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskContext:
    task: str
    n: int
    tokens: GridTokens
    allowed: tuple[int, ...]
    max_new_tokens: int
    enable_thinking: bool
    order: str
    grid_template: str
    correction_template: str
    min_valid_fraction: float

    def instruction(self, sample: Sample, *, correction: bool) -> str:
        tmpl = self.correction_template if correction else self.grid_template
        if self.task == "refseg":
            return tmpl.format(expression=str(sample.fields["expression"]).strip(), n=self.n)
        return tmpl.format(n=self.n)

    def as_dict(self) -> dict[str, Any]:
        return {"task": self.task, "grid_size": self.n, "allowed_ids": list(self.allowed), "max_new_tokens": self.max_new_tokens,
                "stop_id": int(self.tokens.stop_id), "enable_thinking": self.enable_thinking, "order": self.order,
                "grid_template": self.grid_template, "correction_template": self.correction_template,
                "value_ids": list(self.tokens.value_ids(self.task)), "row_id": self.tokens.row_id}


def make_context(cfg: E3Config, tokens: GridTokens, task: str, *, n: int | None = None) -> TaskContext:
    """n 默认取配置的网格边长（32）；测试可给更小的 n（DEV-0025 的过拟合流程用 8×8）。"""
    nn = int(cfg.grid_n if n is None else n)
    tc = cfg.task(task)
    if task == "refseg":
        g, c = tc["grid_template"], tc["correction_template"]
    elif task == "depth":
        g, c = tc["grid_instruction"], tc["correction_instruction"]
    else:
        raise ValueError(f"任务 {task!r} 没有网格评测")
    mvf = float(cfg.doc["tasks"]["depth"]["cell"]["min_valid_fraction"])
    return TaskContext(task=task, n=nn, tokens=tokens, allowed=tuple(tokens.allowed_ids(task, cfg.allowed_scope)),
                       max_new_tokens=seq_len(nn), enable_thinking=cfg.enable_thinking, order=str(tc["order"]),
                       grid_template=str(g), correction_template=str(c), min_valid_fraction=mvf)


# ---------------------------------------------------------------------------
# 阶段：第 r 轮、第 s 次采样
# ---------------------------------------------------------------------------


def stage_names(settings: Mapping[str, Any]) -> list[str]:
    return [f"round{r}" for r in range(1, int(settings["n_rounds"]) + 1)] + \
           [f"sample{s}" for s in range(1, int(settings["n_samples"]) + 1)]


def stage_file(stage: str) -> str:
    return PREDICTIONS if stage == "round1" else f"predictions_{stage}.jsonl"


def stage_round(stage: str) -> int | None:
    return int(stage[5:]) if stage.startswith("round") else None


def chunk_plan(active: Sequence[bool], batch_size: int) -> list[tuple[int, int, list[int]]]:
    """样本 [0, n) 按顺序切成若干段，每段至多 batch_size 个要调用的样本（active），段的边界是续跑的边界：
    返回 [(起, 止, 段内要调用的下标)]，各段首尾相接、覆盖全部样本。"""
    if batch_size < 1:
        raise ValueError("batch_size 必须 ≥ 1")
    out, lo, act = [], 0, []
    for i, a in enumerate(active):
        if a:
            act.append(i)
        if len(act) == batch_size:
            out.append((lo, i + 1, act))
            lo, act = i + 1, []
    if lo < len(active):
        out.append((lo, len(active), act))
    return out


def sampling_seed(base: int, stage: str, chunk: int) -> int:
    h = hashlib.sha256(f"visint-e3-eval-sample|{int(base)}|{stage}|{int(chunk)}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1)


def read_jsonl_tolerant(path: Path, *, repair: bool) -> list[dict[str, Any]]:
    """读 jsonl；末尾不完整的一行丢掉，repair=True 时把文件截到最后一个完整行（同 src/e1/runner）。"""
    from src.e1.runner import read_jsonl_tolerant as _r

    return _r(path, repair=repair)


def _truncate_lines(path: Path, k: int) -> None:
    raw = path.read_bytes()
    pos = 0
    for _ in range(k):
        pos = raw.index(b"\n", pos) + 1
    with open(path, "r+b") as f:
        f.truncate(pos)


def _append(path: Path, lines: Sequence[str]) -> None:
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def _jsonl(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False) + "\n"


def grid_structure(ctx: TaskContext) -> GridStructure:
    """本任务网格的形状（结构强制解码，DEV-0041 §2）：各行的值 token = 本任务的值 token，行分隔与停止 token 取 ctx.tokens。"""
    st = GridStructure.uniform(ctx.tokens.value_ids(ctx.task), ctx.tokens.row_id, ctx.tokens.stop_id, ctx.n)
    if st.length != ctx.max_new_tokens:
        raise RuntimeError(f"结构的长度 {st.length} 与 max_new_tokens {ctx.max_new_tokens} 不同")
    missing = sorted(set(st.all_ids()) - {int(i) for i in ctx.allowed})
    if missing:
        raise RuntimeError(f"结构里的 token {missing[:4]} 不在允许集合里")
    return st


def structured_generate(ad: Any, ctx: TaskContext, inputs: Sequence[Any], *, where: str,
                        sample_kw: Mapping[str, Any] | None = None) -> list[Any]:
    """结构强制的一批（模块说明）：generate_batch(allowed_token_ids, logits_processor = [StructuredGridProcessor])，贪心
    （sample_kw = None）或受限采样；之后逐行核对结构（入口断言），处理器没有生效就报错。"""
    st = grid_structure(ctx)
    proc = StructuredGridProcessor(st)
    kw = dict(sample_kw) if sample_kw is not None else {"do_sample": False}
    res = ad.generate_batch(list(inputs), enable_thinking=ctx.enable_thinking, max_new_tokens=ctx.max_new_tokens,
                            allowed_token_ids=list(ctx.allowed), logits_processor=[proc], **kw)
    if len(res) != len(inputs):
        raise RuntimeError(f"{where}：生成返回 {len(res)} 行，输入 {len(inputs)} 行")
    if inputs and proc.n_calls != st.length:
        raise RuntimeError(f"{where}：结构强制的处理器被调用了 {proc.n_calls} 次，应为 {st.length}")
    for k, r in enumerate(res):
        check_structured_ids(r.token_ids, st, stopped=r.stopped, where=f"{where} 第 {k} 行")
    return res


def generate_chunk(ad: Any, ctx: TaskContext, inputs: Sequence[Any], stage: str, chunk: int,
                   settings: Mapping[str, Any]) -> list[Any]:
    """一批的生成：第 r 轮贪心受限解码（generate_restricted）；第 s 次采样走 generate_batch 的受限采样（模块说明）。
    结构强制（DEV-0041 §2）时两者都改走 structured_generate（贪心 / 同一受限采样，逐位置收窄）。"""
    structured = decode_mode(settings) == "structured"
    if stage.startswith("round"):
        if structured:
            return structured_generate(ad, ctx, inputs, where=f"[{ctx.task} {stage} 第 {chunk} 批]")
        return ad.generate_restricted(list(inputs), list(ctx.allowed), max_new_tokens=ctx.max_new_tokens,
                                      stop_ids=[int(ctx.tokens.stop_id)], enable_thinking=ctx.enable_thinking)
    import torch

    mc = settings["matched_compute"]
    torch.manual_seed(sampling_seed(mc["seed"], stage, chunk))
    if structured:
        return structured_generate(ad, ctx, inputs, where=f"[{ctx.task} {stage} 第 {chunk} 批]",
                                   sample_kw={"do_sample": True, "temperature": float(mc["temperature"]), **mc["sampling_kw"]})
    return ad.generate_batch(list(inputs), enable_thinking=ctx.enable_thinking, max_new_tokens=ctx.max_new_tokens,
                             allowed_token_ids=list(ctx.allowed), do_sample=True, temperature=float(mc["temperature"]),
                             **mc["sampling_kw"])


def prev_states(leaf: Path, stage: str, ids: Sequence[str]) -> list[str | None] | None:
    """第 r ≥ 2 轮的输入状态：第 r − 1 轮之后的状态（必须完整）。其余阶段返回 None。"""
    r = stage_round(stage)
    if r is None or r == 1:
        return None
    recs = read_jsonl_tolerant(leaf / stage_file(f"round{r - 1}"), repair=False)
    if [x["id"] for x in recs] != list(ids):
        raise RuntimeError(f"第 {r} 轮要第 {r - 1} 轮完整，{stage_file(f'round{r - 1}')} 只有 {len(recs)}/{len(ids)} 条")
    return [x["state"] for x in recs]


def run_stage(ad: Any, ctx: TaskContext, settings: Mapping[str, Any], leaf: Path, samples: Sequence[Sample], stage: str, *,
              run_k: int, log: Callable[[str], None]) -> dict[str, Any]:
    """跑一个阶段（续跑规则见模块说明）。返回本次调用的统计。"""
    ids = [s.id for s in samples]
    path = leaf / stage_file(stage)
    states = prev_states(leaf, stage, ids)
    active = [True] * len(samples) if states is None else [st is not None for st in states]
    plan = chunk_plan(active, int(settings["batch_size"]))
    done = read_jsonl_tolerant(path, repair=True)
    if [r["id"] for r in done] != ids[: len(done)]:
        raise RuntimeError(f"{path} 里已完成的 id 不是样本顺序的前缀，不能续跑")
    bounds = [lo for lo, _, _ in plan] + [len(ids)]
    keep = max(b for b in bounds if b <= len(done))
    trimmed = None
    if keep < len(done):
        _truncate_lines(path, keep)
        trimmed = {"from": len(done), "to": keep}
        log(f"[{ctx.task} {stage}] 已完成 {len(done)} 条不在批的边界上，退回到 {keep} 条整批重跑")
    r = stage_round(stage)
    correction = r is not None and r >= 2
    n_calls = n_chunks = 0
    for ci, (lo, hi, act) in enumerate(plan):
        if lo < keep:
            continue
        t0 = time.perf_counter()
        images = {i: load_rgb(samples[i].images[0]) for i in act}
        inputs, instr = [], {}
        for i in act:
            imgs = [images[i]]
            if correction:
                imgs.append(feedback_image(ctx.task, images[i], ES.decode_grid(ctx.task, states[i], ctx.n)))
            instr[i] = ctx.instruction(samples[i], correction=correction)
            inputs.append(ad.build_inputs(imgs, instr[i], ctx.order))
        prep = time.perf_counter() - t0
        results = generate_chunk(ad, ctx, inputs, stage, ci, settings) if act else []
        if len(results) != len(act):
            raise RuntimeError("生成返回的行数与输入不同")
        res_of = dict(zip(act, results))
        lines, tlines = [], []
        for i in range(lo, hi):
            rec: dict[str, Any] = {"id": ids[i], "stage": stage}
            if i not in res_of:
                rec.update(called=False, skipped="no_valid_state", grid=None, failure=None)
                if r is not None:
                    rec["state"] = None
                lines.append(_jsonl(rec))
                continue
            res = res_of[i]
            p = parse_target(ctx.task, res.token_ids, ctx.tokens, n=ctx.n)
            code = None if p.grid is None else ES.encode_grid(ctx.task, p.grid)
            rec.update(called=True, instruction=instr[i], n_images=2 if correction else 1, grid=code,
                       failure=None if p.failure is None else p.failure.value, stopped=res.stopped,
                       n_generated_tokens=int(res.n_generated_tokens), n_prompt_tokens=int(res.n_prompt_tokens),
                       n_visual_tokens=int(res.n_visual_tokens), n_image_pixels=int(res.n_image_pixels or 0))
            if r is not None:
                rec["state"] = code if code is not None else (states[i] if states is not None else None)
                if correction and p.grid is not None:
                    rec["n_changed_cells"] = int((p.grid != ES.decode_grid(ctx.task, states[i], ctx.n)).sum())
            if p.failure is not None:
                rec["token_ids"] = [int(t) for t in res.token_ids]
            lines.append(_jsonl(rec))
            tlines.append(_jsonl({"id": ids[i], "stage": stage, "run": run_k, "chunk": ci, "batch_rows": len(act),
                                  "wall_time_s": float(res.wall_time_s), "batch_wall_time_s": float(res.batch_wall_time_s),
                                  "prep_s": prep / len(act)}))
        _append(leaf / TIMING, tlines)
        _append(path, lines)
        n_calls += len(act)
        n_chunks += 1
        if n_chunks == 1 or n_chunks % 20 == 0 or hi == len(ids):
            log(f"[{ctx.task} {stage}] {hi}/{len(ids)}")
    return {"stage": stage, "n_calls": n_calls, "n_chunks_run": n_chunks, "n_chunks": len(plan), "trimmed": trimmed,
            "n_active": int(sum(active))}


# ---------------------------------------------------------------------------
# 流程：单轮、多轮、匹配计算量的单轮
# ---------------------------------------------------------------------------


def procedures(settings: Mapping[str, Any]) -> list[str]:
    """评分的流程：single（= 第 1 轮）、round<r>（r ≥ 2，第 r 轮之后的状态）、matched_K<K>（K ≥ 2）。"""
    return (["single"] + [f"round{r}" for r in range(2, int(settings["n_rounds"]) + 1)]
            + [f"matched_K{k}" for k in settings["k_values"] if k > 1])


def multi_proc(k: int) -> str:
    """K 轮流程的输出对应的评分流程：K = 1 就是单轮。"""
    return "single" if k == 1 else f"round{k}"


def vote(task: str, grids: Sequence[np.ndarray | None]) -> np.ndarray | None:
    """匹配计算量的单轮的合成（模块说明）：分割逐格多数票、平票取第一张有效网格；深度逐格档号平均后 np.rint。"""
    valid = [np.asarray(g, dtype=np.int64) for g in grids if g is not None]
    if not valid:
        return None
    st = np.stack(valid)
    if task == "refseg":
        ones = st.sum(axis=0)
        zeros = len(valid) - ones
        return np.where(ones > zeros, 1, np.where(zeros > ones, 0, valid[0])).astype(np.int64)
    if task == "depth":
        return np.rint(st.astype(np.float64).mean(axis=0)).astype(np.int64)
    raise ValueError(f"未知任务 {task!r}")


def read_stages(leaf: Path, settings: Mapping[str, Any], ids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for st in stage_names(settings):
        recs = read_jsonl_tolerant(leaf / stage_file(st), repair=False)
        if [r["id"] for r in recs] != list(ids):
            raise RuntimeError(f"{stage_file(st)} 不完整或顺序不对（{len(recs)}/{len(ids)} 条）")
        out[st] = recs
    return out


def procedure_codes(task: str, n: int, settings: Mapping[str, Any], stages: Mapping[str, list[dict[str, Any]]],
                    i: int) -> dict[str, str | None]:
    """第 i 条样本各流程的网格编码。"""
    codes: dict[str, str | None] = {"single": stages["round1"][i]["grid"]}
    for r in range(2, int(settings["n_rounds"]) + 1):
        codes[f"round{r}"] = stages[f"round{r}"][i]["state"]
    for k in settings["k_values"]:
        if k > 1:
            voters = [stages["round1"][i]["grid"]] + [stages[f"sample{s}"][i]["grid"] for s in range(1, k)]
            g = vote(task, [None if c is None else ES.decode_grid(task, c, n) for c in voters])
            codes[f"matched_K{k}"] = None if g is None else ES.encode_grid(task, g)
    return codes


def procedure_stages(proc: str) -> list[str]:
    """一个流程用到的阶段（记账）：single = 第 1 轮；multi_K<K> = 第 1..K 轮；matched_K<K> = 第 1 轮 + 第 1..K−1 次采样。"""
    if proc == "single":
        return ["round1"]
    if proc.startswith("multi_K"):
        k = int(proc[7:])
        return [f"round{r}" for r in range(1, k + 1)]
    if proc.startswith("matched_K"):
        k = int(proc[9:])
        return ["round1"] + [f"sample{s}" for s in range(1, k)]
    raise ValueError(proc)


def compute_procs(settings: Mapping[str, Any]) -> list[str]:
    return ["single"] + [f"multi_K{k}" for k in settings["k_values"]] + [f"matched_K{k}" for k in settings["k_values"]]


def compute_of(proc: str, stages: Mapping[str, list[dict[str, Any]]], timing: Mapping[tuple[str, str], Mapping[str, Any]],
               n: int) -> Compute:
    """一个流程的记账：每条样本把它在这些阶段里的调用相加（Compute.call），样本墙钟 = 各次调用的 GPU 份额 + 准备输入的份额。
    没有调用的阶段（skipped）不计。轮数（src/accounting 的 new_round）：多轮的每一轮各开一轮；匹配计算量的采样与贪心那次互不依赖
    （可以并发），记作同一轮里的其他调用（new_round = False），所以匹配的单轮 n_rounds = 1、n_calls = K。"""
    return compute_of_stages(procedure_stages(proc), stages, timing, n)


def compute_of_stages(stage_list: Sequence[str], stages: Mapping[str, list[dict[str, Any]]],
                      timing: Mapping[tuple[str, str], Mapping[str, Any]], n: int) -> Compute:
    """若干阶段的记账（compute_of 的规则；单个阶段就是"每轮"的记账）。"""
    per = []
    for i in range(n):
        calls, wall = [], 0.0
        for st in stage_list:
            rec = stages[st][i]
            if not rec.get("called"):
                continue
            t = timing.get((st, rec["id"]), {})
            g = float(t.get("wall_time_s", 0.0))
            calls.append(Compute.call(generated_tokens=rec["n_generated_tokens"], prompt_tokens=rec["n_prompt_tokens"],
                                      visual_tokens=rec["n_visual_tokens"], image_pixels=rec["n_image_pixels"],
                                      call_time_s=g, new_round=not st.startswith("sample")))
            wall += g + float(t.get("prep_s", 0.0))
        per.append(merge(calls).as_sample(wall_time_s=wall))
    return merge(per)


def compute_file(proc: str) -> str:
    return COMPUTE if proc == "single" else f"compute_{proc}.json"


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _mean_ci(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"value": None, "ci": None, "n": 0}
    ci = bootstrap_ci(np.asarray(values, dtype=np.float64))
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(values)}


def _ratio_ci(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    if not pairs or sum(u for _, u in pairs) == 0:
        return {"value": None, "ci": None, "n": len(pairs)}
    arr = np.asarray(pairs, dtype=np.float64)
    try:
        ci = bootstrap_ci(arr, statistic=ratio_of_sums)
    except ValueError:  # 某次重采样的累计并为 0（极小样本）
        return {"value": ratio_of_sums(arr), "ci": None, "n": len(pairs)}
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(pairs)}


def _paired(a: Sequence[Any], b: Sequence[Any], statistic: Any = None) -> dict[str, Any] | None:
    if len(a) == 0:
        return None
    try:
        ci = paired_bootstrap_ci(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), statistic=statistic)
    except ValueError:
        return None
    return {"estimate": ci.estimate, **ci_dict(ci), "n": len(a)}


def group_of(task: str, s: Sample) -> str | None:
    """分割按被指代物体是不是 person 分列（与 E1 的 refusal_by_person 同一规则）；深度不分组。"""
    if task != "refseg":
        return None
    cat = s.fields.get("category")
    return "unknown" if cat is None else ("person" if cat == PERSON_CATEGORY else "non_person")


def _seg_block(rows: Sequence[Mapping[str, Any]], proc: str, m: str) -> dict[str, Any]:
    r = [x["procs"][proc] for x in rows]
    n_valid = sum(1 for x in r if x["valid"])
    return {"cIoU": _ratio_ci([(x[m]["inter"], x[m]["union"]) for x in r]), "gIoU": _mean_ci([x[m]["iou"] for x in r]),
            "BIoU": _mean_ci([x[m]["biou"] for x in r]), "n": len(r), "n_valid": n_valid,
            "valid_rate": n_valid / len(r) if r else None}


def _depth_block(rows: Sequence[Mapping[str, Any]], proc: str, m: str) -> dict[str, Any]:
    scorable = [x for x in rows if x["gt"]["n_eval_px"] > 0]
    r = [x["procs"][proc] for x in scorable]
    val = [x[m] for x in r if x["valid"] and x[m] is not None]
    return {"abs_rel": _mean_ci([v["abs_rel"] for v in val]), "d1": _mean_ci([v["d1"] for v in val]),
            "d1_strict": _mean_ci([x[m]["d1"] if (x["valid"] and x[m] is not None) else 0.0 for x in r]),
            "abs_rel_ss": _mean_ci([v["abs_rel_ss"] for v in val]), "d1_ss": _mean_ci([v["d1_ss"] for v in val]),
            "n": len(r), "n_valid": len(val), "valid_rate": len(val) / len(r) if r else None,
            "n_unscorable_gt": len(rows) - len(scorable),
            "note": "abs_rel、d1、*_ss 是条件读数（只在解析成功的图上）；d1_strict 把解析失败的图记 0"}


def _grid_block(task: str, rows: Sequence[Mapping[str, Any]], proc: str) -> dict[str, Any]:
    r = [x["procs"][proc] for x in rows]
    if task == "refseg":
        return {"cell_iou": _mean_ci([x["grid"]["cell_iou"] for x in r]),
                "exact_rate": float(np.mean([x["grid"]["exact"] for x in r])) if r else None,
                "note": "与 GT 网格（mask_to_grid）比较；解析失败记 0"}
    g = [x["grid"] for x in r if x["valid"] and x["grid"] is not None]
    return {"mean_abs_bin_error": _mean_ci([v["mean_abs_bin_error"] for v in g]),
            "exact_cell_rate": _mean_ci([v["exact_cell_rate"] for v in g]),
            "note": "可监督格子上（src/e3/targets.depth_grid），只在解析成功的图上"}


def comparisons(task: str, rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    """默认上采样下的配对 bootstrap（模块说明）。"""
    m = settings["upsample"]["primary"]
    out: dict[str, Any] = {"upsample": m, "definition": "差 = 前者 − 后者，配对 bootstrap（同一组重采样下标）"}
    for k in settings["k_values"]:
        if k == 1:
            continue
        for a, b in ((multi_proc(k), "single"), (multi_proc(k), f"matched_K{k}"), (f"matched_K{k}", "single")):
            key = f"{'multi_K' + str(k) if a == multi_proc(k) else a}_minus_{b}"
            if task == "refseg":
                pa = [x["procs"][a][m] for x in rows]
                pb = [x["procs"][b][m] for x in rows]
                out[key] = {"gIoU": _paired([x["iou"] for x in pa], [x["iou"] for x in pb]),
                            "cIoU": _paired([[x["inter"], x["union"]] for x in pa], [[x["inter"], x["union"]] for x in pb],
                                            statistic=ratio_of_sums),
                            "BIoU": _paired([x["biou"] for x in pa], [x["biou"] for x in pb])}
            else:
                sc = [x for x in rows if x["gt"]["n_eval_px"] > 0]

                def d1s(x: Mapping[str, Any], p: str) -> float:
                    v = x["procs"][p]
                    return v[m]["d1"] if (v["valid"] and v[m] is not None) else 0.0

                both = [x for x in sc if all(x["procs"][p]["valid"] and x["procs"][p][m] is not None for p in (a, b))]
                out[key] = {"d1_strict": _paired([d1s(x, a) for x in sc], [d1s(x, b) for x in sc]),
                            "abs_rel_both_valid": _paired([x["procs"][a][m]["abs_rel"] for x in both],
                                                          [x["procs"][b][m]["abs_rel"] for x in both])}
    return out


def _distribution(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    q = np.percentile(a, [5, 50, 95])
    return {"n": len(values), "mean": float(a.mean()), "p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2]),
            "min": float(a.min()), "max": float(a.max())}


COUNT_FIELDS = ("n_samples", "n_calls", "n_rounds", "generated_tokens", "prompt_tokens", "visual_tokens", "image_pixels")


def compute_summary(computes: Mapping[str, Compute]) -> dict[str, Any]:
    """metrics.json 的 compute 一节：只放由输出决定的计数（调用、轮数、各类 token、像素）与相对单轮的倍数，逐字节可复现；
    墙钟（GPU 的 call_time_s 与端到端的 wall_time_s）在 compute*.json。"""
    single = computes["single"]
    out = {}
    for p, c in computes.items():
        d = {k: getattr(c, k) for k in COUNT_FIELDS}
        d["per_sample_mean"] = {k: getattr(c, k) / c.n_samples for k in COUNT_FIELDS[1:]} if c.n_samples else None
        d["ratio_vs_single"] = {k: (getattr(c, k) / getattr(single, k)) if getattr(single, k) else None
                                for k in ("n_calls", "generated_tokens", "prompt_tokens", "visual_tokens")}
        d["stages"] = procedure_stages(p)
        d["file"] = compute_file(p)
        out[p] = d
    return out


def stage_diagnostics(stages: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for st, recs in stages.items():
        called = [r for r in recs if r.get("called")]
        fails = Counter(r["failure"] for r in called if r["failure"] is not None)
        d: dict[str, Any] = {"n": len(recs), "n_called": len(called), "n_skipped": len(recs) - len(called),
                             "n_valid": sum(1 for r in called if r["grid"] is not None),
                             "parse_failures": dict(sorted(fails.items())),
                             "parse_failure_rate": (sum(fails.values()) / len(called)) if called else None,
                             "n_stopped": sum(1 for r in called if r.get("stopped")),
                             "generated_tokens": _distribution([r["n_generated_tokens"] for r in called])}
        if stage_round(st) is not None and stage_round(st) >= 2:
            ch = [r["n_changed_cells"] for r in called if "n_changed_cells" in r]
            d["changed_cells"] = _distribution(ch)
            d["n_unchanged"] = sum(1 for c in ch if c == 0)
        out[st] = d
    return out


def aggregate(task: str, samples: Sequence[Sample], rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any],
              stages: Mapping[str, list[dict[str, Any]]], computes: Mapping[str, Compute]) -> dict[str, Any]:
    procs = procedures(settings)
    methods = settings["upsample"]["methods"]
    groups = [group_of(task, s) for s in samples]
    by_proc: dict[str, Any] = {}
    for p in procs:
        blk: dict[str, Any] = {}
        for m in methods:
            blk[m] = _seg_block(rows, p, m) if task == "refseg" else _depth_block(rows, p, m)
            if task == "refseg":
                blk[m]["by_group"] = {g: _seg_block([r for r, gg in zip(rows, groups) if gg == g], p, m)
                                      for g in GROUPS if g in groups}
        blk["grid"] = _grid_block(task, rows, p)
        by_proc[p] = blk
    prim = settings["upsample"]["primary"]
    comp = compute_summary(computes)
    head = by_proc["single"][prim]
    headline = ({"cIoU": head["cIoU"], "gIoU": head["gIoU"], "BIoU": head["BIoU"]} if task == "refseg"
                else {"abs_rel": head["abs_rel"], "d1": head["d1"], "d1_strict": head["d1_strict"]})
    return {
        "headline": {"procedure": "single", "upsample": prim, **headline,
                     "note": "单轮是\"LLM 能力\"的主要数字（硬约束 11）；多轮与匹配计算量的单轮见 multi_turn、matched_compute"},
        "single": by_proc["single"],
        "multi_turn": {"k_values": settings["k_values"],
                       "by_K": {str(k): {"procedure": multi_proc(k), "metrics": by_proc[multi_proc(k)],
                                         "compute": comp[f"multi_K{k}"]} for k in settings["k_values"]},
                       "by_round": {str(r): (by_proc["single"] if r == 1 else by_proc[f"round{r}"])
                                    for r in range(1, int(settings["n_rounds"]) + 1)},
                       "definition": "K = 总轮数；第 1 轮就是单轮（K = 1 与单轮逐个相同）；K 轮流程的输出 = 第 K 轮之后的状态"
                                     "（最近一个解析成功的网格）；还没有有效网格的样本不再调用"},
        "matched_compute": {"method": settings["matched_compute"],
                            "by_K": {str(k): {"procedure": "single" if k == 1 else f"matched_K{k}",
                                              "metrics": by_proc["single" if k == 1 else f"matched_K{k}"],
                                              "compute": comp[f"matched_K{k}"]} for k in settings["k_values"]},
                            "definition": "K 次单轮调用：贪心 + K − 1 次受限采样，逐格合成（分割多数票、平票取第一张有效网格；深度档号平均后取整）"},
        "comparisons": comparisons(task, rows, settings),
        "compute": comp,
        "stages": stage_diagnostics(stages),
        "groups": dict(sorted(Counter(g for g in groups if g is not None).items())),
    }


# ---------------------------------------------------------------------------
# 运行身份、README、主流程
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    tiny: bool
    dtype: str
    variant: str
    seed: int
    checkpoint: Mapping[str, Any]
    limit: int | None = None


def leaf_dir(out: str | Path, spec: EvalSpec, task: str, settings: Mapping[str, Any] | None = None,
             check: str | None = None) -> Path:
    """<out>/eval/<变体>/<模型>/seed<k>/<叶目录名>（leaf_name：restricted、没有指令检查时就是任务名）。"""
    return model_dir(out, spec) / leaf_name(task, settings, check)


def model_dir(out: str | Path, spec: EvalSpec) -> Path:
    return Path(out) / EVAL_SUB / spec.variant / spec.model_key / f"seed{int(spec.seed)}"


def ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def run_identity(spec: EvalSpec, cfg: E3Config, settings: Mapping[str, Any], ctx: TaskContext, sel: LoadResult,
                 samples: Sequence[Sample]) -> dict[str, Any]:
    ids = [s.id for s in samples]
    return {
        "experiment": "E3b-eval",
        "framework": "src/e3/evaluate.py（DEV-0030）",
        "variant": spec.variant,
        "seed": int(spec.seed),
        "task": ctx.task,
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny, "dtype": spec.dtype},
        "checkpoint": dict(spec.checkpoint),
        "context": ctx.as_dict(),
        "tokens": {"layout": ctx.tokens.layout()},
        "eval": identity_settings(settings),
        "feedback": {"entry": "src/e3/feedback.feedback_image",
                     "depth_colormap_sha256": hashlib.sha256(DEPTH_COLORMAP.tobytes()).hexdigest()},
        "grid_code": ES.GRID_CODES[ctx.task],
        "config": {"path": cfg.path, "sha256": cfg.sha256, "version": cfg.doc["version"]},
        "data": {"n": len(ids), "ids_sha256": ids_digest(ids), "limit": spec.limit,
                 "stats": json.loads(json.dumps(sel.stats, ensure_ascii=False, default=str))},
    }


def _identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _ci_s(x: Mapping[str, Any] | None) -> str:
    if not x or x.get("value") is None:
        return "—"
    ci = x.get("ci")
    return f"{_fmt(x['value'])}" + (f" [{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else "")


def readme_text(spec: EvalSpec, ident: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    task = ident["task"]
    prim = metrics["headline"]["upsample"]
    names = ("cIoU", "gIoU", "BIoU") if task == "refseg" else ("abs_rel", "d1", "d1_strict")
    lines = [f"# E3b 评测 · {spec.variant} · {spec.model_key} · seed {spec.seed} · {task}", "",
             f"- 检查点 manifest sha256：{spec.checkpoint.get('manifest_sha256')}",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；{spec.dtype}；tiny = {spec.tiny}）",
             f"- 评测集：n = {ident['data']['n']}（id 指纹 {ident['data']['ids_sha256'][:16]}…）",
             f"- 网格 {ident['context']['grid_size']}×{ident['context']['grid_size']}，max_new_tokens = "
             f"{ident['context']['max_new_tokens']}，批大小 {ident['eval']['batch_size']}，允许集合 {len(ident['context']['allowed_ids'])} 个 token",
             f"- 上采样：默认 {prim}；另报 {', '.join(m for m in ident['eval']['upsample']['methods'] if m != prim)}", "",
             f"## 主要数字：单轮（{prim}）", ""]
    for k in names:
        lines.append(f"- {k}：{_ci_s(metrics['headline'][k])}")
    lines += ["", f"## 多轮与匹配计算量的单轮（{prim}；95% CI）", "",
              "| 流程 | " + " | ".join(names) + " | 有效率 | 调用/条 | 轮/条 | 生成 token/条 | 视觉 token/条 | 记账文件 |",
              "|---|" + "---|" * (len(names) + 6)]

    def row(label: str, m: Mapping[str, Any], c: Mapping[str, Any]) -> str:
        pm = c["per_sample_mean"] or {}
        return (f"| {label} | " + " | ".join(_ci_s(m[prim][k]) for k in names) + f" | {_fmt(m[prim]['valid_rate'])} | "
                f"{_fmt(pm.get('n_calls'))} | {_fmt(pm.get('n_rounds'))} | {_fmt(pm.get('generated_tokens'))} | "
                f"{_fmt(pm.get('visual_tokens'))} | {c['file']} |")

    for k, v in metrics["multi_turn"]["by_K"].items():
        lines.append(row(f"多轮 K = {k}", v["metrics"], v["compute"]))
    for k, v in metrics["matched_compute"]["by_K"].items():
        if k != "1":
            lines.append(row(f"匹配的单轮 K = {k}", v["metrics"], v["compute"]))
    ov = ident["eval"].get("k_values_override")
    if ov:  # DEV-0039：只跑了配置 K 那一列的前缀（不覆盖时没有这一行，README 与原来逐字节相同）
        lines += ["", f"- 本次只跑 K = {ov['k_values']}（--k-values 覆盖配置的 {ov['config_k_values']}；{ov['basis']}）"]
    dec = ident["eval"].get("decode")
    if dec:  # DEV-0041 §2：结构强制（restricted 时没有这一行，README 与原来逐字节相同）
        lines += ["", f"- 解码：{dec['mode']}（{dec['rule']}）"]
    mcr = ident["eval"].get("merge_criteria")
    if mcr:  # DEV-0063：判据 e5-v2（v1 时没有这一行，README 与原来逐字节相同）
        lines += ["", f"- 评测前合并核对的判据：{mcr['version']}（{mcr['rule']}；{mcr['basis']}）"]
    lines += ["", "- K = 1 的多轮与单轮是同一次调用（逐个相同）；匹配的单轮 = 贪心 + K − 1 次受限采样，逐格合成（规则见 src/e3/evaluate.py）",
              "- GPU 墙钟（call_time_s）与端到端墙钟（wall_time_s）在记账文件里（compute*.json）；本表只列由输出决定的计数",
              "", "## 文件", "",
              "- predictions.jsonl（第 1 轮 = 单轮）、predictions_round<r>.jsonl、predictions_sample<s>.jsonl：逐条输出；scores.jsonl：逐条指标；",
              "- metrics.json：全部汇总；compute*.json：各流程的记账（compute_by_stage.json 是每轮单独的）；config.yaml：运行身份；",
              "  meta.json；logs/：墙钟与每次调用的记录。", ""]
    return "\n".join(lines)


def score_all(task: str, samples: Sequence[Sample], codes: Sequence[Mapping[str, str | None]], ctx: TaskContext,
              settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    payloads = [{"task": task, "sample": s, "codes": dict(c), "n": ctx.n, "methods": list(settings["upsample"]["methods"]),
                 "threshold": settings["upsample"]["threshold"], "min_valid_fraction": ctx.min_valid_fraction}
                for s, c in zip(samples, codes)]
    w = int(settings["score_workers"])
    if w <= 0 or len(payloads) <= 1:
        return [ES.score_payload(p) for p in payloads]
    with ProcessPoolExecutor(max_workers=w, mp_context=multiprocessing.get_context("spawn")) as ex:
        return list(ex.map(ES.score_payload, payloads, chunksize=max(1, min(16, len(payloads) // (4 * w) or 1))))


def finalize(leaf: Path, spec: EvalSpec, ident: Mapping[str, Any], ctx: TaskContext, settings: Mapping[str, Any],
             samples: Sequence[Sample]) -> dict[str, Any]:
    """由各阶段的 predictions 与 logs/timing.jsonl 写 scores.jsonl、metrics.json、compute*.json、README.md。返回 metrics。"""
    ids = [s.id for s in samples]
    stages = read_stages(leaf, settings, ids)
    codes = [procedure_codes(ctx.task, ctx.n, settings, stages, i) for i in range(len(samples))]
    scored = score_all(ctx.task, samples, codes, ctx, settings)
    rows = []
    for s, sc, c in zip(samples, scored, codes):
        if sc["id"] != s.id:
            raise RuntimeError("评分结果与样本顺序不符")
        rows.append({"id": s.id, "group": group_of(ctx.task, s), "gt": sc["gt"], "codes": c, "procs": sc["procs"]})
    (leaf / SCORES).write_text("".join(_jsonl(r) for r in rows), encoding="utf-8")
    timing: dict[tuple[str, str], dict[str, Any]] = {}
    for t in read_jsonl_tolerant(leaf / TIMING, repair=False):
        timing[(t["stage"], t["id"])] = t
    computes = {p: compute_of(p, stages, timing, len(samples)) for p in compute_procs(settings)}
    for p, c in computes.items():
        write_compute_json(leaf / compute_file(p), c)
    by_stage = {st: compute_of_stages([st], stages, timing, len(samples)) for st in stages}
    (leaf / COMPUTE_BY_STAGE).write_text(json.dumps({st: c.to_json_dict() for st, c in by_stage.items()}, indent=2,
                                                    ensure_ascii=False) + "\n", encoding="utf-8")
    body = aggregate(ctx.task, samples, rows, settings, stages, computes)
    for st, c in by_stage.items():  # 每轮（每个阶段）的计数；墙钟在 compute_by_stage.json
        body["stages"][st]["compute"] = {k: getattr(c, k) for k in COUNT_FIELDS}
    metrics = {"task": ctx.task, "model": spec.model_key, "variant": spec.variant, "seed": int(spec.seed),
               "checkpoint_manifest_sha256": spec.checkpoint.get("manifest_sha256"), "n": len(samples), "complete": True,
               "config_sha256": ident["config"]["sha256"], **body}
    if decode_mode(settings) == "structured":  # DEV-0041 §2：只在结构强制时加（restricted 的 metrics.json 逐字节不变）
        metrics["decode"] = settings["decode"]
    if merge_criteria_version(settings) != DEFAULT_MERGE_CRITERIA:  # DEV-0063：只在 e5-v2 时加（v1 的 metrics.json 逐字节不变）
        metrics["merge_criteria"] = settings["merge_criteria"]
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (leaf / README).write_text(readme_text(spec, ident, metrics), encoding="utf-8")
    return metrics


def _open_leaf(leaf: Path, ident: Mapping[str, Any], cfg: E3Config, spec: EvalSpec, settings: Mapping[str, Any], adapter: Any,
               *, resume: bool, argv: Sequence[str], run_extra: Mapping[str, Any] | None
               ) -> tuple[int, Path, dict[str, Any]]:
    """叶目录的续跑规则（模块说明）与本次的运行记录：返回 (第几次运行, logs/run_<k>.json 的路径, 运行记录)。"""
    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = _identity(yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")))
        new = _identity(ident)
        if old != new:
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"{leaf}：续跑的运行身份与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        files = [REPO_ROOT / cfg.path if not Path(cfg.path).is_absolute() else Path(cfg.path)]
        write_run_metadata(leaf, config=ident, seed=int(spec.seed), data_files=files, repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "device": str(getattr(adapter, "device", None)), "score_workers": settings["score_workers"],
                               **dict(run_extra or {}), "stages": [], "finished_utc": None}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_k, run_path, run_log


def run_eval_task(adapter: Any, cfg: E3Config, settings: Mapping[str, Any], spec: EvalSpec, ctx: TaskContext, sel: LoadResult,
                  out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = (),
                  run_extra: Mapping[str, Any] | None = None, log: Callable[[str], None] | None = None,
                  warmup: bool = True) -> dict[str, Any]:
    """一个任务的评测（规则见模块说明）；adapter 是合并后的检查点（load_trained(merge=True)）。返回 metrics。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    samples = list(sel.records)[: spec.limit] if spec.limit else list(sel.records)
    if not samples:
        raise ValueError(f"{ctx.task} 的评测集是空的")
    leaf = leaf_dir(out_dir, spec, ctx.task, settings)
    ident = run_identity(spec, cfg, settings, ctx, sel, samples)
    run_k, run_path, run_log = _open_leaf(leaf, ident, cfg, spec, settings, adapter, resume=resume, argv=argv,
                                          run_extra=run_extra)
    t_all = time.perf_counter()
    if warmup and hasattr(adapter, "warmup") and not (leaf / stage_file(stage_names(settings)[-1])).exists():
        run_log["warmup_wall_s"] = adapter.warmup([load_rgb(samples[0].images[0])],
                                                  batch_size=min(int(settings["batch_size"]), len(samples)))
    for st in stage_names(settings):
        run_log["stages"].append(run_stage(adapter, ctx, settings, leaf, samples, st, run_k=run_k, log=log))
        run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_score = time.perf_counter()
    metrics = finalize(leaf, spec, ident, ctx, settings, samples)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   wall_s=round(time.perf_counter() - t_all, 3), score_wall_s=round(time.perf_counter() - t_score, 3))
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 指令检查（DEV-0041 §1；规则见模块说明）
# ---------------------------------------------------------------------------


def no_instruction_text(*, tiny: bool) -> dict[str, Any]:
    """E3a 的 no_instruction：configs/e3a_train.json 的 eval.leak_checks.no_instruction，由 src/readout/evaluate.resolve_eval 读出
    （同一个定义，不另写）。返回指令与来源（进运行身份）。"""
    from src.readout.config import load_config as load_e3a_config
    from src.readout.evaluate import resolve_eval as e3a_resolve_eval

    c = load_e3a_config()
    ni = e3a_resolve_eval(c, tiny=tiny)["leak_checks"]["no_instruction"]
    if "refseg" not in ni["tasks"]:
        raise ValueError(f"E3a 的 no_instruction 不含 refseg（tasks = {ni['tasks']}），定义对不上")
    return {"instruction": str(ni["instruction"]),
            "source": "configs/e3a_train.json 的 eval.leak_checks.no_instruction.instruction（src/readout/evaluate.resolve_eval）",
            "e3a_config": {"path": c.path, "sha256": c.sha256}}


def swap_index(sample_id: str, n_candidates: int) -> int:
    """swap 的取法（SWAP_RULE）：sha256('visint-e3b-instr-swap|<SWAP_SEED>|<样本 id>') 的前 8 字节 mod 候选数。"""
    if n_candidates < 1:
        raise ValueError("候选数必须 ≥ 1")
    h = hashlib.sha256(f"visint-e3b-instr-swap|{SWAP_SEED}|{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % int(n_candidates)


def swap_candidates(samples: Sequence[Sample], pool: Sequence[Sample]) -> list[tuple[Sample | None, int]]:
    """每条样本的 (换上去的指代句所在的样本或 None, 候选数)：候选 = pool 里同一张图、不同标注 id 的全部指代句（pool 的顺序）。"""
    by_image: dict[Any, list[Sample]] = {}
    for c in pool:
        by_image.setdefault(c.fields["image_id"], []).append(c)
    out: list[tuple[Sample | None, int]] = []
    for s in samples:
        cands = [c for c in by_image.get(s.fields["image_id"], []) if c.fields["ann_id"] != s.fields["ann_id"]]
        out.append((cands[swap_index(s.id, len(cands))] if cands else None, len(cands)))
    return out


@dataclass(frozen=True)
class InstructionCheckPlan:
    """一项指令检查：kind、送进模型时用的上下文（none：模板换成空指令）、原样本（前 N 条）、送进模型的样本（swap：指代句换成候选的，
    没有候选的不在里面）、swap 的新目标（id → 候选样本）、swap_pairs.jsonl 的行、运行身份里的 instruction_check。"""

    kind: str
    ctx: TaskContext
    samples: tuple[Sample, ...]
    called: tuple[Sample, ...]
    targets: Mapping[str, Sample]
    pairs: tuple[Mapping[str, Any], ...]
    identity: Mapping[str, Any]


def instruction_check_plan(kind: str, ctx: TaskContext, samples: Sequence[Sample], pool: Sequence[Sample], *,
                           tiny: bool) -> InstructionCheckPlan:
    if ctx.task != "refseg":
        raise ValueError(f"指令检查只用于 refseg，收到 {ctx.task!r}")
    base = {"kind": kind, "n_eval": len(samples)}
    if kind == "none":
        ni = no_instruction_text(tiny=tiny)
        cctx = dataclasses.replace(ctx, grid_template=ni["instruction"])
        ident = {**base, "rule": NONE_RULE, **ni, "n_called": len(samples), "n_skipped": 0}
        return InstructionCheckPlan("none", cctx, tuple(samples), tuple(samples), {}, (), ident)
    if kind != "swap":
        raise ValueError(f"指令检查只有 {INSTRUCTION_CHECKS}，收到 {kind!r}")
    called, targets, pairs = [], {}, []
    for s, (c, nc) in zip(samples, swap_candidates(samples, pool)):
        pairs.append({"id": s.id, "image_id": s.fields["image_id"], "ann_id": s.fields["ann_id"], "n_candidates": nc,
                      "skipped": c is None, "candidate_id": None if c is None else c.id,
                      "candidate_ann_id": None if c is None else c.fields["ann_id"],
                      "expression": None if c is None else str(c.fields["expression"]).strip()})
        if c is None:
            continue
        called.append(dataclasses.replace(s, fields={**s.fields, "expression": c.fields["expression"]}))
        targets[s.id] = c
    ident = {**base, "rule": SWAP_RULE, "seed": SWAP_SEED, "n_called": len(called), "n_skipped": len(samples) - len(called),
             "pool": {"n": len(pool), "ids_sha256": ids_digest([p.id for p in pool])},
             "pairs_sha256": hashlib.sha256("".join(_jsonl(p) for p in pairs).encode("utf-8")).hexdigest()}
    return InstructionCheckPlan("swap", ctx, tuple(samples), tuple(called), targets, tuple(pairs), ident)


NORMAL_SAME_KEYS = ("task", "model", "checkpoint", "context", "tokens", "grid_code", "config")
NORMAL_SAME_EVAL = ("version", "batch_size", "upsample", "decode", "merge_criteria")  # merge_criteria：DEV-0063，判据不同不配对


def _read_jsonl_any(leaf: Path, name: str) -> tuple[list[dict[str, Any]], Path] | None:
    """<name> 或服务器压缩后的 <name>.gz（eval_lib.sh 把大于 20 MB 的文件压缩）；都没有时返回 None。"""
    p = leaf / name
    if p.is_file():
        return read_jsonl_tolerant(p, repair=False), p
    gz = leaf / f"{name}.gz"
    if gz.is_file():
        raw = gzip.decompress(gz.read_bytes()).decode("utf-8")
        return [json.loads(x) for x in raw.splitlines() if x.strip()], gz
    return None


def load_normal_side(leaf: str | Path, ident: Mapping[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    """正常评测那一侧（模块说明）：核对运行身份与样本顺序，返回前 N 条的单轮网格、它的逐样本评分（有的话）与指纹。"""
    leaf = Path(leaf)
    if not (leaf / CONFIG).is_file():
        raise FileNotFoundError(f"正常评测的叶目录 {leaf} 里没有 {CONFIG}（--normal-leaf 指向已有的 <任务>/ 叶目录）")
    raw_cfg = (leaf / CONFIG).read_bytes()
    old = _identity(yaml.safe_load(raw_cfg.decode("utf-8")))
    new = _identity(ident)
    bad = [k for k in NORMAL_SAME_KEYS if old.get(k) != new.get(k)]
    bad += [f"eval.{k}" for k in NORMAL_SAME_EVAL if (old.get("eval") or {}).get(k) != (new.get("eval") or {}).get(k)]
    if bad:
        raise ValueError(f"正常评测 {leaf} 的运行身份与本次不同（{bad}），不能配对")
    got = _read_jsonl_any(leaf, PREDICTIONS)
    if got is None:
        raise FileNotFoundError(f"{leaf} 里没有 {PREDICTIONS}（或 .gz）")
    preds, ppath = got
    have = [r["id"] for r in preds]
    if have[: len(ids)] != list(ids):
        raise ValueError(f"正常评测 {leaf} 的样本顺序的前 {len(ids)} 条不是本次的 {len(ids)} 条（它有 {len(have)} 条），不能逐条配对")
    sc = _read_jsonl_any(leaf, SCORES)
    stored = {r["id"]: r["procs"]["single"] for r in sc[0]} if sc is not None else {}
    return {"codes": {r["id"]: r["grid"] for r in preds[: len(ids)]}, "stored": stored,
            "fingerprint": {"config_sha256": hashlib.sha256(raw_cfg).hexdigest(), "predictions_file": ppath.name,
                            "predictions_sha256": hashlib.sha256(ppath.read_bytes()).hexdigest(), "n": len(preds),
                            "decode": decode_mode(old.get("eval") or {})}}


def _instr_block(rows: Sequence[Mapping[str, Any]], target: str, m: str) -> dict[str, Any]:
    rr = [{"procs": {"check": r["targets"][target]["check"], "normal": r["targets"][target]["normal"]}} for r in rows]
    pa = [x["procs"]["normal"][m] for x in rr]
    pb = [x["procs"]["check"][m] for x in rr]
    return {"check": _seg_block(rr, "check", m), "normal": _seg_block(rr, "normal", m),
            "paired_normal_minus_check": {
                "gIoU": _paired([x["iou"] for x in pa], [x["iou"] for x in pb]),
                "cIoU": _paired([[x["inter"], x["union"]] for x in pa], [[x["inter"], x["union"]] for x in pb],
                                statistic=ratio_of_sums),
                "BIoU": _paired([x["biou"] for x in pa], [x["biou"] for x in pb])}}


INSTR_DEFINITION = ("配对差 = 正常 − 检查（同一组 id、同一套评分，配对 bootstrap）。original：两侧都对原来的物体评分（模型跟着指令走时"
                    "差为正）；swapped（只有 swap）：两侧都对换上去的物体评分（跟着指令走时检查一侧更高，差为负）。只用默认上采样")


def finalize_instruction_check(leaf: Path, spec: EvalSpec, ident: Mapping[str, Any], ctx: TaskContext,
                               settings: Mapping[str, Any], plan: InstructionCheckPlan,
                               normal: Mapping[str, Any]) -> dict[str, Any]:
    """由 predictions.jsonl 与正常评测的网格写 scores.jsonl、metrics.json、compute.json、README.md（模块说明）。返回 metrics。"""
    prim = settings["upsample"]["primary"]
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if [r["id"] for r in recs] != [s.id for s in plan.called]:
        raise RuntimeError(f"{PREDICTIONS} 不完整或顺序不对（{len(recs)}/{len(plan.called)} 条）")
    check_code = {r["id"]: r["grid"] for r in recs}
    tsamples, tcodes, keys = [], [], []
    for s in plan.samples:
        if s.id not in check_code:
            continue
        codes = {"check": check_code[s.id], "normal": normal["codes"][s.id]}
        for target, ts in (("original", s), ("swapped", plan.targets.get(s.id))):
            if ts is None:
                continue
            tsamples.append(ts)
            tcodes.append(codes)
            keys.append((s, target))
    one = {**settings, "upsample": {**settings["upsample"], "methods": [prim]}}
    scored = score_all(ctx.task, tsamples, tcodes, ctx, one)
    rows: dict[str, dict[str, Any]] = {}
    n_verified = 0
    for (s, target), ts, sc, codes in zip(keys, tsamples, scored, tcodes):
        if sc["id"] != ts.id:
            raise RuntimeError("评分结果与样本顺序不符")
        row = rows.setdefault(s.id, {"id": s.id, "group": group_of(ctx.task, s), "codes": codes, "targets": {}})
        if target == "swapped":
            row["candidate_id"] = ts.id
        row["targets"][target] = {"gt": sc["gt"], "check": sc["procs"]["check"], "normal": sc["procs"]["normal"]}
        if target == "original" and s.id in normal["stored"]:
            st = normal["stored"][s.id]
            want = json.loads(json.dumps({k: st.get(k) for k in ("valid", "grid", prim)}))
            got = json.loads(json.dumps({k: sc["procs"]["normal"].get(k) for k in ("valid", "grid", prim)}))
            if want != got:
                raise RuntimeError(f"{s.id}：正常评测的网格按本次规则重算的读数与它的 scores.jsonl 不同（{want} vs {got}），配对可能错位")
            n_verified += 1
    rows_l = [rows[s.id] for s in plan.samples if s.id in rows]
    (leaf / SCORES).write_text("".join(_jsonl(r) for r in rows_l), encoding="utf-8")
    timing: dict[tuple[str, str], dict[str, Any]] = {}
    for t in read_jsonl_tolerant(leaf / TIMING, repair=False):
        timing[(t["stage"], t["id"])] = t
    comp = compute_of_stages(["round1"], {"round1": recs}, timing, len(recs))
    write_compute_json(leaf / COMPUTE, comp)
    targets = ["original"] + (["swapped"] if plan.kind == "swap" else [])
    ic = {k: v for k, v in plan.identity.items() if k not in ("pool", "pairs_sha256", "e3a_config")}
    metrics: dict[str, Any] = {
        "task": ctx.task, "model": spec.model_key, "variant": spec.variant, "seed": int(spec.seed),
        "checkpoint_manifest_sha256": spec.checkpoint.get("manifest_sha256"), "n": len(plan.samples), "complete": True,
        "config_sha256": ident["config"]["sha256"],
        "instruction_check": ic,
        "upsample": prim,
        "definition": INSTR_DEFINITION,
        "normal": {**normal["fingerprint"], "n_paired": len(rows_l), "n_verified_against_scores": n_verified,
                   "pairing": "按 id；正常一侧取已有结果的前 N 条（同一检查点、同一批配置），不重跑。批组成：none 与正常评测逐批同一组样本；"
                              "swap 跳过样本后按剩下的样本连续成批，与正常评测的批不再一一对应，跳过的不进配对"},
        "targets": {t: _instr_block(rows_l, t, prim) for t in targets},
        "stages": stage_diagnostics({"round1": recs}),
        "compute": {k: getattr(comp, k) for k in COUNT_FIELDS},
    }
    if decode_mode(settings) == "structured":
        metrics["decode"] = settings["decode"]
    if merge_criteria_version(settings) != DEFAULT_MERGE_CRITERIA:  # DEV-0063：只在 e5-v2 时加（v1 的输出逐字节不变）
        metrics["merge_criteria"] = settings["merge_criteria"]
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (leaf / README).write_text(instruction_readme(spec, ident, metrics), encoding="utf-8")
    return metrics


def instruction_readme(spec: EvalSpec, ident: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    ic = metrics["instruction_check"]
    prim = metrics["upsample"]
    lines = [f"# E3b 指令检查 · {ic['kind']} · {spec.variant} · {spec.model_key} · seed {spec.seed} · {ident['task']}", "",
             f"- 检查点 manifest sha256：{spec.checkpoint.get('manifest_sha256')}",
             f"- 规则：{ic['rule']}",
             f"- 样本：前 {ic['n_eval']} 条，送进模型 {ic['n_called']} 条，跳过 {ic['n_skipped']} 条；配对 {metrics['normal']['n_paired']} 条",
             f"- 正常评测一侧：{metrics['normal']['predictions_file']}（sha256 {metrics['normal']['predictions_sha256'][:16]}…；"
             f"解码 {metrics['normal']['decode']}）；与它的 scores.jsonl 核对一致 {metrics['normal']['n_verified_against_scores']} 条",
             f"- 上采样：{prim}；{metrics['definition']}", ""]
    if metrics.get("decode"):
        lines += [f"- 解码：{metrics['decode']['mode']}（{metrics['decode']['rule']}）", ""]
    if metrics.get("merge_criteria"):  # DEV-0063（v1 时没有这一行）
        lines += [f"- 评测前合并核对的判据：{metrics['merge_criteria']['version']}（{metrics['merge_criteria']['basis']}）", ""]
    lines += ["| 目标 | 检查 cIoU | 正常 cIoU | 检查 gIoU | 正常 gIoU | 正常 − 检查 cIoU | 正常 − 检查 gIoU |", "|---|---|---|---|---|---|---|"]

    def pd(x: Mapping[str, Any] | None) -> str:
        if not x:
            return "—"
        return f"{_fmt(x['estimate'])} [{_fmt(x['low'])}, {_fmt(x['high'])}]"

    for t, b in metrics["targets"].items():
        p = b["paired_normal_minus_check"]
        lines.append(f"| {t} | {_ci_s(b['check']['cIoU'])} | {_ci_s(b['normal']['cIoU'])} | {_ci_s(b['check']['gIoU'])} | "
                     f"{_ci_s(b['normal']['gIoU'])} | {pd(p['cIoU'])} | {pd(p['gIoU'])} |")
    lines += ["", "## 文件", "", "- predictions.jsonl：本次的单轮输出；scores.jsonl：每条在各目标上两侧的统计；metrics.json：汇总；",
              "  swap_pairs.jsonl（swap）：每条样本的候选或跳过；compute.json；config.yaml：运行身份；logs/。", ""]
    return "\n".join(lines)


def run_instruction_check(adapter: Any, cfg: E3Config, settings: Mapping[str, Any], spec: EvalSpec, ctx: TaskContext,
                          sel: LoadResult, out_dir: str | Path, *, check: str, normal_leaf: str | Path | None = None,
                          resume: bool = False, argv: Sequence[str] = (), run_extra: Mapping[str, Any] | None = None,
                          log: Callable[[str], None] | None = None, warmup: bool = True) -> dict[str, Any]:
    """一项指令检查（DEV-0041 §1，规则见模块说明）：只跑单轮（settings 的 k_values 必须是 [1]）。返回 metrics。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if ctx.task != "refseg":
        raise ValueError(f"指令检查只用于 refseg，收到 {ctx.task!r}")
    if list(settings["k_values"]) != [1]:
        raise ValueError(f"指令检查只跑单轮：k_values 必须是 [1]（--k-values 1），收到 {settings['k_values']}")
    pool = list(sel.records)
    samples = pool[: spec.limit] if spec.limit else pool
    if not samples:
        raise ValueError(f"{ctx.task} 的评测集是空的")
    plan = instruction_check_plan(check, ctx, samples, pool, tiny=spec.tiny)
    if not plan.called:
        raise ValueError(f"{check}：{len(samples)} 条样本没有一条能做这项检查（全部没有候选）")
    leaf = leaf_dir(out_dir, spec, ctx.task, settings, check)
    nleaf = Path(normal_leaf) if normal_leaf is not None else leaf_dir(out_dir, spec, ctx.task, settings)
    ident = run_identity(spec, cfg, settings, ctx, sel, samples)
    normal = load_normal_side(nleaf, ident, [s.id for s in samples])
    ident["instruction_check"] = dict(plan.identity)
    ident["normal"] = dict(normal["fingerprint"])
    run_k, run_path, run_log = _open_leaf(leaf, ident, cfg, spec, settings, adapter, resume=resume, argv=argv,
                                          run_extra={**dict(run_extra or {}), "normal_leaf": str(nleaf)})
    if plan.pairs:
        (leaf / SWAP_PAIRS).write_text("".join(_jsonl(p) for p in plan.pairs), encoding="utf-8")
    t_all = time.perf_counter()
    if warmup and hasattr(adapter, "warmup") and not (leaf / PREDICTIONS).exists():
        run_log["warmup_wall_s"] = adapter.warmup([load_rgb(plan.called[0].images[0])],
                                                  batch_size=min(int(settings["batch_size"]), len(plan.called)))
    run_log["stages"].append(run_stage(adapter, plan.ctx, settings, leaf, list(plan.called), "round1", run_k=run_k, log=log))
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_score = time.perf_counter()
    metrics = finalize_instruction_check(leaf, spec, ident, ctx, settings, plan, normal)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   wall_s=round(time.perf_counter() - t_all, 3), score_wall_s=round(time.perf_counter() - t_score, 3))
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


__all__ = [
    "DECODE_MODES",
    "EvalPaths",
    "EvalSpec",
    "INSTRUCTION_CHECKS",
    "InstructionCheckPlan",
    "TaskContext",
    "chunk_plan",
    "compute_of",
    "compute_procs",
    "config_k_values",
    "decode_mode",
    "finalize",
    "finalize_instruction_check",
    "grid_structure",
    "instruction_check_plan",
    "k_values_prefix",
    "leaf_dir",
    "leaf_name",
    "load_eval_set",
    "load_normal_side",
    "make_context",
    "merge_criteria_version",
    "model_dir",
    "multi_proc",
    "no_instruction_text",
    "procedure_codes",
    "procedure_stages",
    "procedures",
    "resolve_eval",
    "run_eval_task",
    "run_instruction_check",
    "run_stage",
    "stage_file",
    "stage_names",
    "structured_generate",
    "swap_candidates",
    "swap_index",
    "vote",
]
