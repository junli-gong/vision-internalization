"""E3b 的评测（DEV-0030 §1；简报 §8.5、§4 第 11 条、§12）：一个检查点 × 一个任务 → <out>/eval/<变体>/<模型>/seed<k>/<任务>/。

**加载**（调用方：scripts/eval_e3b.py）：先跑评测前的合并核对（src/e3/merge_check：check_training_interfaces --checkpoint 的同一套
函数，判据 v1 = D50-3；检查项取配置的 eval.merge_check.checks.grid_eval：merge 加 restricted，restricted 用检查点自己的网格 token 与
本模块的单轮设置，DEV-0032；不通过就不评测），再 factory.load_trained(检查点, merge=True)，attach_vocab 取回网格 token 的布局。

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
"""

from __future__ import annotations

import datetime as dt
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
from src.e3.merge_check import settings_from_config
from src.e3.targets import parse_target, seq_len
from src.e3.tokens import GRID_TASKS, GridTokens
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
                 merge_check_n: int | None = None, k_values: Sequence[int] | None = None) -> dict[str, Any]:
    """configs/e3b_train.json 的 eval 节 → 本次生效的设置（--tiny 时按 eval.tiny 覆盖规模；命令行可再覆盖批大小、评分进程数、
    合并核对的 n，以及只跑配置 K 那一列的前缀：k_values，规则见模块说明）。结构不合规就报错。"""
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
    return out


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


def generate_chunk(ad: Any, ctx: TaskContext, inputs: Sequence[Any], stage: str, chunk: int,
                   settings: Mapping[str, Any]) -> list[Any]:
    """一批的生成：第 r 轮贪心受限解码（generate_restricted）；第 s 次采样走 generate_batch 的受限采样（模块说明）。"""
    if stage.startswith("round"):
        return ad.generate_restricted(list(inputs), list(ctx.allowed), max_new_tokens=ctx.max_new_tokens,
                                      stop_ids=[int(ctx.tokens.stop_id)], enable_thinking=ctx.enable_thinking)
    import torch

    mc = settings["matched_compute"]
    torch.manual_seed(sampling_seed(mc["seed"], stage, chunk))
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


def leaf_dir(out: str | Path, spec: EvalSpec, task: str) -> Path:
    return model_dir(out, spec) / task


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
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (leaf / README).write_text(readme_text(spec, ident, metrics), encoding="utf-8")
    return metrics


def run_eval_task(adapter: Any, cfg: E3Config, settings: Mapping[str, Any], spec: EvalSpec, ctx: TaskContext, sel: LoadResult,
                  out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = (),
                  run_extra: Mapping[str, Any] | None = None, log: Callable[[str], None] | None = None,
                  warmup: bool = True) -> dict[str, Any]:
    """一个任务的评测（规则见模块说明）；adapter 是合并后的检查点（load_trained(merge=True)）。返回 metrics。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    samples = list(sel.records)[: spec.limit] if spec.limit else list(sel.records)
    if not samples:
        raise ValueError(f"{ctx.task} 的评测集是空的")
    leaf = leaf_dir(out_dir, spec, ctx.task)
    ident = run_identity(spec, cfg, settings, ctx, sel, samples)
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


__all__ = [
    "EvalPaths",
    "EvalSpec",
    "TaskContext",
    "chunk_plan",
    "compute_of",
    "compute_procs",
    "config_k_values",
    "finalize",
    "k_values_prefix",
    "leaf_dir",
    "load_eval_set",
    "make_context",
    "model_dir",
    "multi_proc",
    "procedure_codes",
    "procedure_stages",
    "procedures",
    "resolve_eval",
    "run_eval_task",
    "run_stage",
    "stage_file",
    "stage_names",
    "vote",
]
