"""E5 训练配置（configs/e5_train.json；DEV-0037）：读取、结构核对、sha256、生效参数的展开。

配置是运行身份的一部分：它的 sha256、E3b 配置（configs/e3b_train.json：词表、LoRA 与优化器的结构、受限 logits 的规则）的 sha256，
与本次实际生效的各项（resolve_run 的结果）一起写进 config.yaml，续跑时必须完全相同。
- resolve_run：以 E3b 网格监督的生效参数（src/e3/config.resolve_run(supervision = grid)）为底，换上 E5 的几项（training 一节：
  epoch、批、学习率、warmup 比例、检查点间隔、数据线程；--tiny 时按 training.tiny 覆盖），步数由数据量档与 epoch 算出；
  去掉 E3b 专有的键（mix、correction_fraction），supervision / variant 记为 e5。Trainer（src/e3/train.Trainer）读的键都在。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.e3.config import CONFIG_PATH as E3B_CONFIG_PATH
from src.e3.config import E3Config
from src.e3.config import load_config as load_e3b_config
from src.e3.config import resolve_run as resolve_e3b_run
from src.e5.protocol import E5_TASKS
from src.e5.token_grid import PromptFormat

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "e5_train.json"
SCHEMA = 1
STARTS: tuple[str, ...] = ("base_warmup", "e3b", "e3a", "base")
GRID_TOKEN_SOURCES = ("from_checkpoint", "add")
START_LORA_MODES = ("merge_then_new_lora",)
MAX_DATA_SEED = 1 << 16  # 序号 = 数据种子 × seed_stride + j < 2^48（src/synth/constants.SCENE_SEED_STRIDE）


class E5ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class E5TrainConfig:
    path: str
    sha256: str
    doc: Mapping[str, Any]
    e3b: E3Config

    @property
    def grid_n(self) -> int:
        return self.e3b.grid_n

    def tiers(self, tiny: bool) -> tuple[int, ...]:
        d = self.doc["data"]
        return tuple(int(x) for x in (d["tiny"]["tiers"] if tiny else d["tiers"]))

    @property
    def seed_stride(self) -> int:
        return int(self.doc["data"]["seed_stride"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.doc["training"]["seeds"])

    def start(self, kind: str) -> Mapping[str, Any]:
        if kind not in STARTS:
            raise E5ConfigError(f"未知起点 {kind!r}；可选 {STARTS}")
        return self.doc["starts"][kind]

    def prompt_format(self) -> PromptFormat:
        return PromptFormat.from_configs(self.doc["format"], self.e3b.doc)

    def identity(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "version": self.doc["version"],
                "e3b_config": {"path": self.e3b.path, "sha256": self.e3b.sha256, "version": self.e3b.doc["version"]}}


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)


def _posint(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, int) and v >= 1


def load_train_config(path: str | Path = CONFIG_PATH, e3b_path: str | Path | None = None) -> E5TrainConfig:
    """读 E5 训练配置与它引用的 E3b 配置（e3b_path 不给时取配置里的 e3b_config），核对结构。"""
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA:
        raise E5ConfigError(f"{p}：schema {doc.get('schema')!r} 不是 {SCHEMA}")
    e3p = Path(e3b_path) if e3b_path is not None else REPO_ROOT / doc["e3b_config"]
    e3b = load_e3b_config(e3p)
    bad: list[str] = []
    starts = doc.get("starts", {})
    if set(k for k in starts if k != "basis") != set(STARTS):
        bad.append(f"starts 必须恰好是 {STARTS}")
    for k in STARTS:
        s = starts.get(k, {})
        if s.get("grid_tokens") not in GRID_TOKEN_SOURCES or not isinstance(s.get("checkpoint"), bool):
            bad.append(f"starts.{k}：grid_tokens 必须是 {GRID_TOKEN_SOURCES} 之一，checkpoint 必须是 bool")
        if s.get("checkpoint") and not s.get("variant"):
            bad.append(f"starts.{k}：有检查点的起点要写 variant")
    if not starts.get("base", {}).get("checkpoint") is False:
        bad.append("starts.base 不能有检查点")
    if doc.get("start_lora", {}).get("mode") not in START_LORA_MODES:
        bad.append(f"start_lora.mode 只实现了 {START_LORA_MODES}")
    d = doc.get("data", {})
    n5 = len(E5_TASKS)
    for name, tiers in (("data.tiers", d.get("tiers")), ("data.tiny.tiers", (d.get("tiny") or {}).get("tiers"))):
        if not isinstance(tiers, list) or not tiers or any(not _posint(x) or x % n5 for x in tiers) or tiers != sorted(set(tiers)):
            bad.append(f"{name} 必须是严格递增、能被 {n5} 整除的正整数列表，收到 {tiers!r}")
    if d.get("split") != "train":
        bad.append("data.split 必须是 train")
    stride = d.get("seed_stride")
    if not _posint(stride) or stride * MAX_DATA_SEED > (1 << 48):
        bad.append(f"data.seed_stride 必须是正整数且 × {MAX_DATA_SEED} ≤ 2^48")
    elif max(d.get("tiers") or [0]) // n5 > stride:
        bad.append("data.seed_stride 必须不小于每类的最大条数（不同数据种子的序号区间不相交）")
    t = doc.get("training", {})
    for k in ("epochs", "micro_batch_size", "grad_accum_steps", "ckpt_every"):
        if not _posint(t.get(k)) or not _posint((t.get("tiny") or {}).get(k, 1)):
            bad.append(f"training.{k}（与 tiny 里的同名项）必须是正整数")
    lr = t.get("lr", {})
    if set(lr) != {"lora", "token_rows"} or any(not (isinstance(v, (int, float)) and v > 0) for v in lr.values()):
        bad.append("training.lr 必须恰好有 lora、token_rows 两个正数")
    if not 0.0 <= float(t.get("warmup_ratio", -1)) < 1.0:
        bad.append("training.warmup_ratio 必须在 [0, 1)")
    seeds = t.get("seeds", [])
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        bad.append("training.seeds 必须是不重复的非负整数")
    e = doc.get("eval", {})
    npb = e.get("n_per_bin", {})
    if set(npb) != set(E5_TASKS) or any(not _posint(v) for v in npb.values()):
        bad.append(f"eval.n_per_bin 必须对 {E5_TASKS} 各给一个正整数")
    if e.get("subset") not in ("eval", "dev"):
        bad.append("eval.subset 必须是 eval 或 dev")
    if not _posint(e.get("batch_size")) or not _posint((e.get("tiny") or {}).get("batch_size")):
        bad.append("eval.batch_size（与 tiny 里的）必须是正整数")
    mc = e.get("merge_check", {})
    et = e.get("tiny") or {}
    if not _posint(mc.get("n_per_task")) or mc.get("subset") != "dev" or not _posint(mc.get("batch_size")) or \
            not _posint(et.get("merge_check_n_per_task")) or not _posint(et.get("merge_check_batch_size")):
        bad.append("eval.merge_check 要 n_per_task、batch_size（正整数）与 subset = dev（val 划分），tiny 另给 merge_check_n_per_task、"
                   "merge_check_batch_size")
    if bad:
        raise E5ConfigError(f"{p} 不合格：\n- " + "\n- ".join(bad))
    PromptFormat.from_configs(doc["format"], e3b.doc)  # 结尾句、模板、映射
    return E5TrainConfig(path=_rel(p), sha256=hashlib.sha256(raw).hexdigest(), doc=doc, e3b=e3b)


def per_task(n_train: int) -> int:
    """数据量档（五类合计）→ 每类条数。"""
    n5 = len(E5_TASKS)
    if not _posint(n_train) or n_train % n5:
        raise E5ConfigError(f"数据量 {n_train!r} 必须是 {n5} 的正整数倍（五类等量）")
    return n_train // n5


def steps_for(n_examples: int, epochs: int, batch_size: int) -> int:
    """固定 epoch：步数 = ceil(epochs × 训练集条数 / 批大小)（最后一步可能不满）。"""
    return int(math.ceil(int(epochs) * int(n_examples) / int(batch_size)))


def resolve_run(cfg: E5TrainConfig, *, n_examples: int, tiny: bool) -> dict[str, Any]:
    """本次运行实际生效的训练参数（写进 config.yaml；规则见模块说明）。n_examples = 训练集条数（每类 × 任务数）。"""
    base = resolve_e3b_run(cfg.e3b, supervision="grid", tiny=tiny)
    t = cfg.doc["training"]
    over = dict(t["tiny"]) if tiny else {}
    rc = {k: copy.deepcopy(v) for k, v in base.items() if k not in ("mix", "correction_fraction", "supervision", "variant")}
    rc["supervision"] = "e5"
    rc["variant"] = "e5"
    rc["epochs"] = int(over.get("epochs", t["epochs"]))
    rc["micro_batch_size"] = int(over.get("micro_batch_size", t["micro_batch_size"]))
    rc["grad_accum_steps"] = int(over.get("grad_accum_steps", t["grad_accum_steps"]))
    rc["batch_size"] = rc["micro_batch_size"] * rc["grad_accum_steps"]
    rc["ckpt_every"] = int(over.get("ckpt_every", t["ckpt_every"]))
    rc["data_workers"] = int(over.get("data_workers", t["data_workers"]))
    rc["n_examples"] = int(n_examples)
    rc["steps"] = steps_for(n_examples, rc["epochs"], rc["batch_size"])
    rc["optim"]["lora"]["lr"] = float(t["lr"]["lora"])
    rc["optim"]["token_rows"]["lr"] = float(t["lr"]["token_rows"])
    rc["optim"]["warmup_ratio"] = float(t["warmup_ratio"])
    rc["warmup_steps"] = int(over["warmup_steps"]) if (tiny and "warmup_steps" in over) else \
        int(math.ceil(float(t["warmup_ratio"]) * rc["steps"]))
    rc["warmup_steps"] = min(rc["warmup_steps"], rc["steps"])
    rc["start_lora"] = cfg.doc["start_lora"]["mode"]
    if rc["steps"] < 1:
        raise E5ConfigError("步数必须 ≥ 1")
    return rc


__all__ = ["CONFIG_PATH", "E3B_CONFIG_PATH", "E5ConfigError", "E5TrainConfig", "MAX_DATA_SEED", "STARTS", "load_train_config",
           "per_task", "resolve_run", "steps_for"]
