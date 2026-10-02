"""E3b 训练的配置（configs/e3b_train.json）：读取、结构核对、sha256、--tiny 的规模覆盖、生效超参的展开。

配置是运行身份的一部分：它的 sha256 与本次实际生效的各项（resolve_run 的结果）一起写进 config.yaml，续跑时必须完全相同。
数字只写在配置里；与常量（GRID_SIZE_E3B、DEPTH_NUM_BINS、MAX_NEW_GRID_TOKENS）的对应由 load_config 核对、tests 钉住。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.e3.tokens import ALLOWED_SCOPES, GRID_TASKS, GridVocab, check_grid_size, vocab_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "e3b_train.json"
SCHEMA = 1

SUPERVISIONS: tuple[str, ...] = ("grid", "text")  # grid = E3b（GT 网格 + 修正对）；text = C3（稀疏文本）
VARIANT_OF = {"grid": "e3b", "text": "c3"}  # 输出目录里的变体名
TOKEN_ROW_MODES: tuple[str, ...] = ("fp32_master_rows", "dense_param")
_TINY_KEYS = ("steps", "micro_batch_size", "grad_accum_steps", "ckpt_every")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class E3Config:
    path: str
    sha256: str
    doc: Mapping[str, Any]

    @property
    def grid_n(self) -> int:
        return check_grid_size(self.doc["grid"]["size"])

    @property
    def vocab(self) -> GridVocab:
        return vocab_from_config(self.doc)

    @property
    def allowed_scope(self) -> str:
        return str(self.doc["allowed_scope"])

    @property
    def enable_thinking(self) -> bool:
        return bool(self.doc["enable_thinking"])

    def task(self, name: str) -> dict[str, Any]:
        if name not in self.doc["tasks"]:
            raise ConfigError(f"未知任务 {name!r}，配置里有 {sorted(self.doc['tasks'])}")
        return dict(self.doc["tasks"][name])

    @property
    def mix_weights(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.doc["mix"]["weights"].items()}

    @property
    def tasks(self) -> tuple[str, ...]:
        """参与混合的任务（权重 > 0），按 GRID_TASKS 的顺序。"""
        w = self.mix_weights
        return tuple(t for t in GRID_TASKS if w.get(t, 0) > 0)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.doc["seeds"])


def load_config(path: str | Path = CONFIG_PATH) -> E3Config:
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA:
        raise ConfigError(f"{p}：schema {doc.get('schema')!r} 不是 {SCHEMA}")
    check_grid_size(doc["grid"]["size"])
    vocab_from_config(doc)  # 名字、个数、预算
    if doc["allowed_scope"] not in ALLOWED_SCOPES:
        raise ConfigError(f"allowed_scope 必须是 {ALLOWED_SCOPES} 之一")
    w = doc["mix"]["weights"]
    bad = [t for t in w if t not in GRID_TASKS or t not in doc["tasks"]]
    if bad or any(isinstance(v, bool) or int(v) != v or int(v) < 0 for v in w.values()) or sum(int(v) for v in w.values()) < 1:
        raise ConfigError(f"mix.weights 必须是已配置任务的非负整数权重、至少一个为正，收到 {w}")
    if not 0.0 <= float(doc["correction"]["fraction"]) <= 1.0:
        raise ConfigError("correction.fraction 必须在 [0, 1]")
    if doc["optim"]["token_rows"]["mode"] not in TOKEN_ROW_MODES:
        raise ConfigError(f"optim.token_rows.mode 必须是 {TOKEN_ROW_MODES} 之一")
    if float(doc["optim"]["token_rows"]["weight_decay"]) != 0.0:
        raise ConfigError("token 行的参数组必须 weight_decay = 0（D45-3）")
    kinds = doc["c3"]["seg_kinds"]
    if set(kinds) - {"box", "polygon"} or sum(int(v) for v in kinds.values()) < 1:
        raise ConfigError(f"c3.seg_kinds 只能有 box / polygon 且权重和 ≥ 1，收到 {kinds}")
    if len(set(int(s) for s in doc["seeds"])) != len(doc["seeds"]):
        raise ConfigError("seeds 有重复")
    rel = str(p.relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)
    return E3Config(path=rel, sha256=hashlib.sha256(raw).hexdigest(), doc=doc)


def resolve_run(cfg: E3Config, *, supervision: str, tiny: bool) -> dict[str, Any]:
    """本次运行实际生效的训练参数（写进 config.yaml）：--tiny 时按 cfg.tiny 覆盖规模参数。"""
    if supervision not in SUPERVISIONS:
        raise ConfigError(f"supervision 必须是 {SUPERVISIONS} 之一，收到 {supervision!r}")
    d = cfg.doc
    out = {
        "supervision": supervision,
        "variant": VARIANT_OF[supervision],
        "steps": int(d["steps"]),
        "micro_batch_size": int(d["batch"]["micro_batch_size"]),
        "grad_accum_steps": int(d["batch"]["grad_accum_steps"]),
        "ckpt_every": int(d["ckpt_every"]),
        "lora": {k: d["lora"][k] for k in ("r", "alpha", "dropout")},
        "optim": {k: copy.deepcopy(v) for k, v in d["optim"].items() if k != "basis"},
        "loss_reduction": d["loss"]["reduction"],
        "gradient_checkpointing": bool(d["model"]["gradient_checkpointing"]),
        "enable_thinking": cfg.enable_thinking,
        "allowed_scope": cfg.allowed_scope,
        "mix": cfg.mix_weights,
        "correction_fraction": float(d["correction"]["fraction"]) if supervision == "grid" else None,
        "data_workers": int(d["data"]["workers"]),
    }
    if tiny:
        t = d["tiny"]
        for k in _TINY_KEYS:
            out[k] = int(t[k])
    for k in ("steps", "micro_batch_size", "grad_accum_steps", "ckpt_every"):
        if out[k] < 1:
            raise ConfigError(f"{k} 必须 ≥ 1，收到 {out[k]}")
    out["batch_size"] = out["micro_batch_size"] * out["grad_accum_steps"]
    out["warmup_steps"] = int(d["tiny"]["warmup_steps"]) if tiny else int(math.ceil(float(d["optim"]["warmup_ratio"]) * out["steps"]))
    if not 0 <= out["warmup_steps"] <= out["steps"]:
        raise ConfigError(f"warmup_steps {out['warmup_steps']} 不在 [0, steps]")
    return out


__all__ = ["CONFIG_PATH", "ConfigError", "E3Config", "SUPERVISIONS", "TOKEN_ROW_MODES", "VARIANT_OF", "load_config",
           "resolve_run"]
