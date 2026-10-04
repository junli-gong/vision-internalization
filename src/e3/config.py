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

# grid = E3b（GT 网格 + 修正对）；text = C3（稀疏文本）；warmup = 网格格式预热（纯文字给坐标 → 网格，DEV-0035 §2）
SUPERVISIONS: tuple[str, ...] = ("grid", "text", "warmup")
VARIANT_OF = {"grid": "e3b", "text": "c3", "warmup": "warmup"}  # 输出目录里的变体名（每种监督的默认变体）
C1PRIME = "c1prime"  # C1′（DEV-0035 §1）：网格监督、冻结 LLM、只训练网格 token 的行（配置的 c1prime 节）
VARIANTS: tuple[str, ...] = tuple(VARIANT_OF.values()) + (C1PRIME,)
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
    if "warmup" in doc:  # 网格格式预热（DEV-0035）：结构与模板一次核对（规则在 src/e3/warmup.py）
        from src.e3.warmup import WarmupConfigError, WarmupSpec

        try:
            for tiny in (False, True):
                WarmupSpec.from_doc(doc, n=check_grid_size(doc["grid"]["size"]), tiny=tiny)
        except (WarmupConfigError, KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"warmup 节不合规：{e!r}") from None
    rel = str(p.relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)
    return E3Config(path=rel, sha256=hashlib.sha256(raw).hexdigest(), doc=doc)


def resolve_variant(supervision: str, variant: str | None) -> str:
    """变体名：不给时取监督的默认变体（VARIANT_OF）；c1prime 只配网格监督。"""
    if supervision not in SUPERVISIONS:
        raise ConfigError(f"supervision 必须是 {SUPERVISIONS} 之一，收到 {supervision!r}")
    v = VARIANT_OF[supervision] if variant is None else variant
    if v != VARIANT_OF[supervision] and not (v == C1PRIME and supervision == "grid"):
        raise ConfigError(f"变体 {v!r} 不能配 --supervision {supervision}（{C1PRIME} 只配 grid；其余变体由监督决定：{VARIANT_OF}）")
    return v


def resolve_run(cfg: E3Config, *, supervision: str, tiny: bool, variant: str | None = None) -> dict[str, Any]:
    """本次运行实际生效的训练参数（写进 config.yaml）：--tiny 时按 cfg.tiny 覆盖规模参数。
    变体 c1prime（DEV-0035 §1）：另记 frozen_llm = true、lora = null，optim 去掉 lora 一组，token 行必须是 fp32 主副本；其余同 E3b。
    监督 warmup（DEV-0035 §2）：steps = warmup.stop.max_steps（--tiny 取 warmup.tiny.max_steps），mix 取 warmup.mix，没有修正对；
    另记 warmup 一节（停止规则、留出集条数、留出集的批大小）。e3b、c3 的结果与原来逐项相同（不加键）。"""
    v = resolve_variant(supervision, variant)
    d = cfg.doc
    out = {
        "supervision": supervision,
        "variant": v,
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
    if v == C1PRIME:
        if out["optim"]["token_rows"]["mode"] != "fp32_master_rows":
            raise ConfigError("C1′ 的 token 行必须是 fp32_master_rows（配置的 c1prime 节）：没有 LoRA 时不挂梯度掩码，dense_param 会更新"
                              "提示 token 的嵌入行")
        out["lora"] = None
        out["optim"] = {k: x for k, x in out["optim"].items() if k != "lora"}
        out["frozen_llm"] = True
    if supervision == "warmup":
        if "warmup" not in d:
            raise ConfigError("配置里没有 warmup 节")
        from src.e3.warmup import WarmupSpec

        ws = WarmupSpec.from_doc(d, n=cfg.grid_n, tiny=tiny)
        out["steps"] = int(ws.stop["max_steps"])
        out["mix"] = dict(ws.mix)
        out["warmup"] = {"version": str(d["warmup"]["version"]), **dict(ws.stop), "heldout_n": ws.heldout_n,
                         "heldout_batch_size": int(d["eval"]["tiny"]["batch_size"] if tiny else d["eval"]["batch_size"])}
    for k in ("steps", "micro_batch_size", "grad_accum_steps", "ckpt_every"):
        if out[k] < 1:
            raise ConfigError(f"{k} 必须 ≥ 1，收到 {out[k]}")
    out["batch_size"] = out["micro_batch_size"] * out["grad_accum_steps"]
    out["warmup_steps"] = int(d["tiny"]["warmup_steps"]) if tiny else int(math.ceil(float(d["optim"]["warmup_ratio"]) * out["steps"]))
    if not 0 <= out["warmup_steps"] <= out["steps"]:
        raise ConfigError(f"warmup_steps {out['warmup_steps']} 不在 [0, steps]")
    return out


__all__ = ["C1PRIME", "CONFIG_PATH", "ConfigError", "E3Config", "SUPERVISIONS", "TOKEN_ROW_MODES", "VARIANTS", "VARIANT_OF",
           "load_config", "resolve_run", "resolve_variant"]
