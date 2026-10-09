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

# grid = E3b（GT 网格 + 修正对）；text = C3（稀疏文本）；warmup = 网格格式预热（纯文字给坐标 → 网格，DEV-0035 §2）；
# grid+text = E3b+C3（两条样本流：C3 的文本流与 E3b 的网格流，每步各 mixed.per_step 条，DEV-0042 §1；配置的 mixed 节）
MIXED = "grid+text"
STREAMS: tuple[str, ...] = ("text", "grid")  # E3b+C3 的两条样本流（= 各自的单一监督名）
SUPERVISIONS: tuple[str, ...] = ("grid", "text", "warmup", MIXED)
VARIANT_OF = {"grid": "e3b", "text": "c3", "warmup": "warmup", MIXED: "e3b_c3"}  # 输出目录里的变体名（每种监督的默认变体）
C1PRIME = "c1prime"  # C1′（DEV-0035 §1）：网格监督、冻结 LLM、只训练网格 token 的行（配置的 c1prime 节）
# C3+预热（DEV-0042 §2）：预热监督，起点是 C3 的检查点（--start-checkpoint）：先把 C3 的 LoRA 合并进 base，再做网格格式预热；
# 存成一个检查点（C3 与预热的 LoRA 按秩拼接，src/adapters/checkpoint.write_rank_concat）
C3_WARMUP = "c3_warmup"
C3_WARMUP_PARENT = "c3"  # C3+预热的父检查点必须来自这个变体（叶目录 config.yaml 的 variant）
VARIANTS: tuple[str, ...] = tuple(VARIANT_OF.values()) + (C1PRIME, C3_WARMUP)
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
    if "mixed" in doc:  # E3b+C3（DEV-0042）：两条流的条数、权重、步数的来历一次核对
        for tiny in (False, True):
            mixed_spec(doc, tiny=tiny)
    if "warmup" in doc:  # 网格格式预热（DEV-0035）：结构与模板一次核对（规则在 src/e3/warmup.py）
        from src.e3.warmup import WarmupConfigError, WarmupSpec

        try:
            for tiny in (False, True):
                WarmupSpec.from_doc(doc, n=check_grid_size(doc["grid"]["size"]), tiny=tiny)
        except (WarmupConfigError, KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"warmup 节不合规：{e!r}") from None
    rel = str(p.relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)
    return E3Config(path=rel, sha256=hashlib.sha256(raw).hexdigest(), doc=doc)


def mixed_spec(doc: Mapping[str, Any], *, tiny: bool) -> dict[str, Any]:
    """配置的 mixed 节（E3b+C3）→ 本次生效的两条流（规则见配置的 mixed.basis）。核对：
    - per_step 恰好是 text、grid 两项，正整数；两项之和 = 每步条数（batch.micro_batch_size × batch.grad_accum_steps，tiny 取 tiny 节），
      且各自是微批大小的整数倍（每个微批只含一种监督）；
    - steps × per_step.<监督> = 顶层 steps × 每步条数（全程的文本样本恰好是 C3 的、网格样本恰好是 E3b 的，一条不多一条不少）；
    - loss_weights 恰好是 text 1、grid 1（D83-2：1 : 1 事先固定，不调）；order 是两条流的一个排列。"""
    m = doc["mixed"]
    sec = m["tiny"] if tiny else m
    t = doc["tiny"]
    micro = int(t["micro_batch_size"] if tiny else doc["batch"]["micro_batch_size"])
    batch = micro * int(t["grad_accum_steps"] if tiny else doc["batch"]["grad_accum_steps"])
    ref_steps = int(t["steps"] if tiny else doc["steps"])
    per = sec["per_step"]
    where = "mixed.tiny" if tiny else "mixed"
    if set(per) != set(STREAMS) or any(isinstance(v, bool) or int(v) != v or int(v) < 1 for v in per.values()):
        raise ConfigError(f"{where}.per_step 必须恰好是 {list(STREAMS)} 的正整数条数，收到 {per}")
    per = {s: int(per[s]) for s in STREAMS}
    if sum(per.values()) != batch or any(v % micro for v in per.values()):
        raise ConfigError(f"{where}.per_step {per}：两项之和必须是每步条数 {batch}，且各自是微批大小 {micro} 的整数倍"
                          "（每个微批只含一种监督）")
    steps = sec["steps"]
    if isinstance(steps, bool) or int(steps) != steps or any(int(steps) * per[s] != ref_steps * batch for s in STREAMS):
        raise ConfigError(f"{where}.steps = {steps}：steps × per_step.<监督> 必须等于 C3、E3b 的样本数 {ref_steps} × {batch}"
                          f" = {ref_steps * batch}（逐样本相同，D83-2）")
    w = m["loss_weights"]
    if set(w) != set(STREAMS) or any(isinstance(v, bool) or float(v) != 1.0 for v in w.values()):
        raise ConfigError(f"mixed.loss_weights 必须恰好是 text 1、grid 1（D83-2：两种损失 1 : 1，事先固定，不调），收到 {w}")
    order = [str(x) for x in m["order"]]
    if sorted(order) != sorted(STREAMS):
        raise ConfigError(f"mixed.order 必须是 {list(STREAMS)} 的一个排列，收到 {order}")
    return {"version": str(m["version"]), "per_step": per, "order": order,
            "loss_weights": {s: float(w[s]) for s in STREAMS}, "steps": int(steps),
            "micro_batches": {s: per[s] // micro for s in STREAMS},
            "reference": {"steps": ref_steps, "batch_size": batch, "variants": {"text": VARIANT_OF["text"], "grid": VARIANT_OF["grid"]},
                          "samples_per_stream": ref_steps * batch},
            "schedule_length_ratio": int(steps) / ref_steps,
            "streams": {s: f"第 t 步取 {VARIANT_OF[s]} 同一种子的样本流第 {per[s]}t … {per[s]}t + {per[s] - 1} 个槽位"
                        for s in STREAMS}}


def resolve_variant(supervision: str, variant: str | None) -> str:
    """变体名：不给时取监督的默认变体（VARIANT_OF）；c1prime 只配网格监督，c3_warmup 只配预热监督（还要给起点，见 scripts/train_e3b.py）。"""
    if supervision not in SUPERVISIONS:
        raise ConfigError(f"supervision 必须是 {SUPERVISIONS} 之一，收到 {supervision!r}")
    v = VARIANT_OF[supervision] if variant is None else variant
    if v != VARIANT_OF[supervision] and not (v == C1PRIME and supervision == "grid") and \
            not (v == C3_WARMUP and supervision == "warmup"):
        raise ConfigError(f"变体 {v!r} 不能配 --supervision {supervision}（{C1PRIME} 只配 grid，{C3_WARMUP} 只配 warmup；"
                          f"其余变体由监督决定：{VARIANT_OF}）")
    return v


def resolve_run(cfg: E3Config, *, supervision: str, tiny: bool, variant: str | None = None) -> dict[str, Any]:
    """本次运行实际生效的训练参数（写进 config.yaml）：--tiny 时按 cfg.tiny 覆盖规模参数。
    变体 c1prime（DEV-0035 §1）：另记 frozen_llm = true、lora = null，optim 去掉 lora 一组，token 行必须是 fp32 主副本；其余同 E3b。
    监督 warmup（DEV-0035 §2）：steps = warmup.stop.max_steps（--tiny 取 warmup.tiny.max_steps），mix 取 warmup.mix，没有修正对；
    另记 warmup 一节（停止规则、留出集条数、留出集的批大小）。变体 c3_warmup（DEV-0042 §2）另记 parent 一节（起点的规则；起点检查点的
    sha256 等由训练器写进运行身份）。
    监督 grid+text（E3b+C3，DEV-0042 §1）：其余同 E3b 的网格监督（含修正对比例），steps = mixed.steps（--tiny 取 mixed.tiny.steps），
    warmup_steps 按它算；另记 mixed 一节（mixed_spec）。e3b、c3、c1prime、warmup 的结果与原来逐项相同（不加键）。"""
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
        "correction_fraction": float(d["correction"]["fraction"]) if supervision in ("grid", MIXED) else None,
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
    if supervision == MIXED:
        if "mixed" not in d:
            raise ConfigError("配置里没有 mixed 节")
        ms = mixed_spec(d, tiny=tiny)
        out["steps"] = ms["steps"]
        out["mixed"] = ms
    if supervision == "warmup":
        if "warmup" not in d:
            raise ConfigError("配置里没有 warmup 节")
        from src.e3.warmup import WarmupSpec

        ws = WarmupSpec.from_doc(d, n=cfg.grid_n, tiny=tiny)
        out["steps"] = int(ws.stop["max_steps"])
        out["mix"] = dict(ws.mix)
        out["warmup"] = {"version": str(d["warmup"]["version"]), **dict(ws.stop), "heldout_n": ws.heldout_n,
                         "heldout_batch_size": int(d["eval"]["tiny"]["batch_size"] if tiny else d["eval"]["batch_size"])}
        if v == C3_WARMUP:
            out["parent"] = {"variant": C3_WARMUP_PARENT,
                             "merge": "起点 C3 的 LoRA 在 torch.manual_seed 之前合并进 base（src/e5/stack.merge_start_lora，与 E5 合并起点同一串调用），"
                                      "再加网格 token、注入预热的 LoRA",
                             "save": "检查点 = C3 的 LoRA 与预热的 LoRA 按秩拼接（src/adapters/checkpoint.write_rank_concat；alpha / r 不变），"
                                     "网格 token 行取预热之后的"}
    for k in ("steps", "micro_batch_size", "grad_accum_steps", "ckpt_every"):
        if out[k] < 1:
            raise ConfigError(f"{k} 必须 ≥ 1，收到 {out[k]}")
    out["batch_size"] = out["micro_batch_size"] * out["grad_accum_steps"]
    out["warmup_steps"] = int(d["tiny"]["warmup_steps"]) if tiny else int(math.ceil(float(d["optim"]["warmup_ratio"]) * out["steps"]))
    if not 0 <= out["warmup_steps"] <= out["steps"]:
        raise ConfigError(f"warmup_steps {out['warmup_steps']} 不在 [0, steps]")
    return out


__all__ = ["C1PRIME", "C3_WARMUP", "C3_WARMUP_PARENT", "CONFIG_PATH", "ConfigError", "E3Config", "MIXED", "STREAMS", "SUPERVISIONS",
           "TOKEN_ROW_MODES", "VARIANTS", "VARIANT_OF", "load_config", "mixed_spec", "resolve_run", "resolve_variant"]
