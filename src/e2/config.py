"""E2 的配置（configs/e2_probes.json）：读取、sha256、层集合的展开、--tiny 的规模覆盖。

配置是运行身份的一部分：它的 sha256 与本次实际生效的各项（层、参考线、任务参数）一起写进 config.yaml，续跑时必须完全相同。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.constants import BOOTSTRAP_N_RESAMPLES

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "e2_probes.json"

TASKS: tuple[str, ...] = ("depth", "semseg", "normals", "refseg")
PROBE_KINDS: tuple[str, ...] = ("linear", "conv")
REFERENCES: tuple[str, ...] = ("post_proj", "pre_proj")
PRE_PROJ = "pre_proj"  # 参考线的特征键；LLM 层的键是整数层号

# --tiny 能覆盖的规模参数（其余一律取正式配置）
_TINY_TASK_KEYS = ("steps", "batch_size", "eval_every", "ckpt_every")


@dataclass(frozen=True)
class E2Config:
    path: str
    sha256: str
    doc: Mapping[str, Any]

    # ----- 输入 -----
    @property
    def dense_prompt(self) -> str:
        return self.doc["inputs"]["dense_prompt"]

    @property
    def refseg_template(self) -> str:
        return self.doc["inputs"]["refseg_template"]

    def order(self, task: str) -> str:
        inp = self.doc["inputs"]
        return inp["refseg_order"] if task == "refseg" else inp["dense_order"]

    def instruction(self, task: str, expression: str | None = None) -> str:
        """这个任务送进 build_inputs 的指令：稠密任务是固定的一句；refseg 把指代句（去首尾空白）填进模板。"""
        if task == "refseg":
            if expression is None or not str(expression).strip():
                raise ValueError("refseg 要给非空的指代句")
            return self.refseg_template.format(expression=str(expression).strip())
        if expression is not None:
            raise ValueError(f"{task} 的指令是固定的一句，不接受指代句")
        return self.dense_prompt

    # ----- 其余 -----
    @property
    def seed(self) -> int:
        return int(self.doc["seed"])

    def task(self, name: str) -> dict[str, Any]:
        if name not in TASKS:
            raise ValueError(f"未知任务 {name!r}，可用：{TASKS}")
        return dict(self.doc["tasks"][name])

    def default_layer_spec(self, model_key: str) -> str:
        lay = self.doc["layers"]
        return lay["per_model"].get(model_key, lay["default"])


def load_config(path: str | Path = CONFIG_PATH) -> E2Config:
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != 1:
        raise ValueError(f"{p}：schema {doc.get('schema')!r} 不是 1")
    missing = [t for t in TASKS if t not in doc["tasks"]]
    if missing:
        raise ValueError(f"{p}：缺少任务 {missing}")
    rel = str(p.relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)
    return E2Config(path=rel, sha256=hashlib.sha256(raw).hexdigest(), doc=doc)


# ---------------------------------------------------------------------------
# 层
# ---------------------------------------------------------------------------


def parse_layer_spec(tokens: Sequence[str] | str) -> str:
    """命令行的 --layers（all / every k）→ 规范写法 "all" 或 "every k"。"""
    toks = tokens.split() if isinstance(tokens, str) else list(tokens)
    if toks == ["all"]:
        return "all"
    if len(toks) == 2 and toks[0] == "every":
        try:
            k = int(toks[1])
        except ValueError:
            raise ValueError(f"--layers every 后面要跟正整数，收到 {toks[1]!r}") from None
        if k < 1:
            raise ValueError(f"--layers every k 的 k 必须 ≥ 1，收到 {k}")
        return "all" if k == 1 else f"every {k}"
    raise ValueError(f"--layers 只接受 all 或 every k，收到 {' '.join(toks)!r}")


def expand_layers(spec: str, num_layers: int, *, always_include_last: bool = True) -> list[int]:
    """规范写法 → 层号列表（升序）。层号约定见 src/adapters/base.py：有效范围 [0, num_layers]。

    every k = range(0, num_layers + 1, k)；always_include_last 时补上 num_layers（最终归一化之后的那一层）。
    """
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError(f"num_layers 必须是正整数，收到 {num_layers!r}")
    spec = parse_layer_spec(spec)
    k = 1 if spec == "all" else int(spec.split()[1])
    out = list(range(0, num_layers + 1, k))
    if always_include_last and out[-1] != num_layers:
        out.append(num_layers)
    return out


def layer_label(key: int | str) -> str:
    """特征键的显示名：整数层号 → "L12"，参考线 → 原名。"""
    return f"L{key}" if isinstance(key, int) else str(key)


# ---------------------------------------------------------------------------
# 一个任务的生效参数
# ---------------------------------------------------------------------------


def resolve_task(cfg: E2Config, task: str, *, tiny: bool) -> dict[str, Any]:
    """这个任务实际生效的参数（写进 config.yaml）：任务条目 + 公共的 optim / feature_norm / probes / upsample / probe_chunk /
    bootstrap 次数；--tiny 时按 cfg.tiny 覆盖规模参数。"""
    t = copy.deepcopy(cfg.task(task))
    out = {
        "task": task,
        "order": cfg.order(task),
        "out_channels": int(t["out_channels"]),
        "steps": int(t["steps"]),
        "batch_size": int(t["batch_size"]),
        "eval_every": int(t["eval_every"]),
        "ckpt_every": int(t["ckpt_every"]),
        "holdout": dict(t["holdout"]),
        "loss": dict(t["loss"]),
        "optim": copy.deepcopy(dict(cfg.doc["optim"])),
        "feature_norm": {k: v for k, v in cfg.doc["feature_norm"].items() if k != "basis"},
        "probes": {k: {kk: vv for kk, vv in v.items() if kk != "basis"} for k, v in cfg.doc["probes"].items()},
        "upsample": {k: v for k, v in cfg.doc["upsample"].items() if k != "basis"},
        "probe_chunk": int(cfg.doc["probe_chunk"]),
        "probe_chunk_max_elements": int(cfg.doc["probe_chunk_max_elements"]),
        "bootstrap_resamples": int(BOOTSTRAP_N_RESAMPLES),
        "n_eval_expected": t.get("n_eval_expected"),
        "eval_subset": t.get("eval_subset"),
    }
    out["optim"].pop("basis", None)
    if tiny:
        ty = cfg.doc["tiny"]
        for k in _TINY_TASK_KEYS:
            out[k] = int(ty[k])
        out["optim"]["warmup_steps"] = int(ty["warmup_steps"])
        out["feature_norm"]["n_samples"] = int(ty["n_norm_samples"])
        out["holdout"] = {**out["holdout"], "min_items": int(ty["holdout_min_items"]), "n_val": int(ty["n_val"])}
        out["probe_chunk"] = int(ty["probe_chunk"])
        out["bootstrap_resamples"] = int(ty["bootstrap_resamples"])
        out["n_eval_expected"] = None
    for k in ("steps", "batch_size", "eval_every", "ckpt_every", "probe_chunk", "probe_chunk_max_elements"):
        if out[k] < 1:
            raise ValueError(f"{task}.{k} 必须 ≥ 1，收到 {out[k]}")
    return out


def curve_steps(steps: int, eval_every: int) -> list[int]:
    """记曲线的步（完成第 s 步之后）：eval_every 的倍数，最后一步总记。"""
    out = list(range(eval_every, steps + 1, eval_every))
    if not out or out[-1] != steps:
        out.append(steps)
    return out


__all__ = [
    "CONFIG_PATH",
    "E2Config",
    "PRE_PROJ",
    "PROBE_KINDS",
    "REFERENCES",
    "TASKS",
    "curve_steps",
    "expand_layers",
    "layer_label",
    "load_config",
    "parse_layer_spec",
    "resolve_task",
]
