"""E2 的配置（configs/e2_probes.json）：读取、sha256、层集合的展开、--tiny 的规模覆盖。

配置是运行身份的一部分：它的 sha256 与本次实际生效的各项（层、参考线、任务参数）一起写进 config.yaml，续跑时必须完全相同。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
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
PRE_NORM_SUFFIX = "_pre"  # 最后一层归一化之前的特征键 L{N}_pre（pre_norm_key；适配器的层键 base.PRE_NORM_LAST）

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

    @property
    def pre_norm_last(self) -> bool:
        """层集合是否默认加上 L{N}_pre（最后一个 decoder 层的原始输出，最终归一化之前；D53-3）。"""
        return bool(self.doc["layers"]["pre_norm_last"])


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


_EXPLICIT_TOKEN = re.compile(r"L(0|[1-9][0-9]*)(_pre)?")  # 指定层（DEV-0038）：L<层号>，或 L<N>_pre（最后一层归一化之前）


def is_explicit(spec: str) -> bool:
    """规范写法是不是指定层的列表（"L8 L24"；DEV-0038），而不是 all / every k。"""
    return str(spec).startswith("L")


def _explicit(toks: Sequence[str]) -> tuple[list[int], list[int]]:
    """指定层的 token → (层号, 带 _pre 的层号)；格式不对或重复报错。"""
    layers: list[int] = []
    pre: list[int] = []
    for t in toks:
        m = _EXPLICIT_TOKEN.fullmatch(t)
        if m is None:
            raise ValueError(f"--layers 指定层要写成 L<层号>（或 L<N>_pre），收到 {t!r}")
        (pre if m.group(2) else layers).append(int(m.group(1)))
    if len(set(layers)) != len(layers) or len(pre) > 1:
        raise ValueError(f"--layers 有重复：{' '.join(toks)}")
    return layers, pre


def parse_layer_spec(tokens: Sequence[str] | str) -> str:
    """命令行的 --layers（all / every k / 指定层 L8 L24 …）→ 规范写法 "all"、"every k" 或升序的 "L8 L24"（L<N>_pre 排最后）。
    指定层（DEV-0038：E3 的机制测量只看最后一层与 E2 里最好的那一层）只训练给定层上的探针，不补最后一层、不默认加 L<N>_pre。"""
    toks = tokens.split() if isinstance(tokens, str) else list(tokens)
    if toks and all(t.startswith("L") for t in toks):
        layers, pre = _explicit(toks)
        if not layers:
            raise ValueError("--layers 指定层至少要有一个整数层 L<层号>（L<N>_pre 只能附加）")
        return " ".join([f"L{l}" for l in sorted(layers)] + [f"L{p}{PRE_NORM_SUFFIX}" for p in pre])
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
    raise ValueError(f"--layers 只接受 all、every k 或 L<层号> …，收到 {' '.join(toks)!r}")


def expand_layers(spec: str, num_layers: int, *, always_include_last: bool = True) -> list[int]:
    """规范写法 → 层号列表（升序）。层号约定见 src/adapters/base.py：有效范围 [0, num_layers]。

    every k = range(0, num_layers + 1, k)；always_include_last 时补上 num_layers（最终归一化之后的那一层）。
    指定层（"L8 L24"，DEV-0038）：就是这些层（必须在 [0, num_layers] 里），always_include_last 不起作用；L<N>_pre 不是整数层，见
    explicit_pre_norm。
    """
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError(f"num_layers 必须是正整数，收到 {num_layers!r}")
    spec = parse_layer_spec(spec)
    if is_explicit(spec):
        layers, pre = _explicit(spec.split())
        bad = [l for l in layers if not 0 <= l <= num_layers] + [p for p in pre if p != num_layers]
        if bad:
            raise ValueError(f"--layers {spec}：层号 {bad} 不在 [0, {num_layers}] 里（L<N>_pre 的 N 必须是最后一层 {num_layers}）")
        return sorted(layers)
    k = 1 if spec == "all" else int(spec.split()[1])
    out = list(range(0, num_layers + 1, k))
    if always_include_last and out[-1] != num_layers:
        out.append(num_layers)
    return out


def explicit_pre_norm(spec: str, num_layers: int) -> bool | None:
    """指定层时是否要 L<N>_pre（列表里有 L<N>_pre 才要）；不是指定层（all / every k）时返回 None，由配置的 layers.pre_norm_last 决定。"""
    if not is_explicit(parse_layer_spec(spec)):
        return None
    expand_layers(spec, num_layers)  # 核对范围
    return bool(_explicit(parse_layer_spec(spec).split())[1])


def layer_label(key: int | str) -> str:
    """特征键的显示名：整数层号 → "L12"，参考线与 L{N}_pre → 原名。"""
    return f"L{key}" if isinstance(key, int) else str(key)


def pre_norm_key(num_layers: int) -> str:
    """最后一层归一化之前的特征键：L{N}_pre（N = num_layers()；L{N} 是最终归一化之后的值，D19 的层号约定）。"""
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError(f"num_layers 必须是正整数，收到 {num_layers!r}")
    return f"L{num_layers}{PRE_NORM_SUFFIX}"


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
    "PRE_NORM_SUFFIX",
    "PRE_PROJ",
    "PROBE_KINDS",
    "REFERENCES",
    "TASKS",
    "curve_steps",
    "expand_layers",
    "explicit_pre_norm",
    "is_explicit",
    "layer_label",
    "load_config",
    "parse_layer_spec",
    "pre_norm_key",
    "resolve_task",
]
