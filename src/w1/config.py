"""W1 的配置（configs/w1.json）：读取、结构校验、sha256；按设置给出回路协议（Protocol）与渲染好的提示（LoopPrompts）。

约定只写在配置里一处，这里只读它（数字、提示原文、设置 → 关掉的工具组、划分的 salt 与大小、基线的规则）。tests/test_w1_config.py
钉住配置里的数字与这里的读法一致。
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from src.w1.loop import TASKS, LoopPrompts, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
W1_CONFIG_PATH = REPO_ROOT / "configs" / "w1.json"
W1_SCHEMA_VERSION = 1
LOOP_SETTINGS: tuple[str, ...] = ("full", "no_eye", "no_check", "no_snap")  # W1-1、A1、A2、A3
BASELINE_SETTINGS: tuple[str, ...] = ("baseline_e1", "baseline_ownbox", "baseline_poly_structured", "w1f")
SETTINGS: tuple[str, ...] = LOOP_SETTINGS + BASELINE_SETTINGS
# 每种基线适用的 W1 任务（baseline_e1 三个任务都有；其余三种只对指代分割）
BASELINE_TASKS: Mapping[str, tuple[str, ...]] = {"baseline_e1": TASKS, "baseline_ownbox": ("refseg",),
                                                 "baseline_poly_structured": ("refseg",), "w1f": ("refseg",)}
PROMPT_KEYS: tuple[str, ...] = ("system",) + TASKS + ("tool_cap", "tool_call_error", "last_round")


class W1ConfigError(ValueError):
    """configs/w1.json 不合格：列出全部问题。"""


def _posint(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def validate(cfg: Mapping[str, Any]) -> None:
    bad: list[str] = []
    if cfg.get("schema_version") != W1_SCHEMA_VERSION:
        bad.append(f"schema_version 应为 {W1_SCHEMA_VERSION}")
    if not isinstance(cfg.get("frozen"), bool):
        bad.append("frozen 必须是 bool")
    pr = cfg.get("protocol", {})
    for k in ("max_rounds", "max_tool_calls", "thinking_budget", "answer_budget"):
        if not _posint(pr.get(k)):
            bad.append(f"protocol.{k} 必须是正整数")
    st = pr.get("submit_tools")
    if not isinstance(st, list) or not st or len(set(st)) != len(st) or any(not str(x).startswith("submit_") for x in st):
        bad.append("protocol.submit_tools 必须是不重复的 submit_* 工具名")
    if not isinstance(pr.get("submit_exempt_from_cap"), bool):
        bad.append("protocol.submit_exempt_from_cap 必须是 bool")
    bt = cfg.get("batching", {})
    for k in ("batch_size", "chunk_size", "max_batch_tokens", "tool_workers"):
        if not _posint(bt.get(k)):
            bad.append(f"batching.{k} 必须是正整数")
    sets = cfg.get("settings", {})
    if set(k for k in sets if k != "rule") != set(LOOP_SETTINGS):
        bad.append(f"settings 必须恰好给出 {LOOP_SETTINGS}")
    for k in LOOP_SETTINGS:
        g = sets.get(k)
        if not isinstance(g, list) or len(set(g)) != len(g) or any(not isinstance(x, str) or not x for x in g) or "submit" in (g or []):
            bad.append(f"settings.{k} 必须是不重复的组名列表，且不含 submit")
    if sets.get("full") != []:
        bad.append("settings.full（W1-1）不能关任何组")
    p = cfg.get("prompts", {})
    ph = p.get("placeholders", {})
    texts = {"system": p.get("system"), **(p.get("tasks") or {}), **(p.get("messages") or {})}
    if set(p.get("tasks") or {}) != set(TASKS):
        bad.append(f"prompts.tasks 必须恰好是 {TASKS}")
    if set(ph) != set(PROMPT_KEYS):
        bad.append(f"prompts.placeholders 必须恰好给出 {PROMPT_KEYS}")
    for k in PROMPT_KEYS:
        t = texts.get(k)
        if k == "last_round" and t is None:
            continue
        if not isinstance(t, str) or not t.strip():
            bad.append(f"提示 {k} 必须是非空字符串")
            continue
        try:
            used = _fields(t)
        except ValueError as e:
            bad.append(f"提示 {k} 的占位符写法不对：{e}")
            continue
        if used != set(ph.get(k, [])):
            bad.append(f"提示 {k} 用的占位符 {sorted(used)} 与 placeholders 声明的 {ph.get(k)} 不同")
    sp = cfg.get("splits", {})
    for k in ("salt", "purpose", "population", "count_devtune", "count_test"):
        if not isinstance(sp.get(k), str) or not sp.get(k):
            bad.append(f"splits.{k} 必须是非空字符串")
    for k in ("devtune_n", "devcheck_n"):
        if not _posint(sp.get(k)) or not _posint((sp.get("tiny") or {}).get(k)):
            bad.append(f"splits.{k} 与 splits.tiny.{k} 必须是正整数")
    if not isinstance(sp.get("test"), list) or not sp.get("test") or len(set(sp["test"])) != len(sp["test"]):
        bad.append("splits.test 必须是不重复的 E1 poly 子集名（<数据集>_<划分方式>_<划分>）")
    bl = cfg.get("baselines", {})
    if (bl.get("e1_task") or {}) != {"refseg": "refexp_poly", "refexp_box": "refexp_box", "count": "count"}:
        bad.append("baselines.e1_task 必须是 refseg → refexp_poly、refexp_box → refexp_box、count → count（W1-0 = E1 零样本）")
    if (bl.get("poly_structured") or {}).get("variant") != "poly_structured":
        bad.append("baselines.poly_structured.variant 必须是 poly_structured（D103 §三的对照）")
    w1f = bl.get("w1f") or {}
    if w1f.get("function") != "src.w1.workbench.grabcut_mask" or w1f.get("iterations") != "src.w1.workbench.GRABCUT_ITERATIONS":
        bad.append("baselines.w1f 必须用工作台的同一个 GrabCut（function = src.w1.workbench.grabcut_mask，iterations = "
                   "src.w1.workbench.GRABCUT_ITERATIONS；D103：W1-F 用工作台的同一个函数）")
    if not str(w1f.get("rule", "")).strip():
        bad.append("baselines.w1f.rule 不能为空")
    tiny = cfg.get("tiny", {})
    if not _posint(tiny.get("batch_size")) or not _posint(tiny.get("chunk_size")):
        bad.append("tiny.batch_size、tiny.chunk_size 必须是正整数")
    if set(tiny.get("protocol", {})) - {"thinking_budget", "answer_budget"} or any(not _posint(v) for v in tiny.get("protocol", {}).values()):
        bad.append("tiny.protocol 只能缩小 thinking_budget、answer_budget（正整数）")
    if bad:
        raise W1ConfigError("configs/w1.json 不合格：\n- " + "\n- ".join(bad))


@dataclass(frozen=True)
class W1Config:
    data: Mapping[str, Any]
    path: str
    sha256: str

    def protocol(self, setting: str, *, tiny: bool = False) -> Protocol:
        """回路设置（full、no_eye、no_check、no_snap）→ Protocol（关掉的组取 settings）。tiny 时按 tiny.protocol 缩小预算。"""
        if setting not in LOOP_SETTINGS:
            raise ValueError(f"{setting!r} 不是回路设置 {LOOP_SETTINGS}")
        pr = self.data["protocol"]
        p = Protocol(max_rounds=int(pr["max_rounds"]), max_tool_calls=int(pr["max_tool_calls"]),
                     thinking_budget=int(pr["thinking_budget"]), answer_budget=int(pr["answer_budget"]),
                     disabled_groups=frozenset(self.data["settings"][setting]), submit_tools=frozenset(pr["submit_tools"]),
                     submit_exempt_from_cap=bool(pr["submit_exempt_from_cap"]))
        if tiny:
            p = replace(p, **{k: int(v) for k, v in self.data["tiny"]["protocol"].items()})
        return p

    def loop_prompts(self, protocol: Protocol) -> LoopPrompts:
        p = self.data["prompts"]
        caps = {"max_rounds": protocol.max_rounds, "max_tool_calls": protocol.max_tool_calls}
        msgs = p["messages"]
        return LoopPrompts(system=p["system"].format(**caps), tasks=dict(p["tasks"]),
                           tool_cap=msgs["tool_cap"].format(max_tool_calls=protocol.max_tool_calls),
                           tool_call_error=msgs["tool_call_error"], last_round=msgs.get("last_round"))

    def batching(self, *, tiny: bool = False) -> dict[str, int]:
        b = self.data["batching"]
        out = {"batch_size": int(b["batch_size"]), "chunk_size": int(b["chunk_size"]), "max_batch_tokens": int(b["max_batch_tokens"]),
               "tool_workers": int(b["tool_workers"])}
        if tiny:
            out.update(batch_size=int(self.data["tiny"]["batch_size"]), chunk_size=int(self.data["tiny"]["chunk_size"]))
        return out

    @property
    def splits(self) -> Mapping[str, Any]:
        return self.data["splits"]

    @property
    def baselines(self) -> Mapping[str, Any]:
        return self.data["baselines"]

    def e1_task(self, task: str) -> str:
        return self.data["baselines"]["e1_task"][task]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_w1_config(path: str | Path = W1_CONFIG_PATH) -> W1Config:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    validate(data)
    return W1Config(data=data, path=str(p), sha256=file_sha256(p))


__all__ = [
    "BASELINE_SETTINGS",
    "BASELINE_TASKS",
    "LOOP_SETTINGS",
    "PROMPT_KEYS",
    "SETTINGS",
    "W1Config",
    "W1ConfigError",
    "W1_CONFIG_PATH",
    "file_sha256",
    "load_w1_config",
    "validate",
]
