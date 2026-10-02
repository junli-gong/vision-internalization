"""E5 提示配置（src/e5/e5_prompts.json）：读取、结构校验、sha256、逐条渲染（DEV-0018）。

配置只在 JSON 里写一处，这里只读它。三个家族用同一个提示（统一提示）。
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.e5.protocol import E5_GRID_SIZE, E5_TASKS, MODEL_SYSTEMS
from src.synth.seeding import canonical_task

PROMPTS_PATH = Path(__file__).resolve().parent / "e5_prompts.json"
PROMPTS_SCHEMA_VERSION = 1
PLACEHOLDERS = frozenset({"instruction", "target", "n"})
# 格式名 → 解析方式（scoring.parse_answer 按它分派）
FORMAT_KINDS: dict[str, str] = {
    "binary_image": "binary_grid",
    "binary_canvas": "binary_grid",
    "relief_image": "relief_grid",
    "polygon_image": "polygon",
    "polygon_canvas": "polygon",
}
TEXT_ONLY_FORMATS = frozenset({"binary_canvas", "polygon_canvas"})


class PromptConfigError(ValueError):
    pass


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def validate(cfg: Mapping[str, Any]) -> None:
    bad: list[str] = []
    if cfg.get("schema_version") != PROMPTS_SCHEMA_VERSION:
        bad.append(f"schema_version 应为 {PROMPTS_SCHEMA_VERSION}")
    if cfg.get("image_order") not in ("image_first", "instruction_first"):
        bad.append("image_order 必须是 image_first 或 instruction_first")
    dec = cfg.get("decoding", {})
    if dec.get("do_sample") is not False or dec.get("num_beams") != 1:
        bad.append("decoding 必须 do_sample=false、num_beams=1（greedy）")
    bs = cfg.get("batching", {}).get("batch_size")
    if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
        bad.append("batching.batch_size 必须是正整数")
    systems = cfg.get("systems", {})
    if set(systems) != set(MODEL_SYSTEMS):
        bad.append(f"systems 必须恰好是 {MODEL_SYSTEMS}")
    for sysname, spec in systems.items():
        if not isinstance(spec.get("enable_thinking"), bool):
            bad.append(f"{sysname}: enable_thinking 必须是 bool")
        n = spec.get("max_new_tokens")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            bad.append(f"{sysname}: max_new_tokens 必须是正整数")
        fmt = spec.get("format_of_task", {})
        if set(fmt) != set(E5_TASKS):
            bad.append(f"{sysname}: format_of_task 必须对 {E5_TASKS} 各给一个格式")
        templates = spec.get("templates", {})
        for t, f in fmt.items():
            if f not in templates:
                bad.append(f"{sysname}/{t}: 格式 {f!r} 没有模板")
            if f not in FORMAT_KINDS:
                bad.append(f"{sysname}/{t}: 未知格式 {f!r}")
            if (t == "imagery") != (f in TEXT_ONLY_FORMATS):
                bad.append(f"{sysname}/{t}: e 类（只有 imagery）没有图像，必须用 *_canvas 格式")
        for f, tpl in templates.items():
            if not isinstance(tpl, str) or tpl != tpl.strip() or not tpl:
                bad.append(f"{sysname}/{f}: 模板必须是首尾无空白的非空字符串")
                continue
            used = _placeholders(tpl)
            if used - PLACEHOLDERS or "instruction" not in used:
                bad.append(f"{sysname}/{f}: 占位符 {sorted(used)} 不合规（只能用 {sorted(PLACEHOLDERS)}，必须含 instruction）")
        if set(spec.get("parse", {})) != set(templates):
            bad.append(f"{sysname}: parse 必须逐个说明每个模板的解析方式")
    if not isinstance(cfg.get("max_new_tokens_adjustments"), list):
        bad.append("max_new_tokens_adjustments 必须是列表")
    if bad:
        raise PromptConfigError("E5 提示配置不合格：\n- " + "\n- ".join(bad))


@dataclass(frozen=True)
class PromptConfig:
    data: Mapping[str, Any]
    path: str
    sha256: str

    @property
    def order(self) -> str:
        return self.data["image_order"]

    def decoding(self) -> dict[str, Any]:
        return dict(self.data["decoding"])

    def batch_size(self) -> int:
        return int(self.data["batching"]["batch_size"])

    def system(self, system: str) -> Mapping[str, Any]:
        if system not in MODEL_SYSTEMS:
            raise KeyError(f"{system!r} 不是模型系统（{MODEL_SYSTEMS}）")
        return self.data["systems"][system]

    def enable_thinking(self, system: str) -> bool:
        return bool(self.system(system)["enable_thinking"])

    def max_new_tokens(self, system: str) -> int:
        return int(self.system(system)["max_new_tokens"])

    def format_of(self, system: str, task: str) -> str:
        return self.system(system)["format_of_task"][canonical_task(task)]

    def kind_of(self, system: str, task: str) -> str:
        return FORMAT_KINDS[self.format_of(system, task)]

    def template(self, system: str, task: str) -> str:
        return self.system(system)["templates"][self.format_of(system, task)]

    def render(self, system: str, task: str, instruction: str, target: str, n: int = E5_GRID_SIZE) -> str:
        return self.template(system, task).format(instruction=instruction.strip(), target=target, n=n).strip()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prompt_config(path: str | Path = PROMPTS_PATH) -> PromptConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    validate(data)
    return PromptConfig(data=data, path=str(p), sha256=file_sha256(p))
