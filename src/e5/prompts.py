"""E5 提示配置（configs/e5_prompts.json；DEV-0019 从 src/e5/ 搬来）：读取、结构校验、sha256、逐条渲染（DEV-0018）。

配置只在 JSON 里写一处，这里只读它。三个家族用同一个提示（统一提示）。
schema 2（DEV-0028，D59-4、D59-5）：每个系统有 generation 字段——"plain"（generate_batch，上限 max_new_tokens；零样本）或
"budgeted"（generate_batch_budgeted 的预算强制收尾，thinking_budget + answer_budget；思维链，必须开思考）；b 与思维链的 a、b
用 33 符号的起伏网格（只许用在带起伏的任务上：src.synth.relief.RELIEF_TASKS）。
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.e5.protocol import E5_GRID_SIZE, E5_TASKS, MODEL_SYSTEMS
from src.synth.relief import RELIEF_TASKS
from src.synth.seeding import canonical_task

PROMPTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "e5_prompts.json"
PROMPTS_SCHEMA_VERSION = 2
PLACEHOLDERS = frozenset({"instruction", "target", "n"})
# 格式名 → 解析方式（scoring.parse_answer 按它分派）
FORMAT_KINDS: dict[str, str] = {
    "binary_image": "binary_grid",
    "binary_canvas": "binary_grid",
    "relief_image": "relief_grid",
    "relief_image_view": "relief_grid",
    "relief_image_cot": "relief_grid",
    "relief_image_view_cot": "relief_grid",
    "polygon_image": "polygon",
    "polygon_canvas": "polygon",
}
TEXT_ONLY_FORMATS = frozenset({"binary_canvas", "polygon_canvas"})
GENERATION_MODES = ("plain", "budgeted")  # plain = generate_batch；budgeted = generate_batch_budgeted（D59-5）


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
    def _posint(v) -> bool:
        return not isinstance(v, bool) and isinstance(v, int) and v >= 1

    for sysname, spec in systems.items():
        if not isinstance(spec.get("enable_thinking"), bool):
            bad.append(f"{sysname}: enable_thinking 必须是 bool")
        mode = spec.get("generation")
        if mode not in GENERATION_MODES:
            bad.append(f"{sysname}: generation 必须是 {GENERATION_MODES} 之一")
        elif mode == "plain":
            if not _posint(spec.get("max_new_tokens")):
                bad.append(f"{sysname}: max_new_tokens 必须是正整数")
            if "thinking_budget" in spec or "answer_budget" in spec:
                bad.append(f"{sysname}: plain 不用 thinking_budget / answer_budget")
        else:
            if not (_posint(spec.get("thinking_budget")) and _posint(spec.get("answer_budget"))):
                bad.append(f"{sysname}: budgeted 要 thinking_budget 与 answer_budget（正整数）")
            if spec.get("enable_thinking") is not True:
                bad.append(f"{sysname}: budgeted 必须开思考")
            if "max_new_tokens" in spec:
                bad.append(f"{sysname}: budgeted 不用 max_new_tokens（上限是两段预算）")
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
            if FORMAT_KINDS.get(f) == "relief_grid" and t not in RELIEF_TASKS:
                bad.append(f"{sysname}/{t}: 起伏网格只用在带起伏的任务上（{RELIEF_TASKS}）")
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

    def generation(self, system: str) -> str:
        """"plain"（generate_batch）或 "budgeted"（generate_batch_budgeted，预算强制收尾）。"""
        return str(self.system(system)["generation"])

    def max_new_tokens(self, system: str) -> int:
        if self.generation(system) != "plain":
            raise KeyError(f"{system} 是预算强制收尾，没有 max_new_tokens（用 budgets）")
        return int(self.system(system)["max_new_tokens"])

    def budgets(self, system: str) -> tuple[int, int]:
        """(thinking_budget, answer_budget)；只对 budgeted 的系统。"""
        spec = self.system(system)
        if spec["generation"] != "budgeted":
            raise KeyError(f"{system} 不是预算强制收尾")
        return int(spec["thinking_budget"]), int(spec["answer_budget"])

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
