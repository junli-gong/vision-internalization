"""LLMSeg 测试用的假适配器（DEV-0066）：按官方提示的阶段给出脚本化的回复，坐标一律在 0–1000 网格上（Qwen-VL 的约定），
所以官方代码要把它们换回像素。回复只由提示决定（与批的组成无关），用来核对合批不改内容、端到端的计分与输出。

- build_inputs(images, instruction, order)：记下提示；input_ids 的宽度 = len(提示) // 4（只用于切子批）。
- generate / generate_batch：每行按提示里的阶段出回复；token 数 = len(回复) // 4 + 1，超过 max_new_tokens 时截断（stopped = False）。
  calls 记下每次调用的 (方法, 行数, max_new_tokens)。
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from src.adapters.base import GenerationResult
from src.adapters.factory import default_image_policy
from src.adapters import registry

VISUAL = 300
PIXELS = 300 * 32 * 32


def ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 16) -> list[list[float]]:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + rx * np.cos(th), cy + ry * np.sin(th)]).round(1).tolist()


def grid_reply(prompt: str) -> str:
    """一个实例、0–1000 网格上的回复（框在中间，阶段 2 是椭圆，3a 给一个补的焦点框，3b 在框里画一个小椭圆）。"""
    if "Find every DISTINCT instance" in prompt:
        return "```json\n" + json.dumps({"detections": [{"id": 0, "label": "thing", "box": [150, 150, 850, 850], "confidence": 0.9,
                                                         "morphology": {"shape": "blob", "aspect": "square"}}]}) + "\n```"
    if "We have already DETECTED" in prompt:
        return json.dumps({"mask_color": "cyan", "degree": 3, "instances": [
            {"id": 0, "splines": [{"role": "outer", "degree": 3, "closed": True, "control_points": ellipse(450, 500, 250, 300)}]}]})
    if "Find FOCUS BOXES" in prompt:
        return json.dumps({"focus_boxes": [{"type": "positive", "instance_id": 0, "shape": "bump", "box": [650, 350, 840, 650]}]})
    if "FOCUS BOX(ES). For EACH box" in prompt:
        return json.dumps({"refinements": [{"id": 0, "splines": [{"degree": 3, "closed": True,
                                                                  "control_points": ellipse(745, 500, 90, 140, 8)}]}]})
    return json.dumps({"instances": [{"id": 0, "x1": 0, "x2": 0, "y1": 0, "y2": 0}]})


class GridAdapter:
    """假适配器（模块说明）。"""

    family = registry.FAMILY_QWEN35

    def __init__(self, reply=grid_reply):
        self.reply = reply
        self.model_key = "fake-grid"
        self.image_policy = default_image_policy(registry.FAMILY_QWEN35)
        self.calls: list[tuple[str, int, int]] = []
        self._lock = threading.Lock()

    def build_inputs(self, images: Sequence[Any], instruction: str, order: str) -> Any:
        if len(images) != 1 or order != "image_first":
            raise ValueError("假适配器只收一张图、图在前")
        if "<|im_start|>" in instruction:
            raise ValueError("指令里有结构性 token")
        return SimpleNamespace(tensors={"input_ids": np.zeros((1, max(1, len(instruction) // 4)), dtype=np.int64)},
                               prompt=instruction, image_size=images[0].size)

    def _row(self, inp: Any, max_new_tokens: int, n: int) -> GenerationResult:
        text = self.reply(inp.prompt)
        k = len(text) // 4 + 1
        stopped = k <= max_new_tokens
        if not stopped:
            text, k = text[: max_new_tokens * 4], max_new_tokens
        return GenerationResult(text=text, token_ids=tuple(range(k)), n_generated_tokens=k,
                                n_prompt_tokens=int(inp.tensors["input_ids"].shape[1]), n_visual_tokens=VISUAL, wall_time_s=0.001,
                                raw_text=text, stopped=stopped, n_image_pixels=PIXELS, batch_size=n, batch_wall_time_s=0.001 * n)

    def generate(self, inputs: Any, *, enable_thinking: bool, max_new_tokens: int, **_kw: Any) -> GenerationResult:
        assert enable_thinking is False
        with self._lock:
            self.calls.append(("generate", 1, max_new_tokens))
        return self._row(inputs, max_new_tokens, 1)

    def generate_batch(self, inputs_list: Sequence[Any], *, enable_thinking: bool, max_new_tokens: int, **_kw: Any) -> list[GenerationResult]:
        assert enable_thinking is False
        with self._lock:
            self.calls.append(("generate_batch", len(inputs_list), max_new_tokens))
        return [self._row(x, max_new_tokens, len(inputs_list)) for x in inputs_list]


def make_grid_adapter(entry: Any) -> GridAdapter:
    return GridAdapter()


class FailingAdapter(GridAdapter):
    """第一次生成就报错（模拟显存不够）：运行必须中止，不能按空 mask 计分。"""

    def generate_batch(self, inputs_list, **kw):  # noqa: ANN001
        raise RuntimeError("CUDA out of memory (fake)")

    def generate(self, inputs, **kw):  # noqa: ANN001
        raise RuntimeError("CUDA out of memory (fake)")


def make_failing_adapter(entry: Any) -> FailingAdapter:
    return FailingAdapter()
