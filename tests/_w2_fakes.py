"""W2 测试用的假实现（DEV-0059）：候选生成器与渲染照 docs/w2_plan.md §1、§2 的签名写（真实现是 DEV-0058 的 src/w2/candidates.py、
src/w2/render.py，合入之前回路、运行器与命令行都用这里的）；另有挑选一步的假适配器。

- Candidate、CandidateSpec、generate_candidates(image_rgb, box_abs, *, fg_points_abs=None, bg_points_abs=None, spec)：
  - grabcut_default：W1-F 的同一条调用路径（src/w1/baselines.workbench_grabcut，即 src.w1.workbench.grabcut_mask，迭代次数取工作台的常量），
    所以 1 号候选逐位等于 W1-F；报错时用框本身、ok = False；
  - box：框本身；box_scale（percent）：框按中心缩放 (100 + percent)% 后光栅化；points_box：有前景点时取点的外接框（至少 1 像素），
    没有点时 ok = False、用框；其余类型：ok = False、用框（info 记原因）。info 记参数与耗时。
- render_selection(image, candidates, *, layout, style)：grid = 一张横向拼图（每格是原图把候选 mask 的像素染成红色）；separate = K 张；
  style.reference = true 时在前面多一张原图。逐位确定。
- ScriptedSelectAdapter(answers)：按挑选提示里的指代表达（answers 的键是表达式里的一段）给作答原文（或 turn(...) 的字典）；
  chat_inputs 预检结构性 token（与真适配器相同：ValueError），chat_turn_batch 记下每批的行数、工具、预算；build_inputs + generate_batch
  是关思考的路径（记下 enable_thinking 与 max_new_tokens）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from src.executors.raster import box_to_mask
from tests._w1_fakes import STRUCTURAL_TOKENS, ChatTurnResult

VISUAL_TOKENS_PER_IMAGE = 64
PIXELS_PER_VISUAL_TOKEN = 32 * 32


@dataclass(frozen=True)
class Candidate:
    name: str
    mask: np.ndarray
    ok: bool
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSpec:
    generators: tuple[dict, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generators", tuple(dict(g) for g in self.generators))
        if not self.generators or self.generators[0] != {"type": "grabcut_default"}:
            raise ValueError("第一项必须是 grabcut_default")


def _box_mask(box, h: int, w: int) -> np.ndarray:
    return box_to_mask([float(v) for v in box], h, w)


def generate_candidates(image_rgb: np.ndarray, box_abs: Sequence[float], *, fg_points_abs=None, bg_points_abs=None,
                        spec: CandidateSpec) -> list[Candidate]:
    from src.w1 import baselines as B

    img = np.asarray(image_rgb)
    h, w = img.shape[:2]
    out = []
    for g in spec.generators:
        t0 = time.perf_counter()
        typ = g["type"]
        params = {k: v for k, v in g.items() if k != "type"}
        ok, reason, mask = True, None, None
        try:
            if typ == "grabcut_default":
                gc = B.workbench_grabcut()
                mask = gc.fn(img, box_abs=[float(v) for v in box_abs], iterations=gc.iterations)
            elif typ == "box":
                mask = _box_mask(box_abs, h, w)
            elif typ == "box_scale":
                x0, y0, x1, y1 = (float(v) for v in box_abs)
                f = (100.0 + float(params["percent"])) / 100.0
                cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * f, (y1 - y0) / 2 * f
                mask = _box_mask([max(0.0, cx - hw), max(0.0, cy - hh), min(float(w), cx + hw), min(float(h), cy + hh)], h, w)
            elif typ == "points_box":
                if not fg_points_abs:
                    raise ValueError("没有前景点")
                xs = [p[0] for p in fg_points_abs]
                ys = [p[1] for p in fg_points_abs]
                mask = _box_mask([min(xs), min(ys), max(max(xs), min(xs) + 1), max(max(ys), min(ys) + 1)], h, w)
            else:
                raise ValueError(f"未知的生成器类型 {typ!r}")
        except Exception as e:  # noqa: BLE001 — 约定：失败时用框本身，ok = False
            ok, reason, mask = False, f"{type(e).__name__}: {e}", _box_mask(box_abs, h, w)
        name = typ + "".join(f"_{k}{v}" for k, v in sorted(params.items()))
        out.append(Candidate(name=name, mask=np.asarray(mask, dtype=bool), ok=ok,
                             info={"type": typ, "params": params, "elapsed_s": time.perf_counter() - t0, "error": reason}))
    return out


def _tint(image: Image.Image, mask: np.ndarray) -> Image.Image:
    a = np.asarray(image.convert("RGB")).copy()
    a[mask] = (a[mask] // 2) + np.array([127, 0, 0], dtype=np.uint8)
    return Image.fromarray(a)


def render_selection(image: Image.Image, candidates: Sequence[Candidate], *, layout: str, style: Mapping[str, Any]) -> list[Image.Image]:
    if layout not in ("grid", "separate"):
        raise ValueError(f"未知的布局 {layout!r}")
    cells = [_tint(image, c.mask) for c in candidates]
    if style.get("reference"):
        cells = [image.convert("RGB")] + cells
    if layout == "separate":
        return cells
    w, h = image.size
    grid = Image.new("RGB", (w * len(cells), h))
    for i, c in enumerate(cells):
        grid.paste(c, (i * w, 0))
    return [grid]


# ---------------------------------------------------------------------------
# 挑选一步的假适配器
# ---------------------------------------------------------------------------


def _structural(texts: Sequence[str]) -> list[str]:
    return sorted({t for x in texts for t in STRUCTURAL_TOKENS if t in x})


def _texts(msgs) -> list[str]:
    return [c["text"] for m in msgs for c in m["content"] if c["type"] == "text"]


def _n_images(msgs) -> int:
    return sum(1 for m in msgs for c in m["content"] if c["type"] == "image")


class ScriptedSelectAdapter:
    """answers：{表达式里的一段: 作答原文 或 {"text", "thinking_end", "stopped", "thinking"}}；找不到时答 default。"""

    model_key = "fake-select"

    def __init__(self, answers: Mapping[str, Any] | None = None, default: str = "1") -> None:
        self.answers = dict(answers or {})
        self.default = default
        self.batch_sizes: list[int] = []
        self.widths: list[list[int]] = []
        self.tools_seen: list[list[Any]] = []
        self.budgets: list[tuple[int, int]] = []
        self.conversations: list[list[list[dict]]] = []
        self.generate_calls: list[dict] = []
        self.n_chat_inputs = 0

    def warmup(self, images: Sequence[Any] = (), **kw: Any) -> float:
        return 0.0

    def parse_one_box(self, text: str, *, width: float, height: float):
        from src.adapters.qwen35 import parse_boxes
        from src.parsing import one_box

        return one_box(parse_boxes(text, width=width, height=height))

    def parse_box_points(self, text: str, *, width: float, height: float, max_points: int):
        from src.adapters.qwen35 import parse_box_points

        return parse_box_points(text, width=width, height=height, max_points=max_points)

    def _answer(self, prompt: str) -> dict:
        keys = [k for k in self.answers if k in prompt]
        assert len(keys) <= 1, (prompt, keys)
        a = self.answers[keys[0]] if keys else self.default
        a = {"text": a} if isinstance(a, str) else dict(a)
        return {"thinking": "Let me compare.", "thinking_end": "natural", "stopped": True, **a}

    def _width(self, msgs) -> int:
        return sum(len(t) for t in _texts(msgs)) // 4 + VISUAL_TOKENS_PER_IMAGE * _n_images(msgs) + 10

    def chat_inputs(self, messages, *, tools):
        self.n_chat_inputs += 1
        assert tools is None
        bad = _structural(_texts(messages))
        if bad:
            raise ValueError(f"消息里含有结构性 token：{bad}")
        return SimpleNamespace(row=SimpleNamespace(ids=tuple(range(self._width(messages)))))

    def chat_turn_batch(self, conversations, *, tools, thinking_budget, answer_budget):
        assert len(tools) == len(conversations) and all(t is None for t in tools)
        for conv in conversations:
            assert conv[-1]["role"] == "user" and conv[-1]["content"][-1]["type"] == "text"
            assert all(c["type"] == "image" for c in conv[-1]["content"][:-1])  # 挑选图在前、提示在后
            assert not _structural(_texts(conv))
        self.batch_sizes.append(len(conversations))
        self.widths.append([self._width(c) for c in conversations])
        self.tools_seen.append(list(tools))
        self.budgets.append((thinking_budget, answer_budget))
        self.conversations.append([list(c) for c in conversations])
        out = []
        for conv in conversations:
            a = self._answer(conv[-1]["content"][-1]["text"])
            n_img = _n_images(conv)
            vis = VISUAL_TOKENS_PER_IMAGE * n_img
            text = "" if a["thinking_end"] == "unclosed" else a["text"]
            out.append(ChatTurnResult(text=text, thinking=a["thinking"], thinking_end=a["thinking_end"], tool_calls=[], tool_call_errors=[],
                                      stopped=a["stopped"], n_prompt_tokens=self._width(conv),
                                      n_generated_tokens=len(text) // 4 + len(a["thinking"]) // 4 + 1,
                                      n_inserted_tokens=2 if a["thinking_end"] == "forced" else 0, n_visual_tokens=vis, n_images=n_img,
                                      wall_time_s=0.001 * len(conversations), n_image_pixels=vis * PIXELS_PER_VISUAL_TOKEN))
        return out

    # —— 关思考的路径（thinking = false）——
    def build_inputs(self, images, instruction, order):
        bad = _structural([instruction])
        if bad:
            raise ValueError(f"指令里含有结构性 token：{bad}")
        return SimpleNamespace(images=list(images), instruction=instruction, order=order)

    def generate_batch(self, inputs_list, *, enable_thinking, max_new_tokens, **kw):
        self.generate_calls.append({"n": len(inputs_list), "enable_thinking": enable_thinking, "max_new_tokens": max_new_tokens,
                                    "kw": dict(kw)})
        out = []
        for inp in inputs_list:
            a = self._answer(inp.instruction)
            vis = VISUAL_TOKENS_PER_IMAGE * len(inp.images)
            out.append(SimpleNamespace(text=a["text"], raw_text=a["text"], thinking_unclosed=False, stopped=a["stopped"],
                                       n_prompt_tokens=len(inp.instruction) // 4 + vis, n_generated_tokens=len(a["text"]) // 4 + 1,
                                       n_visual_tokens=vis, n_image_pixels=vis * PIXELS_PER_VISUAL_TOKEN,
                                       wall_time_s=0.001 * len(inputs_list), batch_wall_time_s=0.001 * len(inputs_list) ** 2))
        return out

    def generate(self, inputs, *, enable_thinking, max_new_tokens, **kw):
        return self.generate_batch([inputs], enable_thinking=enable_thinking, max_new_tokens=max_new_tokens, **kw)[0]


# 命令行彩排（--adapter-factory，只配 --tiny）：fixture 的表达式是 "the thing <sent_id>"
CLI_ANSWERS = {"the thing 100": "2", "the thing 102": "None.", "the thing 103": "Candidate 2", "the thing 104": "**3**",
               "the thing 105": "7", "the thing 111": "2", "the thing 113": "none", "the thing 115": "4"}


def make_scripted_select_adapter(entry: Any = None) -> ScriptedSelectAdapter:
    return ScriptedSelectAdapter(CLI_ANSWERS, default="1")


__all__ = [
    "CLI_ANSWERS",
    "Candidate",
    "CandidateSpec",
    "ScriptedSelectAdapter",
    "generate_candidates",
    "make_scripted_select_adapter",
    "render_selection",
]
