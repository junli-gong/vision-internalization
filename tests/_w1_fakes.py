"""W1 测试用的假适配器与假工作台（DEV-0051）。按 docs/w1_plan.md §1、§2 的接口约定写，真实现（DEV-0049 的 src/adapters/chat.py 与
chat_turn_batch、DEV-0050 的 src/w1/workbench.py）合入之前，回路、基线与命令行都用它们测；tests/test_w1_contract.py 在真实现存在时核对
字段与签名和这里一致。不放在 tests/_fakes/ 下：那里只给服务器口径的假 fla 用，普通运行里不许有任何模块从那里 import
（tests/test_adapter_qwen35.py::test_fake_fla_package_is_visible_only_in_parity_runs）。

- fake_grabcut_mask：src.w1.workbench.grabcut_mask 的框初始化（同签名、同规则），W1-F 的测试在真工作台合入之前用它。
- FakeWorkbench(image, *, disabled_groups=frozenset(), sandbox=None)：specs() / call(name, arguments) / submission / log()。
  工具（组）：zoom（eye，放大视图）、draw_box（hand，框 → 命名 mask）、grabcut（snap，cv2.grabCut 框初始化）、run_python（code，假的，
  只回显代码长度）、overlay（check，mask 叠加回原图）、submit_mask / submit_box / submit_count（submit）。坐标都是原图的 0–1000，x 在前。
- ScriptedChatAdapter(script)：每条样本按脚本逐轮返回 ChatTurnResult；样本由 user 消息里的任务文字认出（脚本的键是任务文字里的一段）。
- PolicyChatAdapter()：命令行彩排用的确定性策略（按任务文字选工具，不同样本在不同轮结束）。
两个适配器都记下每次 chat_turn_batch 的行数（batch_sizes），并核对约定里的入参。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from src.executors.geometry import norm1000_to_abs
from src.executors.overlay import render_overlay
from src.executors.raster import box_to_mask
from src.parsing import one_box

VISUAL_TOKENS_PER_IMAGE = 300
PIXELS_PER_VISUAL_TOKEN = 32 * 32


# ---------------------------------------------------------------------------
# 约定里的数据结构（docs/w1_plan.md §1、§2）
# ---------------------------------------------------------------------------


try:  # DEV-0049 合入之后用真的数据结构（字段与校验都是真的；check_tools 只收真的 ToolSpec）
    from src.adapters.chat import ChatTurnResult, ToolCall, ToolSpec
except ImportError:  # 合入之前：照 docs/w1_plan.md §1 与 D104 的字段
    @dataclass(frozen=True)
    class ToolSpec:
        name: str
        description: str
        parameters: dict

    @dataclass(frozen=True)
    class ToolCall:
        name: str
        arguments: dict
        raw: str

    @dataclass(frozen=True)
    class ChatTurnResult:
        text: str
        thinking: str
        thinking_end: str
        tool_calls: list
        tool_call_errors: list
        stopped: bool
        n_prompt_tokens: int
        n_generated_tokens: int
        n_inserted_tokens: int
        n_visual_tokens: int
        n_images: int
        wall_time_s: float
        token_ids: tuple = ()
        n_thinking_tokens: int | None = None
        n_image_pixels: int | None = None
        batch_size: int = 1
        batch_wall_time_s: float | None = None

# Qwen3.5 的结构性 token（D104：对话里出现任何一个，chat_inputs 拒收、chat_turn_batch 整批报错）
STRUCTURAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|image_pad|>", "<|video_pad|>", "<|vision_start|>",
                     "<|vision_end|>")


@dataclass
class ToolResult:
    text: str
    images: list
    error: str | None


@dataclass(frozen=True)
class Submission:
    mask: np.ndarray | None = None
    box: tuple | None = None
    count: int | None = None


GROUPS = ("eye", "hand", "snap", "check", "code", "submit")


def _obj(props: Mapping[str, str], required: Sequence[str]) -> dict:
    return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(required)}


TOOLS: dict[str, tuple[str, ToolSpec]] = {
    "zoom": ("eye", ToolSpec("zoom", "Zoomed view of a region [x0, y0, x1, y1] (0-1000, original image).",
                             _obj({"box": "[x0, y0, x1, y1]"}, ["box"]))),
    "draw_box": ("hand", ToolSpec("draw_box", "Rasterize a box into a named mask.", _obj({"box": "[x0, y0, x1, y1]", "out": "name"},
                                                                                          ["box", "out"]))),
    "grabcut": ("snap", ToolSpec("grabcut", "GrabCut initialised with a box; result is a named mask.",
                                 _obj({"box": "[x0, y0, x1, y1]", "iterations": "int", "out": "name"}, ["box", "out"]))),
    "run_python": ("code", ToolSpec("run_python", "Run Python in the sandbox.", _obj({"code": "source"}, ["code"]))),
    "overlay": ("check", ToolSpec("overlay", "Overlay a named mask on the original image.", _obj({"name": "mask name"}, ["name"]))),
    "submit_mask": ("submit", ToolSpec("submit_mask", "Submit a named mask as the final answer.", _obj({"name": "mask name"}, ["name"]))),
    "submit_box": ("submit", ToolSpec("submit_box", "Submit a box [x0, y0, x1, y1] as the final answer.",
                                      _obj({"box": "[x0, y0, x1, y1]"}, ["box"]))),
    "submit_count": ("submit", ToolSpec("submit_count", "Submit a count as the final answer.", _obj({"count": "integer"}, ["count"]))),
}


FAKE_GRABCUT_ITERATIONS = 5


def fake_grabcut_mask(image_rgb: np.ndarray, *, box_abs, iterations: int = FAKE_GRABCUT_ITERATIONS) -> np.ndarray:
    """src.w1.workbench.grabcut_mask 的框初始化部分（同一签名、同一规则：box_to_mask 的框内记可能前景、框外确定背景，种子固定，
    没有前景或背景像素就抛 ValueError），DEV-0050 合入之前给 W1-F 的测试用。"""
    import cv2

    img = np.asarray(image_rgb)
    h, w = img.shape[:2]
    inside = box_to_mask(box_abs, h, w)
    if not inside.any():
        raise ValueError("GrabCut 的初始化里没有前景像素")
    if inside.all():
        raise ValueError("GrabCut 的初始化里没有背景像素")
    gc = np.where(inside, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(np.ascontiguousarray(img[..., ::-1]), gc, None, bgd, fgd, int(iterations), cv2.GC_INIT_WITH_MASK)
    return (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)


class FakeWorkbench:
    def __init__(self, image: Any, *, disabled_groups: frozenset = frozenset(), sandbox: Any = None) -> None:
        bad = set(disabled_groups) - set(GROUPS)
        if bad:
            raise ValueError(f"未知的工具组 {sorted(bad)}")
        self.image = image.convert("RGB")
        self.disabled = frozenset(disabled_groups)
        self.masks: dict[str, np.ndarray] = {}
        self._submission: Submission | None = None
        self._log: list[dict] = []

    def specs(self) -> list[ToolSpec]:
        return [spec for name, (g, spec) in TOOLS.items() if g not in self.disabled]

    @property
    def submission(self) -> Submission | None:
        return self._submission

    def log(self) -> list[dict]:
        return list(self._log)

    def _box(self, s: str) -> tuple[float, float, float, float]:
        v = json.loads(s)
        if not (isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)):
            raise ValueError(f"box 必须是 [x0, y0, x1, y1]，收到 {s!r}")
        w, h = self.image.size
        x0, y0, x1, y1 = (float(x) for x in norm1000_to_abs(np.asarray(v, dtype=np.float64), width=w, height=h))
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"box 退化：{s!r}")
        return x0, y0, x1, y1

    def call(self, name: str, arguments: dict[str, str]) -> ToolResult:
        t0 = time.perf_counter()
        try:
            out = self._call(name, dict(arguments))
        except Exception as e:  # noqa: BLE001 — 报错交给模型看
            out = ToolResult(text=f"Error: {e}", images=[], error=type(e).__name__)
        self._log.append({"name": name, "arguments": dict(arguments), "elapsed_s": time.perf_counter() - t0, "error": out.error})
        return out

    def _call(self, name: str, a: dict[str, str]) -> ToolResult:
        if name not in TOOLS or TOOLS[name][0] in self.disabled:
            return ToolResult(text=f"Error: no tool named {name!r}", images=[], error="unknown_tool")
        w, h = self.image.size
        if name == "zoom":
            x0, y0, x1, y1 = self._box(a["box"])
            crop = self.image.crop((int(x0), int(y0), max(int(x0) + 1, int(np.ceil(x1))), max(int(y0) + 1, int(np.ceil(y1)))))
            return ToolResult(text=f"View of {a['box']} (original-image 0-1000 coordinates).",
                              images=[crop.resize((crop.width * 2, crop.height * 2))], error=None)
        if name == "draw_box":
            self.masks[a["out"]] = box_to_mask(self._box(a["box"]), h, w)
            return ToolResult(text=f"mask {a['out']}: area {int(self.masks[a['out']].sum())} px", images=[], error=None)
        if name == "grabcut":
            self.masks[a["out"]] = fake_grabcut_mask(np.asarray(self.image), box_abs=self._box(a["box"]),
                                                     iterations=int(a.get("iterations", "5")))
            return ToolResult(text=f"mask {a['out']}: area {int(self.masks[a['out']].sum())} px", images=[], error=None)
        if name == "run_python":
            return ToolResult(text=f"stdout: ({len(a['code'])} chars of code ran)", images=[], error=None)
        if name == "overlay":
            m = self.masks[a["name"]]
            ov = render_overlay(np.asarray(self.image), [m])
            from PIL import Image

            return ToolResult(text=f"Overlay of {a['name']} (area {int(m.sum())} px).", images=[Image.fromarray(ov)], error=None)
        if name == "submit_mask":
            if a["name"] not in self.masks:
                return ToolResult(text=f"Error: no mask named {a['name']!r}", images=[], error="unknown_mask")
            self._submission = Submission(mask=self.masks[a["name"]].copy())
            return ToolResult(text="Submitted.", images=[], error=None)
        if name == "submit_box":
            self._submission = Submission(box=self._box(a["box"]))
            return ToolResult(text="Submitted.", images=[], error=None)
        if name == "submit_count":
            n = int(a["count"])
            if n < 0:
                raise ValueError("count < 0")
            self._submission = Submission(count=n)
            return ToolResult(text="Submitted.", images=[], error=None)
        raise AssertionError(name)


# ---------------------------------------------------------------------------
# 假适配器
# ---------------------------------------------------------------------------


def raw_call(name: str, args: Mapping[str, str]) -> str:
    """Qwen3.5 模板的 <tool_call> 写法（XML 参数）。"""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def turn(*, text: str = "", thinking: str = "Let me think.", thinking_end: str = "natural", calls: Sequence[tuple[str, dict]] = (),
         errors: Sequence[str] = (), stopped: bool = True) -> dict:
    """脚本里的一轮。"""
    return {"text": text, "thinking": thinking, "thinking_end": thinking_end, "calls": list(calls), "errors": list(errors),
            "stopped": stopped}


def _n_images(conv: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for m in conv for c in m["content"] if c["type"] == "image")


def _n_text(conv: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(c["text"]) for m in conv for c in m["content"] if c["type"] == "text")


def task_text(conv: Sequence[Mapping[str, Any]]) -> str:
    return next(c["text"] for c in conv[1]["content"] if c["type"] == "text")


class _BaseFake:
    model_key = "fake-chat"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.budgets: list[tuple[int, int]] = []
        self.conversations: list[list[list[dict]]] = []
        self.widths: list[list[int]] = []
        self.n_chat_inputs = 0

    def warmup(self, images: Sequence[Any] = (), **kw: Any) -> float:
        return 0.0

    def parse_one_box(self, text: str, *, width: float, height: float):
        from src.adapters.qwen35 import parse_boxes

        return one_box(parse_boxes(text, width=width, height=height))

    def _check(self, conversations, tools, thinking_budget, answer_budget) -> None:
        assert len(tools) == len(conversations) and conversations
        for conv, tl in zip(conversations, tools):
            assert conv[0]["role"] == "system" and conv[1]["role"] == "user"
            assert conv[1]["content"][0]["type"] == "image"  # 原图在前
            assert conv[-1]["role"] in ("user", "tool")
            assert all(isinstance(t, ToolSpec) for t in tl)
            for m in conv:
                assert set(m) <= {"role", "content"} | ({"reasoning_content"} if m["role"] == "assistant" else set()), m.keys()
                assert all(set(c) == {"type", c["type"]} for c in m["content"])
                if m["role"] == "assistant":  # D104 的回填：作答原文 + 思考段，不给 tool_calls
                    assert "reasoning_content" in m and [c["type"] for c in m["content"]] == ["text"]
        assert isinstance(thinking_budget, int) and isinstance(answer_budget, int)

    @staticmethod
    def _structural(conv) -> list[str]:
        texts = [c["text"] for m in conv for c in m["content"] if c["type"] == "text"]
        texts += [m.get("reasoning_content") or "" for m in conv]
        return sorted({t for x in texts for t in STRUCTURAL_TOKENS if t in x})

    def chat_inputs(self, messages, *, tools):
        """真适配器的预检（D104）：消息里有结构性 token 就报 ValueError（真的是 ChatMessageError，ValueError 的子类）；否则给出提示
        （只用 row.ids 的长度：拆子批用）。"""
        self.n_chat_inputs += 1
        bad = self._structural(messages)
        if bad:
            raise ValueError(f"消息里含有结构性 token：{bad}")
        n = _n_text(messages) // 4 + VISUAL_TOKENS_PER_IMAGE * _n_images(messages) + 10
        return SimpleNamespace(row=SimpleNamespace(ids=tuple(range(n))))

    def _result(self, conv, spec: Mapping[str, Any], batch_rows: int) -> ChatTurnResult:
        calls = [ToolCall(n, {k: str(v) for k, v in a.items()}, raw_call(n, a)) for n, a in spec["calls"]]
        text = (spec["text"] + ("\n" if spec["text"] and calls else "") + "\n".join(c.raw for c in calls)).strip()
        n_img = _n_images(conv)
        vis = VISUAL_TOKENS_PER_IMAGE * n_img
        return ChatTurnResult(text=text, thinking=spec["thinking"], thinking_end=spec["thinking_end"], tool_calls=calls,
                              tool_call_errors=list(spec["errors"]), stopped=spec["stopped"],
                              n_prompt_tokens=_n_text(conv) // 4 + vis + 10, n_generated_tokens=len(text) // 4 + len(spec["thinking"]) // 4 + 1,
                              n_inserted_tokens=2 if spec["thinking_end"] == "forced" else 0, n_visual_tokens=vis, n_images=n_img,
                              wall_time_s=0.001 * batch_rows, n_image_pixels=vis * PIXELS_PER_VISUAL_TOKEN)

    def chat_turn_batch(self, conversations, *, tools, thinking_budget, answer_budget):
        self._check(conversations, tools, thinking_budget, answer_budget)
        bad = [k for k, c in enumerate(conversations) if self._structural(c)]
        if bad:  # 与真适配器相同：整批报错
            raise ValueError(f"conversations{bad}：消息里含有结构性 token")
        self.widths.append([_n_text(c) // 4 + VISUAL_TOKENS_PER_IMAGE * _n_images(c) + 10 for c in conversations])
        self.batch_sizes.append(len(conversations))
        self.budgets.append((thinking_budget, answer_budget))
        self.conversations.append([list(c) for c in conversations])
        return [self._result(c, self.next_turn(c), len(conversations)) for c in conversations]

    def chat_turn(self, messages, *, tools, thinking_budget, answer_budget):
        return self.chat_turn_batch([messages], tools=[tools], thinking_budget=thinking_budget, answer_budget=answer_budget)[0]

    def next_turn(self, conv) -> dict:
        raise NotImplementedError


class ScriptedChatAdapter(_BaseFake):
    """script：{任务文字里的一段: [第 1 轮, 第 2 轮, …]}（每轮是 turn(...)）。轮次 = 对话里已有的 assistant 消息数。"""

    def __init__(self, script: Mapping[str, Sequence[dict]]) -> None:
        super().__init__()
        self.script = {k: list(v) for k, v in script.items()}

    def next_turn(self, conv) -> dict:
        t = task_text(conv)
        keys = [k for k in self.script if k in t]
        assert len(keys) == 1, (t, keys)
        r = sum(1 for m in conv if m["role"] == "assistant")
        return self.script[keys[0]][r]


class PolicyChatAdapter(_BaseFake):
    """命令行彩排用：按任务文字选一套固定的工具调用；样本按任务文字的哈希在不同轮结束（批随之收缩）。"""

    def next_turn(self, conv) -> dict:
        t = task_text(conv)
        r = sum(1 for m in conv if m["role"] == "assistant")
        k = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % 3
        if t.startswith("Task: segment"):
            plan = [turn(text="Fit the object.", calls=[("grabcut", {"box": "[150, 150, 850, 850]", "iterations": "3", "out": "m"})]),
                    turn(text="Check it.", calls=[("overlay", {"name": "m"})]),
                    turn(text="Looks right.", calls=[("submit_mask", {"name": "m"})])]
            if k == 0:
                plan = [plan[0], plan[2]]
            elif k == 1:
                plan = [turn(text="Look first.", calls=[("zoom", {"box": "[0, 0, 500, 500]"})])] + plan
            return plan[min(r, len(plan) - 1)]
        if t.startswith("Task: locate"):
            final = turn(text='```json\n[{"bbox_2d": [100, 120, 900, 880], "label": "object"}]\n```')
            plan = [turn(text="Zoom.", calls=[("zoom", {"box": "[0, 0, 1000, 1000]"})]), final] if k else [final]
            return plan[min(r, len(plan) - 1)]
        plan = [turn(text="Measure.", calls=[("run_python", {"code": "print(1)"})]), turn(calls=[("submit_count", {"count": "2"})])]
        return plan[min(r, len(plan) - 1)] if k else turn(text="3")


def make_policy_adapter(entry: Any = None) -> PolicyChatAdapter:
    """scripts/run_w1.py --adapter-factory 的入口（只配 --tiny）。"""
    return PolicyChatAdapter()


__all__ = [
    "ChatTurnResult",
    "FAKE_GRABCUT_ITERATIONS",
    "FakeWorkbench",
    "GROUPS",
    "PolicyChatAdapter",
    "ScriptedChatAdapter",
    "Submission",
    "TOOLS",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "fake_grabcut_mask",
    "make_policy_adapter",
    "raw_call",
    "task_text",
    "turn",
]
