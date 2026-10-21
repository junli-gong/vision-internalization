"""多轮对话接口（DEV-0049，W1；D103；接口约定 docs/w1_plan.md §1）：数据结构、消息校验、工具调用的解析。

与模型家族无关的部分都在这里；家族相关的（渲染、分词、图像、位置编码、生成）在适配器里（目前只有 Qwen3.5 实现
chat_turn_batch，其余家族抛 NotImplementedError）。

消息（与 HF 对话模板的约定相同）：
    {"role": "system" | "user" | "assistant" | "tool",
     "content": [{"type": "text", "text": str} | {"type": "image", "image": PIL.Image 或 uint8 (H, W, 3) 数组}, ...]}
- assistant 消息还可以带：
  - "tool_calls": [ToolCall, ...]（历史回填用；模板按原生格式重新写出 <tool_call> 块）；
  - "reasoning_content": str（这一轮的思考段；模板只保留最后一个真实 user 之后的 assistant 思考段，之前的去掉）。
  - 回填一轮的结果时，最贴近模型原样输出的写法是 content = [ChatTurnResult.text]（作答段原文，含 <tool_call> 块）、
    reasoning_content = ChatTurnResult.thinking、不给 tool_calls。
- tool 消息的 content 可以含图像（模板渲染为 user 轮里的 <tool_response>…</tool_response>）。
- 校验（check_messages）：role 只能是上面四种；content 是列表，元素只能是 text 与 image，键不多不少；system 只能是第一条、不含图；
  至少一条 user；最后一条必须是 user 或 tool（轮到模型作答）。每段文字（含 reasoning_content、工具调用的名字与参数）交给家族的
  文字检查（Qwen 沿用 _check_instruction：不得含结构性特殊 token）。违反一律报 ChatMessageError（ValueError 的子类），消息里写明
  第几条对话、第几条消息。

工具调用的格式（Qwen3.5 对话模板的 tools 段原文，experiments/P0/0001_env_probe/raw/p4/Qwen__Qwen3.5-*/chat_template.jinja）：
    <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>
parse_tool_calls 从作答段解析，规则写在它的说明里；格式不对的记进错误列表，不抛异常。

ChatTurnResult 的记账口径与 GenerationResult（generate_batch_budgeted）相同：由同一个 GenerationResult 换算而来（from_generation），
补入的 token 不算生成；n_prompt_tokens 是这一轮完整提示（全部历史 + 开思考的助手头）的长度，每一轮都重新预填，不复用上一轮的 KV。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from src.adapters.base import THINKING_ENDS, GenerationResult, ImageTokenInfo
from src.adapters.batching import PromptRow

ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")
CONTENT_TYPES: tuple[str, ...] = ("text", "image")
_ASSISTANT_EXTRA_KEYS = frozenset({"tool_calls", "reasoning_content"})

# Qwen3.5 的工具调用标记（<tool_call> / </tool_call> 是单个 token 248058 / 248059，docs/model_notes.md §5.1）
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "</parameter>"

# 工具名与参数名：写进 <function=NAME>、<parameter=KEY>，不能含空白、尖括号、等号（与 OpenAI 函数名的字符集相同，长度 ≤ 64）
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]{0,63}\Z")


class ChatMessageError(ValueError):
    """对话、消息或工具说明不合规。"""


# ---------------------------------------------------------------------------
# 数据结构（签名照 docs/w1_plan.md §1）
# ---------------------------------------------------------------------------


def _check_name(what: str, name: Any) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ChatMessageError(f"{what} 必须匹配 {NAME_RE.pattern}（字母或下划线开头，至多 64 个字符），收到 {name!r}")
    return name


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的说明，按 Qwen3.5 对话模板的 tools 参数渲染（as_template：OpenAI 的 {"type": "function", "function": …} 写法）。

    parameters 是 JSON Schema：type = "object"，properties 是 参数名 → schema 的字典，required（可缺）是 properties 里的参数名。
    整个说明必须能 JSON 序列化（模板用 tojson 写进 system 块）。
    """

    name: str
    description: str
    parameters: dict

    def __post_init__(self) -> None:
        _check_name("工具名", self.name)
        if not isinstance(self.description, str):
            raise ChatMessageError(f"工具 {self.name}：description 必须是 str，收到 {type(self.description).__name__}")
        p = self.parameters
        if not isinstance(p, dict) or p.get("type") != "object" or not isinstance(p.get("properties"), dict):
            raise ChatMessageError(f"工具 {self.name}：parameters 必须是 type=object、带 properties 字典的 JSON Schema，收到 {p!r}")
        for key in p["properties"]:
            _check_name(f"工具 {self.name} 的参数名", key)
        req = p.get("required", [])
        if not isinstance(req, list) or any(not isinstance(r, str) for r in req) or not set(req) <= set(p["properties"]):
            raise ChatMessageError(f"工具 {self.name}：required 必须是 properties 里的参数名列表，收到 {req!r}")
        try:
            json.dumps(self.as_template(), ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise ChatMessageError(f"工具 {self.name}：说明不能 JSON 序列化：{e}") from None

    def as_template(self) -> dict[str, Any]:
        """传给 apply_chat_template(tools=…) 的一项。"""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用。arguments 的值都是字符串（模板的 <parameter=…> 都是字符串），由调用方按 schema 解析。
    raw 是原文，含 <tool_call> 标记；历史回填时构造的 ToolCall 可以给空串。"""

    name: str
    arguments: dict[str, str]
    raw: str

    def __post_init__(self) -> None:
        _check_name("工具调用的函数名", self.name)
        if not isinstance(self.arguments, dict):
            raise ChatMessageError(f"工具调用 {self.name}：arguments 必须是 dict，收到 {type(self.arguments).__name__}")
        for k, v in self.arguments.items():
            _check_name(f"工具调用 {self.name} 的参数名", k)
            if not isinstance(v, str):
                raise ChatMessageError(f"工具调用 {self.name}：参数 {k} 的值必须是 str，收到 {type(v).__name__}")
        if not isinstance(self.raw, str):
            raise ChatMessageError(f"工具调用 {self.name}：raw 必须是 str")

    def as_template(self) -> dict[str, Any]:
        """assistant 消息的 tool_calls 里的一项（模板读 name 与 arguments）。"""
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ChatTurnResult:
    """一轮（一次 chat_turn）的结果。§1 约定的字段在前；后面五个是 DEV-0049 加的（有默认值，§1 的构造方式照旧可用；
    chat_turn_batch 一律填写）。

    - text：作答段原文（思考段收尾之后；强制收尾时在补的 token 之后），去掉末尾的停止 token 与首尾换行，
      = GenerationResult.answer_text（与 generate_batch_budgeted 同一口径）。unclosed 时为空串。
    - thinking：思考段原文 = 模型在思考段里生成的 token 的解码，不含思考结束标记（natural）与停止 token（unclosed），不 strip；
      Qwen 的思考段开头 <think> 在提示里，不在其中。
    - thinking_end：natural | forced | unclosed | not_opened（同 GenerationResult）。
    - tool_calls / tool_call_errors：parse_tool_calls(text) 的结果（只解析作答段）。
    - stopped：以停止 token 结束；False = 作答段写满 answer_budget 被截断。
    - n_prompt_tokens：这一轮的完整提示（含开思考的助手头）；n_generated_tokens：模型生成的 token（思考段 + 作答段）；
      n_inserted_tokens：强制收尾补入的 token（不计入生成）。n_visual_tokens、n_images：整条对话里的图（每一轮都重新送入）。
    - wall_time_s：整批墙钟按行等分的份额（同 GenerationResult；CUDA 上前后各同步一次，不含建输入）。
    - token_ids：模型生成的 token（= GenerationResult.token_ids，不含补的 token）；() 表示没有记录。
    - n_thinking_tokens：思考段里模型生成的 token 数（含 </think>，凡是模型写出的）；None 表示没有记录。
    - n_image_pixels：模型实际看到的像素数（D28-3；src.accounting.Compute.from_generation 读它）。
    - batch_size / batch_wall_time_s：这一行所在批的行数与整批墙钟。
    """

    text: str
    thinking: str
    thinking_end: str
    tool_calls: list[ToolCall]
    tool_call_errors: list[str]
    stopped: bool
    n_prompt_tokens: int
    n_generated_tokens: int
    n_inserted_tokens: int
    n_visual_tokens: int
    n_images: int
    wall_time_s: float
    token_ids: tuple[int, ...] = ()
    n_thinking_tokens: int | None = None
    n_image_pixels: int | None = None
    batch_size: int = 1
    batch_wall_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.thinking_end not in THINKING_ENDS:
            raise ValueError(f"thinking_end 必须是 {THINKING_ENDS} 之一，收到 {self.thinking_end!r}")
        for name in ("text", "thinking"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} 必须是 str")
        if not isinstance(self.tool_calls, list) or any(not isinstance(c, ToolCall) for c in self.tool_calls):
            raise TypeError("tool_calls 必须是 ToolCall 的列表")
        if not isinstance(self.tool_call_errors, list) or any(not isinstance(e, str) for e in self.tool_call_errors):
            raise TypeError("tool_call_errors 必须是 str 的列表")
        if not isinstance(self.stopped, bool):
            raise TypeError(f"stopped 必须是 bool，收到 {self.stopped!r}")
        for name in ("n_prompt_tokens", "n_generated_tokens", "n_inserted_tokens", "n_visual_tokens", "n_images"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} 必须是非负整数，收到 {v!r}")
        if (self.n_inserted_tokens > 0) != (self.thinking_end == "forced"):
            raise ValueError(f"n_inserted_tokens={self.n_inserted_tokens} 与 thinking_end={self.thinking_end!r} 不符"
                             "（只有强制收尾才补 token）")
        if self.thinking_end == "unclosed" and (self.text or self.tool_calls or self.tool_call_errors):
            raise ValueError("unclosed：没有作答段，text 必须为空串、没有工具调用")
        if (self.n_images == 0) != (self.n_visual_tokens == 0):
            raise ValueError(f"n_images={self.n_images} 与 n_visual_tokens={self.n_visual_tokens} 不相容")
        if not isinstance(self.wall_time_s, (int, float)) or isinstance(self.wall_time_s, bool) or self.wall_time_s < 0:
            raise ValueError(f"wall_time_s 必须是非负数，收到 {self.wall_time_s!r}")
        if not isinstance(self.token_ids, tuple) or any(isinstance(t, bool) or not isinstance(t, int) for t in self.token_ids):
            raise TypeError("token_ids 必须是整数元组")
        if self.token_ids and len(self.token_ids) != self.n_generated_tokens:
            raise ValueError(f"len(token_ids)={len(self.token_ids)} 与 n_generated_tokens={self.n_generated_tokens} 不一致")
        if self.n_thinking_tokens is not None and not 0 <= self.n_thinking_tokens <= self.n_generated_tokens:
            raise ValueError(f"n_thinking_tokens={self.n_thinking_tokens} 不在 [0, n_generated_tokens] 之内")
        if self.n_image_pixels is not None and (self.n_image_pixels == 0) != (self.n_visual_tokens == 0):
            raise ValueError(f"n_image_pixels={self.n_image_pixels} 与 n_visual_tokens={self.n_visual_tokens} 不相容")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size 必须是正整数，收到 {self.batch_size!r}")
        if self.batch_wall_time_s is not None and not math.isclose(
                self.wall_time_s, self.batch_wall_time_s / self.batch_size, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"wall_time_s={self.wall_time_s} 不是整批墙钟 {self.batch_wall_time_s} 的 1/{self.batch_size}")

    @classmethod
    def from_generation(cls, gen: GenerationResult, *, thinking: str, n_images: int) -> "ChatTurnResult":
        """由预算强制收尾的 GenerationResult（BudgetSupport._budget_result 的产物）换算：记账字段原样搬过来，作答段解析工具调用。"""
        if gen.thinking_end is None or gen.answer_text is None:
            raise ValueError("只接受预算强制收尾的生成结果（thinking_end 不为 None）")
        calls, errors = parse_tool_calls(gen.answer_text)
        return cls(
            text=gen.answer_text,
            thinking=thinking,
            thinking_end=gen.thinking_end,
            tool_calls=calls,
            tool_call_errors=errors,
            stopped=bool(gen.stopped),
            n_prompt_tokens=gen.n_prompt_tokens,
            n_generated_tokens=gen.n_generated_tokens,
            n_inserted_tokens=gen.n_inserted_tokens,
            n_visual_tokens=gen.n_visual_tokens,
            n_images=n_images,
            wall_time_s=gen.wall_time_s,
            token_ids=tuple(gen.token_ids),
            n_thinking_tokens=gen.n_thinking_tokens,
            n_image_pixels=gen.n_image_pixels,
            batch_size=gen.batch_size,
            batch_wall_time_s=gen.batch_wall_time_s,
        )


def thinking_token_ids(gen: GenerationResult, *, close_id: int, stop_ids: Sequence[int]) -> list[int]:
    """ChatTurnResult.thinking 的 token：思考段里模型生成的 token，去掉末尾的思考结束标记（natural）或停止 token（unclosed）。"""
    think = [int(t) for t in gen.token_ids[: gen.n_thinking_tokens]]
    if gen.thinking_end == "natural":
        if not think or think[-1] != close_id:
            raise RuntimeError("natural：思考段没有以结束标记结尾")
        think = think[:-1]
    elif gen.thinking_end == "unclosed":
        if not think or think[-1] not in set(int(s) for s in stop_ids):
            raise RuntimeError("unclosed：思考段没有以停止 token 结尾")
        think = think[:-1]
    return think


@dataclass(frozen=True)
class ChatInputs:
    """一条对话送进模型的完整提示（适配器的 chat_inputs 构造；不是 ModelInputs：没有指令区间与查询 token）。

    - tensors：input_ids / attention_mask / mm_token_type_ids（[1, L]，L = 全部历史 + 开思考的助手头），有图时另有各图拼接的
      pixel_values 与 image_grid_thw（按图在模板里出现的顺序）；
    - images：每张图的视觉 token 区间与网格（下标相对这条提示）；
    - row：拼批用的 PromptRow（ids 就是 input_ids）；
    - text：模板渲染的原文（图像占位符展开之前）。
    """

    tensors: Mapping[str, Any]
    images: tuple[ImageTokenInfo, ...]
    row: PromptRow
    text: str


# ---------------------------------------------------------------------------
# 消息与工具的校验、换成模板的写法
# ---------------------------------------------------------------------------


def _is_image(x: Any) -> bool:
    return isinstance(x, (Image.Image, np.ndarray))


def check_messages(messages: Any, *, check_text: Callable[[str], None], where: str = "") -> list[dict]:
    """校验一条对话（规则见模块说明），返回消息列表（原对象，不复制）。check_text 是家族的文字检查（违反时抛 ValueError）。"""
    pre = f"{where}：" if where else ""
    if isinstance(messages, (str, bytes, Mapping)) or not isinstance(messages, Sequence):
        raise ChatMessageError(f"{pre}对话必须是消息的序列，收到 {type(messages).__name__}")
    msgs = list(messages)
    if not msgs:
        raise ChatMessageError(f"{pre}对话为空")

    def text_ok(s: str, loc: str) -> None:
        try:
            check_text(s)
        except ValueError as e:
            raise ChatMessageError(f"{pre}{loc}：{e}") from None

    for i, m in enumerate(msgs):
        loc = f"messages[{i}]"
        if not isinstance(m, Mapping):
            raise ChatMessageError(f"{pre}{loc} 必须是 dict，收到 {type(m).__name__}")
        role = m.get("role")
        if role not in ROLES:
            raise ChatMessageError(f"{pre}{loc} 的 role 必须是 {ROLES} 之一，收到 {role!r}")
        extra = set(m) - {"role", "content"} - (_ASSISTANT_EXTRA_KEYS if role == "assistant" else set())
        if extra:
            raise ChatMessageError(f"{pre}{loc}（{role}）有不认识的键 {sorted(extra)}")
        content = m.get("content")
        if not isinstance(content, list):
            raise ChatMessageError(f"{pre}{loc} 的 content 必须是列表，收到 {type(content).__name__}")
        if not content and role in ("system", "user"):
            raise ChatMessageError(f"{pre}{loc}（{role}）的 content 为空")
        for j, item in enumerate(content):
            iloc = f"{loc}.content[{j}]"
            if not isinstance(item, Mapping) or item.get("type") not in CONTENT_TYPES:
                raise ChatMessageError(f"{pre}{iloc} 必须是 type 为 {CONTENT_TYPES} 之一的 dict，收到 {item!r:.80}")
            if set(item) != {"type", item["type"]}:
                raise ChatMessageError(f"{pre}{iloc} 的键必须恰为 type 与 {item['type']}，收到 {sorted(item)}")
            if item["type"] == "text":
                if not isinstance(item["text"], str):
                    raise ChatMessageError(f"{pre}{iloc} 的 text 必须是 str")
                text_ok(item["text"], iloc)
            else:
                if role == "system":
                    raise ChatMessageError(f"{pre}{iloc}：system 消息不能含图（模板的规则）")
                if not _is_image(item["image"]):
                    raise ChatMessageError(f"{pre}{iloc} 的 image 必须是 PIL.Image 或 numpy 数组，收到 {type(item['image']).__name__}")
        if role == "system" and i != 0:
            raise ChatMessageError(f"{pre}{loc}：system 消息只能是第一条（模板的规则）")
        if role == "assistant":
            rc = m.get("reasoning_content")
            if rc is not None:
                if not isinstance(rc, str):
                    raise ChatMessageError(f"{pre}{loc} 的 reasoning_content 必须是 str")
                text_ok(rc, f"{loc}.reasoning_content")
            calls = m.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, list) or any(not isinstance(c, ToolCall) for c in calls):
                    raise ChatMessageError(f"{pre}{loc} 的 tool_calls 必须是 ToolCall 的列表")
                for k, c in enumerate(calls):
                    for key, v in c.arguments.items():
                        text_ok(v, f"{loc}.tool_calls[{k}].arguments[{key!r}]")
    if not any(m["role"] == "user" for m in msgs):
        raise ChatMessageError(f"{pre}对话里没有 user 消息（模板的规则）")
    if msgs[-1]["role"] not in ("user", "tool"):
        raise ChatMessageError(f"{pre}最后一条消息必须是 user 或 tool（轮到模型作答），收到 {msgs[-1]['role']!r}")
    return msgs


def check_tools(tools: Any, *, where: str = "") -> tuple[ToolSpec, ...]:
    """一条对话的 tools：None（没有工具）或 ToolSpec 的序列，名字不重复。空序列与 None 相同（模板不写 tools 段）。"""
    pre = f"{where}：" if where else ""
    if tools is None:
        return ()
    if isinstance(tools, (str, bytes, ToolSpec, Mapping)) or not isinstance(tools, Sequence):
        raise ChatMessageError(f"{pre}tools 必须是 ToolSpec 的序列或 None，收到 {type(tools).__name__}")
    specs = tuple(tools)
    bad = [type(s).__name__ for s in specs if not isinstance(s, ToolSpec)]
    if bad:
        raise ChatMessageError(f"{pre}tools 的元素必须是 ToolSpec，收到 {bad}")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ChatMessageError(f"{pre}工具名有重复：{names}")
    return specs


def check_chat_batch(conversations: Any, tools: Any) -> list[Sequence[dict]]:
    """chat_turn_batch 的入参形状：conversations 是非空的对话序列；tools 与它等长（每条对话各自的工具，可以为 None）。
    返回对话列表；各条的内容由适配器逐条校验（check_messages / check_tools）。"""
    if isinstance(conversations, (str, bytes, Mapping)) or not isinstance(conversations, Sequence):
        raise ChatMessageError(f"conversations 必须是对话的序列，收到 {type(conversations).__name__}")
    convs = list(conversations)
    if not convs:
        raise ChatMessageError("conversations 为空")
    if convs and isinstance(convs[0], Mapping):
        raise ChatMessageError("conversations 必须是对话（消息的序列）的序列；单条对话请用 chat_turn，或放进列表")
    if isinstance(tools, (str, bytes, Mapping, ToolSpec)) or not isinstance(tools, Sequence):
        raise ChatMessageError(f"tools 必须是与 conversations 等长的序列（每条对话一项，可以为 None），收到 {type(tools).__name__}")
    if len(tools) != len(convs):
        raise ChatMessageError(f"tools 必须与 conversations 等长：tools 有 {len(tools)} 项，conversations 有 {len(convs)} 条")
    return convs


def template_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """校验过的消息 → (apply_chat_template 用的消息, 按模板顺序排列的图)。图在模板里写成 {"type": "image"}（模板只看类型），
    ToolCall 换成 {"name", "arguments"}；reasoning_content 原样传。system 不含图，所以图的顺序 = 消息顺序 × 内容顺序。"""
    out: list[dict[str, Any]] = []
    images: list[Any] = []
    for m in messages:
        content = []
        for item in m["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            else:
                content.append({"type": "image"})
                images.append(item["image"])
        msg: dict[str, Any] = {"role": m["role"], "content": content}
        if m.get("reasoning_content") is not None:
            msg["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            msg["tool_calls"] = [c.as_template() for c in m["tool_calls"]]
        out.append(msg)
    return out, images


# ---------------------------------------------------------------------------
# 工具调用的解析
# ---------------------------------------------------------------------------


def _strip_one_newline(s: str) -> str:
    """参数值去掉紧挨标记的一个换行（模板写成 <parameter=K>\\nV\\n</parameter>；与 vLLM 的 Qwen3 XML 解析器相同）。"""
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith("\n"):
        s = s[:-1]
    return s


def _parse_function(body: str) -> tuple[str | None, dict[str, str] | None, str | None]:
    """一个 <tool_call> 块里面的部分 → (函数名, 参数, 错误)。错误不为 None 时前两项为 None。"""
    s = body.strip()
    if not s.startswith(FUNCTION_OPEN):
        return None, None, f"expected {FUNCTION_OPEN}NAME> right after {TOOL_CALL_OPEN}"
    if s.count(FUNCTION_OPEN) != 1:
        return None, None, f"more than one {FUNCTION_OPEN}…> in one {TOOL_CALL_OPEN}"
    head_end = s.find(">")
    if head_end < 0 or "\n" in s[: head_end]:
        return None, None, f"unterminated {FUNCTION_OPEN}…> tag"
    name = s[len(FUNCTION_OPEN): head_end].strip()
    if not name:
        return None, None, "empty function name"
    if not NAME_RE.match(name):
        return None, None, f"invalid function name {name!r}"
    if not s.endswith(FUNCTION_CLOSE):
        return None, None, f"missing {FUNCTION_CLOSE}"
    inner = s[head_end + 1: len(s) - len(FUNCTION_CLOSE)]
    if FUNCTION_CLOSE in inner:
        return None, None, f"text after {FUNCTION_CLOSE}"
    args: dict[str, str] = {}
    pos = 0
    while True:
        i = inner.find(PARAMETER_OPEN, pos)
        gap = inner[pos: i if i >= 0 else len(inner)]
        if gap.strip():
            return None, None, f"unexpected text {gap.strip()[:40]!r} inside {FUNCTION_OPEN}{name}>"
        if i < 0:
            break
        k_end = inner.find(">", i)
        if k_end < 0 or "\n" in inner[i: k_end]:
            return None, None, f"unterminated {PARAMETER_OPEN}…> tag"
        key = inner[i + len(PARAMETER_OPEN): k_end].strip()
        if not key:
            return None, None, "empty parameter name"
        if not NAME_RE.match(key):
            return None, None, f"invalid parameter name {key!r}"
        v_end = inner.find(PARAMETER_CLOSE, k_end + 1)
        value = inner[k_end + 1: v_end] if v_end >= 0 else ""
        if v_end < 0 or PARAMETER_OPEN in value:
            return None, None, f"missing {PARAMETER_CLOSE} for parameter {key!r}"
        if key in args:
            return None, None, f"duplicate parameter {key!r}"
        args[key] = _strip_one_newline(value)
        pos = v_end + len(PARAMETER_CLOSE)
    return name, args, None


def parse_tool_calls(text: str) -> tuple[list[ToolCall], list[str]]:
    """从作答段解析 Qwen3.5 的工具调用，返回 (调用，按出现顺序；错误，按出现顺序)。不抛异常（text 不是 str 除外）。

    规则：
    - 每个 <tool_call> 到它之后的第一个 </tool_call> 是一个块；块里恰好一个 <function=NAME>…</function>，前后只能有空白。
    - 函数体里是若干 <parameter=KEY>VALUE</parameter>，之间只能有空白；VALUE 去掉紧挨标记的一个换行，其余原样（可以多行、可以是
      JSON）。参数名、函数名不能为空，字符集同 NAME_RE；同一个参数不能写两次。
    - 记错误（这个块不产出调用，继续解析后面的）：没有 </tool_call>（下一个 <tool_call> 先出现，或到结尾都没有）、函数名为空、
      格式不对。块外出现 </tool_call> 或 <function=… 也记错误（标记不配对、调用没包在 <tool_call> 里）。
    - 错误文字用英文，前缀 "tool call K: "（K 从 1 数，按 <tool_call> 出现的顺序），便于原样转给模型。
    """
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    calls: list[ToolCall] = []
    errors: list[str] = []
    pos, k = 0, 0
    while True:
        i = text.find(TOOL_CALL_OPEN, pos)
        outside = text[pos: i if i >= 0 else len(text)]
        if TOOL_CALL_CLOSE in outside:
            errors.append(f"stray {TOOL_CALL_CLOSE} without a matching {TOOL_CALL_OPEN}")
        if FUNCTION_OPEN in outside:
            errors.append(f"{FUNCTION_OPEN}…> outside {TOOL_CALL_OPEN}…{TOOL_CALL_CLOSE}")
        if i < 0:
            break
        k += 1
        start = i + len(TOOL_CALL_OPEN)
        j = text.find(TOOL_CALL_CLOSE, start)
        nxt = text.find(TOOL_CALL_OPEN, start)
        if j < 0 or 0 <= nxt < j:
            errors.append(f"tool call {k}: missing {TOOL_CALL_CLOSE}")
            if nxt < 0:
                break
            pos = nxt
            continue
        raw = text[i: j + len(TOOL_CALL_CLOSE)]
        name, args, err = _parse_function(text[start:j])
        if err is not None:
            errors.append(f"tool call {k}: {err}")
        else:
            calls.append(ToolCall(name=name, arguments=args, raw=raw))
        pos = j + len(TOOL_CALL_CLOSE)
    return calls, errors


__all__ = [
    "CONTENT_TYPES",
    "ChatInputs",
    "ChatMessageError",
    "ChatTurnResult",
    "NAME_RE",
    "ROLES",
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "ToolCall",
    "ToolSpec",
    "check_chat_batch",
    "check_messages",
    "check_tools",
    "parse_tool_calls",
    "template_messages",
    "thinking_token_ids",
]
