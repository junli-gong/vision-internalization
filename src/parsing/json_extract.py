"""从模型输出里取出唯一的一个 JSON 值（DEV-0006）。与模型无关；框与多边形的解析都经过这里。"""

from __future__ import annotations

import json
import re
from typing import Any

from src.parsing.core import ParseFailure, ParseResult

FENCE = "```"
# 代码块开头的信息串（Markdown 的 info string，如 ```json）：紧跟在 ``` 后面、以字母开头的一个词，后面必须是空白
_INFO_STRING = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]*(?=\s)")
_CLOSER = {"[": "]", "{": "}"}


class _DuplicateKey(ValueError):
    pass


def _reject_constant(name: str) -> Any:
    raise ValueError(f"严格 JSON 不允许 {name}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(k)
        out[k] = v
    return out


def _loads_strict(s: str) -> ParseResult:
    """严格 JSON：json.loads，但拒绝 NaN / Infinity / -Infinity（Python 默认接受）；对象里有重复键 → ambiguous
    （同一个字段两个值，不猜取哪个）；其余任何解析错误 → malformed。不用 ast.literal_eval 兜底。"""
    try:
        return ParseResult(json.loads(s, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys), None)
    except _DuplicateKey:
        return ParseResult(None, ParseFailure.AMBIGUOUS)
    except (ValueError, RecursionError):  # json.JSONDecodeError 是 ValueError 的子类
        return ParseResult(None, ParseFailure.MALFORMED)


def _block_body(block: str) -> str:
    """去掉代码块开头的信息串（```json 里的 json）；块内容以换行开头时没有信息串，原样返回。"""
    m = _INFO_STRING.match(block)
    return block[m.end():] if m else block


def _top_level_spans(text: str) -> list[tuple[int, int, bool]]:
    """按括号配对找顶层的 [...] 与 {...}，返回 (start, end, closed)。

    - 顶层（不在任何括号里）的字符都当说明文字，其中的引号不起作用，落单的 ] 与 } 忽略；
    - 进入括号之后按 JSON 的字符串规则跳过 "…"（含反斜杠转义），字符串里的括号不计；
    - 括号错配（如 [ … }）或到结尾仍未闭合：这一段记为 closed=False，从错配处之后重新在顶层扫描。
    """
    spans: list[tuple[int, int, bool]] = []
    stack: list[str] = []
    start = 0
    in_str = escape = False
    for i, ch in enumerate(text):
        if not stack:
            if ch in _CLOSER:
                stack.append(_CLOSER[ch])
                start = i
            continue
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in _CLOSER:
            stack.append(_CLOSER[ch])
        elif ch in "]}":
            if ch == stack[-1]:
                stack.pop()
                if not stack:
                    spans.append((start, i + 1, True))
            else:
                spans.append((start, i + 1, False))
                stack.clear()
    if stack:
        spans.append((start, len(text), False))
    return spans


def extract_json(text: str) -> ParseResult:
    """从模型输出里取出唯一的一个 JSON 值。规则（确定，不猜）：

    1. **有代码块时只看代码块**，块外的文字（说明、停止 token 等）一律忽略。代码块 = 一对 ``` 之间的内容：
       - ``` 的个数是奇数（有代码块没闭合，通常是 max_new_tokens 用完）→ malformed；
       - 多于一个代码块 → ambiguous，即使其中只有一个能解析（同 D10 对网格多块的规则）；
       - 恰好一个：去掉开头的信息串（``` 后紧跟的一个词，如 json、JSON，后面必须是空白），其余内容整体交给严格 JSON 解析
         （_loads_strict）：成功 → 该值（可以是任何 JSON 值）；失败 → malformed（重复键 → ambiguous）。
    2. **没有代码块时**，在全文里按括号配对找顶层的 [...] 或 {...}（规则见 _top_level_spans：说明文字里的引号不起作用，
       括号里按 JSON 字符串规则跳过 "…"）：
       - 一个都没有 → no_json（由调用方换成自己的失败原因，如 no_box、no_polygon）；
       - 多于一个（包括括号错配、未闭合的段）→ ambiguous；
       - 恰好一个：未闭合或错配 → malformed；否则严格 JSON 解析，同上。
       所以裸 JSON 前后的说明文字不影响结果，但说明文字里另有 [...] 或 {...} 时判 ambiguous。
    严格 JSON：单引号、注释、尾逗号、NaN/Infinity 都算 malformed，不用 ast.literal_eval 兜底。
    """
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    if FENCE in text:
        parts = text.split(FENCE)
        if len(parts) % 2 == 0:
            return ParseResult(None, ParseFailure.MALFORMED)
        blocks = parts[1::2]
        if len(blocks) > 1:
            return ParseResult(None, ParseFailure.AMBIGUOUS)
        return _loads_strict(_block_body(blocks[0]))
    spans = _top_level_spans(text)
    if not spans:
        return ParseResult(None, ParseFailure.NO_JSON)
    if len(spans) > 1:
        return ParseResult(None, ParseFailure.AMBIGUOUS)
    start, end, closed = spans[0]
    if not closed:
        return ParseResult(None, ParseFailure.MALFORMED)
    return _loads_strict(text[start:end])
