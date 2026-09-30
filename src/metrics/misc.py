"""其他 T-文本通道指标：计数、相对深度点对、多选题。

输入里的 None 表示解析失败，一律记错（解析失败率由解析器另行统计）。
"""

from __future__ import annotations

import re
from typing import Hashable, NamedTuple, Sequence

_LETTER_ALONE = re.compile(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*[.:]?")
_LETTER_BRACKETED_PREFIX = re.compile(r"[\(\[]\s*([A-Za-z])\s*[\)\]]\s")
_LETTER_DELIM_PREFIX = re.compile(r"([A-Za-z])[.:)]\s")
# 前缀答案之后的余下文本里，只有这三种算"另一个选项"（D12，N2）；"X." 与 "X:" 不算（与 U.S.、句末提到选项无法区分）
_OTHER_BRACKETED = re.compile(r"[\(\[]\s*([A-Za-z])\s*[\)\]]")  # "(B)"、"[b]"
_OTHER_LISTED = re.compile(r"(?<![A-Za-z])([A-Za-z])\)")  # "C)"：独立字母后接右括号
# "answer is X" / "answer: X"（短语不分大小写）。X 为括号里的任意字母、独立的大写字母，
# 或后面紧跟句末标点/结尾的小写字母（避免把 "the answer is a dog" 里的冠词 a 当成选项）
# CHOICE_LETTER_FORM 就是其中 X 的写法，公开给 src.parsing.parse_choice 复用（DEV-0006；三个捕获组，恰有一个非空）
CHOICE_LETTER_FORM = r"(?:[\(\[]\s*([A-Za-z])\s*[\)\]]|([A-Z])(?![A-Za-z])|([a-z])(?=\s*(?:[.,;:!?]|$)))"
_OTHER_STATED = re.compile(r"(?i:answer)\s*(?:(?i:is)|:)\s*" + CHOICE_LETTER_FORM)


def _accuracy(correct: Sequence[bool]) -> float | None:
    if len(correct) == 0:
        return None
    return sum(1 for c in correct if c) / len(correct)


def count_accuracy(preds: Sequence[int | None], gts: Sequence[int]) -> float | None:
    """计数准确率：整数精确匹配。空集合返回 None。

    对齐 CountBench 的常用评测（预测整数 == GT 整数），没有单一官方仓库函数。
    预测必须是 int（bool 不算）；浮点的 3.0 视为类型错误抛出，取整属于解析器。
    """
    if len(preds) != len(gts):
        raise ValueError("preds 与 gts 长度不同")
    for p in preds:
        if p is not None and (not isinstance(p, int) or isinstance(p, bool)):
            raise TypeError(f"计数预测必须是 int 或 None，收到 {p!r}")
    return _accuracy([p is not None and p == g for p, g in zip(preds, gts)])


def pair_accuracy(preds: Sequence[Hashable | None], gts: Sequence[Hashable]) -> float | None:
    """相对深度点对准确率：预测的"哪个点更近"标签与 GT 精确相等的比例。空集合返回 None。

    对齐 BLINK Relative_Depth 的多选判定（选项与 GT 相等即对）；NYUv2 点对由我们构造，标签取值由构造代码定。
    """
    if len(preds) != len(gts):
        raise ValueError("preds 与 gts 长度不同")
    return _accuracy([p is not None and p == g for p, g in zip(preds, gts)])


def normalize_choice(answer: str | None) -> str | None:
    """把多选题回答规范化成单个大写字母；认不出返回 None。

    接受："A"、"a"、"(A)"、"[b]"、"A."、"A:"、"A)"，以及带选项正文的 "(A) cat"、"A. cat"、"A) cat"。
    前缀形式后面的余下文本里若出现另一个选项，是含糊回答，返回 None。"另一个选项"只认三种（D12，N2）：
    - 括号形式 "(X)"/"[X]"，不分大小写：如 "(A) or (B)"；
    - 列表式 "X)"，字母前面不是字母：如 "A) and C) are both right"；
    - 明确给出另一个答案："answer is X"/"answer: X"：如 "A. No, the answer is B."。
    X 等于所选字母的不算。"X." 与 "X:" 不算，所以 "(A) U.S. flag"、"(B) B is closer than A." 照常判为所选字母。
    不从长段自由文本里抽字母（那是解析器的事），例如 "The answer is A" 返回 None。
    """
    if answer is None:
        return None
    s = answer.strip()
    m = _LETTER_ALONE.fullmatch(s)
    if m:
        return m.group(1).upper()
    for pat in (_LETTER_BRACKETED_PREFIX, _LETTER_DELIM_PREFIX):
        m = pat.match(s)
        if m:
            letter = m.group(1).upper()
            rest = s[m.end():]
            found = _OTHER_BRACKETED.findall(rest) + _OTHER_LISTED.findall(rest)
            found += ["".join(g) for g in _OTHER_STATED.findall(rest)]
            others = {x.upper() for x in found} - {letter}
            return None if others else letter
    return None


# ---------------------------------------------------------------------------
# 公开给解析器复用的定义（DEV-0006；src.parsing.parse_choice）。它们直接用 normalize_choice 的那几个正则，
# 约定只在本模块定义一处；normalize_choice 自身不经过这些函数，行为不变。
# ---------------------------------------------------------------------------


def leading_choice_letter(text: str) -> str | None:
    """回答以 normalize_choice 的前缀形式开头时，返回那个字母（大写）；否则 None。只看开头，不看余下文本。

    前缀形式（与 normalize_choice 相同，按同样的顺序尝试）：括号 "(X) …"/"[X] …"，或字母加分隔符 "X. …"/"X: …"/"X) …"，
    两种都要求后面跟空白。整段只有一个字母（如 "B"、"(B)"、"B."）不算前缀形式，那是 normalize_choice 的整段匹配。
    """
    s = text.strip()
    for pat in (_LETTER_BRACKETED_PREFIX, _LETTER_DELIM_PREFIX):
        m = pat.match(s)
        if m:
            return m.group(1).upper()
    return None


def stated_choice_letters(text: str) -> list[str]:
    """文本里 "answer is X"/"answer: X" 明确给出的字母（大写，按出现顺序），与 normalize_choice 认"另一个选项"的规则相同。"""
    return ["".join(g).upper() for g in _OTHER_STATED.findall(text)]


def bracketed_choice_letters(text: str) -> list[str]:
    """文本里括号形式 "(X)"/"[X]" 的字母（不分大小写，返回大写，按出现顺序），与 normalize_choice 认"另一个选项"的规则相同。"""
    return [x.upper() for x in _OTHER_BRACKETED.findall(text)]


class MultipleChoiceResult(NamedTuple):
    accuracy: float
    none_rate: float  # 预测规范化失败（含预测为 None）的比例；报告按系统分列（D12(3)(6)）
    n: int


def multiple_choice_accuracy(preds: Sequence[str | None], gts: Sequence[str]) -> MultipleChoiceResult | None:
    """多选题准确率与 None 率：两边都经 normalize_choice 规范化后相等即对；规范化失败的预测记错并计入 None 率。
    空集合返回 None。

    对齐 CV-Bench / EmbSpatial-Bench 的选项字母匹配判定。GT 规范化失败抛 ValueError。
    """
    if len(preds) != len(gts):
        raise ValueError("preds 与 gts 长度不同")
    norm_gts = []
    for g in gts:
        ng = normalize_choice(g)
        if ng is None:
            raise ValueError(f"GT 选项无法规范化: {g!r}")
        norm_gts.append(ng)
    if not preds:
        return None
    norm_preds = [normalize_choice(p) for p in preds]
    return MultipleChoiceResult(
        accuracy=_accuracy([p == g for p, g in zip(norm_preds, norm_gts)]),
        none_rate=sum(1 for p in norm_preds if p is None) / len(norm_preds),
        n=len(norm_preds),
    )
