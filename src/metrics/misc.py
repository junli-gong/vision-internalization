"""其他 T-文本通道指标：计数、相对深度点对、多选题。

输入里的 None 表示解析失败，一律记错（解析失败率由解析器另行统计）。
"""

from __future__ import annotations

import re
from typing import Hashable, Sequence

_LETTER_ALONE = re.compile(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*[.:]?")
_LETTER_BRACKETED_PREFIX = re.compile(r"[\(\[]\s*([A-Za-z])\s*[\)\]]\s")
_LETTER_DELIM_PREFIX = re.compile(r"([A-Za-z])[.:)]\s")


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
    不从长段自由文本里抽字母（那是解析器的事），例如 "The answer is A" 返回 None。
    """
    if answer is None:
        return None
    s = answer.strip()
    for pat, full in ((_LETTER_ALONE, True), (_LETTER_BRACKETED_PREFIX, False), (_LETTER_DELIM_PREFIX, False)):
        m = pat.fullmatch(s) if full else pat.match(s)
        if m:
            return m.group(1).upper()
    return None


def multiple_choice_accuracy(preds: Sequence[str | None], gts: Sequence[str]) -> float | None:
    """多选题准确率：两边都经 normalize_choice 规范化后相等即对。空集合返回 None。

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
    return _accuracy([normalize_choice(p) == g for p, g in zip(preds, norm_gts)])
