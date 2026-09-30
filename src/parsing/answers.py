"""短答案的解析：计数、多选字母、具名二选一（DEV-0006）。与模型无关。"""

from __future__ import annotations

import re
import string
from typing import Iterable, Sequence

from src.metrics.misc import (
    CHOICE_LETTER_FORM,
    bracketed_choice_letters,
    leading_choice_letter,
    normalize_choice,
    stated_choice_letters,
)
from src.parsing.core import ParseFailure, ParseResult

# ---------------------------------------------------------------------------
# 计数
# ---------------------------------------------------------------------------

NUMBER_WORDS: dict[str, int] = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty".split()
    )
}
# 数字记号：前面不能紧贴字母、数字、下划线或小数点，后面不能紧贴字母、数字、下划线（只认 ASCII）。
# 负号只在它前面也不紧贴这些字符时才算（"3-4" 是两个数 3 与 4，" -4" 是负数）。
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(-?)([0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?![A-Za-z0-9_])")
_NUMBER_WORD = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def parse_count(text: str) -> ParseResult:
    """计数回答 → 非负整数。

    认两种记号（只认 ASCII，整词匹配）：
    - 阿拉伯数字：前后都不紧贴字母、数字或下划线，所以 "2nd"、"3D"、"x2" 不算数；"3." "3," "(3)" "3%" 照常算 3；
    - 英文数词 zero–twenty，不分大小写，整词匹配（"none"、"someone" 里的 one 不算）。
    判定：
    - 出现小数（"3.5"、".5"）或负数（"-2"；"3-4" 里的 4 不是负数）→ malformed，先于下面各条判；
    - 所有记号的数值去重后：一个都没有 → no_number；恰好一个 → 该整数；两个以上 → multiple_numbers。
      "Three (3)." 是一个数值 3；"3 cats and 2 dogs"、"1, 2, 3" 是多个数值。
    已知局限（按规则判失败，不猜）：代词 one（"There are 3, one of them black" → multiple_numbers）；
    连字符数词 "twenty-one" 被读成 20 与 1（→ multiple_numbers）；"no cats"、"a cat" 里没有数（→ no_number）。
    """
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    values: set[int] = set()
    for m in _NUMBER.finditer(text):
        sign, digits = m.groups()
        if sign or "." in digits:
            return ParseResult(None, ParseFailure.MALFORMED)
        values.add(int(digits))
    for m in _NUMBER_WORD.finditer(text):
        values.add(NUMBER_WORDS[m.group(1).lower()])
    if not values:
        return ParseResult(None, ParseFailure.NO_NUMBER)
    if len(values) > 1:
        return ParseResult(None, ParseFailure.MULTIPLE_NUMBERS)
    return ParseResult(values.pop(), None)


# ---------------------------------------------------------------------------
# 多选题：选项字母
# ---------------------------------------------------------------------------

# 选项字母的写法、"answer is X"、括号字母、前缀答案，一律用 metrics.misc 的公开定义（约定只有一处；DEV-0006 审查第 1 轮）。
# 这里只加 parse_choice 自己的扩展：option X、option is X、option: X，choice 同 option（前导词不分大小写），
# X 的写法用 misc.CHOICE_LETTER_FORM。
_OPTION_STATED = re.compile(r"(?i:\b(?:option|choice)\b)\s*(?:(?i:is)\b\s*:?|:)?\s*" + CHOICE_LETTER_FORM)


def _choice_letters(letters: str | Iterable[str]) -> frozenset[str]:
    seq = list(letters)
    if not seq:
        raise ValueError("letters 为空")
    out = []
    for c in seq:
        if not (isinstance(c, str) and len(c) == 1 and c in string.ascii_letters):
            raise ValueError(f"选项字母必须是单个 ASCII 字母，收到 {c!r}")
        out.append(c.upper())
    if len(set(out)) != len(out):
        raise ValueError(f"选项字母有重复（不分大小写）：{seq!r}")
    return frozenset(out)


def parse_choice(text: str, letters: str | Iterable[str]) -> ParseResult:
    """多选题回答 → 选项字母（大写）。letters 是本题的全部选项字母，如 "ABCD" 或 ["A", "B"]，不分大小写。
    只有 letters 里的字母算候选（"Answer: I think …" 里的 I 不是选项）。

    1. metrics.misc.normalize_choice：整段是单个字母或前缀式答案，如 "B"、"(B)"、"B. left"、"(B) left"、
       "B. The answer is B."。它返回的字母在 letters 里 → 就是它。否则往下走。
    2. 明确陈述："The answer is B"、"Answer: (C)"（misc.stated_choice_letters，与 normalize_choice 同一定义），
       以及本函数自己的扩展 "option D"、"the correct option is D"、"My choice is (A)"。
    3. 全文里的括号字母 "(X)"、"[X]"（misc.bracketed_choice_letters）。
    第 2 步认出了字母就用第 2 步，否则用第 3 步：认出的字母，连同开头的前缀式字母（misc.leading_choice_letter，
    即 normalize_choice 的前缀形式，如 "A. …"、"(A) …"），恰好一个不同的字母 → 它；多于一个 → multiple_choices。
    所以回答以前缀式字母开头、后文又认出不同的字母时判 multiple_choices，不取后者（与 D12 N2 一致）：
    "A. No, the answer is B." → multiple_choices；"(A) left (B) right. The answer is B." → multiple_choices；
    回显的选项列表 "(A) left (B) right" → multiple_choices。第 2、3 步都没认出字母 → no_choice
    （normalize_choice 判为含糊的前缀字母，不会单独被接受）。
    "answer is: X"（is 后带冒号）不认：misc 的定义不含它，改那里会改变 normalize_choice 的行为。
    """
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    valid = _choice_letters(letters)
    letter = normalize_choice(text)
    if letter is not None and letter in valid:
        return ParseResult(letter, None)
    lead = {leading_choice_letter(text)} & valid
    stated = set(stated_choice_letters(text))
    stated |= {"".join(m.groups(default="")).upper() for m in _OPTION_STATED.finditer(text)}
    for found in (stated & valid, set(bracketed_choice_letters(text)) & valid):
        if found:
            candidates = found | lead
            if len(candidates) > 1:
                return ParseResult(None, ParseFailure.MULTIPLE_CHOICES)
            return ParseResult(candidates.pop(), None)
    return ParseResult(None, ParseFailure.NO_CHOICE)


# ---------------------------------------------------------------------------
# 具名二选一："left/right"、"television/dining table"
# ---------------------------------------------------------------------------

# 规范化时从首尾去掉的字符：ASCII 标点（含引号、括号、* 与 `）、弯引号，以及空白
_EDGE_CHARS = string.punctuation + "\u2018\u2019\u201c\u201d\u00ab\u00bb" + string.whitespace
_ARTICLE = re.compile(r"(?:a|an|the)\s+")
_WORD_CHAR = r"[a-z0-9]"


def normalize_phrase(s: str) -> str:
    """小写，连续空白合成一个空格，反复去掉首尾的标点、引号与开头的冠词（a / an / the），直到不再变化。"""
    s = " ".join(s.lower().split())
    while True:
        t = s.strip(_EDGE_CHARS)
        m = _ARTICLE.match(t)
        if m:
            t = t[m.end():]
        if t == s:
            return s
        s = t


def _choice_names(names: Sequence[str]) -> list[tuple[str, str]]:
    if isinstance(names, str):
        raise TypeError("names 必须是名字的序列，不能是单个字符串")
    out = []
    for n in names:
        if not isinstance(n, str):
            raise TypeError(f"名字必须是 str，收到 {n!r}")
        norm = normalize_phrase(n)
        if not norm:
            raise ValueError(f"名字规范化后为空：{n!r}")
        out.append((n, norm))
    if not out:
        raise ValueError("names 为空")
    if len({norm for _, norm in out}) != len(out):
        raise ValueError(f"名字规范化后有重复：{list(names)!r}")
    return out


def parse_named_choice(text: str, names: Sequence[str]) -> ParseResult:
    """具名选择题的回答 → names 里的某个名字（原样返回调用方给的写法）。

    规范化（normalize_phrase）：小写、合并空白、去掉首尾的标点与引号、去掉开头的冠词 a / an / the。
    1. 规范化后的整段回答等于某个名字（同样规范化）→ 该名字。例："Left."、"**left**"、"The television"、"'dining table'"。
    2. 否则在（小写、合并空白后的）全文里找各名字作为整词短语的出现（前后不紧贴 ASCII 字母或数字）。长名字优先：
       名字按规范化后的长度从长到短依次匹配，匹配到的片段随即遮掉，所以 "dining table" 里的 "table" 不再算一次出现。
       恰好一个名字出现（可以出现多次）→ 该名字；
    3. 一个都没有 → no_choice；多于一个 → multiple_choices。
    例：names = ["table", "dining table"]，"The dining table is closer." → dining table；
    "The table is closer than the dining table." → multiple_choices；"It's the left one, not the right." → multiple_choices。
    已知局限（按规则判，不猜）：否定句只含一个名字时照样取它（"not the left" → left）；复数与变形不算（"televisions"）。
    """
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    pairs = _choice_names(names)
    whole = normalize_phrase(text)
    for name, norm in pairs:
        if whole == norm:
            return ParseResult(name, None)
    hay = " ".join(text.lower().split())
    found = []
    for name, norm in sorted(pairs, key=lambda p: len(p[1]), reverse=True):
        pat = re.compile(rf"(?<!{_WORD_CHAR}){re.escape(norm)}(?!{_WORD_CHAR})")
        hits = list(pat.finditer(hay))
        if hits:
            found.append(name)
            for m in hits:
                hay = hay[: m.start()] + "\0" * (m.end() - m.start()) + hay[m.end():]
    if not found:
        return ParseResult(None, ParseFailure.NO_CHOICE)
    if len(found) > 1:
        return ParseResult(None, ParseFailure.MULTIPLE_CHOICES)
    return ParseResult(found[0], None)
