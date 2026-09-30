"""src/parsing 与模型无关的部分（DEV-0006）：失败原因表、记账、parse_generation、extract_json、计数、选择、多边形。

各家族原生框格式的解析在 tests/test_parsing_boxes.py。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.adapters.base import GenerationResult
from src.executors.raster import polygons_to_mask
from src.grid.codec import GridParse, GridParseFailure, parse_grid_text
from src.parsing import (
    BoxParse,
    OneBoxParse,
    ParseFailure,
    ParseResult,
    ParseTally,
    extract_json,
    normalize_phrase,
    one_box,
    parse_choice,
    parse_count,
    parse_generation,
    parse_named_choice,
    parse_polygons,
)

PF = ParseFailure


def _gen(text="", *, unclosed=False):
    return GenerationResult(text, (1,), 1, 5, 0, 0.0, raw_text=text, thinking_unclosed=unclosed)


# ---------------------------------------------------------------------------
# 失败原因表与结果类型
# ---------------------------------------------------------------------------


def test_failure_value_table_is_pinned():
    """取值表钉死（完成总结里的表）：改动要有意为之。前 6 个是 DEV-0005 的 BoxParseFailure 原值，顺序不变。"""
    assert [m.value for m in ParseFailure] == [
        "no_box", "unbalanced_tags", "malformed", "out_of_range", "inverted", "multiple_boxes",
        "empty", "thinking_unclosed", "ambiguous", "no_json",
        "no_number", "multiple_numbers", "no_choice", "multiple_choices", "no_polygon", "too_few_points",
    ]
    assert all(isinstance(m, str) and m == m.value for m in ParseFailure)
    # 与网格解析的失败原因不重名：两者可以记进同一个 ParseTally 而不混淆
    assert not {m.value for m in ParseFailure} & {m.value for m in GridParseFailure}


def test_result_types_have_failure_last_and_values_none_on_failure():
    for cls in (ParseResult, BoxParse, OneBoxParse):
        assert cls._fields[-1] == "failure"
    assert BoxParse._fields == ("boxes", "labels", "failure")
    assert OneBoxParse._fields == ("box", "failure")


def test_one_box_rules():
    box = (1.0, 2.0, 3.0, 4.0)
    assert one_box(BoxParse((box,), ("cat",), None)) == (box, None)
    assert one_box(BoxParse((), (), None)) == (None, PF.NO_BOX)  # 明确的"没有框"
    assert one_box(BoxParse((box, box), (None, None), None)) == (None, PF.MULTIPLE_BOXES)  # 相同的框也不合并
    assert one_box(BoxParse(None, None, PF.INVERTED)) == (None, PF.INVERTED)  # 失败原样给出


# ---------------------------------------------------------------------------
# ParseTally
# ---------------------------------------------------------------------------


def test_tally_counts_and_rate():
    t = ParseTally()
    assert t.to_dict() == {"n": 0, "n_ok": 0, "n_failed": 0, "failure_rate": None, "failures": {}}  # 空集合报 null
    t.add(ParseResult(3, None))
    t.add(OneBoxParse(None, PF.NO_BOX))
    t.add(BoxParse(None, None, PF.MALFORMED))
    t.add(GridParse(None, GridParseFailure.NO_GRID))  # 其他 str 枚举
    t.add(PF.NO_BOX)  # 直接给失败原因
    t.add(None)  # 直接给"成功"
    assert (t.n, t.n_ok, t.n_failed) == (6, 2, 4)
    assert t.failure_rate == pytest.approx(4 / 6)
    assert t.counts == {"malformed": 1, "no_box": 2, "no_grid": 1}
    d = t.to_dict()
    assert list(d) == ["n", "n_ok", "n_failed", "failure_rate", "failures"]
    assert list(d["failures"]) == ["malformed", "no_box", "no_grid"]  # 按原因字符串的字典序
    assert json.loads(json.dumps(d)) == d


def test_tally_key_order_independent_of_insertion_order():
    items = [PF.NO_BOX, PF.EMPTY, None, PF.AMBIGUOUS, PF.NO_BOX, GridParseFailure.MULTIPLE_BLOCKS]
    a, b = ParseTally(items), ParseTally(reversed(items))
    assert a == b
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())  # 逐字相同


def test_tally_merge():
    a = ParseTally([PF.NO_BOX, None, PF.EMPTY])
    b = ParseTally([None, None, PF.NO_BOX, GridParseFailure.NO_GRID])
    m = a.merge(b)
    assert m == a + b == b + a
    assert (m.n, m.n_ok) == (7, 3)
    assert m.counts == {"empty": 1, "no_box": 2, "no_grid": 1}
    assert (a.n, b.n) == (3, 4)  # 合并不改动原对象
    assert ParseTally() + ParseTally() == ParseTally()
    with pytest.raises(TypeError):
        a.merge({"n": 1})


@pytest.mark.parametrize("bad", ["no_box", 3, ParseResult(None, "malformed"), (None, PF.NO_BOX)])
def test_tally_rejects_non_enum_failures(bad):
    with pytest.raises(TypeError):
        ParseTally().add(bad)


# ---------------------------------------------------------------------------
# parse_generation
# ---------------------------------------------------------------------------


def test_parse_generation_passes_text_and_kwargs():
    assert parse_generation(_gen("3 cats"), parse_count) == (3, None)
    assert parse_generation(_gen(" (B) "), parse_choice, letters="AB") == ("B", None)
    one = parse_generation(_gen("[[0, 0], [1000, 0], [1000, 1000]]"), parse_polygons, width=10, height=20)
    assert one.failure is None and one.value == (((0.0, 0.0), (10.0, 0.0), (10.0, 20.0)),)


@pytest.mark.parametrize("fn, kw, cls", [
    (parse_count, {}, ParseResult),
    (parse_choice, {"letters": "AB"}, ParseResult),
    (parse_polygons, {"width": 4, "height": 3}, ParseResult),
    (extract_json, {}, ParseResult),
    (parse_grid_text, {"n": 4}, GridParse),
])
def test_parse_generation_prechecks_keep_the_result_type(fn, kw, cls):
    unclosed = parse_generation(_gen("", unclosed=True), fn, **kw)
    assert type(unclosed) is cls and unclosed.failure is PF.THINKING_UNCLOSED
    assert all(v is None for v in unclosed[:-1])
    for blank in ("", " \n\t "):
        empty = parse_generation(_gen(blank), fn, **kw)
        assert type(empty) is cls and empty.failure is PF.EMPTY and all(v is None for v in empty[:-1])


def test_parse_generation_rejects_parse_fns_that_break_the_contract():
    with pytest.raises(TypeError, match="NamedTuple"):
        parse_generation(_gen(""), lambda text: (None, PF.NO_BOX))
    with pytest.raises(TypeError, match="空串"):
        parse_generation(_gen(""), lambda text: ParseResult(0, None))  # 对空串"成功"
    with pytest.raises(TypeError, match="空串"):
        parse_generation(_gen(""), lambda text: ParseResult(0, PF.NO_NUMBER))  # 失败却带着值
    # 正常文本不做这些检查，原样返回
    assert parse_generation(_gen("x"), lambda text: ParseResult(text, None)) == ("x", None)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, value", [
    # 代码块：信息串 json / JSON / 无；块外的说明文字与停止 token 忽略
    ('```json\n[1, 2]\n```', [1, 2]),
    ('```JSON\n{"a": 1}\n```', {"a": 1}),
    ('```\n[1, 2]\n```', [1, 2]),
    ('Sure! Here it is:\n```json\n[{"a": [1, 2]}]\n```\nHope this helps.<|im_end|>', [{"a": [1, 2]}]),
    ('```json\n3\n```', 3),  # 代码块里可以是任何 JSON 值
    ('```json [1, 2]```', [1, 2]),  # 单行代码块：信息串后面是空格
    ('[9] outside ```json\n[1]\n``` and {"b": 2}', [1]),  # 有代码块时只看代码块
    # 没有代码块：全文里唯一的顶层 [...] 或 {...}
    ('[1, 2]', [1, 2]),
    ('The box is [10, 20, 30, 40].', [10, 20, 30, 40]),
    ("Here's the result: {\"k\": \"it's [not] a list\"} ok", {"k": "it's [not] a list"}),  # 字符串里的括号不计
    ('a ] stray closer, then [1, [2, 3]]', [1, [2, 3]]),  # 顶层落单的 ] 忽略
    ('  \n[1]\n\n', [1]),
    ('{"a": "\\"}"}', {"a": '"}'}),  # 转义的引号
])
def test_extract_json_accepts(text, value):
    assert extract_json(text) == (value, None)


@pytest.mark.parametrize("text, failure", [
    ("no json here", PF.NO_JSON),
    ("", PF.NO_JSON),
    ("answer: 42", PF.NO_JSON),  # 只认 [...] 与 {...}
    # 多个候选 → ambiguous，即使只有一个能解析
    ('```json\n[1]\n```\n```json\n[2]\n```', PF.AMBIGUOUS),
    ('```python\nprint(1)\n```\n```json\n[2]\n```', PF.AMBIGUOUS),
    ('[1, 2] and [3, 4]', PF.AMBIGUOUS),
    ('{"a": 1}\n{"b": 2}', PF.AMBIGUOUS),
    ('[see note] [1, 2, 3, 4]', PF.AMBIGUOUS),
    ('[1, 2} then [3]', PF.AMBIGUOUS),  # 错配的一段也算一个候选
    ('{"a": 1, "a": 2}', PF.AMBIGUOUS),  # 重复键：同一字段两个值
    # 唯一的候选坏了 → malformed
    ('```json\n[1, 2]\n', PF.MALFORMED),  # 代码块没闭合（max_new_tokens 用完）
    ('```json\n[1, 2,]\n```', PF.MALFORMED),  # 尾逗号
    ("```json\n{'a': 1}\n```", PF.MALFORMED),  # 单引号：严格 JSON，不用 literal_eval 兜底
    ("{'a': 1}", PF.MALFORMED),
    ('```json\n// boxes\n[1]\n```', PF.MALFORMED),  # 注释
    ('[1, NaN]', PF.MALFORMED),
    ('[Infinity]', PF.MALFORMED),
    ('{"a": -Infinity}', PF.MALFORMED),
    ('[1, 2}', PF.MALFORMED),  # 括号错配
    ('[{"a": [1, 2]}', PF.MALFORMED),  # 到结尾仍未闭合
    ('```json\n```', PF.MALFORMED),  # 空代码块
    ('```json[1]```', PF.MALFORMED),  # 信息串后面不是空白：整块当内容，解析失败
    ('[' * 100000, PF.MALFORMED),
    ('```json\n' + '[' * 100000 + ']' * 100000 + '\n```', PF.MALFORMED),  # 递归过深
])
def test_extract_json_failures(text, failure):
    assert extract_json(text) == (None, failure)


def test_extract_json_type_error():
    with pytest.raises(TypeError):
        extract_json(None)


# ---------------------------------------------------------------------------
# parse_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, n", [
    ("3", 3), ("3.", 3), ("(3)", 3), ("There are 3 cats.", 3), ("0", 0), ("03", 3), ("12", 12),
    ("three", 3), ("Three.", 3), ("THREE", 3), ("There are twelve apples.", 12), ("zero", 0), ("Twenty", 20),
    ("Three (3).", 3), ("3 cats: 3 in total", 3),  # 同一个数值出现多次
    ("Answer: 7\n", 7), ("50%", 50),
    ("There are 3 cats in the 2nd row", 3),  # 2nd 不是数
    ("3D view: 4 boxes", 4),  # 3D 不是数
    ("I see 5 cats. v1.2", 5),  # 版本号里的数字不算（前面紧贴小数点或字母）
])
def test_parse_count_accepts(text, n):
    got = parse_count(text)
    assert got == (n, None) and type(got.value) is int


@pytest.mark.parametrize("text, failure", [
    ("no cats", PF.NO_NUMBER), ("a cat", PF.NO_NUMBER), ("", PF.NO_NUMBER), ("none", PF.NO_NUMBER),
    ("someone", PF.NO_NUMBER), ("often", PF.NO_NUMBER), ("２", PF.NO_NUMBER),  # 只认 ASCII 数字
    ("3 cats and 2 dogs", PF.MULTIPLE_NUMBERS), ("1, 2, 3", PF.MULTIPLE_NUMBERS),
    ("three or 4", PF.MULTIPLE_NUMBERS), ("3-4", PF.MULTIPLE_NUMBERS),
    ("There are 3, one of them black", PF.MULTIPLE_NUMBERS),  # 已知局限：代词 one
    ("twenty-one", PF.MULTIPLE_NUMBERS),  # 已知局限：连字符数词
    ("3.5", PF.MALFORMED), ("0.9", PF.MALFORMED), (".5", PF.MALFORMED), ("-2", PF.MALFORMED),
    ("count: -1", PF.MALFORMED), ("4 cats (confidence 0.9)", PF.MALFORMED),  # 小数先于多数值判
])
def test_parse_count_failures(text, failure):
    assert parse_count(text) == (None, failure)


# ---------------------------------------------------------------------------
# parse_choice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, letter", [
    # 第 1 步：normalize_choice
    ("B", "B"), ("b", "B"), ("(B)", "B"), ("[b]", "B"), ("B.", "B"), ("(B) right", "B"), ("B. right", "B"),
    # 第 2 步：明确陈述
    ("The answer is B", "B"), ("The answer is B.", "B"), ("Answer: (C)", "C"), ("answer: c.", "C"),
    ("option D", "D"), ("Option (A) is correct.", "A"), ("The correct option is D.", "D"),
    ("My choice is (A).", "A"), ("The answer is option B", "B"), ("The option is: B", "B"),
    ("The answer is B. So the answer is B.", "B"),  # 同一个字母说两次
    ("Answer: I think the answer is B", "B"),  # I 不在 letters 里，不算候选
    ("A. Answer: I think (A).", "A"),  # normalize_choice 因 "I" 判含糊；I 不是选项，候选只剩前缀字母 A
    ("C) I think option D", "C"),  # 第 1 步：normalize_choice 按 D12 N2 不把 "option D" 算另一个选项，直接给出 C
    # 第 3 步：全文里唯一的括号字母
    ("I pick (c).", "C"), ("It is [B], the right one.", "B"), ("Looking at the image, ( D ) fits.", "D"),
    ("(A) (A)", "A"),
])
def test_parse_choice_accepts(text, letter):
    assert parse_choice(text, "ABCD") == (letter, None)


@pytest.mark.parametrize("text, failure", [
    ("(A) left (B) right", PF.MULTIPLE_CHOICES),  # 回显的选项列表
    ("(A) left\n(B) right\n(C) up", PF.MULTIPLE_CHOICES),
    ("The answer is A or the answer is B", PF.MULTIPLE_CHOICES),
    ("Option A is wrong, option B is right", PF.MULTIPLE_CHOICES),
    ("Either (a) or (b).", PF.MULTIPLE_CHOICES),
    ("the answer is a dog", PF.NO_CHOICE),  # 冠词 a 不是选项
    ("I think it is on the left.", PF.NO_CHOICE),
    ("", PF.NO_CHOICE),
    ("E", PF.NO_CHOICE),  # 不在 letters 里
    ("The answer is E", PF.NO_CHOICE),
    ("(E) nothing", PF.NO_CHOICE),
    ("The answer is: D", PF.NO_CHOICE),  # is 后带冒号：misc 的 "answer is X" 定义不含它
    # 以前缀式字母开头、后文又认出不同的字母 → multiple_choices（D12 N2）
    ("A. No, the answer is B.", PF.MULTIPLE_CHOICES),
    ("(A) left (B) right. The answer is B.", PF.MULTIPLE_CHOICES),
    ("(C) is wrong, the answer is D", PF.MULTIPLE_CHOICES),
    ("A) and C) are both right", PF.NO_CHOICE),  # normalize_choice 判含糊；第 2、3 步不认 "C)"，前缀字母不单独接受
])
def test_parse_choice_failures(text, failure):
    assert parse_choice(text, "ABCD") == (None, failure)


def test_parse_choice_review_examples():
    """DEV-0006 审查第 1 轮给的 4 个例子：前缀式字母与后文不同的字母并存时不取后者，与 D12 N2 一致。"""
    assert parse_choice("A. No, the answer is B.", "ABCD") == (None, PF.MULTIPLE_CHOICES)
    assert parse_choice("(A) left (B) right. The answer is B.", "ABCD") == (None, PF.MULTIPLE_CHOICES)
    assert parse_choice("B. The answer is B.", "ABCD") == ("B", None)  # normalize_choice 直接给出 B
    assert parse_choice("The answer is B", "ABCD") == ("B", None)


def test_parse_choice_reuses_the_misc_definitions():
    """约定只在 metrics.misc 定义一处：parse_choice 用它公开的函数与字母写法，normalize_choice 的正则一字未变。"""
    from src.metrics import misc
    from src.parsing import answers

    # _OTHER_STATED 由 CHOICE_LETTER_FORM 拼出，编译后的正则与 DEV-0006 之前的原文逐字相同
    assert misc._OTHER_STATED.pattern == (
        r"(?i:answer)\s*(?:(?i:is)|:)\s*(?:[\(\[]\s*([A-Za-z])\s*[\)\]]|([A-Z])(?![A-Za-z])|([a-z])(?=\s*(?:[.,;:!?]|$)))"
    )
    assert answers._OPTION_STATED.pattern.endswith(misc.CHOICE_LETTER_FORM)
    for name in ("_LETTER_FORM", "_STATED", "_BRACKETED"):  # 原先的重复副本已删掉
        assert not hasattr(answers, name)
    assert misc.leading_choice_letter("(b) left") == "B" and misc.leading_choice_letter("C: yes") == "C"
    assert misc.leading_choice_letter("D) x") == "D" and misc.leading_choice_letter("  A. no") == "A"
    for no_prefix in ("B", "(B)", "B.", "The answer is B", "A dog", "I think (B)"):
        assert misc.leading_choice_letter(no_prefix) is None
    assert misc.stated_choice_letters("The answer is b. Answer: (C), answer is a dog") == ["B", "C"]
    assert misc.bracketed_choice_letters("(a) x [B] y ( c ) z C)") == ["A", "B", "C"]
    # 前缀式答案、后文没有别的选项时，normalize_choice 取的就是 leading_choice_letter
    for text in ("(b) left", "C: yes", "D) x", "A. the one on the left"):
        assert misc.normalize_choice(text) == misc.leading_choice_letter(text)


def test_parse_choice_letters_argument():
    assert parse_choice("(b)", ["a", "b"]) == ("B", None)
    assert parse_choice("(C)", "AB") == (None, PF.NO_CHOICE)
    # 括号字母只看 letters 里的：C 不是选项，所以 (A) 是唯一的候选
    assert parse_choice("(A) since (C) is not an option", "AB") == ("A", None)
    for bad in ("", "AA", "Ab a", ["AB"], [1], "A1"):
        with pytest.raises(ValueError):
            parse_choice("A", bad)
    with pytest.raises(TypeError):
        parse_choice(None, "AB")


# ---------------------------------------------------------------------------
# parse_named_choice
# ---------------------------------------------------------------------------

LR = ["left", "right"]
OBJ = ["television", "dining table", "table"]


@pytest.mark.parametrize("text, names, name", [
    ("left", LR, "left"), ("Left.", LR, "left"), ("**Right**", LR, "right"), ('"left"', LR, "left"),
    ("The right.", LR, "right"), ("  RIGHT \n", LR, "right"),
    ("The television", OBJ, "television"), ("'dining table'", OBJ, "dining table"), ("A table.", OBJ, "table"),
    ("the  dining\ntable", OBJ, "dining table"),  # 空白合并
    ("The dining table is closer.", OBJ, "dining table"),  # 长名字优先："table" 不再命中
    ("The table is closer to the camera.", OBJ, "table"),
    ("It is on the left side.", LR, "left"),
    ("The answer is left. left!", LR, "left"),  # 同一个名字出现多次
    ("It is not on the left.", LR, "left"),  # 已知局限：否定句只含一个名字
])
def test_parse_named_choice_accepts(text, names, name):
    assert parse_named_choice(text, names) == (name, None)


@pytest.mark.parametrize("text, names, failure", [
    ("It's the left one, not the right.", LR, PF.MULTIPLE_CHOICES),
    ("The table is closer than the dining table.", OBJ, PF.MULTIPLE_CHOICES),
    ("Which is closer, the television or the dining table? The television.", OBJ, PF.MULTIPLE_CHOICES),  # 回显问题
    ("neither", LR, PF.NO_CHOICE),
    ("leftmost", LR, PF.NO_CHOICE),  # 整词匹配
    ("the tablet", OBJ, PF.NO_CHOICE),
    ("televisions", OBJ, PF.NO_CHOICE),  # 复数不算
    ("", LR, PF.NO_CHOICE),
])
def test_parse_named_choice_failures(text, names, failure):
    assert parse_named_choice(text, names) == (None, failure)


def test_parse_named_choice_returns_callers_spelling_and_validates_names():
    assert parse_named_choice("the dining table", ["Dining Table", "TV"]) == ("Dining Table", None)
    assert normalize_phrase('  "The  Dining Table!" ') == "dining table"
    assert normalize_phrase("the the left.") == "left"
    for bad in (["left", "Left"], ["left", "the left"], ["", "x"], ["...", "x"], []):
        with pytest.raises(ValueError):
            parse_named_choice("left", bad)
    with pytest.raises(TypeError):
        parse_named_choice("left", "left")
    with pytest.raises(TypeError):
        parse_named_choice("left", ["left", 3])


# ---------------------------------------------------------------------------
# parse_polygons
# ---------------------------------------------------------------------------

W, H = 200, 100  # 故意 W ≠ H，轴序写反会得到不同的数


def test_parse_polygons_single_xy():
    got = parse_polygons("```json\n[[100, 200], [500, 200], [500, 800]]\n```", width=W, height=H)
    # x = a/1000·200，y = b/1000·100：(20, 20), (100, 20), (100, 80)
    assert got == ((((20.0, 20.0), (100.0, 20.0), (100.0, 80.0)),), None)


def test_parse_polygons_yx_swaps_before_scaling():
    got = parse_polygons("[[100, 200], [500, 200], [500, 800]]", width=W, height=H, axis_order="yx")
    # 点是 [y, x]：x = 200/1000·200 = 40，y = 100/1000·100 = 10；(40, 10), (40, 50), (160, 50)
    assert got == ((((40.0, 10.0), (40.0, 50.0), (160.0, 50.0)),), None)
    xy = parse_polygons("[[100, 200], [500, 200], [500, 800]]", width=W, height=H)
    assert xy.value != got.value  # 轴序写反一定能被抓住


def test_parse_polygons_multiple_and_decimal():
    text = "Two parts:\n[[[0, 0], [1000, 0], [1000, 500]], [[0, 500.5], [250, 1000], [0, 1000], [0.0, 750]]]"
    got = parse_polygons(text, width=W, height=H)
    assert got.failure is None and len(got.value) == 2
    assert got.value[0] == ((0.0, 0.0), (200.0, 0.0), (200.0, 50.0))
    # 500.5/1000·100 = 50.05，250/1000·200 = 50，750/1000·100 = 75
    np.testing.assert_allclose(got.value[1], [(0.0, 50.05), (50.0, 100.0), (0.0, 100.0), (0.0, 75.0)], rtol=0, atol=1e-9)
    assert all(isinstance(c, float) for poly in got.value for p in poly for c in p)


def test_parse_polygons_feed_the_rasterizer():
    """输出可直接交给 raster.polygons_to_mask：0–1000 的整幅矩形 → 整幅 mask。"""
    got = parse_polygons("[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]", width=W, height=H)
    mask = polygons_to_mask(got.value, H, W)
    assert mask.shape == (H, W) and mask.all()


@pytest.mark.parametrize("text, failure", [
    ("no polygon", PF.NO_POLYGON),
    ("[]", PF.NO_POLYGON),
    ("```json\n[]\n```", PF.NO_POLYGON),
    ("[[1, 2], [3, 4]] and [[5, 6]]", PF.AMBIGUOUS),
    ("```json\n[[1, 2]\n```", PF.MALFORMED),  # JSON 坏了
    ("[10, 20, 30, 40, 50, 60]", PF.MALFORMED),  # 扁平列表不是接受的形状
    ('{"polygon": [[1, 2], [3, 4], [5, 6]]}', PF.MALFORMED),
    ("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", PF.MALFORMED),  # 点不是两个数
    ('[[1, "2"], [3, 4], [5, 6]]', PF.MALFORMED),
    ("[[1, true], [3, 4], [5, 6]]", PF.MALFORMED),
    ("[[1, null], [3, 4], [5, 6]]", PF.MALFORMED),
    ("[[1, 1e999], [3, 4], [5, 6]]", PF.MALFORMED),  # json 读成 inf：不是有限数
    ("[[[1, 2], [3, 4], [5, 6]], [7, 8]]", PF.MALFORMED),  # 层数混杂
    ("[[]]", PF.MALFORMED),
    ("[[1, -2], [3, 4], [5, 6]]", PF.OUT_OF_RANGE),
    ("[[1, 2], [1000.5, 4], [5, 6]]", PF.OUT_OF_RANGE),
    ("[[1, 2], [3, 4]]", PF.TOO_FEW_POINTS),
    ("[[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4]]]", PF.TOO_FEW_POINTS),  # 一个坏了整条判失败
    ("[[[1, 2], [3, 4], [5, 6]], []]", PF.TOO_FEW_POINTS),
    # 检查顺序：malformed 先于 out_of_range 先于 too_few_points
    ('[[[1, 2], [3, 4]], [[2000, 0], [0, 0], [0, 0]], [[1, "x"], [0, 0], [0, 0]]]', PF.MALFORMED),
    ("[[[1, 2], [3, 4]], [[2000, 0], [0, 0], [0, 0]]]", PF.OUT_OF_RANGE),
])
def test_parse_polygons_failures(text, failure):
    assert parse_polygons(text, width=W, height=H) == (None, failure)


def test_parse_polygons_boundaries_and_arguments():
    ok = parse_polygons("[[0, 0], [1000, 0], [0, 1000]]", width=W, height=H)
    assert ok.failure is None and ok.value == (((0.0, 0.0), (200.0, 0.0), (0.0, 100.0)),)
    with pytest.raises(ValueError):
        parse_polygons("[]", width=W, height=H, axis_order="xyz")
    with pytest.raises(ValueError):
        parse_polygons("[]", width=0, height=H)
    with pytest.raises(TypeError):
        parse_polygons(None, width=W, height=H)
