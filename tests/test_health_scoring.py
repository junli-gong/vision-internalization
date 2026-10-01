"""健康检查的解析、逐题评分、汇总与比较（src/health/scoring.py，DEV-0008）。IFEval 的官方 checker 另见 test_health_ifeval.py。"""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from src import constants as C
from src.adapters.base import GenerationResult
from src.health import protocol as P
from src.health import scoring as S
from src.parsing import ParseFailure


def gen(text: str, *, unclosed: bool = False, n: int = 3) -> GenerationResult:
    return GenerationResult(text="" if unclosed else text, token_ids=tuple(range(n)), n_generated_tokens=n,
                            n_prompt_tokens=10, n_visual_tokens=0, wall_time_s=0.0, raw_text=text,
                            thinking_unclosed=unclosed)


# ---------------------------------------------------------------------------
# GSM8K 的两种解析
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, value", [
    ("3 + 4 = 7 apples.\n#### 7", Decimal(7)),
    ("Total: 1,234 + 1 = 1,235\n#### 1,235", Decimal(1235)),  # 千分位逗号
    ("He loses 5.\n#### -5", Decimal(-5)),  # 负数
    ("#### 12\nwait, let me redo it.\n#### 18", Decimal(18)),  # 多个 ####：取最后一个
    ("#### $1,000.50 total", Decimal("1000.50")),
    ("####\n  42 is the answer", Decimal(42)),  # #### 之后的第一个数，可以换行
    ("#####  9", Decimal(9)),
    ("#### 18 (or 20 if tax) ", Decimal(18)),  # 之后的第一个数，不是最后一个
])
def test_gsm8k_strict(text, value):
    assert S.parse_gsm8k_strict(text) == (value, None)


@pytest.mark.parametrize("text, failure", [
    ("The answer is 18.", S.HealthParseFailure.NO_MARKER),
    ("", S.HealthParseFailure.NO_MARKER),
    ("#### eighteen", ParseFailure.NO_NUMBER),
    ("5 apples then ####", ParseFailure.NO_NUMBER),
])
def test_gsm8k_strict_failures(text, failure):
    assert S.parse_gsm8k_strict(text) == (None, failure)


@pytest.mark.parametrize("text, value", [
    ("The answer is 1,234.", Decimal(1234)),
    ("costs $5 and then 7 more", Decimal(7)),
    ("pages 3-4", Decimal(4)),  # 负号前面紧挨数字时不算负号
    ("x = -12", Decimal(-12)),
    ("1,2,3", Decimal(3)),  # 逗号后不是 3 位数字：分隔符
    ("12,34", Decimal(34)),
    ("1,234,567 people", Decimal(1234567)),
    ("#### 18 dollars (18.5?)", Decimal("18.5")),
    ("Step 2: 3.14. Done", Decimal("3.14")),
    ("1.2.3", Decimal("1.2")),  # 紧挨小数点的数字不算新数的开头
])
def test_gsm8k_flexible(text, value):
    assert S.parse_gsm8k_flexible(text) == (value, None)


def test_gsm8k_flexible_failures_and_types():
    assert S.parse_gsm8k_flexible("no digits here") == (None, ParseFailure.NO_NUMBER)
    assert S.parse_gsm8k_flexible("") == (None, ParseFailure.NO_NUMBER)
    assert S.parse_gsm8k_flexible("about .5 of it") == (None, ParseFailure.NO_NUMBER)  # ".5" 不认
    for fn in (S.parse_gsm8k_flexible, S.parse_gsm8k_strict):
        with pytest.raises(TypeError):
            fn(None)


def test_gsm8k_integer_comparison():
    assert S.gsm8k_is_correct(Decimal("18.00"), 18)
    assert S.gsm8k_is_correct(Decimal("18"), 18)
    assert not S.gsm8k_is_correct(Decimal("18.5"), 18)
    assert not S.gsm8k_is_correct(Decimal("-18"), 18)
    assert not S.gsm8k_is_correct(None, 18)
    for bad in (True, 18.0, "18"):
        with pytest.raises(TypeError):
            S.gsm8k_is_correct(Decimal(18), bad)


def test_score_gsm8k_both_modes():
    item = {"id": 3, "question": "q", "answer": 1235}
    r = S.score_gsm8k(item, gen("1,000 + 235 = 1,235\n#### 1,235"))
    assert r == {"gold": 1235, "strict": {"parsed": "1235", "failure": None, "correct": True},
                 "flexible": {"parsed": "1235", "failure": None, "correct": True}}
    r = S.score_gsm8k(item, gen("So the answer is 1235 apples."))
    assert r["strict"] == {"parsed": None, "failure": "no_marker", "correct": False}
    assert r["flexible"] == {"parsed": "1235", "failure": None, "correct": True}
    r = S.score_gsm8k(item, gen("", unclosed=True))
    assert r["strict"]["failure"] == r["flexible"]["failure"] == "thinking_unclosed"
    r = S.score_gsm8k(item, gen("  \n"))
    assert r["strict"]["failure"] == r["flexible"]["failure"] == "empty"


# ---------------------------------------------------------------------------
# MMLU-Pro：parse_choice，字母上限按选项数截断
# ---------------------------------------------------------------------------

ITEM4 = {"id": 1, "question": "q", "options": ["a", "b", "c", "d"], "answer": "C", "answer_index": 2}
ITEM10 = {"id": 2, "question": "q", "options": list("abcdefghij"), "answer": "J", "answer_index": 9}


@pytest.mark.parametrize("item, text, parsed, failure, correct", [
    (ITEM4, "C", "C", None, True),
    (ITEM4, "The answer is (C).", "C", None, True),
    (ITEM4, "B. something", "B", None, False),
    (ITEM4, "E", None, "no_choice", False),  # 只有 4 个选项：E 不是候选
    (ITEM4, "The answer is (E).", None, "no_choice", False),
    (ITEM10, "E", "E", None, False),  # 10 个选项时 E 是合法选项
    (ITEM10, "J", "J", None, True),
    (ITEM10, "A. No, the answer is B.", None, "multiple_choices", False),
    (ITEM4, "", None, "empty", False),
])
def test_score_mmlu_pro(item, text, parsed, failure, correct):
    r = S.score_mmlu_pro(item, gen(text))
    assert (r["parsed"], r["failure"], r["correct"], r["gold"]) == (parsed, failure, correct, item["answer"])
    assert r["letters"] == "ABCDEFGHIJ"[: len(item["options"])]


def test_score_mmlu_pro_thinking_unclosed():
    r = S.score_mmlu_pro(ITEM10, gen("<think>long", unclosed=True))
    assert r["failure"] == "thinking_unclosed" and r["correct"] is False


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def test_pct1_rounding_and_empty():
    assert S.pct1(1, 3) == 33.3 and S.pct1(2, 3) == 66.7
    assert S.pct1(1, 8) == 12.5 and S.pct1(1, 80) == 1.3  # 1.25 四舍五入到 1.3（不是银行家舍入）
    assert S.pct1(3, 80) == 3.8 and S.pct1(0, 50) == 0.0 and S.pct1(50, 50) == 100.0
    assert S.pct1(0, 0) is None  # 空集合报 null，不与"全错"同形
    assert S.round1(Fraction(-5, 2)) == -2.5 and S.round1(Fraction(-25, 100)) == -0.3


def _rec(benchmark: str, i: int, ok: bool, **extra):
    base = {"id": i, "text": "x", "thinking_unclosed": False, "reached_max_new_tokens": False}
    if benchmark == "mmlu_pro":
        base.update(correct=ok, failure=None if ok else "no_choice")
    elif benchmark == "gsm8k":
        base.update(strict={"correct": ok, "failure": None if ok else "no_marker"},
                    flexible={"correct": ok, "failure": None if ok else "no_number"})
    else:
        base.update(strict={"follow_all": ok, "follow_list": [ok, True]}, loose={"follow_all": True, "follow_list": [True, True]})
    base.update(extra)
    return base


def _records(n_ok: dict[str, int], n: dict[str, int] | None = None, *, version: int = P.PROTOCOL_VERSION):
    """按 version 的布局造逐题记录。v2 的 mmlu_pro_fmt32 默认与 mmlu_pro 相同（n_ok 里可另给）。"""
    n = n or dict(P.COUNTS)
    out = {}
    for key, reading in P.record_readings(version).items():
        b = reading.benchmark
        out[key] = [_rec(b, i, i < n_ok.get(key, n_ok[b])) for i in range(n[b])]
    return out


def test_summarize_counts_everything_from_records():
    recs = _records({"mmlu_pro": 57, "gsm8k": 31, "ifeval": 20})
    recs["gsm8k"][0]["strict"] = {"correct": False, "failure": "no_marker"}  # strict 与 flexible 不同
    recs["mmlu_pro"][-1]["thinking_unclosed"] = True
    recs["ifeval"][-1]["reached_max_new_tokens"] = True
    s = S.summarize(recs)
    assert s["headline"] == {"mmlu_pro": 57.0, "gsm8k": 62.0, "ifeval": 40.0}
    assert s["scores"]["gsm8k"]["strict"] == {"n": 50, "n_correct": 30, "pct": 60.0}
    # instruction-level：每题两条指令，第二条恒为 True
    assert s["scores"]["ifeval"]["instruction_level_strict"] == {"n": 100, "n_correct": 70, "pct": 70.0}
    assert s["scores"]["ifeval"]["prompt_level_loose"]["pct"] == 100.0
    assert s["parse_tally"]["mmlu_pro"] == {"n": 100, "n_ok": 57, "n_failed": 43, "failure_rate": 0.43,
                                            "failures": {"no_choice": 43}}
    assert s["parse_tally"]["gsm8k_strict"]["failures"] == {"no_marker": 20}
    assert s["parse_tally"]["gsm8k_flexible"]["failures"] == {"no_number": 19}
    assert s["diagnostics"]["mmlu_pro"]["n_thinking_unclosed"] == 1
    assert s["diagnostics"]["ifeval"]["n_reached_max_new_tokens"] == 1


def test_summarize_empty_is_null():
    s = S.summarize({})
    assert s["headline"] == {"mmlu_pro": None, "gsm8k": None, "ifeval": None}
    assert s["parse_tally"]["mmlu_pro"]["failure_rate"] is None
    assert s["unscored"] == {}


def _unscore_ifeval(recs):
    for r in recs["ifeval"]:
        r["strict"] = r["loose"] = None
        r["instruction_id_list"] = ["punctuation:no_comma", "detectable_format:title"]
    return recs


def test_summarize_unscored_ifeval_is_null_with_reason():
    recs = _unscore_ifeval(_records({"mmlu_pro": 57, "gsm8k": 31, "ifeval": 20}))
    s = S.summarize(recs)
    assert s["headline"] == {"mmlu_pro": 57.0, "gsm8k": 62.0, "ifeval": None}
    assert s["unscored"] == {"ifeval": S.IFEVAL_SKIP_REASON} and "--rescore" in S.IFEVAL_SKIP_REASON
    assert s["scores"]["ifeval"]["prompt_level_strict"] == {"n": 50, "n_correct": None, "pct": None}
    assert s["scores"]["ifeval"]["instruction_level_loose"] == {"n": 100, "n_correct": None, "pct": None}
    assert list(s) == ["scores", "headline", "unscored", "parse_tally", "diagnostics"]
    recs["ifeval"][0]["strict"] = {"follow_all": True, "follow_list": [True, True]}  # 混杂：有的评了、有的没评
    with pytest.raises(ValueError, match="混杂|有的"):
        S.summarize(recs)


def test_unknown_failure_string_is_rejected():
    recs = _records({"mmlu_pro": 0, "gsm8k": 0, "ifeval": 0}, {"mmlu_pro": 1, "gsm8k": 0, "ifeval": 0})
    recs["mmlu_pro"][0]["failure"] = "no_choise"
    with pytest.raises(KeyError):
        S.summarize(recs)


# ---------------------------------------------------------------------------
# 比较：标红规则
# ---------------------------------------------------------------------------


def _report(n_ok, *, sha="a" * 64, version=P.PROTOCOL_VERSION, n=None, key="m"):
    recs = _records(n_ok, n, version=version if version in P.SUPPORTED_PROTOCOL_VERSIONS else P.PROTOCOL_VERSION)
    summary = S.summarize(recs, protocol_version=version) if version in P.SUPPORTED_PROTOCOL_VERSIONS else {}
    return {"protocol_version": version, "items_file": {"sha256": sha}, "limit": None, "model": {"key": key},
            "records": recs, **summary}


def _row(res, reading, metric):
    return next(r for r in res["rows"] if r["reading"] == reading and r["metric"] == metric)


def test_compare_flags_drops_strictly_greater_than_5pp():
    base = _report({"mmlu_pro": 60, "gsm8k": 30, "ifeval": 25}, key="base")
    exactly5 = _report({"mmlu_pro": 55, "gsm8k": 28, "ifeval": 25}, key="merged")
    res = S.compare(base, exactly5)
    assert _row(res, "mmlu_pro", "accuracy")["delta_pp"] == -5.0
    assert _row(res, "mmlu_pro", "accuracy")["flag"] is False  # 恰好 5 个点不标
    assert _row(res, "gsm8k", "flexible")["delta_pp"] == -4.0 and not _row(res, "gsm8k", "flexible")["flag"]
    assert res["any_flag"] is False and res["flagged"] == []

    worse = _report({"mmlu_pro": 54, "gsm8k": 27, "ifeval": 28}, key="merged")
    res = S.compare(base, worse)
    mm, gs, ie = _row(res, "mmlu_pro", "accuracy"), _row(res, "gsm8k", "flexible"), _row(res, "ifeval", "prompt_level_strict")
    assert (mm["delta_pp"], mm["flag"], mm["lost"], mm["gained"]) == (-6.0, True, 6, 0)
    assert (gs["delta_pp"], gs["flag"]) == (-6.0, True)
    assert (ie["delta_pp"], ie["flag"], ie["gained"]) == (6.0, False, 3)
    assert res["flagged"] == ["mmlu_pro", "gsm8k"] and res["any_flag"] is True
    assert (res["base_model"], res["merged_model"], res["threshold_pp"]) == ("base", "merged", 5)
    # 非标题口径只报差值，不标
    assert _row(res, "gsm8k", "strict")["flag"] is None and _row(res, "gsm8k", "strict")["delta_pp"] == -6.0
    assert _row(res, "ifeval", "instruction_level_strict")["n"] == 100


def test_compare_lost_and_gained_are_paired():
    base = _report({"mmlu_pro": 50, "gsm8k": 25, "ifeval": 25})
    merged = copy.deepcopy(base)
    # 同样 50 题对，但换了一批：丢 10、得 10，差值为 0
    for i in range(40, 60):
        merged["records"]["mmlu_pro"][i]["correct"] = not merged["records"]["mmlu_pro"][i]["correct"]
    r = _row(S.compare(base, merged), "mmlu_pro", "accuracy")
    assert (r["delta_pp"], r["lost"], r["gained"], r["flag"]) == (0.0, 10, 10, False)


@pytest.mark.parametrize("better, worse", [(0, 0), (0, 6), (1, 9), (9, 1), (5, 5), (5, 15), (0, 1), (3, 3), (12, 30),
                                           (0, 50)])
def test_sign_test_matches_scipy_binomtest(better, worse):
    from scipy.stats import binomtest

    expected = 1.0 if better + worse == 0 else binomtest(better, better + worse, 0.5).pvalue
    assert S.sign_test_p(better, worse) == pytest.approx(expected, rel=1e-12, abs=0)


def test_sign_test_exact_values_and_errors():
    assert S.sign_test_p(0, 0) == 1.0  # 没有不一致的题
    assert S.sign_test_p(0, 6) == 2 / 64 == S.sign_test_p(6, 0)  # 双侧、对称
    assert S.sign_test_p(1, 9) == 22 / 1024
    assert S.sign_test_p(5, 15) == 43400 / 1048576
    for bad in ((-1, 2), (1.0, 2), (True, 2)):
        with pytest.raises(ValueError):
            S.sign_test_p(*bad)


def test_compare_ci_and_sign_test_are_reported_alongside_flag():
    from src.stats import paired_bootstrap_ci, ratio_of_sums

    base = _report({"mmlu_pro": 60, "gsm8k": 30, "ifeval": 25})
    merged = _report({"mmlu_pro": 50, "gsm8k": 30, "ifeval": 25})
    # GSM8K：同样 30 题对，但换了 6 题（3 丢 3 得）；IFEval 第二条指令在 4 题上变成未遵循
    for i in (0, 1, 2):
        merged["records"]["gsm8k"][i]["flexible"] = {"correct": False, "failure": "no_number"}
    for i in (30, 31, 32):
        merged["records"]["gsm8k"][i]["flexible"] = {"correct": True, "failure": None}
    for i in (40, 41, 42, 43):
        merged["records"]["ifeval"][i]["strict"]["follow_list"] = [False, False]
    res = S.compare(base, merged)
    mm = _row(res, "mmlu_pro", "accuracy")
    assert (mm["delta_pp"], mm["flag"]) == (-10.0, True)
    assert mm["sign_test"] == {"n_better": 0, "n_worse": 10, "p_value": 2 / 1024}
    # CI 就是 src.stats.paired_bootstrap_ci（默认种子、1000 次）对逐题 0/1 的结果，换成百分点、一位小数
    b01 = [float(r["correct"]) for r in base["records"]["mmlu_pro"]]
    m01 = [float(r["correct"]) for r in merged["records"]["mmlu_pro"]]
    ci = paired_bootstrap_ci(m01, b01)
    assert mm["ci95_pp"] == [S.round1(Fraction(ci.low) * 100), S.round1(Fraction(ci.high) * 100)]
    assert mm["ci95_pp"][0] < mm["delta_pp"] < mm["ci95_pp"][1] < 0
    gs = _row(res, "gsm8k", "flexible")
    assert (gs["delta_pp"], gs["lost"], gs["gained"], gs["flag"]) == (0.0, 3, 3, False)
    assert gs["sign_test"] == {"n_better": 3, "n_worse": 3, "p_value": 1.0}
    assert gs["ci95_pp"][0] < 0 < gs["ci95_pp"][1]
    # instruction-level：按题重采样，统计量是 Σ遵循数 / Σ指令数；符号检验按题计（4 题各少遵循 1 条）
    il = _row(res, "ifeval", "instruction_level_strict")
    assert (il["n"], il["delta_pp"], il["lost"], il["sign_test"]["n_worse"]) == (100, -4.0, 4, 4)
    per = lambda rep: [(sum(r["strict"]["follow_list"]), 2) for r in rep["records"]["ifeval"]]
    ci = paired_bootstrap_ci(np.array(per(merged), float), np.array(per(base), float), statistic=ratio_of_sums)
    assert il["ci95_pp"] == [S.round1(Fraction(ci.low) * 100), S.round1(Fraction(ci.high) * 100)]
    assert il["flag"] is None  # 非标题口径不标
    # 标红规则不变：只看差值；统计量与之并列
    assert res["flagged"] == ["mmlu_pro"]
    assert res["statistics"]["bootstrap"] == {"n_resamples": C.BOOTSTRAP_N_RESAMPLES, "level": C.BOOTSTRAP_CI_LEVEL,
                                              "seed": C.BOOTSTRAP_SEED}
    assert [k for k in mm][-3:] == ["flag", "ci95_pp", "sign_test"]


def test_compare_with_unscored_ifeval():
    base = _report({"mmlu_pro": 60, "gsm8k": 30, "ifeval": 25})
    merged_recs = _unscore_ifeval(_records({"mmlu_pro": 50, "gsm8k": 30, "ifeval": 0}))
    merged_recs["ifeval"][0]["reached_max_new_tokens"] = True
    merged = {**base, "records": merged_recs, **S.summarize(merged_recs)}
    res = S.compare(base, merged)
    assert res["unscored"] == ["ifeval"] and res["flagged"] == ["mmlu_pro"]
    rows = [r for r in res["rows"] if r["benchmark"] == "ifeval"]
    assert len(rows) == 4 and all(not r["scored"] and r["flag"] is None and r["ci95_pp"] is None
                                  and r["sign_test"] is None and r["delta_pp"] is None for r in rows)
    assert rows[0]["base_pct"] == 50.0 and rows[0]["merged_pct"] is None  # 评了分的一边照常给出
    # 截断率与评分无关：未评分的项照样给出
    t = next(t for t in res["truncation"]["rows"] if t["reading"] == "ifeval")
    assert (t["base_n_reached"], t["merged_n_reached"], t["delta_pp"], t["notice"]) == (0, 1, 2.0, False)


# ---------------------------------------------------------------------------
# 协议 v2（D28-7）：读数、诊断读数不标红、截断率提示、与 v1 的输出比较
# ---------------------------------------------------------------------------

FMT32 = P.MMLU_PRO_FMT32


def test_summarize_v2_has_both_mmlu_pro_readings():
    recs = _records({"mmlu_pro": 57, FMT32: 31, "gsm8k": 31, "ifeval": 20})
    s = S.summarize(recs)
    assert list(s["scores"]) == list(s["diagnostics"]) == [r.name for r in P.READINGS]
    assert s["headline"] == {"mmlu_pro": 57.0, "gsm8k": 62.0, "ifeval": 40.0}  # 标题：MMLU-Pro 取 256 的读数
    assert s["scores"][FMT32]["accuracy"] == {"n": 100, "n_correct": 31, "pct": 31.0}
    assert list(s["parse_tally"]) == ["mmlu_pro", FMT32, "gsm8k_strict", "gsm8k_flexible"]
    assert s["parse_tally"][FMT32]["failures"] == {"no_choice": 69}
    with pytest.raises(ValueError, match="v2"):
        S.summarize({**recs, "mmlu_pro_fmt64": []})


def test_fmt32_is_diagnostic_never_flagged():
    """MMLU-Pro 的 256 读数不变、32 读数掉 20 个点（"变啰嗦了"）：fmt32 给出差值、CI 与符号检验，不标红。"""
    base = _report({"mmlu_pro": 60, FMT32: 50, "gsm8k": 30, "ifeval": 25})
    merged = _report({"mmlu_pro": 60, FMT32: 30, "gsm8k": 30, "ifeval": 25})
    res = S.compare(base, merged)
    fmt = _row(res, FMT32, "accuracy")
    assert (fmt["delta_pp"], fmt["flag"], fmt["headline"], fmt["diagnostic"], fmt["max_new_tokens"]) == (
        -20.0, None, False, True, 32)
    assert fmt["ci95_pp"][1] < 0 and fmt["sign_test"]["n_worse"] == 20
    head = _row(res, "mmlu_pro", "accuracy")
    assert (head["delta_pp"], head["flag"], head["headline"], head["max_new_tokens"]) == (0.0, False, True, 256)
    assert res["any_flag"] is False and res["flagged"] == []
    # 256 的读数掉 6 个点：标红
    res = S.compare(base, _report({"mmlu_pro": 54, FMT32: 50, "gsm8k": 30, "ifeval": 25}))
    assert res["flagged"] == ["mmlu_pro"] and _row(res, FMT32, "accuracy")["flag"] is None


def _set_reached(report, reading, k):
    for i, r in enumerate(report["records"][reading]):
        r["reached_max_new_tokens"] = i < k
    return report


def test_truncation_notice_strictly_more_than_10pp():
    """写满 max_new_tokens 的比例变化：恰好 10 个点不提示，超过 10 个点（任一方向）提示；与标红分开。"""
    base = _report({"mmlu_pro": 60, "gsm8k": 30, "ifeval": 25})
    _set_reached(base, FMT32, 40)
    _set_reached(base, "gsm8k", 10)
    merged = _report({"mmlu_pro": 60, "gsm8k": 30, "ifeval": 25})
    _set_reached(merged, FMT32, 30)  # 100 题少 10 题：-10.0，不提示
    _set_reached(merged, "gsm8k", 16)  # 50 题多 6 题：+12.0，提示
    _set_reached(merged, "ifeval", 6)  # 50 题多 6 题：+12.0，提示（基座 0）
    res = S.compare(base, merged)
    tr = {t["reading"]: t for t in res["truncation"]["rows"]}
    assert (tr[FMT32]["delta_pp"], tr[FMT32]["notice"]) == (-10.0, False)
    assert (tr["gsm8k"]["base_pct"], tr["gsm8k"]["merged_pct"], tr["gsm8k"]["delta_pp"], tr["gsm8k"]["notice"]) == (
        20.0, 32.0, 12.0, True)
    assert (tr["mmlu_pro"]["delta_pp"], tr["mmlu_pro"]["max_new_tokens"]) == (0.0, 256)
    assert res["truncation"]["notices"] == ["gsm8k", "ifeval"] and res["truncation"]["threshold_pp"] == 10
    assert res["any_flag"] is False  # 截断率提示不是标红
    _set_reached(merged, FMT32, 29)  # -11.0：提示
    assert S.compare(base, merged)["truncation"]["notices"] == [FMT32, "gsm8k", "ifeval"]


def test_compare_v1_reports_map_32_to_fmt32():
    """v1 的 MMLU-Pro（32 token）按口径对上 v2 的 mmlu_pro_fmt32；256 的标题读数只在 v2 一边：不可比，写明原因。"""
    v1 = _report({"mmlu_pro": 40, "gsm8k": 30, "ifeval": 25}, version=1)
    assert list(v1["records"]) == list(P.BENCHMARKS) and v1["headline"]["mmlu_pro"] == 40.0
    v2 = _report({"mmlu_pro": 70, FMT32: 30, "gsm8k": 30, "ifeval": 25})
    res = S.compare(v1, v2)
    assert res["protocol_versions"] == {"base": 1, "merged": 2} and res["incomparable"] == ["mmlu_pro"]
    head = _row(res, "mmlu_pro", "accuracy")
    assert (head["comparable"], head["base_pct"], head["merged_pct"], head["flag"]) == (False, None, 70.0, None)
    assert S.INCOMPARABLE in head["note"] and head["note"] == res["notes"][0]
    fmt = _row(res, FMT32, "accuracy")
    assert (fmt["base_pct"], fmt["merged_pct"], fmt["delta_pp"], fmt["flag"]) == (40.0, 30.0, -10.0, None)
    assert res["any_flag"] is False  # v1 的 32 掉 10 个点：只是诊断
    # 两边都是 v1：同样不标红（D28-7 起 32 token 只是格式诊断），256 的标题两边都没有
    res = S.compare(v1, _report({"mmlu_pro": 20, "gsm8k": 30, "ifeval": 25}, version=1))
    assert res["incomparable"] == ["mmlu_pro"] and res["any_flag"] is False
    assert _row(res, FMT32, "accuracy")["delta_pp"] == -20.0
    assert "没有这个读数" in res["notes"][0] and "协议 v1" in res["notes"][0]
    assert [t["reading"] for t in res["truncation"]["rows"]] == [FMT32, "gsm8k", "ifeval"]


def test_compare_refuses_incomparable_runs(tmp_path):
    base = _report({"mmlu_pro": 50, "gsm8k": 25, "ifeval": 25})
    with pytest.raises(ValueError, match="题目文件"):
        S.compare(base, _report({"mmlu_pro": 50, "gsm8k": 25, "ifeval": 25}, sha="b" * 64))
    for bad_version in (P.PROTOCOL_VERSION + 1, 0, None, True):
        with pytest.raises(ValueError, match="协议版本"):
            S.compare(base, _report({"mmlu_pro": 50, "gsm8k": 25, "ifeval": 25}, version=bad_version))
    limited = _report({"mmlu_pro": 3, "gsm8k": 3, "ifeval": 3}, n={"mmlu_pro": 3, "gsm8k": 3, "ifeval": 3})
    with pytest.raises(ValueError, match="limit"):
        S.compare(base, limited)
    # 也接受文件路径
    p = tmp_path / "b.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    assert S.compare(str(p), p)["any_flag"] is False
