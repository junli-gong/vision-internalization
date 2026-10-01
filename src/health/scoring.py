"""健康检查的解析、逐题评分、汇总与比较（简报 §8.11；DEV-0008）。

- MMLU-Pro：src.parsing.parse_choice，letters 取本题的实际选项字母（最多到 J），经 parse_generation 先判思考段未闭合与空回答。
- GSM8K：本模块的两个解析函数（不改 src/parsing）：
  - strict：回答里最后一个 "####" 之后的第一个数；没有 "####" 判 no_marker，"####" 之后没有数判 no_number；
  - flexible：全文最后一个数；一个数都没有判 no_number。
  数的写法：可带负号（负号前面紧挨数字时不算负号，"3-4" 是 3 与 4）、可带千分位逗号（逗号之后必须恰好 3 位数字，
  否则逗号当分隔符："1,2,3" 是三个数）、可带小数部分；前面紧挨数字或小数点的不算新数的开头（".5" 不认）。
  去掉千分位逗号后按十进制精确值与参考答案（整数）比较，相等即对："18.00" 对 18 算对，"18.5" 对 18 算错。
- IFEval：官方 checker（src/health/ifeval，只改 import 路径）的 test_instruction_following_strict / _loose，
  回答取 GenerationResult.text（剥离思考段之后；思考段未闭合时为空串，官方规则下全部指令判未遵循）。
  评分前把 langdetect 的种子固定为 protocol.LANGDETECT_SEED（langdetect 默认每次随机，同一回答可能判出不同语言）。
- 汇总：每个口径都从逐题记录重新数出来（同一条路径给出分数、比较与门报告的数），百分数保留一位小数（四舍五入）。
  IFEval 可以先不评分（--skip-ifeval-scoring），这时它的分数为 null，unscored 里写明原因；之后用 --rescore 补评。
- 比较：compare(base, merged) 按精确分数算差值（百分点），标题读数的标题口径下降严格大于 RED_FLAG_DROP_PP 即标出；
  与标红并列给出按题配对的 bootstrap 95% CI 与不一致题对上的精确符号检验 p 值（审查第 1 轮）。
- 协议 v2（D28-7，DEV-0013）：汇总与比较的单位是读数（protocol.READINGS）。MMLU-Pro 的标题读数 mmlu_pro 是 256 token，
  诊断读数 mmlu_pro_fmt32 是同一次生成的前 32 个 token：给出差值与 CI，不标红。比较另给截断率提示（写满 max_new_tokens
  的比例变化严格大于 TRUNCATION_NOTICE_PP 个百分点），与标红分开。v1 的输出照旧可读、可比：v1 的 MMLU-Pro（32 token）
  按口径对上 v2 的 mmlu_pro_fmt32，256 的标题读数只在 v2 一边时写明"口径不同、标题不可比"。
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED
from src.health import protocol as P
from src.parsing import ParseFailure, ParseResult, ParseTally, parse_choice, parse_generation
from src.stats import paired_bootstrap_ci, ratio_of_sums

# ---------------------------------------------------------------------------
# GSM8K 解析
# ---------------------------------------------------------------------------


class HealthParseFailure(str, enum.Enum):
    """健康检查自己的失败原因（ParseFailure 里没有的）。ParseTally 按取值字符串计数，可与 ParseFailure 混收。"""

    NO_MARKER = "no_marker"  # GSM8K strict：回答里没有 "####"


GSM8K_MARKER = "####"
_NUMBER = re.compile(r"(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)")


def _to_decimal(token: str) -> Decimal:
    return Decimal(token.replace(",", ""))


def parse_gsm8k_strict(text: str) -> ParseResult:
    """strict 口径：最后一个 "####" 之后的第一个数（Decimal）。"""
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    k = text.rfind(GSM8K_MARKER)
    if k < 0:
        return ParseResult(None, HealthParseFailure.NO_MARKER)
    m = _NUMBER.search(text, k + len(GSM8K_MARKER))
    if m is None:
        return ParseResult(None, ParseFailure.NO_NUMBER)
    return ParseResult(_to_decimal(m.group()), None)


def parse_gsm8k_flexible(text: str) -> ParseResult:
    """flexible 口径：全文最后一个数（Decimal）。"""
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    last = None
    for last in _NUMBER.finditer(text):
        pass
    if last is None:
        return ParseResult(None, ParseFailure.NO_NUMBER)
    return ParseResult(_to_decimal(last.group()), None)


def gsm8k_is_correct(value: Decimal | None, gold: int) -> bool:
    """解析值与参考答案（整数）按精确数值相等即对；解析失败记错。"""
    if isinstance(gold, bool) or not isinstance(gold, int):
        raise TypeError(f"GSM8K 参考答案必须是 int，收到 {gold!r}")
    return value is not None and value == gold


# ---------------------------------------------------------------------------
# 逐题评分：输入题目与 GenerationResult，输出可写进 JSON 的记录
# ---------------------------------------------------------------------------


def _failure(p: Any) -> str | None:
    return None if p.failure is None else p.failure.value


def score_mmlu_pro(item: Mapping[str, Any], result: Any) -> dict[str, Any]:
    letters = P.option_letters(len(item["options"]))
    p = parse_generation(result, parse_choice, letters=letters)
    return {"letters": letters, "parsed": p.value, "failure": _failure(p), "gold": item["answer"],
            "correct": p.value is not None and p.value == item["answer"]}


def score_gsm8k(item: Mapping[str, Any], result: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"gold": item["answer"]}
    for name, fn in (("strict", parse_gsm8k_strict), ("flexible", parse_gsm8k_flexible)):
        p = parse_generation(result, fn)
        out[name] = {"parsed": None if p.value is None else str(p.value), "failure": _failure(p),
                     "correct": gsm8k_is_correct(p.value, item["answer"])}
    return out


def ifeval_follow(item: Mapping[str, Any], response: str) -> dict[str, Any]:
    """官方 checker 的 strict 与 loose 两种判定。item 用题目文件里的原始 prompt、instruction_id_list、kwargs。"""
    from langdetect import DetectorFactory

    from src.health.ifeval import evaluation_lib

    DetectorFactory.seed = P.LANGDETECT_SEED
    inp = evaluation_lib.InputExample(
        key=item["id"],
        instruction_id_list=list(item["instruction_id_list"]),
        prompt=item["prompt"],
        kwargs=[dict(kw) for kw in item["kwargs"]],
    )
    prompt_to_response = {item["prompt"]: response}
    out = {}
    for name, fn in (("strict", evaluation_lib.test_instruction_following_strict),
                     ("loose", evaluation_lib.test_instruction_following_loose)):
        o = fn(inp, prompt_to_response)
        out[name] = {"follow_all": bool(o.follow_all_instructions),
                     "follow_list": [bool(x) for x in o.follow_instruction_list]}
    return out


def score_ifeval(item: Mapping[str, Any], result: Any) -> dict[str, Any]:
    return {"instruction_id_list": list(item["instruction_id_list"]), **ifeval_follow(item, result.text)}


SCORERS: Mapping[str, Callable[[Mapping[str, Any], Any], dict[str, Any]]] = {
    "mmlu_pro": score_mmlu_pro,
    "gsm8k": score_gsm8k,
    "ifeval": score_ifeval,
}

# ---------------------------------------------------------------------------
# 汇总：口径 → 逐单位的对错（单位 = 题，或 IFEval instruction-level 的 (题, 第几条指令)）
# ---------------------------------------------------------------------------

METRICS: Mapping[str, tuple[str, ...]] = {
    "mmlu_pro": ("accuracy",),
    "gsm8k": ("flexible", "strict"),
    "ifeval": ("prompt_level_strict", "instruction_level_strict", "prompt_level_loose", "instruction_level_loose"),
}


IFEVAL_SKIP_REASON = (
    "--skip-ifeval-scoring: IFEval generated but not scored (generate first, score later; the punkt_tab license is "
    "pending the project lead's answer, DECISIONS_NEEDED N12). Score later with scripts/health_check.py --rescore."
)


def is_scored(benchmark: str, records: list[Mapping[str, Any]]) -> bool:
    """该项的记录是否评了分。只有 IFEval 可能没评（--skip-ifeval-scoring：strict、loose 为 None）；
    同一项的记录要么全评、要么全没评，混杂时报错。没有记录时算已评（分数为 null 的原因是没有题）。"""
    if benchmark != "ifeval":
        return True
    flags = {r["strict"] is not None for r in records}
    if len(flags) > 1:
        raise ValueError("IFEval 的记录有的评了分、有的没评")
    return flags != {False}


def unit_outcomes(benchmark: str, metric: str, records: list[Mapping[str, Any]]) -> dict[str, bool]:
    """某口径下每个评分单位的对错，键为 "id" 或 "id#k"（IFEval instruction-level 的第 k 条指令）。"""
    out: dict[str, bool] = {}
    for r in records:
        if benchmark == "mmlu_pro" and metric == "accuracy":
            out[str(r["id"])] = bool(r["correct"])
        elif benchmark == "gsm8k" and metric in ("flexible", "strict"):
            out[str(r["id"])] = bool(r[metric]["correct"])
        elif benchmark == "ifeval" and metric.startswith("prompt_level_"):
            out[str(r["id"])] = bool(r[metric.removeprefix("prompt_level_")]["follow_all"])
        elif benchmark == "ifeval" and metric.startswith("instruction_level_"):
            for k, ok in enumerate(r[metric.removeprefix("instruction_level_")]["follow_list"]):
                out[f"{r['id']}#{k}"] = bool(ok)
        else:
            raise KeyError(f"未知口径 {benchmark}/{metric}")
    return out


def round1(x: int | Fraction) -> float:
    """精确分数 → 保留一位小数（四舍五入，.x5 远离零进位）。"""
    x = Fraction(x)
    return float((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def pct1(num: int, den: int) -> float | None:
    """百分数 100·num/den，保留一位小数；den = 0 时为 None（空集合报 null，不与"全错"同形）。"""
    return None if den == 0 else round1(Fraction(100 * num, den))


_FAILURES = {f.value: f for f in (*ParseFailure, *HealthParseFailure)}


def _tally(failures: list[str | None]) -> ParseTally:
    return ParseTally(None if f is None else _FAILURES[f] for f in failures)


def item_counts(benchmark: str, metric: str, records: list[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """按题汇总：每题 (对的单位数, 单位数)，顺序同 records。二值口径每题一个单位；IFEval instruction-level 每条指令一个。"""
    per: dict[str, list[int]] = {str(r["id"]): [0, 0] for r in records}
    for key, ok in unit_outcomes(benchmark, metric, records).items():
        c_n = per[key.split("#")[0]]
        c_n[0] += int(ok)
        c_n[1] += 1
    return [(c, n) for c, n in per.values()]


def summarize(records: Mapping[str, list[Mapping[str, Any]]], *,
              protocol_version: int = P.PROTOCOL_VERSION) -> dict[str, Any]:
    """逐题记录 → 分数（各口径）、标题分数、未评分的项及原因、ParseTally（MMLU-Pro 与 GSM8K 两个口径）、诊断计数。

    键按 protocol_version 那一版的布局（protocol.RECORD_KEYS_BY_VERSION）：v2 是四个读数（mmlu_pro、mmlu_pro_fmt32、
    gsm8k、ifeval），headline 只列三项的标题读数（mmlu_pro 为 256 token）；v1 是三项，结构与 v1 的输出逐字节相同
    （重评 v1 的输出不改变它的任何汇总）。
    未评分的 IFEval（--skip-ifeval-scoring）：各口径的 n 照常给出，n_correct 与 pct 为 None，unscored 里写明原因。
    """
    layout = P.record_readings(protocol_version)
    extra = [k for k in records if k not in layout]
    if extra:
        raise ValueError(f"协议 v{protocol_version} 的记录里没有 {extra}（应为 {list(layout)}）")
    scores: dict[str, Any] = {}
    unscored: dict[str, str] = {}
    tally: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for key, reading in layout.items():
        bench = reading.benchmark
        recs = records.get(key, [])
        scored = is_scored(bench, recs)
        if not scored:
            unscored[key] = IFEVAL_SKIP_REASON
        scores[key] = {}
        for metric in METRICS[bench]:
            if scored:
                units = unit_outcomes(bench, metric, recs)
                n, c = len(units), sum(units.values())
                scores[key][metric] = {"n": n, "n_correct": c, "pct": pct1(c, n)}
            else:
                n = len(recs) if metric.startswith("prompt_level_") else sum(len(r["instruction_id_list"]) for r in recs)
                scores[key][metric] = {"n": n, "n_correct": None, "pct": None}
        if bench == "mmlu_pro":
            tally[key] = _tally([r["failure"] for r in recs]).to_dict()
        elif bench == "gsm8k":
            tally[f"{key}_strict"] = _tally([r["strict"]["failure"] for r in recs]).to_dict()
            tally[f"{key}_flexible"] = _tally([r["flexible"]["failure"] for r in recs]).to_dict()
        diagnostics[key] = {
            "n": len(recs),
            "n_thinking_unclosed": sum(r["thinking_unclosed"] for r in recs),
            "n_empty_text": sum(r["text"].strip() == "" for r in recs),
            # 生成数达到该读数的 max_new_tokens：被截断的上界（恰好在最后一步生成停止 token 的也算在内）
            "n_reached_max_new_tokens": sum(r["reached_max_new_tokens"] for r in recs),
        }
    # 标题：每项一个数（v2 的 mmlu_pro 为 256 token；v1 的输出里 mmlu_pro 是当时的 32 token 标题，原样保留）
    headline = {key: scores[key][P.HEADLINE[key]]["pct"] for key in layout if key in P.BENCHMARKS}
    return {"scores": scores, "headline": headline, "unscored": unscored, "parse_tally": tally,
            "diagnostics": diagnostics}


# 由逐题记录算出的部分：跑完即评与事后重评（--rescore）这些部分逐字节相同（tests 核对）
SCORED_SECTIONS = ("records", "scores", "headline", "unscored", "parse_tally", "diagnostics")


# ---------------------------------------------------------------------------
# 比较：基座 vs 合并后
# ---------------------------------------------------------------------------


def _load_report(x: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(x, (str, Path)):
        return json.loads(Path(x).read_text(encoding="utf-8"))
    return x


def sign_test_p(n_better: int, n_worse: int) -> float:
    """不一致题对上的精确符号检验（双侧；H0：变好与变坏的概率各 1/2）。

    p = min(1, 2·P(X ≤ min(变好, 变坏)))，X ~ Binomial(变好 + 变坏, 1/2)，按精确分数算。没有不一致的题时 p = 1。
    （p = 1/2 时二项分布对称，这与"概率不大于观测值的结果之和"的双侧定义相同，即 scipy.stats.binomtest 的 p 值。）
    """
    for v in (n_better, n_worse):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"题数必须是非负整数，收到 {v!r}")
    n, k = n_better + n_worse, min(n_better, n_worse)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2**n)
    return float(min(Fraction(1), 2 * tail))


def paired_ci_pp(metric: str, base_counts: list[tuple[int, int]],
                 merged_counts: list[tuple[int, int]]) -> tuple[float, float] | None:
    """按题配对的 bootstrap 95% CI（合并后 − 基座，百分点，一位小数），用 src.stats.paired_bootstrap_ci 与
    src.constants 的默认重采样次数、置信水平与种子。重采样的单位是题：二值口径的统计量是均值；IFEval
    instruction-level 是 Σ遵循的指令数 / Σ指令数（src.stats.ratio_of_sums，同一题的几条指令一起抽）。没有题时为 None。"""
    if metric.startswith("instruction_level_"):
        ci = paired_bootstrap_ci(np.array(merged_counts, dtype=np.float64), np.array(base_counts, dtype=np.float64),
                                 statistic=ratio_of_sums)
    else:
        ci = paired_bootstrap_ci(np.array([c for c, _ in merged_counts], dtype=np.float64),
                                 np.array([c for c, _ in base_counts], dtype=np.float64))
    if ci is None:
        return None
    return round1(Fraction(ci.low) * 100), round1(Fraction(ci.high) * 100)


INCOMPARABLE = "口径不同、标题不可比"


def _view(report: Mapping[str, Any]) -> tuple[int, dict[str, list[Mapping[str, Any]]]]:
    """一份输出 → (协议版本, {v2 读数名: 逐题记录})。v1 的 "mmlu_pro"（32 token）落到 mmlu_pro_fmt32。"""
    version = P.check_protocol_version(report.get("protocol_version"))
    records = report["records"]
    return version, {name: records[key] for key, name in P.RECORD_KEYS_BY_VERSION[version].items() if key in records}


def _side_pct(bench: str, metric: str, recs: list[Mapping[str, Any]] | None) -> float | None:
    if recs is None or not is_scored(bench, recs):
        return None
    units = unit_outcomes(bench, metric, recs)
    return pct1(sum(units.values()), len(units))


def _incomparable_note(reading: P.Reading, vb: int, vm: int, in_b: bool, in_m: bool) -> str:
    def side(v: int, has: bool) -> str:
        return f"协议 v{v}，{'有' if has else '没有'}这个读数"

    return (f"{INCOMPARABLE}：{reading.name}（{reading.benchmark}，max_new_tokens={reading.max_new_tokens}）"
            f"基座{side(vb, in_b)}、合并后{side(vm, in_m)}。协议 v1 的 MMLU-Pro 只有 32 token 的读数，"
            f"只与 v2 的 {P.MMLU_PRO_FMT32} 比对（D28-7）。")


def compare(base: Mapping[str, Any] | str | Path, merged: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """两次健康检查（scripts/health_check.py 的输出 JSON 或其路径）→ 每个读数每个口径的差值（合并后 − 基座，百分点）。

    只比较同一题目文件（sha256 相同）、同一批题目（逐题的键相同，--limit 不同的运行不可比）的两次运行，协议版本须在
    protocol.SUPPORTED_PROTOCOL_VERSIONS 里，否则 ValueError。比较的单位是 v2 的读数（protocol.READINGS），两边各自
    按自己的版本布局取记录：口径（项与 max_new_tokens）相同才比；v1 的 MMLU-Pro（32 token）对上 v2 的 mmlu_pro_fmt32，
    256 的标题读数 mmlu_pro 只要有一边没有，就不比：该读数各行 comparable=False、数值为 None、note 写明
    "口径不同、标题不可比"，列在 incomparable 与 notes 里（两边都是 v1 时也一样：D28-7 起 32 token 只是格式诊断）。
    每行给出：
    - 两边的分数、按精确分数算的差值（一位小数）、丢题数（基座对、合并后错）与得题数（instruction-level 按指令数）；
    - flag：标题读数的标题口径（protocol.HEADLINE）下降严格大于 RED_FLAG_DROP_PP 个百分点时为 True（简报 §8.11 的
      标红规则，只看差值）；非标题口径与诊断读数（mmlu_pro_fmt32）为 None（给出差值、CI 与符号检验，不标红）；
    - 与 flag 并列的两项统计（不改变标红规则）：ci95_pp 为按题配对的 bootstrap 95% CI（paired_ci_pp）；
      sign_test 为按题的精确符号检验：变好 / 变坏的题数（该题对的单位数合并后更多 / 更少；二值口径即得题数 / 丢题数）
      与双侧 p 值（sign_test_p）。
    任一边的某项没有评分（IFEval 用了 --skip-ifeval-scoring）时，该项各行 scored=False，差值、flag、CI、p 值都为 None，
    该项列在 unscored 里；any_flag 只看可比、评了分的标题口径。
    截断率提示（D28-7，与标红分开）：每个可比的读数给出两边写满该读数 max_new_tokens 的题所占比例（reached_max_new_tokens）
    与差值；差值的绝对值严格大于 TRUNCATION_NOTICE_PP 个百分点即列进 truncation.notices。未评分的项照样给出。
    """
    b, m = _load_report(base), _load_report(merged)
    vb, view_b = _view(b)
    vm, view_m = _view(m)
    if b["items_file"]["sha256"] != m["items_file"]["sha256"]:
        raise ValueError("两次运行用的题目文件不同（sha256 不一致）")
    rows = []
    unscored: list[str] = []
    incomparable: list[str] = []
    notes: list[str] = []
    truncation = []
    for reading in P.READINGS:
        name, bench = reading.name, reading.benchmark
        rb, rm = view_b.get(name), view_m.get(name)
        diagnostic = reading.role == P.ROLE_DIAGNOSTIC
        common = {"reading": name, "benchmark": bench}
        if rb is None or rm is None:
            incomparable.append(name)
            notes.append(_incomparable_note(reading, vb, vm, rb is not None, rm is not None))
            for metric in METRICS[bench]:
                rows.append({**common, "metric": metric, "headline": not diagnostic and metric == P.HEADLINE[bench],
                             "diagnostic": diagnostic, "max_new_tokens": reading.max_new_tokens, "comparable": False,
                             "scored": None, "note": notes[-1], "n": None, "base_pct": _side_pct(bench, metric, rb),
                             "merged_pct": _side_pct(bench, metric, rm), "delta_pp": None, "lost": None,
                             "gained": None, "flag": None, "ci95_pp": None, "sign_test": None})
            continue
        if [r["id"] for r in rb] != [r["id"] for r in rm]:
            raise ValueError(f"{name}：两次运行的题目不同（--limit 不同？）")
        n_items = len(rb)
        tb, tm = sum(bool(r["reached_max_new_tokens"]) for r in rb), sum(bool(r["reached_max_new_tokens"]) for r in rm)
        t_delta = Fraction(100 * (tm - tb), n_items) if n_items else None
        truncation.append({"reading": name, "max_new_tokens": reading.max_new_tokens, "n": n_items,
                           "base_n_reached": tb, "merged_n_reached": tm,
                           "base_pct": pct1(tb, n_items), "merged_pct": pct1(tm, n_items),
                           "delta_pp": None if t_delta is None else round1(t_delta),
                           "notice": t_delta is not None and abs(t_delta) > P.TRUNCATION_NOTICE_PP})
        scored_b, scored_m = is_scored(bench, rb), is_scored(bench, rm)
        if not (scored_b and scored_m):
            unscored.append(name)
        for metric in METRICS[bench]:
            headline = not diagnostic and metric == P.HEADLINE[bench]
            row: dict[str, Any] = {**common, "metric": metric, "headline": headline, "diagnostic": diagnostic,
                                   "max_new_tokens": reading.max_new_tokens, "comparable": True,
                                   "scored": scored_b and scored_m, "note": None}
            if not row["scored"]:
                rows.append({**row, "n": None, "base_pct": _side_pct(bench, metric, rb),
                             "merged_pct": _side_pct(bench, metric, rm), "delta_pp": None, "lost": None,
                             "gained": None, "flag": None, "ci95_pp": None, "sign_test": None})
                continue
            ub, um = unit_outcomes(bench, metric, rb), unit_outcomes(bench, metric, rm)
            if set(ub) != set(um):
                raise ValueError(f"{name}/{metric}：评分单位不同")
            n = len(ub)
            cb, cm = sum(ub.values()), sum(um.values())
            delta = Fraction(100 * (cm - cb), n) if n else None
            per_b, per_m = item_counts(bench, metric, rb), item_counts(bench, metric, rm)
            better = sum(1 for (x, _), (y, _) in zip(per_b, per_m) if y > x)
            worse = sum(1 for (x, _), (y, _) in zip(per_b, per_m) if y < x)
            ci = paired_ci_pp(metric, per_b, per_m)
            rows.append({
                **row,
                "n": n,
                "base_pct": pct1(cb, n),
                "merged_pct": pct1(cm, n),
                "delta_pp": None if delta is None else round1(delta),
                "lost": sum(1 for k in ub if ub[k] and not um[k]),
                "gained": sum(1 for k in ub if um[k] and not ub[k]),
                "flag": (delta is not None and delta < -P.RED_FLAG_DROP_PP) if headline else None,
                "ci95_pp": None if ci is None else list(ci),
                "sign_test": {"n_better": better, "n_worse": worse, "p_value": sign_test_p(better, worse)},
            })
    flagged = [r["reading"] for r in rows if r["flag"]]
    return {
        "protocol_versions": {"base": vb, "merged": vm},
        "threshold_pp": P.RED_FLAG_DROP_PP,
        "rule": (f"headline metric of a headline reading drops by more than {P.RED_FLAG_DROP_PP} percentage points "
                 f"(merged - base < -{P.RED_FLAG_DROP_PP}); MMLU-Pro's headline reading is mmlu_pro at "
                 f"max_new_tokens={P.READINGS_BY_NAME['mmlu_pro'].max_new_tokens} (D28-7); the diagnostic reading "
                 f"{P.MMLU_PRO_FMT32} gets delta, CI and sign test but is never flagged"),
        "statistics": {
            "ci95_pp": "paired bootstrap over questions, merged - base, percentile interval (src.stats.paired_bootstrap_ci; "
                       "instruction-level metrics resample questions with src.stats.ratio_of_sums)",
            "bootstrap": {"n_resamples": BOOTSTRAP_N_RESAMPLES, "level": BOOTSTRAP_CI_LEVEL, "seed": BOOTSTRAP_SEED},
            "sign_test": "exact two-sided binomial test (p = 1/2) on questions whose count of correct units differs",
            "note": "reported alongside the flag; the flag rule itself is unchanged (brief §8.11)",
        },
        "base_model": b.get("model", {}).get("key"),
        "merged_model": m.get("model", {}).get("key"),
        "partial_run": b.get("limit") is not None or m.get("limit") is not None,
        "incomparable": incomparable,
        "notes": notes,
        "unscored": unscored,
        "rows": rows,
        "flagged": flagged,
        "any_flag": bool(flagged),
        "truncation": {
            "rule": (f"share of questions whose generation reached the reading's max_new_tokens; listed in notices when "
                     f"|merged - base| > {P.TRUNCATION_NOTICE_PP} percentage points (D28-7); separate from the red flag"),
            "threshold_pp": P.TRUNCATION_NOTICE_PP,
            "rows": truncation,
            "notices": [t["reading"] for t in truncation if t["notice"]],
        },
    }


# ---------------------------------------------------------------------------
# IFEval 的 nltk 数据
# ---------------------------------------------------------------------------


class PunktDataError(RuntimeError):
    """找不到 punkt_tab（english），或文件与协议钉住的哈希不同。"""


def punkt_provenance() -> dict[str, Any]:
    """nltk 实际会用的 punkt_tab/english：路径与四个文件的 sha256，以及是否与 protocol.PUNKT_TAB_ENGLISH_SHA256 相同。"""
    import nltk

    try:
        ptr = nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError as e:
        raise PunktDataError(
            "找不到 nltk 的 punkt_tab 数据（IFEval 的句子切分要用）。下载到项目目录并用 NLTK_DATA 指过去，不写全局 ~/nltk_data：\n"
            "  python -m nltk.downloader -d $PROJ/nltk_data punkt_tab\n  export NLTK_DATA=$PROJ/nltk_data"
        ) from e
    files = {}
    for name in P.PUNKT_TAB_ENGLISH_SHA256:
        with ptr.join(name).open() as f:
            files[name] = hashlib.sha256(f.read()).hexdigest()
    return {
        "path": str(getattr(ptr, "path", ptr)),
        "nltk_data_env": os.environ.get("NLTK_DATA"),
        "files_sha256": files,
        "matches_pinned": files == dict(P.PUNKT_TAB_ENGLISH_SHA256),
    }


def require_punkt(*, pinned: bool) -> dict[str, Any]:
    """运行前检查 punkt_tab。pinned=True（真模型）时要求与钉住的哈希逐个相同；False（--tiny 彩排）只记录。"""
    prov = punkt_provenance()
    if pinned and not prov["matches_pinned"]:
        raise PunktDataError(f"punkt_tab/english 与协议钉住的哈希不同：{prov}")
    return prov
