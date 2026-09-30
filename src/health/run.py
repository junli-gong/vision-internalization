"""健康检查的逐题运行与事后重评（简报 §8.11；DEV-0008）。只经适配器接口（build_inputs / generate），与模型家族无关。

每题一次 generate，单轮；记账按 src/accounting.py：每题记一次调用（new_round=True）并标为一个样本，样本墙钟 =
build_inputs + generate 的实际耗时（评分不计入）。计时前先做一次预热调用，不计入记账（D21）。

先生成、后评分（DEV-0008 审查第 1 轮）：逐题记录分两部分——生成部分（GENERATION_FIELDS，只来自模型输出）与评分部分
（score_record 的返回值）。评分只读生成部分里的 text 与 thinking_unclosed，跑完即评（run_one）与事后重评
（rescore_records）调用的是同一个 score_record、输入同一份生成部分，所以规则完全相同，结果逐字节相同（tests 核对）。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Iterable, Mapping, NamedTuple

from src.accounting import Compute, merge
from src.health import protocol as P
from src.health.scoring import SCORERS

# 逐题记录的生成部分，按这个顺序写在记录的最前面；其后是评分部分
GENERATION_FIELDS = ("id", "prompt_sha256", "raw_text", "text", "thinking_unclosed", "n_prompt_tokens",
                     "n_generated_tokens", "reached_max_new_tokens", "wall_time_s")


class TextOnlyViolation(RuntimeError):
    """纯文本输入却出现了图像或视觉 token。"""


class RescoreError(ValueError):
    """重评的输入与当前题目或协议对不上。"""


class _Answer(NamedTuple):
    """评分用到的回答字段（GenerationResult 的子集；解析器经 parse_generation 读这两个字段）。"""

    text: str
    thinking_unclosed: bool


def prompt_sha256(benchmark: str, item: Mapping[str, Any]) -> str:
    return hashlib.sha256(P.PROMPT_BUILDERS[benchmark](item).encode("utf-8")).hexdigest()


def generate_text_only(adapter: Any, prompt: str, *, max_new_tokens: int) -> tuple[Any, float]:
    """协议规定的一次生成：零张图、instruction_first、关思考、greedy（每次显式传入）。返回 (GenerationResult, 墙钟秒)。"""
    t0 = time.perf_counter()
    inputs = adapter.build_inputs([], prompt, P.ORDER)
    result = adapter.generate(inputs, enable_thinking=P.ENABLE_THINKING, max_new_tokens=max_new_tokens,
                              **dict(P.GREEDY_KWARGS))
    wall = time.perf_counter() - t0
    if inputs.images or result.n_visual_tokens != 0:
        raise TextOnlyViolation(f"纯文本输入却有 {len(inputs.images)} 张图、{result.n_visual_tokens} 个视觉 token")
    return result, wall


def score_record(benchmark: str, item: Mapping[str, Any], generation: Mapping[str, Any], *,
                 score_ifeval: bool) -> dict[str, Any]:
    """逐题记录的评分部分，只读 generation["text"] 与 generation["thinking_unclosed"]。

    score_ifeval=False 时 IFEval 不评分（strict、loose 记 None；原因见 scoring.IFEVAL_SKIP_REASON），其余两项照常。
    """
    if benchmark == "ifeval" and not score_ifeval:
        return {"instruction_id_list": list(item["instruction_id_list"]), "strict": None, "loose": None}
    return SCORERS[benchmark](item, _Answer(generation["text"], generation["thinking_unclosed"]))


def run_one(adapter: Any, benchmark: str, item: Mapping[str, Any], *,
            score_ifeval: bool = True) -> tuple[dict[str, Any], Compute]:
    """一题：构造提示、生成、评分。返回 (逐题记录, 这一题的记账)。"""
    prompt = P.PROMPT_BUILDERS[benchmark](item)
    max_new = P.MAX_NEW_TOKENS[benchmark]
    result, wall = generate_text_only(adapter, prompt, max_new_tokens=max_new)
    generation = {
        "id": item["id"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_text": result.raw_text,
        "text": result.text,
        "thinking_unclosed": result.thinking_unclosed,
        "n_prompt_tokens": result.n_prompt_tokens,
        "n_generated_tokens": result.n_generated_tokens,
        "reached_max_new_tokens": result.n_generated_tokens >= max_new,
        "wall_time_s": round(wall, 4),
    }
    assert tuple(generation) == GENERATION_FIELDS
    record = {**generation, **score_record(benchmark, item, generation, score_ifeval=score_ifeval)}
    return record, Compute.from_generation(result, new_round=True).as_sample(wall_time_s=wall)


def run_items(adapter: Any, items_obj: Mapping[str, Any], *, limit: int | None = None, score_ifeval: bool = True,
              state: dict[str, Any] | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """跑全部题目（limit 给定时每项只跑前 limit 题），把结果逐题写进 state（中途出错时调用方仍能落盘已有结果）。

    state 里的键：warmup、records（项 → 逐题记录列表）、compute（项 → compute.json 字典，外加 total）。
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError(f"limit 必须是正整数，收到 {limit!r}")
    state = {} if state is None else state
    warm, warm_wall = generate_text_only(adapter, P.WARMUP_PROMPT, max_new_tokens=P.WARMUP_MAX_NEW_TOKENS)
    state["warmup"] = {"prompt": P.WARMUP_PROMPT, "max_new_tokens": P.WARMUP_MAX_NEW_TOKENS,
                       "n_generated_tokens": warm.n_generated_tokens, "wall_time_s": round(warm_wall, 4),
                       "counted_in_compute": False}
    state["records"] = {}
    state["compute"] = {}
    totals: dict[str, Compute] = {}
    for bench in P.BENCHMARKS:
        items = list(items_obj["items"][bench])
        if limit is not None:
            items = items[:limit]
        records = state["records"][bench] = []
        totals[bench] = Compute()
        for k, item in enumerate(items, 1):
            record, compute = run_one(adapter, bench, item, score_ifeval=score_ifeval)
            records.append(record)
            totals[bench] = totals[bench] + compute
            state["compute"][bench] = totals[bench].to_json_dict()
            if progress is not None and (k % 10 == 0 or k == len(items)):
                progress(f"{bench} {k}/{len(items)}  生成 token 累计 {totals[bench].generated_tokens}")
    state["compute"]["total"] = merge(totals.values()).to_json_dict()
    return state


def rescore_records(records: Mapping[str, list[Mapping[str, Any]]], items_obj: Mapping[str, Any], *,
                    benchmarks: Iterable[str] = P.BENCHMARKS) -> dict[str, list[dict[str, Any]]]:
    """从已保存的逐题记录重算 benchmarks 里各项的评分部分（其余项原样保留），规则与 run_one 相同。

    逐题核对：记录的 id 必须是题目文件里该项的前若干题（--limit 跑出来的也行），prompt_sha256 必须等于按当前协议
    构造的提示的哈希（防止拿别的题目或别的提示模板跑出来的输出来评分）。
    """
    benchmarks = set(benchmarks)
    unknown = benchmarks - set(P.BENCHMARKS)
    if unknown:
        raise RescoreError(f"未知的项 {sorted(unknown)}")
    out: dict[str, list[dict[str, Any]]] = {}
    for bench in P.BENCHMARKS:
        recs = list(records.get(bench, []))
        items = items_obj["items"][bench]
        if [r["id"] for r in recs] != [it["id"] for it in items[: len(recs)]]:
            raise RescoreError(f"{bench}：记录的 id 不是题目文件的前 {len(recs)} 题")
        if bench not in benchmarks:
            out[bench] = [dict(r) for r in recs]
            continue
        out[bench] = []
        for r, item in zip(recs, items):
            missing = [k for k in GENERATION_FIELDS if k not in r]
            if missing:
                raise RescoreError(f"{bench} {r.get('id')}：记录缺少生成字段 {missing}")
            if r["prompt_sha256"] != prompt_sha256(bench, item):
                raise RescoreError(f"{bench} {r['id']}：prompt_sha256 与按当前协议构造的提示不符")
            generation = {k: r[k] for k in GENERATION_FIELDS}
            out[bench].append({**generation, **score_record(bench, item, generation, score_ifeval=True)})
    return out
