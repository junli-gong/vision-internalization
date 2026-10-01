"""健康检查的逐题运行与事后重评（简报 §8.11；DEV-0008；协议 v2：DEV-0013；协议 v3：DEV-0016）。只经适配器接口
（build_inputs / generate / option_logits / split_answer），与模型家族无关。

每题一次 generate，单轮；v3 的 MMLU-Pro 另有一次 option_logits（受限字母读数，D36-4）。记账按 src/accounting.py：每次模型调用
记一次调用（new_round=True：各读数都是单轮流程），一题标为一个样本，样本墙钟 = 各次 build_inputs + 模型调用的实际耗时之和
（评分不计入）。计时前先做预热调用（generate 一次、option_logits 一次），不计入记账（D21）。

先生成、后评分（DEV-0008 审查第 1 轮）：逐题记录分两部分——生成部分（GENERATION_FIELDS，只来自模型输出）与评分部分
（score_record 的返回值）。评分只读生成部分里的 text 与 thinking_unclosed，跑完即评（run_one）与事后重评
（rescore_records）调用的是同一个 score_record、输入同一份生成部分，所以规则完全相同，结果逐字节相同（tests 核对）。

受限字母读数（协议 v3，D36-4）：mmlu_pro_letter 的生成部分是 option_logits 的结果（选项字母、各字母所用 token、logit、
选项集合上的概率与 argmax），评分只读其中的 options、probs 与 choice（scoring.score_mmlu_pro_letter）。各字母实际用到的写法与
token id 在预热时查一次，记进 state["option_spellings"]，每题核对相同。

前缀读数（协议 v2 起，D28-7）：MMLU-Pro 只自由生成一次（256，v3 里叫 mmlu_pro_free256），诊断读数 mmlu_pro_fmt32 由
prefix_generation 从同一次生成的前 32 个 token 重建：文本由适配器的 split_answer 对前 32 个 token 解码（与适配器生成时
拆分回答的是同一个函数），stopped 为"前 32 个 token 里出现了停止 token"，评分与 v1 的 32 token 口径完全相同。greedy 下这与真的用 max_new_tokens=32 生成
逐字段相同（墙钟除外；tests 在三个家族的 --tiny 上核对，audit_prefix_readings 在算力机上核对）。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Iterable, Mapping, NamedTuple

from src.accounting import Compute, merge
from src.health import protocol as P
from src.health.scoring import SCORERS, score_mmlu_pro_letter

# 逐题记录的生成部分，按这个顺序写在记录的最前面；其后是评分部分。v2 加了 stopped（只记录，不进入评分）
GENERATION_FIELDS_V1 = ("id", "prompt_sha256", "raw_text", "text", "thinking_unclosed", "n_prompt_tokens",
                        "n_generated_tokens", "reached_max_new_tokens", "wall_time_s")
GENERATION_FIELDS = ("id", "prompt_sha256", "raw_text", "text", "thinking_unclosed", "stopped", "n_prompt_tokens",
                     "n_generated_tokens", "reached_max_new_tokens", "wall_time_s")
GENERATION_FIELDS_BY_VERSION = {1: GENERATION_FIELDS_V1, 2: GENERATION_FIELDS, 3: GENERATION_FIELDS}
assert set(GENERATION_FIELDS_BY_VERSION) == set(P.SUPPORTED_PROTOCOL_VERSIONS)
# 受限字母读数（v3）的生成部分：option_logits 的一行。token_ids、logits、probs 与 options 一一对应
LETTER_GENERATION_FIELDS = ("id", "prompt_sha256", "options", "token_ids", "logits", "probs", "choice", "n_prompt_tokens",
                            "wall_time_s")


def generation_fields(reading: P.Reading, version: int) -> tuple[str, ...]:
    return LETTER_GENERATION_FIELDS if reading.kind == P.KIND_OPTION_LETTER else GENERATION_FIELDS_BY_VERSION[version]


class TextOnlyViolation(RuntimeError):
    """纯文本输入却出现了图像或视觉 token。"""


class RescoreError(ValueError):
    """重评的输入与当前题目或协议对不上。"""


class PrefixReadingError(RuntimeError):
    """无法从一次生成重建前缀读数（适配器没有 split_answer、没有记录 stopped，或 split_answer 与生成时的拆分不一致）。"""


class LetterReadingError(RuntimeError):
    """受限字母读数的结果与请求对不上（行数、选项），或某个字母用到的写法与预热时查到的不同。"""


class _Answer(NamedTuple):
    """评分用到的回答字段（GenerationResult 的子集；解析器经 parse_generation 读这两个字段）。

    故意不带 stopped：健康检查与 v1 相同，截断的回答照样解析（parse_generation 见不到 stopped 就跳过"截断判失败"）。
    """

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


def letter_text_only(adapter: Any, prompt: str, letters: str) -> tuple[Any, float]:
    """受限字母读数的一次调用：零张图、instruction_first、关思考，选项为 letters。返回 (OptionScores, 墙钟秒)。"""
    t0 = time.perf_counter()
    inputs = adapter.build_inputs([], prompt, P.ORDER)
    results = adapter.option_logits([inputs], [list(letters)], enable_thinking=P.ENABLE_THINKING)
    wall = time.perf_counter() - t0
    if inputs.images or len(results) != 1 or results[0].n_visual_tokens != 0:
        raise TextOnlyViolation(f"纯文本输入却有 {len(inputs.images)} 张图，或 option_logits 返回了 {len(results)} 行 / 视觉 token")
    if list(results[0].options) != list(letters):
        raise LetterReadingError(f"option_logits 的选项 {results[0].options} 不是请求的 {list(letters)}")
    return results[0], wall


def spelling_table(scores: Any) -> dict[str, dict[str, Any]]:
    """OptionScores → {字母: {"spellings": [{"text", "token_id"}, …], "skipped": [...]}}（写进输出，供核对各家族的写法）。"""
    return {o: {"spellings": [{"text": x.text, "token_id": x.token_id} for x in sp], "skipped": list(sk)}
            for o, sp, sk in zip(scores.options, scores.spellings, scores.skipped_spellings)}


def check_spellings(table: Mapping[str, Any] | None, scores: Any, item_id: Any) -> None:
    """这一题各字母用到的写法必须与预热时查到的（table）相同；table 为 None 时不核对。"""
    if table is None:
        return
    for letter, entry in spelling_table(scores).items():
        if table.get(letter) != entry:
            raise LetterReadingError(f"{item_id}：字母 {letter} 的写法 {entry} 与预热时查到的 {table.get(letter)} 不同")


def letter_record(item: Mapping[str, Any], prompt_hash: str, scores: Any, *, wall: float | None) -> dict[str, Any]:
    """一次 option_logits → 受限字母读数的生成部分（LETTER_GENERATION_FIELDS 的顺序）。"""
    generation = {
        "id": item["id"],
        "prompt_sha256": prompt_hash,
        "options": list(scores.options),
        "token_ids": list(scores.token_ids),
        "logits": list(scores.logits),
        "probs": list(scores.probs),
        "choice": scores.choice,
        "n_prompt_tokens": scores.n_prompt_tokens,
        "wall_time_s": None if wall is None else round(wall, 4),
    }
    assert tuple(generation) == LETTER_GENERATION_FIELDS
    return generation


def score_reading(reading: P.Reading, item: Mapping[str, Any], generation: Mapping[str, Any], *,
                  score_ifeval: bool) -> dict[str, Any]:
    """按读数的类型评分：受限字母读数用 scoring.score_mmlu_pro_letter，其余用 score_record。"""
    if reading.kind == P.KIND_OPTION_LETTER:
        return score_mmlu_pro_letter(item, generation)
    return score_record(reading.benchmark, item, generation, score_ifeval=score_ifeval)


def score_record(benchmark: str, item: Mapping[str, Any], generation: Mapping[str, Any], *,
                 score_ifeval: bool) -> dict[str, Any]:
    """逐题记录的评分部分，只读 generation["text"] 与 generation["thinking_unclosed"]。

    score_ifeval=False 时 IFEval 不评分（strict、loose 记 None；原因见 scoring.IFEVAL_SKIP_REASON），其余两项照常。
    """
    if benchmark == "ifeval" and not score_ifeval:
        return {"instruction_id_list": list(item["instruction_id_list"]), "strict": None, "loose": None}
    return SCORERS[benchmark](item, _Answer(generation["text"], generation["thinking_unclosed"]))


def _require_stopped(result: Any) -> bool:
    if not isinstance(result.stopped, bool):
        raise PrefixReadingError(f"适配器没有记录 stopped（收到 {result.stopped!r}；D29：三个适配器一律填 True/False）")
    return result.stopped


def generation_record(item: Mapping[str, Any], prompt_hash: str, result: Any, *, max_new_tokens: int,
                      wall: float | None) -> dict[str, Any]:
    """一次生成 → 逐题记录的生成部分（GENERATION_FIELDS 的顺序）。wall 为 None 表示不单独计时。"""
    generation = {
        "id": item["id"],
        "prompt_sha256": prompt_hash,
        "raw_text": result.raw_text,
        "text": result.text,
        "thinking_unclosed": result.thinking_unclosed,
        "stopped": _require_stopped(result),
        "n_prompt_tokens": result.n_prompt_tokens,
        "n_generated_tokens": result.n_generated_tokens,
        "reached_max_new_tokens": result.n_generated_tokens >= max_new_tokens,
        "wall_time_s": None if wall is None else round(wall, 4),
    }
    assert tuple(generation) == GENERATION_FIELDS
    return generation


def prefix_generation(adapter: Any, item: Mapping[str, Any], prompt_hash: str, result: Any, *,
                      max_new_tokens: int) -> dict[str, Any]:
    """同一次 greedy 生成的前 k 个 token（k = max_new_tokens），按"只生成 k 个"重建生成部分（D28-7）。

    - token：result.token_ids 的前 k 个。适配器的 token_ids 截到第一个停止 token 为止（含），停止 token 只可能是最后一个；
    - raw_text、text、thinking_unclosed：adapter.split_answer(前 k 个 token)，与适配器生成时拆分回答的是同一个函数
      （先核对它对全部 token 给出的恰是 result 里的这三个字段，不一致就报错）；
    - stopped：前 k 个 token 里出现了停止 token，即 result.stopped 且生成数 ≤ k；
    - n_generated_tokens：min(生成数, k)；reached_max_new_tokens：它 ≥ k；n_prompt_tokens 不变；
    - wall_time_s：None（与生成读数同一次生成，不单独计时）。
    """
    split = getattr(adapter, "split_answer", None)
    if split is None:
        raise PrefixReadingError(f"{type(adapter).__name__} 没有 split_answer，无法重建前缀读数")
    stopped = _require_stopped(result)
    ids = tuple(int(t) for t in result.token_ids)
    whole = tuple(split(ids, enable_thinking=P.ENABLE_THINKING))
    if whole != (result.text, result.raw_text, result.thinking_unclosed):
        raise PrefixReadingError(f"{item['id']}：split_answer 对全部 token 的结果与生成时不同：{whole!r} vs "
                                 f"{(result.text, result.raw_text, result.thinking_unclosed)!r}")
    head = ids[:max_new_tokens]
    text, raw_text, unclosed = split(head, enable_thinking=P.ENABLE_THINKING)
    generation = {
        "id": item["id"],
        "prompt_sha256": prompt_hash,
        "raw_text": raw_text,
        "text": text,
        "thinking_unclosed": unclosed,
        "stopped": stopped and len(ids) <= max_new_tokens,
        "n_prompt_tokens": result.n_prompt_tokens,
        "n_generated_tokens": len(head),
        "reached_max_new_tokens": len(head) >= max_new_tokens,
        "wall_time_s": None,
    }
    assert tuple(generation) == GENERATION_FIELDS
    return generation


def run_one(adapter: Any, benchmark: str, item: Mapping[str, Any], *, score_ifeval: bool = True,
            spellings: Mapping[str, Any] | None = None) -> tuple[dict[str, dict[str, Any]], Compute]:
    """一题：构造提示、按读数调用模型、评分。返回 ({读数名: 逐题记录}, 这一题的记账)。

    读数是 protocol.readings_of(benchmark)，按顺序：
    - 受限字母读数（v3 的 MMLU-Pro：mmlu_pro_letter）：一次 option_logits，选项为本题的实际字母；各字母的写法与 spellings
      （预热时查到的表）核对；
    - 自由生成读数（上限 MAX_NEW_TOKENS[benchmark]）：一次 generate；
    - 它的前缀读数（MMLU-Pro 的 mmlu_pro_fmt32：同一次生成的前 32 个 token，见 prefix_generation）：不另外调用模型，不另记账。
    """
    prompt = P.PROMPT_BUILDERS[benchmark](item)
    max_new = P.MAX_NEW_TOKENS[benchmark]
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    out: dict[str, dict[str, Any]] = {}
    calls = Compute()
    walls = 0.0
    free: dict[str, Any] = {}
    for reading in P.readings_of(benchmark):
        if reading.kind == P.KIND_OPTION_LETTER:
            scores, wall = letter_text_only(adapter, prompt, P.option_letters(len(item["options"])))
            check_spellings(spellings, scores, item["id"])
            generation = letter_record(item, prompt_hash, scores, wall=wall)
            calls, walls = calls + Compute.from_generation(scores, new_round=True), walls + wall
        elif reading.prefix_of is None:
            assert reading.max_new_tokens == max_new
            result, wall = generate_text_only(adapter, prompt, max_new_tokens=max_new)
            free[reading.name] = result
            generation = generation_record(item, prompt_hash, result, max_new_tokens=max_new, wall=wall)
            calls, walls = calls + Compute.from_generation(result, new_round=True), walls + wall
        else:
            generation = prefix_generation(adapter, item, prompt_hash, free[reading.prefix_of],
                                           max_new_tokens=reading.max_new_tokens)
        out[reading.name] = {**generation, **score_reading(reading, item, generation, score_ifeval=score_ifeval)}
    return out, calls.as_sample(wall_time_s=walls)


def run_items(adapter: Any, items_obj: Mapping[str, Any], *, limit: int | None = None, score_ifeval: bool = True,
              state: dict[str, Any] | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """跑全部题目（limit 给定时每项只跑前 limit 题），把结果逐题写进 state（中途出错时调用方仍能落盘已有结果）。

    state 里的键：warmup、option_spellings（受限字母读数用到的 A–J 写法与 token id，预热时查到）、records（读数 → 逐题记录
    列表，顺序同 protocol.READINGS）、compute（项 → compute.json 字典，外加 total；前缀读数与它的生成读数是同一次调用，
    不另记；MMLU-Pro 每题两次调用：option_logits 与 generate）。
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError(f"limit 必须是正整数，收到 {limit!r}")
    state = {} if state is None else state
    warm, warm_wall = generate_text_only(adapter, P.WARMUP_PROMPT, max_new_tokens=P.WARMUP_MAX_NEW_TOKENS)
    letters, letters_wall = letter_text_only(adapter, P.WARMUP_PROMPT, P.OPTION_LETTERS)
    state["warmup"] = {"prompt": P.WARMUP_PROMPT, "max_new_tokens": P.WARMUP_MAX_NEW_TOKENS,
                       "n_generated_tokens": warm.n_generated_tokens, "wall_time_s": round(warm_wall, 4),
                       "option_logits": {"options": P.OPTION_LETTERS, "wall_time_s": round(letters_wall, 4)},
                       "counted_in_compute": False}
    state["option_spellings"] = spelling_table(letters)
    state["records"] = {}
    state["compute"] = {}
    totals: dict[str, Compute] = {}
    for bench in P.BENCHMARKS:
        items = list(items_obj["items"][bench])
        if limit is not None:
            items = items[:limit]
        for reading in P.readings_of(bench):  # mmlu_pro 之后紧跟 mmlu_pro_fmt32
            state["records"][reading.name] = []
        totals[bench] = Compute()
        for k, item in enumerate(items, 1):
            records, compute = run_one(adapter, bench, item, score_ifeval=score_ifeval,
                                       spellings=state["option_spellings"])
            for name, record in records.items():
                state["records"][name].append(record)
            totals[bench] = totals[bench] + compute
            state["compute"][bench] = totals[bench].to_json_dict()
            if progress is not None and (k % 10 == 0 or k == len(items)):
                progress(f"{bench} {k}/{len(items)}  生成 token 累计 {totals[bench].generated_tokens}")
    state["compute"]["total"] = merge(totals.values()).to_json_dict()
    return state


def rescore_records(records: Mapping[str, list[Mapping[str, Any]]], items_obj: Mapping[str, Any], *,
                    benchmarks: Iterable[str] = P.BENCHMARKS,
                    protocol_version: int = P.PROTOCOL_VERSION) -> dict[str, list[dict[str, Any]]]:
    """从已保存的逐题记录重算 benchmarks 里各项的评分部分（其余项原样保留），规则与 run_one 相同。

    records 按 protocol_version 那一版的布局（protocol.RECORD_KEYS_BY_VERSION；v1 只有三个键、记录里没有 stopped；v2 四个；
    v3 五个，受限字母读数的生成部分是 LETTER_GENERATION_FIELDS），一项的全部读数一起重评（v3 的 MMLU-Pro：mmlu_pro_letter、
    mmlu_pro_free256 与 mmlu_pro_fmt32）。逐题核对：记录的 id 必须是题目文件里该项的前若干题（--limit 跑出来的也行），
    prompt_sha256 必须等于按当前协议构造的提示的哈希（防止拿别的题目或别的提示模板跑出来的输出来评分；v1、v2、v3 的提示
    模板相同）。前缀读数另核对它与生成读数的对应：同一批 id 与 prompt_sha256，生成数 = min(生成读数的生成数, k)，
    stopped = 生成读数 stopped 且生成数 ≤ k。
    """
    benchmarks = set(benchmarks)
    unknown = benchmarks - set(P.BENCHMARKS)
    if unknown:
        raise RescoreError(f"未知的项 {sorted(unknown)}")
    try:
        layout = P.record_readings(protocol_version)
    except ValueError as e:
        raise RescoreError(str(e)) from None
    if list(records) != list(layout):
        raise RescoreError(f"协议 v{protocol_version} 的记录应有 {list(layout)}（按此顺序），收到 {list(records)}")
    out: dict[str, list[dict[str, Any]]] = {}
    for key, reading in layout.items():
        fields = generation_fields(reading, protocol_version)
        bench = reading.benchmark
        recs = list(records[key])
        items = items_obj["items"][bench]
        if [r["id"] for r in recs] != [it["id"] for it in items[: len(recs)]]:
            raise RescoreError(f"{key}：记录的 id 不是题目文件的前 {len(recs)} 题")
        if bench not in benchmarks:
            out[key] = [dict(r) for r in recs]
            continue
        out[key] = []
        for r, item in zip(recs, items):
            missing = [k for k in fields if k not in r]
            if missing:
                raise RescoreError(f"{key} {r.get('id')}：记录缺少生成字段 {missing}")
            if r["prompt_sha256"] != prompt_sha256(bench, item):
                raise RescoreError(f"{key} {r['id']}：prompt_sha256 与按当前协议构造的提示不符")
            generation = {k: r[k] for k in fields}
            try:
                scored = score_reading(reading, item, generation, score_ifeval=True)
            except ValueError as e:
                raise RescoreError(f"{key} {r['id']}：{e}") from None
            out[key].append({**generation, **scored})
        # 前缀读数：只在它截取的读数也在这份输出里时核对（v1 的 "mmlu_pro" 口径同 fmt32，但那是直接生成的 32 个 token）
        source = [k for k, r in layout.items() if reading.prefix_of is not None and r.name == reading.prefix_of]
        if source:
            _check_prefix_records(key, reading.max_new_tokens, out[key], records[source[0]])
    return out


def _check_prefix_records(key: str, k: int, prefix: list[Mapping[str, Any]], source: list[Mapping[str, Any]]) -> None:
    """前缀读数的记录与它截取的生成读数对得上：同一批 id 与 prompt_sha256，生成数与 stopped 符合截取规则。"""
    if [r["id"] for r in prefix] != [r["id"] for r in source]:
        raise RescoreError(f"{key}：题目与它截取的读数不同")
    for p, s in zip(prefix, source):
        if not (p["prompt_sha256"] == s["prompt_sha256"]
                and p["n_generated_tokens"] == min(s["n_generated_tokens"], k)
                and p["stopped"] == bool(s["stopped"] and s["n_generated_tokens"] <= k)):
            raise RescoreError(f"{key} {p['id']}：与它截取的读数对不上（prompt_sha256、生成数或 stopped）")


def audit_prefix_readings(adapter: Any, items_obj: Mapping[str, Any], records: Mapping[str, list[Mapping[str, Any]]],
                          *, n: int) -> dict[str, Any]:
    """前缀读数的运行期审计（算力机用：开发机 CPU fp32 上前缀必然相同，GPU bf16 上要实测）。

    对每个前缀读数，取前 n 题真的用 max_new_tokens = k 再生成一次，按生成读数的同一套规则记录与评分，与截出来的记录
    逐字段比较（墙钟除外）。不改变已有记录与分数；审计的调用单独记账，不计入 compute。返回 {读数名: 审计结果}。
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n 必须是正整数，收到 {n!r}")
    out: dict[str, Any] = {}
    for reading in P.READINGS:
        if reading.prefix_of is None:
            continue
        bench = reading.benchmark
        derived = list(records.get(reading.name, []))[:n]
        total = Compute()
        mismatches = []
        for item, rec in zip(items_obj["items"][bench], derived):
            prompt = P.PROMPT_BUILDERS[bench](item)
            result, wall = generate_text_only(adapter, prompt, max_new_tokens=reading.max_new_tokens)
            total = total + Compute.from_generation(result, new_round=True).as_sample(wall_time_s=wall)
            gen = generation_record(item, rec["prompt_sha256"], result, max_new_tokens=reading.max_new_tokens, wall=None)
            real = {**gen, **score_record(bench, item, gen, score_ifeval=True)}
            diff = sorted(k for k in set(real) | set(rec) if k != "wall_time_s" and real.get(k) != rec.get(k))
            if diff:
                mismatches.append({"id": item["id"], "fields": diff, "derived": {k: rec.get(k) for k in diff},
                                   "generated": {k: real.get(k) for k in diff}})
        out[reading.name] = {
            "max_new_tokens": reading.max_new_tokens, "prefix_of": reading.prefix_of, "n": len(derived),
            "n_identical": len(derived) - len(mismatches), "mismatches": mismatches,
            "compute": total.to_json_dict(), "counted_in_compute": False,
        }
    return out
