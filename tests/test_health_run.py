"""健康检查的运行流程（src/health/run.py 与 scripts/health_check.py，DEV-0008）。

- 假适配器：核对每次调用都按协议（零张图、instruction_first、关思考、greedy 参数、各项的 max_new_tokens），分数与记账
  从逐题记录数出来，预热不计入，中途出错时已有结果留在 state 里。
- 三家族的真实 tokenizer / 模板（小型随机模型）：200 题的提示都能零张图 build_inputs，指令区间解码回来就是提示。
- 端到端：三家族各跑 `--tiny --limit 3`，退出码 0，输出 JSON 的结构齐全；`--compare` 命令行；真模型模式在加载模型之前
  就核对 punkt_tab。子进程的 NLTK_DATA 指向 tests/fixtures/health/nltk_data：一份自造的空 punkt_tab（不是 nltk 的数据，
  哈希与协议钉住的不同），--tiny 只记录哈希，所以彩排不依赖下载；真模型模式会拒绝它。
- 先生成、后评分（审查第 1 轮）：--skip-ifeval-scoring 在拿不到 IFEval 依赖与 punkt_tab 时照样跑完；--rescore 与
  跑完即评的结果逐字节相同（子进程的 --tiny 输出，与假适配器上的全部 200 题）。
- 协议 v2（DEV-0013，D28-7）：MMLU-Pro 生成一次（256），截出前 32 个 token 的诊断读数 mmlu_pro_fmt32。三家族的 --tiny 上，
  截出的读数与真的用 max_new_tokens=32 生成的结果逐字段相同（停止 token 用 logits 处理器强制放在第 1、6、32、33、34、101 步，
  或一直禁止）；--audit-fmt32 的运行期审计；RUN-0009 的 v1 输出照旧能重评、能比较。
- 协议 v3（DEV-0016，D36-4）：MMLU-Pro 另有受限字母读数 mmlu_pro_letter（标题）：每题一次 option_logits，选项为本题的实际字母；
  256 的自由作答改名 mmlu_pro_free256（诊断）。预热时查一次 A–J 的写法，写进输出的 option_spellings，每题核对相同。
  RUN-0011 的 v2 输出照旧能重评（逐字节相同）、能比较。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.adapters import batching, gemma4, internvl35, qwen35, registry
from src.adapters.base import GenerationResult, ModelInputs, OptionSpelling, Span
from src.health import protocol as P
from src.health.items import load_items
from src.health.run import (
    LetterReadingError,
    PrefixReadingError,
    RescoreError,
    TextOnlyViolation,
    audit_prefix_readings,
    generate_text_only,
    generation_record,
    prefix_generation,
    rescore_records,
    run_items,
    run_one,
    score_record,
)
from src.health.scoring import IFEVAL_SKIP_REASON, SCORED_SECTIONS, compare, ifeval_follow, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/health_check.py"
SYNTHETIC_NLTK_DATA = REPO_ROOT / "tests/fixtures/health/nltk_data"
FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
LETTER, FREE256, FMT32 = P.MMLU_PRO_LETTER, P.MMLU_PRO_FREE256, P.MMLU_PRO_FMT32
K32 = P.FMT32_MAX_NEW_TOKENS
READING_NAMES = [r.name for r in P.READINGS]
RUN0009 = REPO_ROOT / "experiments/P0/0009_ifeval_rescore"
V1_FILES = sorted(RUN0009.glob("health_*.json"))
RUN0011 = REPO_ROOT / "experiments/P0/0011_health_v2_base"
V2_FILES = sorted(RUN0011.glob("health_*.json"))


@pytest.fixture(scope="module")
def items():
    obj, _ = load_items()
    return obj


# ---------------------------------------------------------------------------
# 假适配器
# ---------------------------------------------------------------------------


FAKE_STOP = 0  # 假 tokenizer：一个字符一个 token（码位），0 是停止 token


class FakeAdapter:
    """只实现 build_inputs / generate / option_logits / split_answer：回答由 responder(prompt) 给出，受限字母读数选
    letter(prompt, options)（默认第一个选项），调用逐条记下。

    一个字符一个 token；回答写完再生成停止 token，被 max_new_tokens 截断时没有停止 token（与适配器的 token_ids 约定相同：
    截到第一个停止 token 为止，含）。split_answer 去掉末尾的停止 token 得 text，raw_text 在末尾记 "<stop>"。
    option_logits：字母 X 的写法 "X"（id 1000 + 码位）与 " X"（id 2000 + 码位）；选中的字母两种写法的 logit 是 2、1，其余 0、-1；
    经 batching.make_option_scores 组装（与真适配器同一段代码）。
    """

    def __init__(self, responder, *, letter=None, visual_tokens=0, fail_on_call=None):
        self.responder, self.visual_tokens, self.fail_on_call = responder, visual_tokens, fail_on_call
        self.letter = letter or (lambda prompt, options: options[0])
        self.calls = []

    def option_logits(self, inputs_list, options, *, enable_thinking):
        rows, table = [], []
        logits = torch.zeros(len(inputs_list), FAKE_VOCAB, dtype=torch.float32)
        for b, (inp, opts) in enumerate(zip(inputs_list, options)):
            prompt = inp.tensors["prompt"]
            self.calls.append(("option_logits", prompt, tuple(opts), enable_thinking))
            pick = self.letter(prompt, list(opts))
            table.append([((OptionSpelling(o, 1000 + ord(o)), OptionSpelling(" " + o, 2000 + ord(o))), ()) for o in opts])
            for o in opts:
                logits[b, 1000 + ord(o)], logits[b, 2000 + ord(o)] = (2.0, 1.0) if o == pick else (0.0, -1.0)
            rows.append(batching.PromptRow(ids=(1,) * (5 + len(prompt) // 4), n_visual_tokens=self.visual_tokens,
                                           n_image_pixels=1024 * self.visual_tokens))
        return batching.make_option_scores(rows, logits, [tuple(o) for o in options], table, wall=0.001)

    def build_inputs(self, images, instruction, order):
        self.calls.append(("build_inputs", list(images), instruction, order))
        return ModelInputs(tensors={"prompt": instruction}, order=order, images=(), instruction_span=Span(0, 1))

    def generate(self, inputs, *, enable_thinking, max_new_tokens, **kw):
        prompt = inputs.tensors["prompt"]
        self.calls.append(("generate", prompt, enable_thinking, max_new_tokens, kw))
        if self.fail_on_call is not None and sum(c[0] == "generate" for c in self.calls) == self.fail_on_call:
            raise RuntimeError("boom")
        ids = (*(ord(c) for c in self.responder(prompt)), FAKE_STOP)[:max_new_tokens]
        text, raw_text, unclosed = self.split_answer(ids, enable_thinking=enable_thinking)
        return GenerationResult(text=text, token_ids=ids, n_generated_tokens=len(ids),
                                n_prompt_tokens=5 + len(prompt) // 4, n_visual_tokens=self.visual_tokens,
                                wall_time_s=0.001, raw_text=raw_text, thinking_unclosed=unclosed,
                                stopped=ids[-1:] == (FAKE_STOP,))

    def split_answer(self, token_ids, *, enable_thinking):
        ids = list(token_ids)
        stopped = ids[-1:] == [FAKE_STOP]
        body = "".join(chr(i) for i in (ids[:-1] if stopped else ids))
        return body, body + ("<stop>" if stopped else ""), False


FAKE_VOCAB = 2100


def canned_letter(items):
    """受限字母读数：偶数位置选答案字母，奇数位置选第一个不是答案的字母（准确率恰为 50%）。"""
    pick = {}
    for k, it in enumerate(items["items"]["mmlu_pro"]):
        letters = P.option_letters(len(it["options"]))
        pick[P.mmlu_pro_prompt(it)] = it["answer"] if k % 2 == 0 else next(x for x in letters if x != it["answer"])
    return lambda prompt, options: pick.get(prompt, options[0])


def canned_responder(items):
    """MMLU-Pro：偶数位置答对；GSM8K：每 3 题一轮（带 ####、不带 ####、答错）；IFEval：空回答（官方规则下全部未遵循）。"""
    answers = {P.WARMUP_PROMPT: "OK"}
    for k, it in enumerate(items["items"]["mmlu_pro"]):
        answers[P.mmlu_pro_prompt(it)] = f"The answer is ({it['answer']})." if k % 2 == 0 else "I am not sure."
    for k, it in enumerate(items["items"]["gsm8k"]):
        answers[P.gsm8k_prompt(it)] = [f"Work.\n#### {it['answer']:,}", f"So it is {it['answer']}.",
                                       f"#### {it['answer'] + 1}"][k % 3]
    for it in items["items"]["ifeval"]:
        answers[P.ifeval_prompt(it)] = ""
    return answers.__getitem__


def test_full_run_follows_protocol(items):
    ad = FakeAdapter(canned_responder(items), letter=canned_letter(items))
    state = run_items(ad, items)
    builds = [c for c in ad.calls if c[0] == "build_inputs"]
    gens = [c for c in ad.calls if c[0] == "generate"]
    letters = [c for c in ad.calls if c[0] == "option_logits"]
    assert len(gens) == 201 and len(letters) == 101  # 预热各 1 次 + 200 次生成 + MMLU-Pro 的 100 次受限字母读数
    assert len(builds) == 302
    assert gens[0][1:4] == (P.WARMUP_PROMPT, False, P.WARMUP_MAX_NEW_TOKENS)
    assert letters[0] == ("option_logits", P.WARMUP_PROMPT, tuple(P.OPTION_LETTERS), False)
    expected = [(b, P.PROMPT_BUILDERS[b](it)) for b in P.BENCHMARKS for it in items["items"][b]]
    for (bench, prompt), g in zip(expected, gens[1:]):
        assert g[1:4] == (prompt, False, P.MAX_NEW_TOKENS[bench])
    for it, c in zip(items["items"]["mmlu_pro"], letters[1:]):  # 选项为本题的实际字母，关思考
        assert c == ("option_logits", P.mmlu_pro_prompt(it), tuple(P.option_letters(len(it["options"]))), False)
    assert all(b[1:] == ([], b[2], "instruction_first") for b in builds)
    for g in gens:
        assert g[4] == dict(P.GREEDY_KWARGS) and g[4]["do_sample"] is False  # 每次都显式给 greedy
    assert P.MAX_NEW_TOKENS["mmlu_pro"] == 256  # MMLU-Pro 只自由生成一次，256
    # 逐题记录按读数分键、按题目文件的顺序；MMLU-Pro 的回答都在 32 个 token 内写完，两个自由生成读数只差墙钟
    assert list(state["records"]) == READING_NAMES
    for name in READING_NAMES:
        bench = P.READINGS_BY_NAME[name].benchmark
        assert [r["id"] for r in state["records"][name]] == [it["id"] for it in items["items"][bench]]
    for full, short in zip(state["records"][FREE256], state["records"][FMT32]):
        assert full["stopped"] is True and full["n_generated_tokens"] < K32
        assert {**full, "wall_time_s": None} == short
    rec = state["records"][LETTER][0]
    assert list(rec) == ["id", "prompt_sha256", "options", "token_ids", "logits", "probs", "choice", "n_prompt_tokens",
                         "wall_time_s", "letters", "gold", "correct", "p_gold"]
    assert rec["prompt_sha256"] == state["records"][FREE256][0]["prompt_sha256"]
    assert rec["choice"] == rec["gold"] and rec["correct"] is True and rec["token_ids"][0] == 1000 + ord("A")
    s = summarize(state["records"])
    assert s["headline"] == {"mmlu_pro": 50.0, "gsm8k": 68.0, "ifeval": 0.0}
    # 正确选项的概率：选中的字母 logit 2、其余 0，n 个选项时选中的概率 e²/(e² + n − 1)，没选中的 1/(e² + n − 1)
    want = []
    for k, it in enumerate(items["items"]["mmlu_pro"]):
        z = math.exp(2) + len(it["options"]) - 1
        want.append(math.exp(2) / z if k % 2 == 0 else 1 / z)
    assert s["scores"][LETTER]["mean_p_gold"]["mean"] == pytest.approx(sum(want) / 100, abs=1e-6)
    assert s["scores"][LETTER]["accuracy"] == {"n": 100, "n_correct": 50, "pct": 50.0}
    assert s["scores"][FREE256]["accuracy"] == s["scores"][FMT32]["accuracy"] == {"n": 100, "n_correct": 50, "pct": 50.0}
    assert s["scores"]["gsm8k"]["strict"] == {"n": 50, "n_correct": 17, "pct": 34.0}
    assert s["parse_tally"][FREE256]["failures"] == {"no_choice": 50}
    assert s["parse_tally"]["gsm8k_strict"]["failures"] == {"no_marker": 17}
    assert s["scores"]["ifeval"]["instruction_level_loose"]["n"] == 77
    # 记账：每题一个样本；MMLU-Pro 每题两次调用（option_logits 与 generate），各算一轮；预热不计入
    comp = state["compute"]
    assert [comp[b]["n_samples"] for b in P.BENCHMARKS] == [100, 50, 50]
    assert comp["mmlu_pro"]["n_calls"] == comp["mmlu_pro"]["n_rounds"] == 200
    assert comp["total"]["n_calls"] == comp["total"]["n_rounds"] == 300 and comp["total"]["n_samples"] == 200
    assert comp["total"]["visual_tokens"] == 0
    assert comp["total"]["generated_tokens"] == 100 + sum(r["n_generated_tokens"] for b in (FREE256, "gsm8k", "ifeval")
                                                           for r in state["records"][b])
    assert state["warmup"]["counted_in_compute"] is False
    assert state["option_spellings"] == {x: {"spellings": [{"text": x, "token_id": 1000 + ord(x)},
                                                           {"text": " " + x, "token_id": 2000 + ord(x)}], "skipped": []}
                                         for x in P.OPTION_LETTERS}


def test_limited_run_records(items):
    responses = {P.WARMUP_PROMPT: "OK"}
    ifeval_texts = ["First paragraph.\n\nSecond one here", "<<A Title>>\nbody", "*one* and *two*, " + "word " * 30]
    for it, text in zip(items["items"]["ifeval"][:3], ifeval_texts):
        responses[P.ifeval_prompt(it)] = text
    ad = FakeAdapter(lambda p: responses.get(p, "B"))
    msgs = []
    state = run_items(ad, items, limit=3, progress=msgs.append)
    assert {b: len(v) for b, v in state["records"].items()} == {LETTER: 3, FREE256: 3, FMT32: 3, "gsm8k": 3, "ifeval": 3}
    for it, rec in zip(items["items"]["ifeval"][:3], state["records"]["ifeval"]):
        assert {k: rec[k] for k in ("strict", "loose")} == ifeval_follow(it, rec["text"])
        assert rec["instruction_id_list"] == it["instruction_id_list"]
    rec = state["records"][FREE256][0]
    assert rec["raw_text"] == "B<stop>" and rec["text"] == "B" and rec["parsed"] == "B"
    assert rec["reached_max_new_tokens"] is False and len(rec["prompt_sha256"]) == 64
    assert rec["stopped"] is True and rec["n_generated_tokens"] == 2
    assert msgs == ["mmlu_pro 3/3  生成 token 累计 9", "gsm8k 3/3  生成 token 累计 6",  # MMLU-Pro：3 × (2 + 受限字母读数 1)
                    f"ifeval 3/3  生成 token 累计 {sum(r['n_generated_tokens'] for r in state['records']['ifeval'])}"]


def test_failure_keeps_partial_results(items):
    ad = FakeAdapter(lambda p: "A", fail_on_call=5)  # 预热是第 1 次 generate，第 4 题出错（它的受限字母读数已做完）
    state = {}
    with pytest.raises(RuntimeError, match="boom"):
        run_items(ad, items, state=state)
    assert len(state["records"][LETTER]) == len(state["records"][FREE256]) == 3
    assert state["compute"]["mmlu_pro"]["n_samples"] == 3
    assert "warmup" in state and "total" not in state["compute"]


def test_visual_tokens_are_rejected(items):
    with pytest.raises(TextOnlyViolation):
        run_items(FakeAdapter(lambda p: "A", visual_tokens=1), items, limit=1)


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_limit_must_be_positive_int(items, bad):
    with pytest.raises(ValueError):
        run_items(FakeAdapter(lambda p: "A"), items, limit=bad)


# ---------------------------------------------------------------------------
# 协议 v2：mmlu_pro_fmt32 = 同一次生成的前 32 个 token（假适配器，一个字符一个 token）
# ---------------------------------------------------------------------------


def _real_32(adapter, item):
    """真的用 max_new_tokens=32 生成一次，按生成读数的同一套规则记录与评分（墙钟记 None，便于逐字段比较）。"""
    result, _ = generate_text_only(adapter, P.mmlu_pro_prompt(item), max_new_tokens=K32)
    gen = generation_record(item, hashlib.sha256(P.mmlu_pro_prompt(item).encode()).hexdigest(), result,
                            max_new_tokens=K32, wall=None)
    return {**gen, **score_record("mmlu_pro", item, gen, score_ifeval=True)}


def test_fmt32_is_the_32_token_prefix_on_fake(items):
    """停止 token 落在第 32 步之前、恰在第 32 步、恰在第 33 步、32 与 256 之间、不停止（截断在 256）：截出的读数与真的
    只生成 32 个逐字段相同（墙钟除外），评分规则与 v1 的 32 口径相同（截断的回答照样解析）。"""
    mm = items["items"]["mmlu_pro"][:6]
    pad = "x" * 40
    texts = [
        f"({mm[0]['answer']})",                                # 3 个字符 + 停止：都在 32 内
        f"{pad} The answer is ({mm[1]['answer']}).",           # 答案在第 32 个 token 之后：256 对、fmt32 判 no_choice
        "y" * (K32 - 1),                                       # 31 + 停止 = 32 个 token：fmt32 停在第 32 步
        "z" * K32,                                             # 32 + 停止 = 33 个 token：fmt32 写满、没有停止 token
        f"({mm[4]['answer']}) " + "w" * 300,                   # 不停止，截断在 256；截断的回答照样解析，两个读数都对
        "q" * 100,                                             # 101 个 token
    ]
    by_prompt = {P.mmlu_pro_prompt(it): t for it, t in zip(mm, texts)}
    ad = FakeAdapter(lambda p: by_prompt.get(p, "OK"))
    rows = []
    for it in mm:
        recs, compute = run_one(ad, "mmlu_pro", it)
        assert list(recs) == [LETTER, FREE256, FMT32]
        full, short = recs[FREE256], recs[FMT32]
        assert short == _real_32(ad, it)
        assert compute.n_calls == 2  # 受限字母读数一次、自由生成一次；截出的读数不另调用模型
        rows.append((full["n_generated_tokens"], full["stopped"], short["n_generated_tokens"], short["stopped"],
                     short["reached_max_new_tokens"], full["correct"], short["correct"], short["failure"]))
    assert rows == [
        (4, True, 4, True, False, True, True, None),
        (len(texts[1]) + 1, True, 32, False, True, True, False, "no_choice"),
        (32, True, 32, True, True, False, False, "no_choice"),
        (33, True, 32, False, True, False, False, "no_choice"),
        (256, False, 32, False, True, True, True, None),
        (101, True, 32, False, True, False, False, "no_choice"),
    ]
    assert recs[FMT32]["raw_text"] == "q" * 32 and recs[FMT32]["wall_time_s"] is None


def test_prefix_generation_refuses_what_it_cannot_reconstruct(items):
    it = items["items"]["mmlu_pro"][0]
    ad = FakeAdapter(lambda p: "A")
    result, _ = generate_text_only(ad, P.mmlu_pro_prompt(it), max_new_tokens=256)
    ok = prefix_generation(ad, it, "h", result, max_new_tokens=K32)
    assert (ok["text"], ok["stopped"], ok["n_generated_tokens"]) == ("A", True, 2)

    class NoSplit:  # 没有 split_answer 的适配器
        pass

    with pytest.raises(PrefixReadingError, match="split_answer"):
        prefix_generation(NoSplit(), it, "h", result, max_new_tokens=K32)
    unrecorded = GenerationResult(text="A", token_ids=result.token_ids, n_generated_tokens=2, n_prompt_tokens=1,
                                  n_visual_tokens=0, wall_time_s=0.0, raw_text="A<stop>")  # stopped=None：未记录
    with pytest.raises(PrefixReadingError, match="stopped"):
        prefix_generation(ad, it, "h", unrecorded, max_new_tokens=K32)
    with pytest.raises(PrefixReadingError, match="stopped"):
        generation_record(it, "h", unrecorded, max_new_tokens=256, wall=0.0)
    other = GenerationResult(text="B", token_ids=result.token_ids, n_generated_tokens=2, n_prompt_tokens=1,
                             n_visual_tokens=0, wall_time_s=0.0, raw_text="A<stop>", stopped=True)
    with pytest.raises(PrefixReadingError, match="split_answer"):  # split_answer 与生成时的拆分不一致
        prefix_generation(ad, it, "h", other, max_new_tokens=K32)


def test_audit_prefix_readings_on_fake(items):
    long = {P.mmlu_pro_prompt(it): "x" * (15 * k) for k, it in enumerate(items["items"]["mmlu_pro"][:4])}
    ad = FakeAdapter(lambda p: long.get(p, "OK"))
    state = run_items(ad, items, limit=4, score_ifeval=False)  # IFEval 的评分与此无关（也免得要 punkt_tab）
    n_calls = len([c for c in ad.calls if c[0] == "generate"])
    audit = audit_prefix_readings(ad, items, state["records"], n=10)  # n 大于已有的题数：只审已有的
    assert list(audit) == [FMT32]
    a = audit[FMT32]
    assert (a["n"], a["n_identical"], a["mismatches"], a["max_new_tokens"], a["counted_in_compute"]) == (
        4, 4, [], K32, False)
    assert a["compute"]["n_calls"] == 4 and len([c for c in ad.calls if c[0] == "generate"]) == n_calls + 4
    assert all(c[3] == K32 for c in ad.calls[-8:] if c[0] == "generate")
    # 截出的记录被改过：审计报出不一致的字段
    bad = copy.deepcopy(state["records"])
    bad[FMT32][2]["text"] = "tampered"
    a = audit_prefix_readings(ad, items, bad, n=4)[FMT32]
    assert a["n_identical"] == 3 and a["mismatches"][0]["id"] == items["items"]["mmlu_pro"][2]["id"]
    assert a["mismatches"][0]["fields"] == ["text"]
    with pytest.raises(ValueError):
        audit_prefix_readings(ad, items, state["records"], n=0)


# ---------------------------------------------------------------------------
# 三家族：零张图输入与全部 200 个提示（真实 tokenizer 与模板，小型随机模型）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("_health_check_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_every_prompt_builds_with_zero_images(script, items, key):
    args = script.parse_args(["--tiny", "--model", key, "--out", "unused.json"])
    ad, entry, _ = script.make_adapter(args)
    assert entry.family == registry.get_model(key).family and ad.model_key == f"tiny({key})"
    for bench in P.BENCHMARKS:
        for it in items["items"][bench]:
            prompt = P.PROMPT_BUILDERS[bench](it)
            inp = ad.build_inputs([], prompt, P.ORDER)
            assert inp.images == () and inp.query_span is None
            ids = inp.tensors["input_ids"][0].tolist()
            assert ad.tokenizer.decode(ids[inp.instruction_span.start : inp.instruction_span.end]) == prompt, (bench, it["id"])
            assert "pixel_values" not in inp.tensors


FAMILY_MODULES = {registry.FAMILY_QWEN35: qwen35, registry.FAMILY_GEMMA4: gemma4, registry.FAMILY_INTERNVL35: internvl35}
# 强制的停止步（从 0 起）：第 1 步、32 之前、恰为第 32 个 token、第 33、34 个、32 与 256 之间；None = 一直不许停（截断在 256）
FORCED_STOP_STEPS = (0, 5, 31, 32, 33, 100, None)


class _ForceStop:
    """测试用 logits 处理器：第 step 步只许生成第一个停止 token，此前各步禁止全部停止 token；step=None 时一直禁止。

    每次 generate 用一个新实例（按调用次数计步）。小型随机模型自己几乎不会生成停止 token，用它把三种情形都造出来：
    停在前 32 个 token 里、停在 32 之后、不停。
    """

    def __init__(self, step, stop_ids):
        self.step, self.stop_ids, self.calls = step, list(stop_ids), 0

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        if self.step is not None and self.calls == self.step:
            scores[:] = float("-inf")
            scores[:, self.stop_ids[0]] = 0.0
        else:
            scores[:, self.stop_ids] = float("-inf")
        self.calls += 1
        return scores


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_tiny_fmt32_equals_real_32_token_generation(script, items, key, monkeypatch):
    """真实 tokenizer、模板与 split_answer（小型随机模型，CPU）：MMLU-Pro 生成一次 256、截出的 32 读数，与真的用
    max_new_tokens=32 生成的结果逐字段相同（墙钟除外：截出的读数不单独计时，真的生成这里也记 None）。"""
    ad, entry, _ = script.make_adapter(script.parse_args(["--tiny", "--model", key, "--out", "unused.json"]))
    stop_ids = [ad.tokenizer.convert_tokens_to_ids(t) for t in FAMILY_MODULES[entry.family].STOP_TOKENS]
    assert all(isinstance(i, int) for i in stop_ids)
    orig = ad.generate
    it = items["items"]["mmlu_pro"][0]
    for step in FORCED_STOP_STEPS:
        def forced(inputs, _step=step, **kw):
            return orig(inputs, logits_processor=[_ForceStop(_step, stop_ids)], **kw)

        monkeypatch.setattr(ad, "generate", forced)
        recs, _ = run_one(ad, "mmlu_pro", it)
        full, short = recs[FREE256], recs[FMT32]
        assert short == _real_32(ad, it), (key, step)
        n_full = 256 if step is None else step + 1
        assert (full["n_generated_tokens"], full["stopped"]) == (n_full, step is not None), (key, step)
        assert (short["n_generated_tokens"], short["stopped"], short["reached_max_new_tokens"]) == (
            min(n_full, K32), step is not None and n_full <= K32, n_full >= K32), (key, step)
        if n_full <= K32:
            assert (short["text"], short["raw_text"]) == (full["text"], full["raw_text"])  # 前缀就是全部


def test_parse_args_rules(script):
    a = script.parse_args(["--tiny", "--out", "x.json"])
    assert (a.device, a.dtype, a.model) == ("cpu", "fp32", None)
    a = script.parse_args(["--model", "qwen3.5-2b", "--out", "x.json"])
    assert (a.device, a.dtype) == ("cuda", "bf16")
    assert a.audit_fmt32 is None
    for bad in (["--tiny"], ["--out", "x.json"], ["--tiny", "--out", "x.json", "--limit", "0"],
                ["--tiny", "--out", "x.json", "--dtype", "bfloat16"], ["--tiny", "--out", "x.json", "--audit-fmt32", "0"]):
        with pytest.raises(SystemExit):
            script.parse_args(bad)
    assert script.parse_args(["--tiny", "--out", "x.json", "--audit-fmt32", "5"]).audit_fmt32 == 5


# ---------------------------------------------------------------------------
# 端到端（子进程，与服务器同一流程）
# ---------------------------------------------------------------------------


def _run_script(args, *, nltk_data=SYNTHETIC_NLTK_DATA, timeout=900):
    env = {**os.environ, "NLTK_DATA": str(nltk_data)}
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    cache = {}
    root = tmp_path_factory.mktemp("health_tiny")

    def get(key):
        if key not in cache:
            out = root / f"{key}.json"
            cache[key] = (_run_script(["--tiny", "--model", key, "--limit", "3", "--audit-fmt32", "3",
                                       "--out", str(out)]), out)
        return cache[key]

    return get


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_tiny_end_to_end(tiny_run, items, key):
    proc, out = tiny_run(key)
    assert out.exists(), proc.stderr[-3000:]
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert proc.returncode == 0 and rep["ok"] is True, (rep.get("error"), rep.get("traceback"))
    assert (rep["mode"], rep["limit"], rep["protocol_version"]) == ("tiny", 3, P.PROTOCOL_VERSION)
    entry = registry.get_model(key)
    assert (rep["model"]["key"], rep["model"]["family"], rep["model"]["revision"]) == (entry.key, entry.family, entry.revision)
    assert rep["model"]["device"] == "cpu" and rep["model"]["dtype"] == "torch.float32"
    gen = rep["generation"]
    assert {k: gen[k] for k in ("order", "n_images", "enable_thinking", "greedy_kwargs", "max_new_tokens")} == {
        "order": "instruction_first", "n_images": 0, "enable_thinking": False, "greedy_kwargs": dict(P.GREEDY_KWARGS),
        "max_new_tokens": {"mmlu_pro": 256, "gsm8k": 512, "ifeval": 1024}}
    assert gen["readings"] == {
        LETTER: {"benchmark": "mmlu_pro", "kind": "option_letter", "max_new_tokens": 1, "role": "headline",
                 "prefix_of": None},
        FREE256: {"benchmark": "mmlu_pro", "kind": "generate", "max_new_tokens": 256, "role": "diagnostic",
                  "prefix_of": None},
        FMT32: {"benchmark": "mmlu_pro", "kind": "generate", "max_new_tokens": 32, "role": "diagnostic",
                "prefix_of": FREE256},
        "gsm8k": {"benchmark": "gsm8k", "kind": "generate", "max_new_tokens": 512, "role": "headline", "prefix_of": None},
        "ifeval": {"benchmark": "ifeval", "kind": "generate", "max_new_tokens": 1024, "role": "headline",
                   "prefix_of": None}}
    assert gen["option_letters"]["spelling_templates"] == ["{}", " {}"]
    assert rep["items_file"] == {"path": P.ITEMS_PATH, "sha256": P.ITEMS_SHA256}
    assert list(rep["records"]) == list(rep["scores"]) == list(rep["diagnostics"]) == READING_NAMES
    for reading in P.READINGS:
        recs = rep["records"][reading.name]
        assert [r["id"] for r in recs] == [it["id"] for it in items["items"][reading.benchmark][:3]]
        for r, it in zip(recs, items["items"][reading.benchmark]):
            if reading.kind == P.KIND_OPTION_LETTER:  # 选项为本题的实际字母；概率在选项集合上
                assert r["options"] == list(P.option_letters(len(it["options"]))) and r["choice"] in r["options"]
                assert abs(sum(r["probs"]) - 1) < 1e-5 and r["p_gold"] == r["probs"][r["options"].index(it["answer"])]
                assert r["correct"] == (r["choice"] == it["answer"])
                continue
            assert isinstance(r["raw_text"], str) and isinstance(r["text"], str) and isinstance(r["stopped"], bool)
            assert 1 <= r["n_generated_tokens"] <= reading.max_new_tokens
    for full, short in zip(rep["records"][FREE256], rep["records"][FMT32]):
        assert short["n_generated_tokens"] == min(full["n_generated_tokens"], K32) and short["wall_time_s"] is None
    assert list(rep["headline"]) == list(P.BENCHMARKS)
    assert rep["headline"]["mmlu_pro"] == rep["scores"][LETTER]["accuracy"]["pct"]
    assert {k: v["n"] for k, v in rep["parse_tally"].items()} == {FREE256: 3, FMT32: 3, "gsm8k_strict": 3,
                                                                   "gsm8k_flexible": 3}
    # 这次运行实际用到的字母写法：A–J 各有 "X" 与 " X" 两个单 token 写法（id 见 test_adapter_option_logits）
    sp = rep["option_spellings"]
    assert list(sp) == list(P.OPTION_LETTERS)
    assert all([x["text"] for x in v["spellings"]] == [k, " " + k] and v["skipped"] == [] for k, v in sp.items())
    # --audit-fmt32 3：截出的读数与真的只生成 32 个逐字段相同；审计的调用不计入记账
    audit = rep["prefix_audit"][FMT32]
    assert (audit["n"], audit["n_identical"], audit["mismatches"], audit["counted_in_compute"]) == (3, 3, [], False)
    assert audit["compute"]["n_calls"] == 3 and rep["run"]["config"]["audit_fmt32"] == 3
    assert "prefix audit mmlu_pro_fmt32: 3/3 identical" in proc.stdout
    assert rep["scores"]["ifeval"]["instruction_level_strict"]["n"] == sum(
        len(it["instruction_id_list"]) for it in items["items"]["ifeval"][:3])
    total = rep["compute"]["total"]
    assert (total["n_samples"], total["n_calls"], total["visual_tokens"]) == (9, 12, 0)  # MMLU-Pro 每题两次调用
    assert rep["warmup"]["counted_in_compute"] is False
    scoring = rep["scoring"]
    assert (scoring["mode"], scoring["benchmarks_scored"], scoring["source"]) == ("inline", list(P.BENCHMARKS), None)
    assert scoring["code"]["commit"] == rep["run"]["meta"]["git"]["commit"] and scoring["scored_at_utc"]
    punkt = scoring["ifeval"]["punkt_tab"]
    assert str(SYNTHETIC_NLTK_DATA) in punkt["path"] and punkt["matches_pinned"] is False
    assert (scoring["ifeval"]["scored"], scoring["ifeval"]["skip_reason"], scoring["ifeval"]["langdetect_seed"]) == (
        True, None, 0)
    assert rep["unscored"] == {}
    run = rep["run"]
    assert run["config"]["seed"] == run["meta"]["seed"] == P.SEED
    assert list(run["meta"]["data_sha256"].values()) == [P.ITEMS_SHA256]
    assert run["meta"]["git"]["commit"] is None or len(run["meta"]["git"]["commit"]) == 40
    assert set(run["packages_ifeval"]) == {"nltk", "langdetect", "immutabledict"}
    # 汇总与逐题记录一致（门报告的数从记录重算得到同一结果）
    assert summarize(rep["records"])["headline"] == rep["headline"]


def test_compare_cli(tiny_run, tmp_path):
    _, out = tiny_run("internvl3.5-2b-hf")
    cmp_out = tmp_path / "cmp.json"
    proc = _run_script(["--compare", str(out), str(out), "--out", str(cmp_out)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(cmp_out.read_text(encoding="utf-8"))
    assert res["any_flag"] is False and res["partial_run"] is True and res["unscored"] == []
    assert res["protocol_versions"] == {"base": 3, "merged": 3} and res["incomparable"] == [] and res["notes"] == []
    assert [(r["reading"], r["metric"]) for r in res["rows"]][:4] == [(LETTER, "accuracy"), (LETTER, "mean_p_gold"),
                                                                      (FREE256, "accuracy"), (FMT32, "accuracy")]
    assert [r["reading"] for r in res["rows"]][4:] == ["gsm8k", "gsm8k"] + ["ifeval"] * 4
    assert all(r["comparable"] and r["delta_pp"] == 0.0 and r["ci95_pp"] == [0.0, 0.0] for r in res["rows"])
    assert all(r["sign_test"] == ({"n_better": 0, "n_worse": 0, "p_value": 1.0} if r["metric"] != "mean_p_gold" else None)
               for r in res["rows"])
    assert [t["reading"] for t in res["truncation"]["rows"]] == READING_NAMES[1:] and res["truncation"]["notices"] == []
    assert "any_flag=False" in proc.stdout and "CI95=[+0.0, +0.0]" in proc.stdout and "[诊断，不标红]" in proc.stdout


# ---------------------------------------------------------------------------
# 先生成、后评分（审查第 1 轮）：--skip-ifeval-scoring 与 --rescore
# ---------------------------------------------------------------------------

SKIP_KEY = "internvl3.5-2b-hf"  # 三家族里最快的；生成是确定的（同一小模型、greedy、CPU）


def _blocked_ifeval_env(tmp_path: Path) -> dict:
    """子进程环境：nltk、langdetect、immutabledict 一 import 就报错；NLTK_DATA 指向空目录（没有 punkt_tab）。"""
    blocked = tmp_path / "blocked"
    for name in ("nltk", "langdetect", "immutabledict"):
        (blocked / name).mkdir(parents=True)
        (blocked / name / "__init__.py").write_text(f"raise ImportError('{name} is blocked in this test')\n")
    (tmp_path / "empty_nltk_data").mkdir()
    pythonpath = os.pathsep.join(p for p in (str(blocked), os.environ.get("PYTHONPATH", "")) if p)
    return {**os.environ, "PYTHONPATH": pythonpath, "NLTK_DATA": str(tmp_path / "empty_nltk_data")}


@pytest.fixture(scope="module")
def skip_run(tmp_path_factory):
    """--skip-ifeval-scoring：IFEval 的依赖与 punkt_tab 都拿不到时照样跑完。"""
    root = tmp_path_factory.mktemp("health_skip")
    out = root / "skip.json"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--tiny", "--model", SKIP_KEY, "--limit", "3",
                           "--skip-ifeval-scoring", "--out", str(out)], cwd=REPO_ROOT, env=_blocked_ifeval_env(root),
                          capture_output=True, text=True, timeout=900)
    return proc, out


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)


def _scored_view(rep: dict, *, drop_timing: bool) -> str:
    """跑完即评与重评应当逐字节相同的部分；比较两次不同的运行时去掉逐题的墙钟（唯一与评分无关、每次都变的逐题字段）。"""
    view = {s: rep[s] for s in SCORED_SECTIONS}
    if drop_timing:
        view["records"] = {b: [{k: v for k, v in r.items() if k != "wall_time_s"} for r in recs]
                           for b, recs in rep["records"].items()}
    return _dump(view)


def test_skip_ifeval_scoring_needs_no_ifeval_dependencies(skip_run, items):
    proc, out = skip_run
    assert out.exists(), proc.stderr[-3000:]
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert proc.returncode == 0 and rep["ok"] is True, (rep.get("error"), rep.get("traceback"))
    assert "is blocked in this test" not in proc.stderr
    # IFEval：生成照常、原始输出照常保存；分数为 null 并写明原因
    ife = rep["records"]["ifeval"]
    assert [r["id"] for r in ife] == [it["id"] for it in items["items"]["ifeval"][:3]]
    assert all(r["raw_text"] and r["strict"] is None and r["loose"] is None for r in ife)
    assert rep["headline"]["ifeval"] is None and rep["unscored"] == {"ifeval": IFEVAL_SKIP_REASON}
    assert all(v["n_correct"] is None and v["pct"] is None for v in rep["scores"]["ifeval"].values())
    assert rep["scores"]["ifeval"]["instruction_level_strict"]["n"] == sum(
        len(it["instruction_id_list"]) for it in items["items"]["ifeval"][:3])
    # 另两项照常评分
    assert rep["headline"]["mmlu_pro"] is not None and rep["headline"]["gsm8k"] is not None
    assert len(rep["records"][LETTER]) == 3
    assert rep["parse_tally"][FREE256]["n"] == 3 and rep["headline"]["mmlu_pro"] is not None
    ifeval = rep["scoring"]["ifeval"]
    assert rep["scoring"]["benchmarks_scored"] == ["mmlu_pro", "gsm8k"]
    assert (ifeval["scored"], ifeval["skip_reason"], ifeval["punkt_tab"], ifeval["langdetect_seed"]) == (
        False, IFEVAL_SKIP_REASON, None, None)
    assert rep["run"]["config"]["skip_ifeval_scoring"] is True
    assert "未评分" in proc.stdout


def test_rescore_is_byte_identical_to_inline_scoring(tiny_run, skip_run, tmp_path):
    """同一套规则：一次跑完即评的输出，与"只存输出、事后 --rescore"的结果，由逐题记录算出的部分逐字节相同。"""
    _, inline_path = tiny_run(SKIP_KEY)
    inline_raw = inline_path.read_bytes()
    inline = json.loads(inline_raw)
    # 1) 对一次跑完的输出重评三项：连逐题墙钟在内逐字节相同
    re_all = tmp_path / "rescored_inline.json"
    proc = _run_script(["--rescore", str(inline_path), "--out", str(re_all)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = json.loads(re_all.read_text(encoding="utf-8"))
    assert _scored_view(rep, drop_timing=False) == _scored_view(inline, drop_timing=False)
    for key in ("model", "run", "generation", "compute", "warmup", "items_file", "argv", "started_utc"):
        assert rep[key] == inline[key], key  # 生成侧的记录原样保留
    scoring = rep["scoring"]
    assert (scoring["mode"], scoring["benchmarks_scored"]) == ("rescore", list(P.BENCHMARKS))
    assert scoring["source"] == {"path": str(inline_path), "sha256": hashlib.sha256(inline_raw).hexdigest()}
    assert scoring["scored_at_utc"] and "commit" in scoring["code"]
    assert scoring["ifeval"]["punkt_tab"]["files_sha256"] == inline["scoring"]["ifeval"]["punkt_tab"]["files_sha256"]
    assert rep["scoring_history"] == [inline["scoring"]]
    assert inline_path.read_bytes() == inline_raw  # 输入原样不动

    # 2) 先生成（不评 IFEval）、后评分：与一次跑完的评分逐字节相同（两次运行只差逐题墙钟）
    _, skip_path = skip_run
    skipped = json.loads(skip_path.read_text(encoding="utf-8"))
    for name in READING_NAMES:  # 前提：两次运行的生成相同
        field = "logits" if name == LETTER else "raw_text"
        assert [r[field] for r in skipped["records"][name]] == [r[field] for r in inline["records"][name]]
    for extra in ([], ["--only-ifeval"]):
        out = tmp_path / f"rescored_skip{'_ifeval' if extra else ''}.json"
        proc = _run_script(["--rescore", str(skip_path), "--out", str(out), *extra], timeout=300)
        assert proc.returncode == 0, proc.stderr[-3000:]
        rep = json.loads(out.read_text(encoding="utf-8"))
        assert _scored_view(rep, drop_timing=True) == _scored_view(inline, drop_timing=True), extra
        assert rep["unscored"] == {} and rep["headline"]["ifeval"] == inline["headline"]["ifeval"]
        assert rep["scoring"]["benchmarks_scored"] == (["ifeval"] if extra else list(P.BENCHMARKS))
        assert rep["scoring_history"] == [skipped["scoring"]] and rep["scoring_history"][0]["ifeval"]["scored"] is False
    # 3) 未评 IFEval 的一边参与比较：该项不比，另两项照常
    res = compare(inline, skipped)
    assert res["unscored"] == ["ifeval"] and not any(r["scored"] for r in res["rows"] if r["benchmark"] == "ifeval")
    assert all(r["scored"] and r["delta_pp"] == 0.0 for r in res["rows"] if r["benchmark"] != "ifeval")


def test_rescore_real_output_requires_pinned_punkt(tiny_run, tmp_path):
    _, inline_path = tiny_run(SKIP_KEY)
    rep = json.loads(inline_path.read_text(encoding="utf-8"))
    rep["mode"] = "real"  # 真模型的输出：重评 IFEval 时要求钉住的 punkt_tab（这里只有自造的空 punkt_tab）
    fake_real = tmp_path / "real.json"
    fake_real.write_text(json.dumps(rep), encoding="utf-8")
    out = tmp_path / "out.json"
    proc = _run_script(["--rescore", str(fake_real), "--out", str(out)], timeout=300)
    assert proc.returncode == 2
    err = json.loads(out.read_text(encoding="utf-8"))
    assert err["ok"] is False and "PunktDataError" in err["error"]


def test_rescore_rejects_mismatched_records(items):
    ad = FakeAdapter(canned_responder(items))
    records = run_items(ad, items, limit=2)["records"]
    assert rescore_records(records, items) == records
    bad = copy.deepcopy(records)
    bad["gsm8k"][0]["prompt_sha256"] = "0" * 64
    with pytest.raises(RescoreError, match="prompt_sha256"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)
    bad[FREE256].reverse()
    with pytest.raises(RescoreError, match="前 2 题"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)
    del bad["ifeval"][1]["text"]
    with pytest.raises(RescoreError, match="text"):
        rescore_records(bad, items, benchmarks=("ifeval",))
    rescore_records(bad, items, benchmarks=("mmlu_pro",))  # 不重评的项原样保留，不核对生成字段
    with pytest.raises(RescoreError, match="未知"):
        rescore_records(records, items, benchmarks=("mmlu",))
    # v2：前缀读数必须与它截取的读数对得上；布局与版本必须对
    bad = copy.deepcopy(records)
    bad[FMT32][0]["n_generated_tokens"] += 1
    with pytest.raises(RescoreError, match="截取"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)
    bad[FMT32][1]["stopped"] = not bad[FMT32][1]["stopped"]
    with pytest.raises(RescoreError, match="截取"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)
    del bad[FREE256][0]["stopped"]  # v2 起的记录必须有 stopped
    with pytest.raises(RescoreError, match="stopped"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)
    del bad[FMT32]
    with pytest.raises(RescoreError, match="v3"):
        rescore_records(bad, items)
    with pytest.raises(RescoreError, match="v1"):  # v3 的记录不能按 v1、v2 的布局读
        rescore_records(records, items, protocol_version=1)
    with pytest.raises(RescoreError, match="v2"):
        rescore_records(records, items, protocol_version=2)
    with pytest.raises(RescoreError, match="不受支持"):
        rescore_records(records, items, protocol_version=4)
    # v3 的受限字母读数：生成部分的字段齐全、选项是本题的实际字母
    bad = copy.deepcopy(records)
    del bad[LETTER][0]["probs"]
    with pytest.raises(RescoreError, match="probs"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)
    bad[LETTER][1]["options"] = bad[LETTER][1]["options"][:-1]
    with pytest.raises(RescoreError, match="实际字母"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)  # 评分部分被改过：重评从生成部分重新算出来
    bad[LETTER][0]["correct"], bad[LETTER][0]["p_gold"] = not bad[LETTER][0]["correct"], -1.0
    assert rescore_records(bad, items) == records


def test_rescore_in_process_equals_inline(items):
    """全部 200 题（假适配器）：先不评 IFEval、再重评，与跑完即评的逐题记录与汇总逐字节相同。
    需要 punkt_tab 的题给空回答（官方规则下空回答直接判未遵循，不调用句子切分），其余题给会触发各种 checker 的回答。"""
    punkt_users = {"length_constraints:number_sentences", "change_case:capital_word_frequency"}
    texts = {}
    for k, it in enumerate(items["items"]["ifeval"]):
        texts[P.ifeval_prompt(it)] = "" if punkt_users & set(it["instruction_id_list"]) else [
            "<<A Title>>\n*first* and *second* part, with a comma.\nP.S. done",
            '"THIS IS ALL CAPS AND QUOTED"',
            "lowercase only here\n\n***\n\nSECTION 1\nmore text"][k % 3]
    base = canned_responder(items)

    def responder(prompt):
        return texts[prompt] if prompt in texts else base(prompt)

    inline = run_items(FakeAdapter(responder), items)["records"]
    skipped = run_items(FakeAdapter(responder), items, score_ifeval=False)["records"]
    assert all(r["strict"] is None for r in skipped["ifeval"])
    assert summarize(skipped)["unscored"] == {"ifeval": IFEVAL_SKIP_REASON}
    no_timing = lambda recs: {b: [{k: v for k, v in r.items() if k != "wall_time_s"} for r in rs] for b, rs in recs.items()}
    for benchmarks in (P.BENCHMARKS, ("ifeval",)):
        rescored = rescore_records(skipped, items, benchmarks=benchmarks)
        assert _dump(no_timing(rescored)) == _dump(no_timing(inline))
        assert _dump(summarize(rescored)) == _dump(summarize(inline))
    assert any(r["strict"]["follow_all"] for r in inline["ifeval"])  # 确实有题被判遵循，不是全 False 的平凡情形


def test_cli_mode_rules(script, tmp_path):
    ok = tmp_path / "in.json"
    for bad in (["--rescore", str(ok)],  # 缺 --out
                ["--rescore", str(ok), "--out", str(ok)],  # 覆盖输入
                ["--rescore", str(ok), "--out", "x.json", "--tiny"],
                ["--rescore", str(ok), "--out", "x.json", "--skip-ifeval-scoring"],
                ["--tiny", "--out", "x.json", "--only-ifeval"],
                ["--compare", "a.json", "b.json", "--rescore", str(ok), "--out", "x.json"],
                ["--compare", "a.json", "b.json", "--limit", "3"],
                ["--compare", "a.json", "b.json", "--audit-fmt32", "3"],
                ["--rescore", str(ok), "--out", "x.json", "--audit-fmt32", "3"]):
        with pytest.raises(SystemExit):
            script.parse_args(bad)
    a = script.parse_args(["--rescore", str(ok), "--out", "x.json", "--only-ifeval"])
    assert a.only_ifeval and a.rescore == str(ok)
    a = script.parse_args(["--tiny", "--out", "x.json", "--skip-ifeval-scoring"])
    assert a.skip_ifeval_scoring and not a.only_ifeval


def test_rescore_refuses_failed_run(tmp_path):
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"ok": False, "error": "RuntimeError('boom')"}), encoding="utf-8")
    out = tmp_path / "out.json"
    proc = _run_script(["--rescore", str(failed), "--out", str(out)], timeout=300)
    assert proc.returncode == 2
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False


def test_real_mode_checks_punkt_before_loading_model(tmp_path):
    out = tmp_path / "real.json"
    proc = _run_script(["--model", "qwen3.5-0.8b", "--device", "cpu", "--dtype", "fp32", "--out", str(out)], timeout=300)
    assert proc.returncode == 2
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["ok"] is False and "PunktDataError" in rep["error"]
    assert "model" not in rep and "records" not in rep  # 在加载模型之前就停了


# ---------------------------------------------------------------------------
# 兼容 v1 的输出（RUN-0009，入库的 8 个基座）：能读、能重评、能比较
# ---------------------------------------------------------------------------

needs_v1 = pytest.mark.skipif(not V1_FILES, reason="RUN-0009 的输出不在仓库里")
SUMMARY_KEYS = ("scores", "headline", "unscored", "parse_tally", "diagnostics")


@needs_v1
def test_v1_outputs_rescore_byte_identically(items):
    """当前代码按 v1 的布局重评 RUN-0009 的 8 份输出：MMLU-Pro 与 GSM8K 的逐题记录、以及全部汇总，与文件里的逐字节相同
    （IFEval 的重评要钉住的 punkt_tab，开发机上没有，这里原样保留；它的规则没有变）。"""
    assert len(V1_FILES) == 8
    for path in V1_FILES:
        rep = json.loads(path.read_text(encoding="utf-8"))
        assert rep["protocol_version"] == 1 and list(rep["records"]) == list(P.BENCHMARKS), path.name
        assert rep["generation"]["max_new_tokens"]["mmlu_pro"] == P.READINGS_BY_NAME[FMT32].max_new_tokens == 32
        recs = rescore_records(rep["records"], items, benchmarks=("mmlu_pro", "gsm8k"), protocol_version=1)
        assert _dump(recs) == _dump(rep["records"]), path.name
        assert _dump(summarize(recs, protocol_version=1)) == _dump({k: rep[k] for k in SUMMARY_KEYS}), path.name
    with pytest.raises(ValueError, match="v1"):  # 不按版本读就报错：v2 的读数放不进 v1 的布局
        summarize({**recs, FMT32: []}, protocol_version=1)


@needs_v1
def test_v1_rescore_cli_keeps_v1(tmp_path):
    """--rescore 一份 v1 输出：照常完成，输出仍是 v1 的布局，MMLU-Pro 与 GSM8K 的部分与输入逐字节相同。
    （真模型的输出重评 IFEval 要钉住的 punkt_tab；开发机没有，这里把 mode 改成 tiny 只为走通命令行路径。）"""
    rep = json.loads(V1_FILES[0].read_text(encoding="utf-8"))
    src = tmp_path / "v1.json"
    src.write_text(json.dumps({**rep, "mode": "tiny"}), encoding="utf-8")
    out = tmp_path / "v1_rescored.json"
    proc = _run_script(["--rescore", str(src), "--out", str(out)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    new = json.loads(out.read_text(encoding="utf-8"))
    assert new["protocol_version"] == 1 and list(new["records"]) == list(P.BENCHMARKS)
    for bench in ("mmlu_pro", "gsm8k"):
        assert _dump(new["records"][bench]) == _dump(rep["records"][bench])
        assert new["scores"][bench] == rep["scores"][bench] and new["headline"][bench] == rep["headline"][bench]
    assert new["parse_tally"] == rep["parse_tally"] and new["scoring_history"][-1] == rep["scoring"]


def _v3_report(items, key="fake-v3"):
    """假适配器在全部 200 题上跑出的 v3 输出（与 RUN-0009、RUN-0011 同一题目文件）。"""
    state = run_items(FakeAdapter(canned_responder(items), letter=canned_letter(items)), items)
    return {"protocol_version": 3, "items_file": {"sha256": P.ITEMS_SHA256}, "limit": None, "model": {"key": key},
            "records": state["records"], **summarize(state["records"])}


@needs_v1
def test_compare_v1_with_v3_only_matches_same_reading(items):
    """v1 对 v3：MMLU-Pro 只比 v1 的 32 对 fmt32（诊断，不标红）；受限字母读数与 256 的自由作答写明"口径不同、标题不可比"。"""
    v1 = json.loads(V1_FILES[0].read_text(encoding="utf-8"))
    v3 = _v3_report(items)
    for base, merged, (vb, vm) in ((v1, v3, (1, 3)), (v3, v1, (3, 1))):
        res = compare(base, merged)
        assert res["protocol_versions"] == {"base": vb, "merged": vm}
        assert res["incomparable"] == [LETTER, FREE256] and len(res["notes"]) == 2
        assert all("口径不同、标题不可比" in n and "协议 v1" in n for n in res["notes"])
        head = next(r for r in res["rows"] if r["reading"] == LETTER)
        assert (head["comparable"], head["flag"], head["delta_pp"], head["ci95_pp"]) == (False, None, None, None)
        assert head["note"] == res["notes"][0] and head["headline"] is True
        v3_side, v1_side = ("merged_pct", "base_pct") if vm == 3 else ("base_pct", "merged_pct")
        assert head[v3_side] == v3["headline"]["mmlu_pro"] and head[v1_side] is None
        fmt = next(r for r in res["rows"] if r["reading"] == FMT32)
        assert fmt["comparable"] and fmt["diagnostic"] and fmt["flag"] is None and fmt["ci95_pp"] is not None
        v1_pct, v3_pct = v1["scores"]["mmlu_pro"]["accuracy"]["pct"], v3["scores"][FMT32]["accuracy"]["pct"]
        assert (fmt["base_pct"], fmt["merged_pct"]) == ((v1_pct, v3_pct) if vb == 1 else (v3_pct, v1_pct))
        assert LETTER not in res["flagged"] and FMT32 not in res["flagged"]
        assert [t["reading"] for t in res["truncation"]["rows"]] == [FMT32, "gsm8k", "ifeval"]


@needs_v1
def test_compare_cli_on_two_v1_outputs(tmp_path):
    """RUN-0009 的一对 v1 输出照旧能比：MMLU-Pro 的 32 读数按 fmt32 比、不标红；256 的标题两边都没有，写明不可比。"""
    a, b = V1_FILES[0], V1_FILES[1]
    out = tmp_path / "cmp_v1.json"
    proc = _run_script(["--compare", str(a), str(b), "--out", str(out)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(out.read_text(encoding="utf-8"))
    ra, rb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    assert res["protocol_versions"] == {"base": 1, "merged": 1} and res["incomparable"] == [LETTER, FREE256]
    fmt = next(r for r in res["rows"] if r["reading"] == FMT32)
    assert (fmt["base_pct"], fmt["merged_pct"]) == (ra["scores"]["mmlu_pro"]["accuracy"]["pct"],
                                                     rb["scores"]["mmlu_pro"]["accuracy"]["pct"])
    assert fmt["flag"] is None and fmt["n"] == 100
    gs = next(r for r in res["rows"] if r["reading"] == "gsm8k" and r["metric"] == "flexible")
    assert (gs["base_pct"], gs["merged_pct"]) == (ra["headline"]["gsm8k"], rb["headline"]["gsm8k"])
    trunc = {t["reading"]: t for t in res["truncation"]["rows"]}
    assert trunc[FMT32]["base_n_reached"] == ra["diagnostics"]["mmlu_pro"]["n_reached_max_new_tokens"]
    assert "口径不同、标题不可比" in proc.stdout


# ---------------------------------------------------------------------------
# 兼容 v2 的输出（RUN-0011，入库的 8 个基座）：能读、能重评（逐字节相同）、能比较
# ---------------------------------------------------------------------------

needs_v2 = pytest.mark.skipif(not V2_FILES, reason="RUN-0011 的输出不在仓库里")


@needs_v2
def test_v2_outputs_rescore_byte_identically(items):
    """当前代码按 v2 的布局重评 RUN-0011 的 8 份输出：MMLU-Pro 的两个读数与 GSM8K 的逐题记录、以及全部汇总，与文件里的
    逐字节相同（IFEval 的重评要钉住的 punkt_tab，开发机上没有，这里原样保留；它的规则没有变）。"""
    assert len(V2_FILES) == 8
    for path in V2_FILES:
        rep = json.loads(path.read_text(encoding="utf-8"))
        assert rep["protocol_version"] == 2 and list(rep["records"]) == ["mmlu_pro", FMT32, "gsm8k", "ifeval"], path.name
        recs = rescore_records(rep["records"], items, benchmarks=("mmlu_pro", "gsm8k"), protocol_version=2)
        assert _dump(recs) == _dump(rep["records"]), path.name
        assert _dump(summarize(recs, protocol_version=2)) == _dump({k: rep[k] for k in SUMMARY_KEYS}), path.name


@needs_v2
def test_v2_rescore_cli_keeps_v2(tmp_path):
    rep = json.loads(V2_FILES[0].read_text(encoding="utf-8"))
    src = tmp_path / "v2.json"
    src.write_text(json.dumps({**rep, "mode": "tiny"}), encoding="utf-8")  # 只为走通命令行路径（见 v1 的同名用例）
    out = tmp_path / "v2_rescored.json"
    proc = _run_script(["--rescore", str(src), "--out", str(out)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    new = json.loads(out.read_text(encoding="utf-8"))
    assert new["protocol_version"] == 2 and list(new["records"]) == ["mmlu_pro", FMT32, "gsm8k", "ifeval"]
    for key in ("mmlu_pro", FMT32, "gsm8k"):
        assert _dump(new["records"][key]) == _dump(rep["records"][key])
        assert new["scores"][key] == rep["scores"][key]
    assert new["headline"]["mmlu_pro"] == rep["headline"]["mmlu_pro"] and new["parse_tally"] == rep["parse_tally"]
    assert "mmlu_pro" in proc.stdout and "mmlu_pro_letter" not in proc.stdout


@needs_v2
def test_compare_cli_on_two_v2_outputs(tmp_path):
    """RUN-0011 的一对 v2 输出：受限字母读数两边都没有，写明不可比；256 的自由作答（v3 的 free256）与 fmt32 照比，不标红。"""
    a, b = V2_FILES[0], V2_FILES[1]
    out = tmp_path / "cmp_v2.json"
    proc = _run_script(["--compare", str(a), str(b), "--out", str(out)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(out.read_text(encoding="utf-8"))
    ra, rb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    assert res["protocol_versions"] == {"base": 2, "merged": 2} and res["incomparable"] == [LETTER]
    free = next(r for r in res["rows"] if r["reading"] == FREE256)
    assert (free["base_pct"], free["merged_pct"]) == (ra["headline"]["mmlu_pro"], rb["headline"]["mmlu_pro"])
    assert free["diagnostic"] and free["flag"] is None and free["n"] == 100
    assert LETTER not in res["flagged"] and FREE256 not in res["flagged"] and FMT32 not in res["flagged"]
    gs = next(r for r in res["rows"] if r["reading"] == "gsm8k" and r["metric"] == "flexible")
    assert (gs["base_pct"], gs["merged_pct"]) == (ra["headline"]["gsm8k"], rb["headline"]["gsm8k"])
    trunc = {t["reading"]: t for t in res["truncation"]["rows"]}
    assert trunc[FREE256]["base_n_reached"] == ra["diagnostics"]["mmlu_pro"]["n_reached_max_new_tokens"]
    assert "口径不同、标题不可比" in proc.stdout


@needs_v2
def test_compare_v2_with_v3(items):
    """v2 对 v3：v2 的 "mmlu_pro"（256）对上 free256（诊断），fmt32 照比；受限字母读数只在 v3 一边：不可比、不标红。"""
    v2 = json.loads(V2_FILES[0].read_text(encoding="utf-8"))
    v3 = _v3_report(items)
    for base, merged, (vb, vm) in ((v2, v3, (2, 3)), (v3, v2, (3, 2))):
        res = compare(base, merged)
        assert res["protocol_versions"] == {"base": vb, "merged": vm} and res["incomparable"] == [LETTER]
        head = next(r for r in res["rows"] if r["reading"] == LETTER and r["metric"] == "accuracy")
        v3_side = "merged_pct" if vm == 3 else "base_pct"
        assert head["comparable"] is False and head[v3_side] == v3["headline"]["mmlu_pro"]
        free = next(r for r in res["rows"] if r["reading"] == FREE256)
        v2_pct, v3_pct = v2["scores"]["mmlu_pro"]["accuracy"]["pct"], v3["scores"][FREE256]["accuracy"]["pct"]
        assert (free["base_pct"], free["merged_pct"]) == ((v2_pct, v3_pct) if vb == 2 else (v3_pct, v2_pct))
        assert free["comparable"] and free["flag"] is None
        assert [t["reading"] for t in res["truncation"]["rows"]] == [FREE256, FMT32, "gsm8k", "ifeval"]


def test_letter_reading_checks_spellings_against_warmup(items):
    """每题各字母用到的写法必须与预热时查到的（输出里的 option_spellings）相同。"""
    ad = FakeAdapter(lambda p: "A")
    state = run_items(ad, items, limit=1, score_ifeval=False)
    table = copy.deepcopy(state["option_spellings"])
    it = items["items"]["mmlu_pro"][0]
    run_one(ad, "mmlu_pro", it, spellings=table)  # 相同：通过
    table["A"]["spellings"][1]["token_id"] += 1
    with pytest.raises(LetterReadingError, match="写法"):
        run_one(ad, "mmlu_pro", it, spellings=table)
