"""DEV-0016：三家族的 option_logits（受限选项读数，D36-4、D36-6）。CPU 小模型。

- 写法：A–J 的 "X" 与 " X" 在三家族的真实 tokenizer 里都是单个 token，id 钉在 SPELLING_IDS（真实模型的 tokenizer 与这里的
  固定 revision 小文件相同；健康检查的输出另记下每次运行实际用的表）。
- 批量与逐条：choice 与所用 token 逐个相同，logits 与 probs 在 fp32 下只差填充带来的舍入（实测 ≤ 2e-6，容差 1e-5）。
- 与 HF 的 logits 处理器路径对得上：generate 第一步送进处理器的 scores 在各写法上与 option_logits 的 logit 逐位相同，
  argmax 与 generate(allowed_token_ids=全部写法, max_new_tokens=1) 生成的 token 相同。
- 第一个生成位置：不经 generate、直接对完整提示做一次前向，最后一个位置的 logits 与 option_logits 相同（倒数第二个位置明显不同，
  错一位能被测出）；Qwen 关思考时提示以空思考块 `<think>\\n\\n</think>\\n\\n` 结尾，开思考时以 `<think>\\n` 结尾。
- bf16：logits 是 lm_head 的 bf16 输出转成的 fp32（每个值都能用 bf16 精确表示，与直接前向逐位相同），softmax 在 fp32 下做。
- 规则：两种写法取较大者、平局取 id 小的 token（与 torch.argmax 相同）；写法查找、入参检查、入口断言。
"""

from __future__ import annotations

import contextlib
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import batching, registry
from src.adapters.base import OPTION_SPELLING_TEMPLATES, ModelAdapter, OptionScores, OptionSpelling
from src.adapters.batching import OptionLogitsError, OptionSpellingError
from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy
from src.adapters.qwen35 import AdapterDeviceError

FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=True, max_patches=4),
}
INIT_STD = 0.05  # 同 test_adapter_batching：HF 默认初始化下小模型的输出几乎与输入无关
LETTERS = "ABCDEFGHIJ"
# (不带空格, 带一个前导空格) 的 token id，A..J。固定 revision 的 tokenizer（tests 取 config/tokenizer 小文件）
SPELLING_IDS = {
    "qwen3.5-0.8b": [(32, 357), (33, 417), (34, 351), (35, 414), (36, 458), (37, 426), (38, 469), (39, 462), (40, 353),
                     (41, 604)],
    "gemma-4-e2b-it": [(236776, 562), (236799, 603), (236780, 565), (236796, 622), (236788, 645), (236811, 633),
                       (236823, 667), (236814, 640), (236777, 564), (236863, 730)],
    "internvl3.5-2b-hf": [(32, 362), (33, 425), (34, 356), (35, 422), (36, 468), (37, 434), (38, 479), (39, 472),
                          (40, 358), (41, 619)],
}
BATCH_ATOL = 1e-5


@pytest.fixture(scope="module", params=FAMILY_KEYS)
def ad(request):
    key = request.param
    try:
        return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True,
                            init_std=INIT_STD)
    except OSError as e:
        pytest.fail(f"取不到 {key} 的 config/tokenizer/处理器小文件：{e!r}（需要联网下载一次）", pytrace=False)


def _img(rng, h, w):
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[h // 4: h // 2, w // 4: w // 2, 1] = 230
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def samples():
    """(图, 指令, 顺序, 选项)：提示长短不一（左填充的长度各不相同），图横、竖、两张、没有，选项个数各不相同。"""
    rng = np.random.default_rng(20261016)
    land, port, wide = _img(rng, 100, 150), _img(rng, 150, 100), _img(rng, 64, 256)
    return [
        ([land], "Is the cat on the left? A. yes B. no", "instruction_first", list("AB")),
        ([], "Which is a prime? A. 4 B. 6 C. 7 D. 9\nAnswer with the letter only.", "instruction_first", list("ABCD")),
        ([port, wide], "Which image is brighter, the first or the second one?", "image_first", list("AB")),
        ([], "Pick one.", "instruction_first", list(LETTERS)),
        ([wide], "Count the objects. A. 1 B. 2 C. 3", "instruction_first", list("ABC")),
        ([], "x", "instruction_first", list("ABCDEFG")),
    ]


@pytest.fixture(scope="module")
def inputs(ad, samples):
    return [ad.build_inputs(ims, ins, order) for ims, ins, order, _ in samples]


@pytest.fixture(scope="module")
def options(samples):
    return [opts for *_, opts in samples]


def _all_ids(s: OptionScores) -> list[int]:
    return [sp.token_id for x in s.spellings for sp in x]


# ---------------------------------------------------------------------------
# 写法
# ---------------------------------------------------------------------------


def test_letter_spellings_are_single_tokens_with_pinned_ids(ad):
    key = ad.model_key.removeprefix("tiny(").removesuffix(")")
    vocab_rows = ad.model.get_output_embeddings().weight.shape[0]
    assert OPTION_SPELLING_TEMPLATES == ("{}", " {}")
    for letter, (bare, spaced) in zip(LETTERS, SPELLING_IDS[key]):
        usable, skipped = batching.option_spellings(ad.tokenizer, letter, vocab_rows=vocab_rows)
        assert usable == (OptionSpelling(letter, bare), OptionSpelling(" " + letter, spaced)) and skipped == ()
        assert ad.tokenizer.decode([bare]) == letter and ad.tokenizer.decode([spaced]) == " " + letter


# ---------------------------------------------------------------------------
# 批量与逐条；与 HF 的 logits 处理器路径、generate(allowed_token_ids) 一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thinking", [False, True])
def test_batch_matches_single(ad, inputs, options, thinking):
    batch = ad.option_logits(inputs, options, enable_thinking=thinking)
    single = [ad.option_logits([i], [o], enable_thinking=thinking)[0] for i, o in zip(inputs, options)]
    assert len({s.n_prompt_tokens for s in single}) >= 5  # 前提：有填充，且各行填充长度不同
    for inp, opts, s, b in zip(inputs, options, single, batch):
        assert (b.options, b.choice, b.token_ids, b.spellings, b.skipped_spellings) == (
            s.options, s.choice, s.token_ids, s.spellings, s.skipped_spellings) and b.options == tuple(opts)
        np.testing.assert_allclose(b.logits, s.logits, rtol=0, atol=BATCH_ATOL)
        np.testing.assert_allclose(b.probs, s.probs, rtol=0, atol=BATCH_ATOL)
        assert (b.n_prompt_tokens, b.n_visual_tokens, b.n_image_pixels) == (s.n_prompt_tokens, s.n_visual_tokens,
                                                                            s.n_image_pixels)
        assert b.n_visual_tokens == sum(len(i.span) for i in inp.images)
        assert b.batch_size == len(inputs) and s.batch_size == 1 and b.n_generated_tokens == 1
        assert math.isclose(b.wall_time_s * len(inputs), b.batch_wall_time_s, rel_tol=1e-9)
    # 提示长度与 generate_batch 记的相同（共用同一段拼批）
    gen = ad.generate_batch(inputs, enable_thinking=thinking, max_new_tokens=1)
    assert [g.n_prompt_tokens for g in gen] == [b.n_prompt_tokens for b in batch]


class _Recorder:
    """logits 处理器：记下每一步送进来的 scores，原样返回。"""

    def __init__(self):
        self.scores = []

    def __call__(self, input_ids, scores):
        self.scores.append(scores.detach().clone())
        return scores


@pytest.mark.parametrize("thinking", [False, True])
def test_matches_hf_processor_path_and_generate_allowed_token_ids(ad, inputs, options, thinking):
    """同一位置的数值与 HF 的 logits 处理器路径逐位相同；argmax 与 generate(allowed_token_ids=全部写法, max_new_tokens=1) 相同。"""
    batch = ad.option_logits(inputs, options, enable_thinking=thinking)
    for inp, s in zip(inputs, batch):
        single = ad.option_logits([inp], [s.options], enable_thinking=thinking)[0]
        rec = _Recorder()
        ad.generate(inp, enable_thinking=thinking, max_new_tokens=1, logits_processor=[rec])
        assert len(rec.scores) == 1 and rec.scores[0].dtype == torch.float32
        row = rec.scores[0][0]
        for sp, lg in zip(single.spellings, single.spelling_logits):
            assert [float(row[x.token_id]) for x in sp] == list(lg)  # 逐位相同
        g = ad.generate(inp, enable_thinking=thinking, max_new_tokens=1, allowed_token_ids=_all_ids(single))
        assert g.token_ids == (single.choice_token_id,) == (s.choice_token_id,)
        owner = next(o for o, sp in zip(single.options, single.spellings) if g.token_ids[0] in [x.token_id for x in sp])
        assert owner == single.choice == s.choice
    # 不是平凡情形：各行选中的字母不全相同；带空格与不带空格的写法都有胜出的时候（取最大值的规则真的起作用）
    assert len({s.choice for s in batch}) >= 2
    wins = [t == sp[k].token_id for s in batch for t, sp in zip(s.token_ids, s.spellings) for k in (0, 1)]
    assert any(wins[0::2]) and any(wins[1::2])


# ---------------------------------------------------------------------------
# 第一个生成位置
# ---------------------------------------------------------------------------

HEADER_TAIL = {  # 提示最后几个 token（第一个生成位置就在它们之后）
    (registry.FAMILY_QWEN35, False): ["<|im_start|>", "assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ"],
    (registry.FAMILY_QWEN35, True): ["<|im_start|>", "assistant", "Ċ", "<think>", "Ċ"],
    (registry.FAMILY_GEMMA4, False): ["<turn|>", "\n", "<|turn>", "model", "\n"],
    (registry.FAMILY_GEMMA4, True): ["<turn|>", "\n", "<|turn>", "model", "\n"],
    (registry.FAMILY_INTERNVL35, False): ["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ"],
    (registry.FAMILY_INTERNVL35, True): ["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ"],
}


def _direct_last_logits(ad, inp, thinking):
    """不经 generate：对 generate 会送进模型的完整提示（同一个 _prompt_row）做一次不带缓存的前向，返回最后两个位置的 logits。"""
    row = ad._prompt_row(inp, thinking)
    ids = torch.tensor([row.ids])
    kw = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if row.mm_token_type_ids is not None:
        kw["mm_token_type_ids"] = torch.tensor([row.mm_token_type_ids])
    scope = ad.gdn_scope() if hasattr(ad, "gdn_scope") else contextlib.nullcontext()
    with scope, torch.no_grad():
        logits = ad.model(**kw, use_cache=False).logits[0]
    return row, logits[-1], logits[-2]


@pytest.mark.parametrize("thinking", [False, True])
def test_first_generated_position(ad, thinking):
    """第一个生成位置 = 提示（含 assistant 头；Qwen 关思考时含空思考块）最后一个 token 之后。纯文本、没有填充。"""
    inp = ad.build_inputs([], "Which letter comes first? A. b B. a C. c", "instruction_first")
    s = ad.option_logits([inp], [list("ABC")], enable_thinking=thinking)[0]
    row, last, prev = _direct_last_logits(ad, inp, thinking)
    tail = HEADER_TAIL[(ad.family, thinking)]
    assert ad.tokenizer.convert_ids_to_tokens(list(row.ids[-len(tail):])) == tail
    assert s.n_prompt_tokens == len(row.ids)
    mine = [v for lg in s.spelling_logits for v in lg]
    np.testing.assert_allclose([float(last[t]) for t in _all_ids(s)], mine, rtol=0, atol=1e-6)
    off_by_one = [float(prev[t]) for t in _all_ids(s)]
    assert max(abs(a - b) for a, b in zip(off_by_one, mine)) > 0.01  # 错一位的定义会被测出


def test_qwen_thinking_off_header_is_the_empty_think_block():
    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=PixelBudget(64), tiny=True,
                      init_std=INIT_STD)
    tok = ad.tokenizer
    assert tok.decode(list(ad.assistant_header_ids(False))) == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert tok.decode(list(ad.assistant_header_ids(True))) == "<|im_start|>assistant\n<think>\n"


# ---------------------------------------------------------------------------
# 精度：bf16 模型的 logits 取 lm_head 输出转成的 fp32，softmax 在 fp32 下做
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_bf16_logits_are_fp32_of_the_bf16_lm_head_output(key):
    ad = make_adapter(key, device="cpu", dtype=torch.bfloat16, image_policy=POLICIES[key], tiny=True, init_std=INIT_STD)
    inp = ad.build_inputs([], "Pick A or B. A. left B. right", "instruction_first")
    s = ad.option_logits([inp], [list("ABCD")], enable_thinking=False)[0]
    _, last, _ = _direct_last_logits(ad, inp, False)
    assert last.dtype == torch.bfloat16  # lm_head 输出是 bf16
    mine = [v for lg in s.spelling_logits for v in lg]
    assert [float(last[t].float()) for t in _all_ids(s)] == mine  # 就是 bf16 输出转成 fp32，逐位相同
    assert all(float(torch.tensor(v).bfloat16().float()) == v for v in mine)
    want = torch.softmax(torch.tensor(s.logits, dtype=torch.float32), dim=0).tolist()
    assert list(s.probs) == want  # fp32 下的 softmax
    assert math.isclose(sum(s.probs), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 规则：取最大值、平局、写法查找（纯函数，不需要模型）
# ---------------------------------------------------------------------------


def _sp(*pairs):
    return tuple(OptionSpelling(t, i) for t, i in pairs)


def test_score_options_takes_the_larger_spelling_and_breaks_ties_like_argmax():
    logits = torch.zeros(20, dtype=torch.float32)
    spellings = [_sp(("A", 1), (" A", 11)), _sp(("B", 2), (" B", 12)), _sp(("C", 3), (" C", 13))]
    logits[1], logits[11] = 1.0, 3.0  # A：带空格的更大
    logits[2], logits[12] = 2.5, 0.0  # B：不带空格的更大
    logits[3], logits[13] = -1.0, -1.0  # C：两种写法相等 → 取 id 小的
    s = batching.score_options(logits, spellings)
    assert s["logits"] == (3.0, 2.5, -1.0) and s["token_ids"] == (11, 2, 3) and s["top"] == 0
    assert s["spelling_logits"] == ((1.0, 3.0), (2.5, 0.0), (-1.0, -1.0))
    want = torch.softmax(torch.tensor([3.0, 2.5, -1.0]), dim=0).tolist()
    assert list(s["probs"]) == want
    # 选项之间平局：取所用 token id 较小的选项（= greedy 在允许集合上的 torch.argmax）
    logits[11], logits[2] = 2.0, 2.0
    logits[12] = 2.0  # B 的两种写法也都是 2.0 → B 用 id 2
    s = batching.score_options(logits, spellings)
    assert s["logits"][:2] == (2.0, 2.0) and s["token_ids"][:2] == (11, 2) and s["top"] == 1
    masked = torch.full_like(logits, float("-inf"))
    allowed = [x.token_id for sp in spellings for x in sp]
    masked[allowed] = logits[allowed]
    assert int(torch.argmax(masked)) == 2
    with pytest.raises(OptionSpellingError):  # 两个选项共用一个 token
        batching.score_options(logits, [_sp(("A", 1)), _sp(("B", 1))])
    with pytest.raises(OptionLogitsError):
        batching.score_options(logits.double(), spellings)


class _FakeTok:
    """encode 按表给出 id；decode 按表解码，表里没有的解码成 "?"。"""

    def __init__(self, enc, dec):
        self.enc, self.dec = enc, dec

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self.enc[text])

    def decode(self, ids):
        return self.dec.get(tuple(ids), "?")


def test_option_spellings_rules():
    enc = {"A": [65], " A": [1065], "Q": [81], " Q": [7, 81], "B": [66], " B": [1066], "Z": [8, 90], " Z": [7, 90],
           "W": [10 ** 6], " W": [1087]}
    dec = {(65,): "A", (1065,): " A", (81,): "Q", (66,): "b", (1066,): " B", (10 ** 6,): "W", (1087,): " W"}
    tok = _FakeTok(enc, dec)
    usable, skipped = batching.option_spellings(tok, "A", vocab_rows=5000)
    assert usable == (OptionSpelling("A", 65), OptionSpelling(" A", 1065)) and skipped == ()
    usable, skipped = batching.option_spellings(tok, "Q", vocab_rows=5000)
    assert usable == (OptionSpelling("Q", 81),) and skipped == (" Q",)  # 不是单 token 的写法不用，记下来
    usable, skipped = batching.option_spellings(tok, "B", vocab_rows=5000)
    assert usable == (OptionSpelling(" B", 1066),) and skipped == ("B",)  # 解码回来不一样：不用
    with pytest.raises(OptionSpellingError, match="都不是单个 token"):
        batching.option_spellings(tok, "Z", vocab_rows=5000)
    with pytest.raises(OptionSpellingError, match="超出"):
        batching.option_spellings(tok, "W", vocab_rows=5000)


# ---------------------------------------------------------------------------
# 入参检查、入口断言、签名
# ---------------------------------------------------------------------------


def test_option_logits_validation(ad, inputs, options):
    with pytest.raises(TypeError):
        ad.option_logits(inputs, options)  # 思考开关必填
    with pytest.raises(TypeError):
        ad.option_logits(inputs, options, enable_thinking=None)
    with pytest.raises(TypeError):
        ad.option_logits(inputs[0], options[:1], enable_thinking=False)  # 单条要放进列表
    with pytest.raises(ValueError, match="行"):
        ad.option_logits(inputs, options[:-1], enable_thinking=False)
    with pytest.raises(TypeError, match="单个字符串"):
        ad.option_logits(inputs[:1], ["ABCD"], enable_thinking=False)
    with pytest.raises(TypeError, match="单个字符串"):
        ad.option_logits(inputs[:1], "AB", enable_thinking=False)
    for bad in (["A"], ["A", "A"], ["A", ""], ["A", " B"], ["A", 1]):
        with pytest.raises(ValueError):
            ad.option_logits(inputs[:1], [bad], enable_thinking=False)
    with pytest.raises(ValueError):
        ad.option_logits([], [], enable_thinking=False)


def test_option_logits_entry_assertion_checks_every_row(ad, inputs, options):
    """入口断言逐行生效（lessons：开发机用 meta 设备代理）：第 2 行的张量不在模型的设备上 → AdapterDeviceError。"""
    bad = inputs[0]
    moved = type(bad)(tensors={k: v.to("meta") if torch.is_tensor(v) else v for k, v in bad.tensors.items()},
                      order=bad.order, images=bad.images, instruction_span=bad.instruction_span)
    with pytest.raises(AdapterDeviceError, match="option_logits"):
        ad.option_logits([inputs[1], moved], options[:2], enable_thinking=False)


def test_query_inputs_are_rejected():
    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=PixelBudget(64), tiny=True)
    ad.inject_query_tokens(4, "grid2d")
    inp = ad.build_inputs([], "hi", "instruction_first")
    with pytest.raises(ValueError, match="option_logits"):
        ad.option_logits([inp], [list("AB")], enable_thinking=False)


class _StubModel:
    """first_step_logits 的入口断言用：generate 返回给定的 logits 与 sequences。"""

    def __init__(self, logits, sequences, rows=7):
        self.out = SimpleNamespace(logits=logits, sequences=sequences)
        self.rows = rows
        self.kwargs = None

    def generate(self, **kw):
        self.kwargs = kw
        return self.out


def test_first_step_logits_entry_assertions():
    ids = torch.tensor([[5, 6, 7]])
    kw = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    call = dict(device=torch.device("cpu"), stop_ids=[1], pad_id=0, vocab_rows=7)
    seq = torch.tensor([[5, 6, 7, 3]])
    good = _StubModel((torch.zeros(1, 7),), seq)
    logits, wall = batching.first_step_logits(good, kw, **call)
    assert logits.shape == (1, 7) and wall >= 0
    gk = good.kwargs  # 显式 greedy、一步、原始 logits、不带处理器
    assert (gk["max_new_tokens"], gk["do_sample"], gk["num_beams"], gk["output_logits"], gk["return_dict_in_generate"]) == (
        1, False, 1, True, True) and len(gk["logits_processor"]) == 0
    with pytest.raises(OptionLogitsError, match="bfloat16"):
        batching.first_step_logits(_StubModel((torch.zeros(1, 7, dtype=torch.bfloat16),), seq), kw, **call)
    with pytest.raises(OptionLogitsError, match="形状"):
        batching.first_step_logits(_StubModel((torch.zeros(1, 9),), seq), kw, **call)
    with pytest.raises(OptionLogitsError, match="1 步"):
        batching.first_step_logits(_StubModel((torch.zeros(1, 7), torch.zeros(1, 7)), seq), kw, **call)
    with pytest.raises(OptionLogitsError, match="前缀"):
        batching.first_step_logits(_StubModel((torch.zeros(1, 7),), torch.tensor([[5, 9, 7, 3]])), kw, **call)


def test_signature_rules():
    from src.adapters import gemma4, internvl35, qwen35

    for cls in (qwen35.Qwen35Adapter, gemma4.Gemma4Adapter, internvl35.InternVL35Adapter):
        assert cls.option_logits is not ModelAdapter.option_logits
        par = inspect.signature(cls.option_logits).parameters["enable_thinking"]
        assert par.kind is inspect.Parameter.KEYWORD_ONLY and par.default is inspect.Parameter.empty
    assert "option_logits" in ModelAdapter.__abstractmethods__
    with pytest.raises(TypeError, match="enable_thinking"):

        class _Bad(ModelAdapter):
            def option_logits(self, inputs_list, options, *, enable_thinking=False): ...


def test_option_scores_validation():
    sp = (_sp(("A", 1), (" A", 11)), _sp(("B", 2), (" B", 12)))
    ok = dict(options=("A", "B"), spellings=sp, skipped_spellings=((), ()), spelling_logits=((1.0, 2.0), (0.5, 0.0)),
              logits=(2.0, 0.5), token_ids=(11, 2), probs=(0.8175744761936437, 0.18242552380635635), choice="A",
              n_prompt_tokens=5, n_visual_tokens=0, n_image_pixels=0, wall_time_s=0.5, batch_size=2,
              batch_wall_time_s=1.0)
    s = OptionScores(**ok)
    assert (s.choice_token_id, s.prob("B"), s.n_generated_tokens) == (11, 0.18242552380635635, 1)
    for bad in ({"choice": "B"}, {"logits": (1.0, 0.5)}, {"token_ids": (1, 2)}, {"probs": (0.5, 0.6)},
                {"options": ("A", "A")}, {"wall_time_s": 0.4}, {"n_visual_tokens": 3},
                {"spellings": (_sp(("A", 1), (" A", 2)), _sp(("B", 2), (" B", 12)))}):
        with pytest.raises(ValueError):
            OptionScores(**{**ok, **bad})
