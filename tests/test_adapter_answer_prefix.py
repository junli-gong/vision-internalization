"""DEV-0033：作答开头预填（generate / generate_batch 的 answer_prefix；D66-2 的格式强制诊断用）。三家族，CPU 小模型 fp32。

- 不给预填（不传，或 answer_prefix=None）：送进 model.generate 的 input_ids、attention_mask、mm_token_type_ids 与以前逐 token 相同——
  以前的构造独立重算（Qwen：user 轮 + assistant_header_ids；Gemma、InternVL：prompt_ids），结果逐字段相同。
- 给了预填：每行提示 = 原提示 + 预填的 token（mm_token_type_ids 记 0），n_prompt_tokens 多出预填的长度；等价于"预填的 token 是模型已经
  生成的前几个 token"：用 logits 处理器逼模型先 greedy 写出预填的 token、再续写，续写的 token 与 answer_prefix 的结果逐 token 相同，
  text、raw_text（split_answer 作用在 预填 + 生成 上）、stopped 也相同；批量（有左填充）与逐条都核对。
- 规则：只用于关思考；空串、解码回不来、含停止 token 的预填报错；GenerationResult.answer_prefix_ids 的校验。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import LogitsProcessor

from src.adapters import batching, registry
from src.adapters.base import GenerationResult
from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy

FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=True, max_patches=4),
}
INIT_STD = 0.05  # 同 test_adapter_batching：重取参数，greedy 输出才随输入变化
PREFIXES = ("[[", "[[120, 3")  # 一个短的（E1 多边形用的就是它）、一个多 token 的
N_NEW = 12


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
def inputs(ad):
    """长短不一的三条（批里有左填充），两条有图。"""
    rng = np.random.default_rng(20261033)
    land, port = _img(rng, 100, 150), _img(rng, 150, 100)
    items = [([land], "Outline the cat with a polygon. Output JSON.", "image_first"),
             ([port], "Segment the dog in the image as a 16x16 binary grid. Output only the grid, nothing else please.",
              "image_first"),
             ([], "Say hi.", "instruction_first")]
    return [ad.build_inputs(ims, ins, order) for ims, ins, order in items]


def _old_prompt_ids(ad, inp) -> list[int]:
    """预填之前的构造（独立重算，不经 _batch_kw）：Qwen = user 轮 + assistant 头；Gemma、InternVL = prompt_ids。"""
    if ad.family == registry.FAMILY_QWEN35:
        return inp.tensors["input_ids"][0].tolist() + list(ad.assistant_header_ids(False))
    if ad.family == registry.FAMILY_GEMMA4:
        return ad.prompt_ids(inp, enable_thinking=False)[0].tolist()
    return list(ad.prompt_ids(inp, False))


@pytest.fixture()
def capture(monkeypatch):
    """记下每次 batching.run_generate 收到的 kw（送进 model.generate 的张量），照常生成。"""
    seen = []
    real = batching.run_generate

    def spy(model, kw, **rest):
        seen.append({k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in kw.items()})
        return real(model, kw, **rest)

    monkeypatch.setattr(batching, "run_generate", spy)
    return seen


def _rows(kw):
    """左填充的批 → 每行非填充部分的 input_ids（与 mm_token_type_ids，没有时为 None）。"""
    out = []
    for b in range(kw["input_ids"].shape[0]):
        m = kw["attention_mask"][b].bool()
        mm = kw["mm_token_type_ids"][b][m].tolist() if "mm_token_type_ids" in kw else None
        out.append((kw["input_ids"][b][m].tolist(), mm))
    return out


def test_no_prefix_inputs_unchanged(ad, inputs, capture):
    plain = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=4)
    none = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=4, answer_prefix=None)
    single = ad.generate(inputs[0], enable_thinking=False, max_new_tokens=4, answer_prefix=None)
    k_plain, k_none, k_single = capture
    assert set(k_plain) == set(k_none)
    for k in k_plain:
        assert torch.equal(k_plain[k], k_none[k]) if torch.is_tensor(k_plain[k]) else k_plain[k] == k_none[k], k
    for inp, (ids, mm) in zip(inputs, _rows(k_plain)):
        assert ids == _old_prompt_ids(ad, inp)
        if mm is not None and ad.family == registry.FAMILY_QWEN35:
            assert mm == inp.tensors["mm_token_type_ids"][0].tolist() + [0] * len(ad.assistant_header_ids(False))
    assert _rows(k_single)[0][0] == _old_prompt_ids(ad, inputs[0])
    for a, b in zip(plain, none):
        assert a.token_ids == b.token_ids and a.text == b.text and a.raw_text == b.raw_text
        assert a.answer_prefix_ids == b.answer_prefix_ids == ()
    assert single.answer_prefix_ids == ()


class _ForcePrefix(LogitsProcessor):
    """前 len(ids) 步只允许 ids[t]（逼 greedy 先写出预填的 token），之后不限制。"""

    def __init__(self, ids):
        self.ids, self.start = list(ids), None

    def __call__(self, input_ids, scores):
        if self.start is None:
            self.start = input_ids.shape[1]
        t = input_ids.shape[1] - self.start
        if t >= len(self.ids):
            return scores
        keep = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        keep[self.ids[t]] = True
        return scores.masked_fill(~keep, float("-inf"))


@pytest.mark.parametrize("prefix", PREFIXES)
def test_prefix_equals_greedy_continuation_after_forced_prefix(ad, inputs, capture, prefix):
    pids = list(ad._answer_prefix_ids(prefix, False))
    assert pids and ad.tokenizer.decode(pids) == prefix
    m = len(pids)
    forced = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=m + N_NEW, logits_processor=[_ForcePrefix(pids)])
    pre = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=N_NEW, answer_prefix=prefix)
    k_forced, k_pre = capture
    for (ids0, mm0), (ids1, mm1) in zip(_rows(k_forced), _rows(k_pre)):  # 输入 = 原提示 + 预填（预填记文本）
        assert ids1 == ids0 + pids
        if mm0 is not None:
            assert mm1 == mm0 + [0] * m
    assert len({len(r[0]) for r in _rows(k_pre)}) > 1  # 批里确有左填充
    assert len({f.token_ids[m:m + 4] for f in forced}) >= 2, "小模型的续写与输入无关，逐 token 核对没有意义"
    for f, p in zip(forced, pre):
        assert list(f.token_ids[:m]) == pids
        assert p.token_ids == f.token_ids[m:]
        assert (p.text, p.raw_text, p.stopped, p.thinking_unclosed) == (f.text, f.raw_text, f.stopped, f.thinking_unclosed)
        assert p.raw_text.startswith(prefix)
        assert p.n_prompt_tokens == f.n_prompt_tokens + m and p.n_generated_tokens == f.n_generated_tokens - m
        assert p.answer_prefix_ids == tuple(pids)
    single = [ad.generate(i, enable_thinking=False, max_new_tokens=N_NEW, answer_prefix=prefix) for i in inputs]
    for s, p in zip(single, pre):  # 逐条与批量逐 token 相同（左填充不影响预填之后的续写）
        assert (s.token_ids, s.text, s.raw_text, s.stopped) == (p.token_ids, p.text, p.raw_text, p.stopped)


def test_prefix_rules(ad, inputs):
    with pytest.raises(ValueError, match="关思考"):
        ad.generate_batch(inputs[:1], enable_thinking=True, max_new_tokens=2, answer_prefix="[[")
    with pytest.raises(ValueError, match="非空字符串"):
        ad.generate_batch(inputs[:1], enable_thinking=False, max_new_tokens=2, answer_prefix="")
    stop = ad.tokenizer.decode([ad.stop_token_ids()[0]])
    with pytest.raises(ValueError, match="停止 token"):
        ad.generate_batch(inputs[:1], enable_thinking=False, max_new_tokens=2, answer_prefix="x" + stop)


def test_encode_answer_prefix_unit():
    enc = {"ab": [1, 2], "z": [9]}.__getitem__
    dec = {(1, 2): "ab", (9,): "z"}

    def decode(ids):
        return dec[tuple(ids)]

    kw = dict(encode=enc, decode=decode, forbidden_ids=[9])
    assert batching.encode_answer_prefix(None, enable_thinking=True, **kw) == ()
    assert batching.encode_answer_prefix("ab", enable_thinking=False, **kw) == (1, 2)
    with pytest.raises(ValueError):
        batching.encode_answer_prefix("z", enable_thinking=False, **kw)  # 停止 token
    with pytest.raises(ValueError):
        batching.encode_answer_prefix("ab", enable_thinking=False, encode=enc, decode=lambda ids: "a b", forbidden_ids=[])
    with pytest.raises(TypeError):
        batching.encode_answer_prefix("ab", enable_thinking=None, **kw)
    row = batching.PromptRow(ids=(5, 6), n_visual_tokens=0, n_image_pixels=0, mm_token_type_ids=(1, 0))
    assert batching.with_answer_prefix(row, ()) is row
    got = batching.with_answer_prefix(row, (1, 2))
    assert got.ids == (5, 6, 1, 2) and got.mm_token_type_ids == (1, 0, 0, 0)


def test_generation_result_answer_prefix_ids_field():
    base = dict(text="x", token_ids=(1,), n_generated_tokens=1, n_prompt_tokens=3, n_visual_tokens=0, wall_time_s=0.0)
    assert GenerationResult(**base).answer_prefix_ids == ()
    assert GenerationResult(**base, answer_prefix_ids=(4, 5)).answer_prefix_ids == (4, 5)
    for bad in ([4, 5], (True,), (-1,)):
        with pytest.raises(ValueError):
            GenerationResult(**base, answer_prefix_ids=bad)
