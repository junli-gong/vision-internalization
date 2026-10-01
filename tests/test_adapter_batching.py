"""DEV-0011：三家族的 generate_batch 与逐条 generate 的一致性（CPU 小模型，fp32）。

- 小模型经 factory.make_adapter(tiny=True, init_std=…) 构造：HF 默认初始化下小模型的 greedy 输出几乎与输入无关（Qwen 每步都是
  同一个换行 token），"逐个相同"会是空话；重取参数后输出随指令与图变化（用例里先核对这一点）。
- 8 条混合样本：指令长短不一（左填充的长度各不相同），图有横、竖、方、宽、两张、没有；Qwen 用像素预算（每张图网格不同），
  InternVL 用多 tile（每行 tile 数不同），Gemma 每张图按长宽比各有网格。
- 位置编码：forward-pre-hook 挂在语言模型上，记下 prefill 与每个解码步实际送进去的 position_ids，逐行与单独生成时比较
  （Qwen 的 M-RoPE 三维位置，另与本适配器自己的规则 mrope_rows 比较）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import LogitsProcessor

from src.adapters import qwen35, registry
from src.adapters.base import ModelAdapter
from src.adapters.factory import make_adapter
from src.adapters.image_policy import INTERNVL_NATIVE_TILES, PixelBudget, SoftTokenBudget, TilePolicy
from src.adapters.qwen35 import AdapterDeviceError

FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=True, max_patches=4),
}
INIT_STD = 0.05
MAX_NEW = 20  # ≥ 16 步（任务书）


def _stop_ids(ad):
    from src.adapters import gemma4, internvl35

    mod = {registry.FAMILY_QWEN35: qwen35, registry.FAMILY_GEMMA4: gemma4, registry.FAMILY_INTERNVL35: internvl35}
    return {ad.token_id(s) for s in mod[ad.family].STOP_TOKENS}


@pytest.fixture(scope="module", params=FAMILY_KEYS)
def ad(request):
    key = request.param
    try:
        return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True,
                            init_std=INIT_STD)
    except OSError as e:  # 取不到小文件时给出原因后失败，不跳过
        pytest.fail(f"取不到 {key} 的 config/tokenizer/处理器小文件：{e!r}（需要联网下载一次）", pytrace=False)


def _img(rng, h, w):
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[h // 4: h // 2, w // 4: w // 2, 1] = 230
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def samples():
    """8 条：(图, 指令, 顺序)。指令长短不一，图横、竖、方、宽、两张（网格不同）、没有。"""
    rng = np.random.default_rng(20261001)
    land, port, square, wide, tall = (_img(rng, 100, 150), _img(rng, 150, 100), _img(rng, 120, 120),
                                      _img(rng, 64, 256), _img(rng, 300, 200))
    return [
        ([land], "Describe the cat.", "instruction_first"),
        ([port], "What is in this picture? Please answer in detail, mentioning every colour you see.", "instruction_first"),
        ([square, wide], "Compare.", "instruction_first"),
        ([], "Say hi.", "instruction_first"),
        ([tall], "Where is the dog?", "image_first"),
        ([wide], "Count the objects in the image and output a number.", "instruction_first"),
        ([land], "x", "image_first"),
        ([port, land], "Which image is brighter, the first or the second one?", "instruction_first"),
    ]


@pytest.fixture(scope="module")
def inputs(ad, samples):
    return [ad.build_inputs(ims, ins, order) for ims, ins, order in samples]


def _same_row(s, b):
    assert b.token_ids == s.token_ids
    assert (b.text, b.raw_text, b.thinking_unclosed, b.stopped) == (s.text, s.raw_text, s.thinking_unclosed, s.stopped)
    assert (b.n_prompt_tokens, b.n_visual_tokens, b.n_image_pixels, b.n_generated_tokens) == (
        s.n_prompt_tokens, s.n_visual_tokens, s.n_image_pixels, s.n_generated_tokens)


# ---------------------------------------------------------------------------
# 一致性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thinking", [False, True])
def test_batch_matches_single_greedy_token_for_token(ad, inputs, thinking):
    single = [ad.generate(i, enable_thinking=thinking, max_new_tokens=MAX_NEW) for i in inputs]
    batch = ad.generate_batch(inputs, enable_thinking=thinking, max_new_tokens=MAX_NEW)
    # 前提：这批输入确实混合（提示长度各不相同 → 有填充；视觉网格不止一种），小模型的输出随输入变化
    assert len({s.n_prompt_tokens for s in single}) >= 6
    assert len({i.grid_hw for x in inputs for i in x.images}) >= 2
    assert len({s.token_ids[:5] for s in single}) >= 4, "小模型的输出与输入无关，逐个相同的核对没有意义"
    assert all(s.n_generated_tokens >= 16 or s.stopped for s in single)
    for inp, s, b in zip(inputs, single, batch):
        _same_row(s, b)
        assert isinstance(b.stopped, bool) and b.batch_size == len(inputs) and s.batch_size == 1
        assert b.batch_wall_time_s == batch[0].batch_wall_time_s
        assert math.isclose(b.wall_time_s * len(inputs), b.batch_wall_time_s, rel_tol=1e-9)
        assert s.batch_wall_time_s == s.wall_time_s
        assert b.n_image_pixels == sum(i.n_pixels for i in inp.images)
        assert b.n_visual_tokens == sum(len(i.span) for i in inp.images)


def _pad_id(ad):
    from src.adapters import gemma4, internvl35

    pad = {registry.FAMILY_QWEN35: qwen35.END_OF_TEXT, registry.FAMILY_GEMMA4: gemma4.PAD,
           registry.FAMILY_INTERNVL35: internvl35.END_OF_TEXT}[ad.family]
    return ad.token_id(pad)


class _ForceStop(LogitsProcessor):
    """与批次无关的提前停止规则：按一行自己的提示长度 n（不含左填充）决定——n % 3 != 0 的行在第 2 + n % 5 步只留停止 token，
    其余的行不干预（跑满）。规则只看这一行自己，所以逐条与批量下行为相同；用它让各行在不同的步数停下，检验逐行拆分与填充。"""

    def __init__(self, stop_id: int, pad_id: int):
        self.stop_id, self.pad_id, self.start, self.target = stop_id, pad_id, None, None

    def __call__(self, input_ids, scores):
        if self.start is None:
            self.start = input_ids.shape[1]
            n = (input_ids != self.pad_id).sum(dim=1)
            self.target = torch.where(n % 3 != 0, 2 + n % 5, torch.full_like(n, -1))
        hit = self.target == input_ids.shape[1] - self.start
        out = scores.clone()
        out[hit] = float("-inf")
        out[hit, self.stop_id] = 0.0
        return out


def test_rows_stop_at_different_steps_and_padding_is_not_counted(ad, inputs):
    stop, pad = min(_stop_ids(ad)), _pad_id(ad)
    single = [ad.generate(i, enable_thinking=False, max_new_tokens=MAX_NEW, logits_processor=[_ForceStop(stop, pad)])
              for i in inputs]
    batch = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=MAX_NEW,
                              logits_processor=[_ForceStop(stop, pad)])
    lengths = {s.n_generated_tokens for s in single if s.stopped}
    assert len(lengths) >= 2 and any(not s.stopped for s in single), "规则没能让各行在不同步数停下"
    for s, b in zip(single, batch):
        _same_row(s, b)
        if b.stopped:
            assert b.token_ids[-1] == stop and stop not in b.token_ids[:-1]
        else:
            assert b.n_generated_tokens == MAX_NEW and not (set(b.token_ids) & _stop_ids(ad))


def test_allowed_token_ids_in_batch(ad, inputs):
    allowed = sorted(set(ad.tokenizer("hello world.", add_special_tokens=False)["input_ids"]))
    stop = min(_stop_ids(ad))
    for allow in (allowed, allowed + [stop]):
        single = [ad.generate(i, enable_thinking=False, max_new_tokens=8, allowed_token_ids=allow) for i in inputs]
        batch = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=8, allowed_token_ids=allow)
        for s, b in zip(single, batch):
            _same_row(s, b)
            assert set(b.token_ids) <= set(allow)
            if stop not in allow:  # 停止 token 不在允许集合里：一定跑满，stopped 为 False
                assert b.stopped is False and b.n_generated_tokens == 8


# ---------------------------------------------------------------------------
# 位置编码：每一步实际送进语言模型的 position_ids
# ---------------------------------------------------------------------------


def _capture_positions(ad, fn):
    lm = ad.model.model.language_model
    seen = []

    def hook(mod, args, kwargs):
        pos = kwargs.get("position_ids")
        seen.append(None if pos is None else pos.detach().cpu().clone())

    h = lm.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        out = fn()
    finally:
        h.remove()
    return out, seen


def _as_3d(pos):
    """统一成 (rows, B, L)：Qwen 是 (4, B, L)（第 0 行是文本位置），其余家族是 (B, L) → (1, B, L)。"""
    return pos if pos.dim() == 3 else pos.unsqueeze(0)


@pytest.mark.parametrize("thinking", [False, True])
def test_positions_of_non_pad_tokens_equal_single_generation_at_every_step(ad, inputs, thinking):
    batch, seen_b = _capture_positions(ad, lambda: ad.generate_batch(inputs, enable_thinking=thinking, max_new_tokens=MAX_NEW))
    assert all(p is not None for p in seen_b), "有一步没有把 position_ids 显式送进语言模型"
    prefill = _as_3d(seen_b[0])
    width = prefill.shape[-1]
    for b, inp in enumerate(inputs):
        s, seen_s = _capture_positions(ad, lambda: ad.generate(inp, enable_thinking=thinking, max_new_tokens=MAX_NEW))
        n = s.n_prompt_tokens
        p = width - n
        single_prefill = _as_3d(seen_s[0])
        assert single_prefill.shape[-1] == n
        assert torch.equal(prefill[:, b, p:], single_prefill[:, 0, :]), f"第 {b} 行 prefill 的位置不同"
        if ad.family == registry.FAMILY_QWEN35:  # 与本适配器自己的 M-RoPE 规则逐个相等
            own = qwen35.mrope_rows(n, [(i.span.start, i.span.end, i.grid_hw) for i in inp.images])
            assert torch.equal(single_prefill[1:, 0, :], own) and torch.equal(single_prefill[0, 0], torch.arange(n))
        steps = min(len(seen_s), len(seen_b)) - 1
        assert steps >= s.n_generated_tokens - 1
        for t in range(1, steps + 1):
            assert torch.equal(_as_3d(seen_b[t])[:, b, -1], _as_3d(seen_s[t])[:, 0, -1]), f"第 {b} 行第 {t} 步的位置不同"


def test_qwen_layouts_for_the_mrope_check():
    """M-RoPE 的专门用例要覆盖的布局：横图与竖图网格不同、带与不带思考头、长度不同（左填充的长度不同）。"""
    rng = np.random.default_rng(3)
    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=PixelBudget(96), tiny=True,
                      init_std=INIT_STD)
    land, port = _img(rng, 90, 210), _img(rng, 210, 90)
    items = [ad.build_inputs([land], "Describe.", "instruction_first"),
             ad.build_inputs([port], "Describe this tall picture in many many words please.", "image_first"),
             ad.build_inputs([port, land], "Compare the two.", "instruction_first")]
    grids = [i.grid_hw for x in items for i in x.images]
    assert grids[0] == tuple(reversed(grids[1])) and grids[0][0] != grids[0][1]
    assert ad.assistant_header_ids(True) != ad.assistant_header_ids(False)
    for thinking in (False, True):
        batch, seen_b = _capture_positions(ad, lambda: ad.generate_batch(items, enable_thinking=thinking, max_new_tokens=4))
        width = seen_b[0].shape[-1]
        for b, inp in enumerate(items):
            n = batch[b].n_prompt_tokens
            own = qwen35.mrope_rows(n, [(i.span.start, i.span.end, i.grid_hw) for i in inp.images])
            assert width - n >= 0
            assert torch.equal(seen_b[0][1:, b, width - n:], own)
            assert torch.equal(seen_b[0][0, b, width - n:], torch.arange(n))
            # 解码步：(T, H, W) 都从提示末尾的位置接着加 1
            for t in range(1, len(seen_b)):
                assert seen_b[t][1:, b, -1].tolist() == [int(own[0, -1]) + t] * 3


def test_qwen_runtime_position_check_catches_a_wrong_rope_index(monkeypatch):
    """生成前的逐行核对（_check_batch_positions）不是摆设：get_rope_index 若在填充下给出错位的位置，generate_batch 报错。"""
    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=PixelBudget(64), tiny=True)
    rng = np.random.default_rng(4)
    items = [ad.build_inputs([_img(rng, 80, 160)], "a", "instruction_first"),
             ad.build_inputs([_img(rng, 160, 80)], "a much longer instruction here", "instruction_first")]
    real = ad.model.model.get_rope_index

    def shifted(input_ids, **kw):
        pos, delta = real(input_ids, **kw)
        return pos + 1, delta

    monkeypatch.setattr(ad.model.model, "get_rope_index", shifted)
    with pytest.raises(RuntimeError, match="M-RoPE"):
        ad.generate_batch(items, enable_thinking=False, max_new_tokens=2)
    ad.generate(items[0], enable_thinking=False, max_new_tokens=2)  # 没有填充的单条不受影响（也不调这项核对）


def test_n_pixels_formula_per_family(ad, inputs):
    """模型实际看到的像素数（D28-3）：Qwen = (h·32)(w·32)；Gemma = (h·48)(w·48)；InternVL = tile 数 × 448²（含缩略图）。"""
    for inp in inputs:
        for info in inp.images:
            h, w = info.grid_hw
            if ad.family == registry.FAMILY_QWEN35:
                want = h * 32 * w * 32
            elif ad.family == registry.FAMILY_GEMMA4:
                want = h * 48 * w * 48
            else:
                want = (h // 16) * 448 * 448
            assert info.n_pixels == want and len(info.span) == h * w


# ---------------------------------------------------------------------------
# 接口与入口检查
# ---------------------------------------------------------------------------


def test_generate_batch_signature_and_validation(ad, inputs):
    assert type(ad).generate_batch is not ModelAdapter.generate_batch
    with pytest.raises(TypeError):
        ad.generate_batch(inputs, max_new_tokens=4)  # 思考开关必填
    with pytest.raises(TypeError):
        ad.generate_batch(inputs, enable_thinking=False)  # max_new_tokens 必填
    with pytest.raises(ValueError):
        ad.generate_batch([], enable_thinking=False, max_new_tokens=4)
    with pytest.raises(TypeError):
        ad.generate_batch(inputs[0], enable_thinking=False, max_new_tokens=4)  # 单条要放进列表
    with pytest.raises(TypeError):
        ad.generate_batch([inputs[0], "x"], enable_thinking=False, max_new_tokens=4)
    with pytest.raises(ValueError, match="gen_kwargs"):
        ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=4, attention_mask=None)
    with pytest.raises(ValueError):
        ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=0)


def test_generate_batch_entry_assertion_checks_every_row(ad, inputs):
    """入口断言逐行生效（lessons：开发机用 meta 设备代理）：第 2 行的张量不在模型的设备上 → AdapterDeviceError，报出两个值。"""
    bad = inputs[1]
    moved = type(bad)(tensors={k: v.to("meta") if torch.is_tensor(v) else v for k, v in bad.tensors.items()},
                      order=bad.order, images=bad.images, instruction_span=bad.instruction_span)
    with pytest.raises(AdapterDeviceError, match="meta"):
        ad.generate_batch([inputs[0], moved], enable_thinking=False, max_new_tokens=2)


def test_query_inputs_cannot_be_batched():
    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=PixelBudget(64), tiny=True)
    ad.inject_query_tokens(4, "grid2d")
    inp = ad.build_inputs([], "hi", "instruction_first")
    with pytest.raises(ValueError, match="generate_batch"):
        ad.generate_batch([inp], enable_thinking=False, max_new_tokens=2)
    with pytest.raises(ValueError, match="generate"):
        ad.generate(inp, enable_thinking=False, max_new_tokens=2)


def test_warmup_does_not_change_results(ad, inputs, samples):
    before = ad.generate(inputs[0], enable_thinking=False, max_new_tokens=6)
    assert ad.warmup(samples[0][0]) >= 0 and ad.warmup(samples[0][0], batch_size=3) >= 0
    after = ad.generate(inputs[0], enable_thinking=False, max_new_tokens=6)
    assert after.token_ids == before.token_ids


# ---------------------------------------------------------------------------
# Gemma：滑窗层在左填充下（E2B 35 层里 28 层是 512 滑窗，D21-3；思考预算 4096 一定超过窗口）
# ---------------------------------------------------------------------------


def test_gemma_sliding_window_layers_under_left_padding(samples):
    """小模型把滑窗改成 16（提示 9–136 个 token，再生成 40 个），各行的提示与生成都跨过窗口；批量与逐条照样逐个相同。"""
    from src.adapters import gemma4
    from src.adapters.factory import reinit_parameters

    comps = gemma4.load_pinned_components()
    model = gemma4.build_tiny_model(comps.config, sliding_window=16, generation_config=comps.generation_config)
    reinit_parameters(model, std=INIT_STD, seed=0)
    ad = gemma4.make_tiny_adapter(comps, model=model)
    assert ad.model.config.text_config.sliding_window == 16
    items = [ad.build_inputs(ims, ins, order) for ims, ins, order in samples]
    assert min(x.tensors["input_ids"].shape[1] for x in items) < 16 < max(x.tensors["input_ids"].shape[1] for x in items)
    for thinking in (False, True):
        single = [ad.generate(i, enable_thinking=thinking, max_new_tokens=40) for i in items]
        batch = ad.generate_batch(items, enable_thinking=thinking, max_new_tokens=40)
        assert len({s.token_ids[:5] for s in single}) >= 4
        for s, b in zip(single, batch):
            _same_row(s, b)


# ---------------------------------------------------------------------------
# InternVL：原生动态 tile（D28-3）
# ---------------------------------------------------------------------------


def test_internvl_native_tiles_640x480_is_13_tiles_and_generation_works():
    ad = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, image_policy=INTERNVL_NATIVE_TILES,
                      tiny=True, init_std=INIT_STD)
    assert ad.image_policy == TilePolicy(True, 12) and ad.processor_kwargs()["max_patches"] == 12
    rng = np.random.default_rng(5)
    big, small = _img(rng, 480, 640), _img(rng, 448, 448)
    a = ad.build_inputs([big], "Describe.", "instruction_first")
    info = a.images[0]
    assert info.grid_hw == (16 * 13, 16) and len(info.span) == 3328 and info.n_pixels == 13 * 448 * 448
    b = ad.build_inputs([small], "Describe the picture.", "instruction_first")
    assert b.images[0].grid_hw == (16, 16)  # 方图只切出 1 个 tile
    single = [ad.generate(x, enable_thinking=False, max_new_tokens=8) for x in (a, b)]
    batch = ad.generate_batch([a, b], enable_thinking=False, max_new_tokens=8)
    for s, r in zip(single, batch):
        _same_row(s, r)
    assert batch[0].n_visual_tokens == 3328 and batch[0].n_image_pixels == 13 * 448 * 448
    with pytest.raises(ValueError, match="tile"):  # 探针类方法照旧拒绝多 tile
        ad.hidden_states(a, [0])
    with pytest.raises(ValueError, match="tile"):
        ad.vision_features(a)
