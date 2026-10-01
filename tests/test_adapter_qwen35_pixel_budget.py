"""DEV-0011：Qwen3.5 的像素预算口径（D28-3）与 token_id()。

- build_inputs 与官方 Qwen3VLProcessor 类（qwen35.reference_processor；与服务器上 AutoProcessor 同一个类，开发机只绕过视频处理器
  的类型检查）在同样的 min_pixels = max_pixels 下逐项相等：input_ids（含 assistant 头）、attention_mask、mm_token_type_ids、
  image_grid_thw、pixel_values。
- 每张图的网格 = 图像处理器模块自己的 smart_resize；640×480 在 300 档恰为 15×20、不缩放。
- 像素预算下 hidden_states、vision_features、位置编码、查询 token 按每张图自己的网格工作；多张图网格不同也可以。
- 固定尺寸的全局属性（image_size、llm_grid_hw、resize_image）在像素预算下报错。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import qwen35, registry
from src.adapters.image_policy import (
    E1_PIXEL_BUDGET_TOKENS,
    E1_SUBSET_PIXEL_BUDGET_TOKENS,
    FixedSize,
    PixelBudget,
    SoftTokenBudget,
)
from src.adapters.qwen35 import Qwen35Adapter

INSTR = "Describe the cat."


@pytest.fixture(scope="module")
def components():
    entry = registry.get_model(qwen35.TINY_SOURCE_KEY)
    try:
        return qwen35.load_pinned_components(entry.key)
    except Exception as e:  # noqa: BLE001 — 取不到小文件时给出原因后失败，不跳过
        pytest.fail(f"取不到 {entry.repo}@{entry.revision} 的小文件：{e!r}（需要联网下载一次）", pytrace=False)


@pytest.fixture(scope="module")
def base_model(components):
    return qwen35.build_tiny_model(components[0], tie_word_embeddings=True)


def _budget(components, base_model, n, *, fresh=False):
    config, tok, ip, template = components
    if fresh:
        tok = copy.deepcopy(tok)
    model = copy.deepcopy(base_model) if fresh else base_model
    return qwen35.make_tiny_adapter((config, tok, ip, template), image_policy=PixelBudget(n), model=model)


@pytest.fixture(scope="module")
def ad300(components, base_model):
    return _budget(components, base_model, E1_PIXEL_BUDGET_TOKENS)


@pytest.fixture(scope="module")
def ref(components):
    _, tok, ip, template = components
    return qwen35.reference_processor(tok, ip, template)


@pytest.fixture(scope="module")
def imgs():
    rng = np.random.default_rng(11)
    out = {}
    for name, (h, w) in {"vga": (480, 640), "land": (360, 540), "port": (500, 300), "small": (100, 150)}.items():
        a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        a[h // 3: h // 2, w // 4: w // 2, 0] = 250
        out[name] = Image.fromarray(a)
    return out


def _content(n_images, order):
    imgs = [{"type": "image"}] * n_images
    text = {"type": "text", "text": INSTR}
    return [text, *imgs] if order == "instruction_first" else [*imgs, text]


def _compare(ad, ref, images, order, thinking):
    inp = ad.build_inputs(images, INSTR, order)
    px = ad.processor_pixels()
    text = ref.apply_chat_template([{"role": "user", "content": _content(len(images), order)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)
    enc = ref(text=[text], images=list(images), min_pixels=px, max_pixels=px, return_tensors="pt")
    t = inp.tensors
    header = torch.tensor([ad.assistant_header_ids(thinking)], dtype=torch.long)
    ids = torch.cat([t["input_ids"], header], dim=1)
    mm = torch.cat([t["mm_token_type_ids"], torch.zeros_like(header)], dim=1)
    assert torch.equal(enc["input_ids"], ids)
    assert torch.equal(enc["attention_mask"], torch.ones_like(ids))
    assert torch.equal(enc["mm_token_type_ids"], mm)
    assert torch.equal(enc["image_grid_thw"], t["image_grid_thw"])
    assert torch.equal(enc["pixel_values"], t["pixel_values"])
    return inp


# ---------------------------------------------------------------------------
# 构造与口径
# ---------------------------------------------------------------------------


def test_constructor_takes_exactly_one_image_policy():
    with pytest.raises(ValueError):
        Qwen35Adapter(device="cpu")
    with pytest.raises(ValueError):
        Qwen35Adapter(device="cpu", image_size=(64, 64), image_policy=PixelBudget(300))
    with pytest.raises(TypeError):
        Qwen35Adapter(device="cpu", image_policy=SoftTokenBudget(280))
    with pytest.raises(ValueError):
        PixelBudget(0)
    with pytest.raises(ValueError):
        FixedSize(480, 630)  # 不是 32 的倍数
    assert Qwen35Adapter(device="cpu", image_size=(64, 96)).image_policy == FixedSize(64, 96)
    assert Qwen35Adapter(device="cpu", image_policy=FixedSize(64, 96)).image_size == (64, 96)
    pb = Qwen35Adapter(device="cpu", image_policy=PixelBudget(300))
    assert pb.image_policy == PixelBudget(300) and pb.image_policy.pixels == 300 * 32 * 32 == 307200
    assert PixelBudget(300).as_dict() == {"pixel_budget_tokens": 300, "min_pixels": 307200, "max_pixels": 307200}


def test_global_grid_attributes_raise_in_pixel_budget_mode(ad300, imgs):
    for attr in ("image_size", "llm_grid_hw"):
        with pytest.raises(ValueError, match="固定尺寸"):
            getattr(ad300, attr)
    with pytest.raises(ValueError, match="固定尺寸"):
        ad300.resize_image(imgs["vga"])


def test_640x480_at_300_tokens_is_15x20_without_rescaling(ad300, imgs, ref):
    inp = _compare(ad300, ref, [imgs["vga"]], "image_first", False)
    info = inp.images[0]
    assert info.grid_hw == (15, 20) and len(info.span) == 300 and info.n_pixels == 480 * 640
    assert inp.tensors["image_grid_thw"].tolist() == [[1, 30, 40]]
    # 不缩放：处理器直接拿原图（640×480 本来就是 32 的倍数且恰为预算）得到的像素值与这里相同
    direct = ad300.image_processor(images=[imgs["vga"]], min_pixels=307200, max_pixels=307200, do_resize=False,
                                   return_tensors="pt")
    assert torch.equal(direct["pixel_values"], inp.tensors["pixel_values"])


@pytest.mark.parametrize("n", [E1_PIXEL_BUDGET_TOKENS, E1_SUBSET_PIXEL_BUDGET_TOKENS])
@pytest.mark.parametrize("names", [("land",), ("port",), ("land", "port"), ("small", "vga")])
def test_build_inputs_equals_official_processor(components, base_model, ref, imgs, n, names):
    ad = _budget(components, base_model, n)
    images = [imgs[k] for k in names]
    for order in ("instruction_first", "image_first"):
        for thinking in (False, True):
            inp = _compare(ad, ref, images, order, thinking)
    merge = ad.image_processor.merge_size
    want = [tuple(g // merge for g in ad.expected_grid((im.height, im.width))) for im in images]
    assert [i.grid_hw for i in inp.images] == want
    assert [i.n_pixels for i in inp.images] == [h * 32 * w * 32 for h, w in want]
    if len(names) == 2:
        assert want[0] != want[1]  # 两张图网格不同


def test_smart_resize_rounding_gives_approximately_n_tokens(ad300, imgs):
    """网格的 token 数只是约等于预算：放大时向上取整（540×360 → 22×15 = 330），缩小时向下取整。"""
    inp = ad300.build_inputs([imgs["land"]], INSTR, "image_first")
    assert inp.images[0].grid_hw == (15, 22)
    big = Image.fromarray(np.zeros((1000, 1500, 3), np.uint8))
    h, w = ad300.build_inputs([big], INSTR, "image_first").images[0].grid_hw
    assert h * w <= 300


def test_fixed_size_mode_is_unchanged_and_records_pixels(components, base_model, imgs):
    config, tok, ip, template = components
    ad = qwen35.make_tiny_adapter((config, tok, ip, template), image_size=(128, 192), model=base_model)
    inp = ad.build_inputs([imgs["port"], imgs["land"]], INSTR, "instruction_first")
    assert [i.grid_hw for i in inp.images] == [(4, 6), (4, 6)] == [ad.llm_grid_hw] * 2
    assert [i.n_pixels for i in inp.images] == [128 * 192] * 2


# ---------------------------------------------------------------------------
# 前向：每张图自己的网格
# ---------------------------------------------------------------------------


def test_positions_follow_each_image_grid(ad300, imgs):
    inp = ad300.build_inputs([imgs["land"], imgs["port"]], INSTR, "instruction_first")
    pos = ad300.position_ids(inp)  # 内部已与 get_rope_index 逐项比对
    t = inp.tensors
    ref_pos, _ = ad300.model.model.get_rope_index(t["input_ids"], mm_token_type_ids=t["mm_token_type_ids"],
                                                  image_grid_thw=t["image_grid_thw"], attention_mask=t["attention_mask"])
    assert torch.equal(pos[1:], ref_pos)
    for info in inp.images:  # 每张图：T 恒定，H/W 按这张图自己的网格展开
        s, e = info.span.start, info.span.end
        gh, gw = info.grid_hw
        block = pos[1:, 0, s:e]
        assert block[0].unique().numel() == 1
        assert torch.equal(block[1] - block[0], torch.arange(gh).repeat_interleave(gw))
        assert torch.equal(block[2] - block[0], torch.arange(gw).repeat(gh))


def test_hidden_states_and_vision_features_single_and_multi_image(ad300, imgs):
    for images in ([imgs["port"]], [imgs["land"], imgs["port"]]):
        inp = ad300.build_inputs(images, INSTR, "instruction_first")
        n_tok = sum(len(i.span) for i in inp.images)
        with torch.no_grad():
            hs = ad300.hidden_states(inp, [0, ad300.num_layers()])
            vf = ad300.vision_features(inp)
        assert tuple(hs.image[0].shape) == (n_tok, ad300.hidden_size())
        assert torch.equal(vf, hs.image[0])  # 与第 0 层图像位置逐位相同


def test_query_tokens_with_pixel_budget(components, base_model, imgs):
    ad = _budget(components, base_model, E1_PIXEL_BUDGET_TOKENS, fresh=True)
    ad.inject_query_tokens(16, "grid2d")
    inp = ad.build_inputs([imgs["land"], imgs["port"]], INSTR, "instruction_first")
    pos = ad.position_ids(inp)
    qs, qe = inp.query_span.start, inp.query_span.end
    t0 = int(pos[1, 0, qs - 1]) + 1
    assert pos[1:, 0, qs:qe].tolist() == [[t0] * 16, [t0 + r for r in range(4) for _ in range(4)],
                                          [t0 + c for _ in range(4) for c in range(4)]]
    q = torch.randn(16, ad.hidden_size()) * 0.02
    with torch.no_grad():
        hs = ad.hidden_states(inp, [0, 1], query_embeds=q)
    assert torch.equal(hs.query[0], q) and hs.image[1].shape[0] == sum(len(i.span) for i in inp.images)


def test_generate_records_visual_tokens_and_pixels(ad300, imgs):
    inp = ad300.build_inputs([imgs["land"], imgs["small"]], INSTR, "instruction_first")
    res = ad300.generate(inp, enable_thinking=False, max_new_tokens=4)
    assert res.n_visual_tokens == sum(len(i.span) for i in inp.images)
    assert res.n_image_pixels == sum(i.n_pixels for i in inp.images) > 0
    assert isinstance(res.stopped, bool) and res.batch_size == 1
    text_only = ad300.generate(ad300.build_inputs([], INSTR, "instruction_first"), enable_thinking=False,
                               max_new_tokens=2)
    assert (text_only.n_visual_tokens, text_only.n_image_pixels) == (0, 0)


# ---------------------------------------------------------------------------
# token_id()：词表里的任意 token（RUN-0005）
# ---------------------------------------------------------------------------


def test_token_id_looks_up_any_vocab_token(ad300):
    tok = ad300.tokenizer
    assert ad300.token_id("<think>") == tok.convert_tokens_to_ids("<think>")
    assert tok.convert_ids_to_tokens(ad300.token_id("<think>")) == "<think>"
    assert ad300.token_id(qwen35.IM_END) == tok.convert_tokens_to_ids(qwen35.IM_END)  # attach 时缓存的照旧
    word = tok.convert_ids_to_tokens(1000)  # 普通词表项（字面是词表里的写法）
    assert ad300.token_id(word) == 1000
    for bad in ("<definitely_not_a_token>", "two words here", ""):
        with pytest.raises(KeyError):
            ad300.token_id(bad)
