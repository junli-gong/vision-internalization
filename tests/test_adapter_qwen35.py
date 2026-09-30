"""Qwen3.5 适配器（DEV-0003）的 CPU 测试。

模型用本地构造的小型随机 Qwen3.5：取固定 revision 的真实 config，缩到 4 层（3 linear + 1 full）、hidden 64、
视觉塔 1 层；tokenizer / 图像处理器 / 聊天模板从固定 revision 下载（小文件，不含权重）。取不到就失败并说明原因，
不静默跳过。fixture 的图 H≠W，两张图内容不同；两条指令内容不同、token 长度相同。
"""

from __future__ import annotations

import copy
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import qwen35, registry
from src.adapters.base import GenerationResult, ModelAdapter, check_grid_token_names
from src.adapters.qwen35 import (
    END_OF_TEXT,
    IM_END,
    IM_START,
    IMAGE_PAD,
    QUERY_PLACEHOLDER,
    THINK_END,
    VISION_END,
    VISION_START,
    AdapterDeviceError,
    Qwen35Adapter,
)
from src.constants import MAX_NEW_GRID_TOKENS

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_P4 = REPO_ROOT / "experiments/P0/0001_env_probe/raw/p4"

IMG_HW = (128, 192)  # H≠W；每张图的 LLM 网格 4×6 = 24 个视觉 token
GRID_HW = (4, 6)
N_IMG_TOK = 24
INSTR_A = "Describe the cat."
INSTR_B = "Describe the sofa."

# 0.8B 固定 revision 的词表事实（docs/model_notes.md §5.1），测试里钉住；适配器本身不写死它们
LEN_TOKENIZER = 248077
N_BASE_VOCAB = 248044
N_ROWS = 248320


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def components():
    entry = registry.get_model(qwen35.TINY_SOURCE_KEY)
    try:
        return qwen35.load_pinned_components(entry.key)
    except Exception as e:  # noqa: BLE001 — 取不到小文件时给出原因后失败，不跳过
        pytest.fail(
            f"取不到 {entry.repo}@{entry.revision} 的 config/tokenizer/处理器/模板小文件：{e!r}。"
            "测试需要联网下载一次（之后走本地 HF 缓存）。",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def base_models(components):
    return {tied: qwen35.build_tiny_model(components[0], tie_word_embeddings=tied) for tied in (True, False)}


def _adapter(components, base_models, *, tied=True, fresh_tokenizer=False, image_size=IMG_HW, model=None):
    config, tok, ip, template = components
    if fresh_tokenizer:
        tok = copy.deepcopy(tok)
    m = model if model is not None else copy.deepcopy(base_models[tied])
    return qwen35.make_tiny_adapter((config, tok, ip, template), image_size=image_size, model=m)


@pytest.fixture(scope="module")
def adapter(components, base_models):
    """只读共享：不许在它上面加 token 或注入查询（用例结束时检查）。"""
    ad = _adapter(components, base_models)
    yield ad
    assert ad.n_grid_tokens == 0 and ad.query_token_id is None, "共享适配器被改动过"
    assert len(ad.tokenizer) == LEN_TOKENIZER


@pytest.fixture
def fresh(components, base_models):
    """每次一个独立的适配器（tokenizer 副本 + 模型副本），可以加 token、注入查询。"""
    return lambda tied=True, image_size=IMG_HW: _adapter(
        components, base_models, tied=tied, fresh_tokenizer=True, image_size=image_size
    )


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(20260930)
    a = np.zeros((100, 150, 3), np.uint8)  # 横图：水平渐变 + 一个矩形 + 噪声
    a[..., 0] = np.linspace(0, 255, 150, dtype=np.uint8)[None, :]
    a[20:60, 30:90, 1] = 200
    a = (a + rng.integers(0, 30, a.shape)).clip(0, 255).astype(np.uint8)
    b = rng.integers(0, 256, (150, 100, 3), dtype=np.uint8)  # 竖图：纯噪声
    return [a, Image.fromarray(b)]


def _all_layers(ad):
    return list(range(ad.num_layers() + 1))


def _collapse_image_runs(ids, image_id):
    out = []
    for t in ids:
        if t == image_id and out and out[-1] == image_id:
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# 注册表与固定 revision
# ---------------------------------------------------------------------------


def test_registry_has_the_eight_d9_models():
    assert registry.keys_for_family(registry.FAMILY_QWEN35) == (
        "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b")
    assert registry.keys_for_family(registry.FAMILY_GEMMA4) == ("gemma-4-e2b-it", "gemma-4-e4b-it")
    assert registry.keys_for_family(registry.FAMILY_INTERNVL35) == ("internvl3.5-2b-hf", "internvl3.5-4b-hf")
    assert len(registry.MODELS) == 8
    assert [e.repo for e in registry.MODELS.values()] == [
        "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B",
        "google/gemma-4-E2B-it", "google/gemma-4-E4B-it",
        "OpenGVLab/InternVL3_5-2B-HF", "OpenGVLab/InternVL3_5-4B-HF",
    ]


def test_registry_revisions_match_model_notes_table():
    text = (REPO_ROOT / "docs/model_notes.md").read_text()
    table = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{40})` \|", text, flags=re.M))
    assert len(table) >= 8
    for e in registry.MODELS.values():
        assert table[e.repo] == e.revision, e.key


def test_registry_rejects_unknown_key_and_bad_revision():
    with pytest.raises(KeyError, match="未登记"):
        registry.get_model("Qwen/Qwen3.5-0.8B")
    with pytest.raises(ValueError):
        registry.ModelEntry("x", registry.FAMILY_QWEN35, "a/b", "main")
    with pytest.raises(ValueError):
        registry.ModelEntry("x", "llama", "a/b", "0" * 40)


def test_load_accepts_only_registered_qwen_keys():
    ad = Qwen35Adapter(device="cpu", image_size=IMG_HW)
    with pytest.raises(KeyError):
        ad.load("Qwen/Qwen3.5-0.8B", torch.float32)  # 裸仓库名不行：必须走注册表的 revision
    with pytest.raises(ValueError, match="gemma4"):
        ad.load("gemma-4-e2b-it", torch.float32)
    with pytest.raises(TypeError):
        ad.load("qwen3.5-0.8b", "bf16")


def test_pinned_files_match_run0001_copies(components):
    config, tok, _, template = components
    assert template == (RAW_P4 / "Qwen__Qwen3.5-0.8B/chat_template.jinja").read_text()
    raw = json.loads((RAW_P4 / "Qwen__Qwen3.5-0.8B/config.json").read_text())
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id", "tie_word_embeddings"):
        assert getattr(config, key) == raw[key], key
    tc = config.text_config
    assert tc.vocab_size == raw["text_config"]["vocab_size"] == N_ROWS
    assert list(tc.layer_types) == raw["text_config"]["layer_types"]
    assert len(tok) == LEN_TOKENIZER and tok.vocab_size == N_BASE_VOCAB


# ---------------------------------------------------------------------------
# 构造与挂载
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(100, 192), (128, 200), (0, 32), (128.0, 192), (True, 32), (128,)])
def test_image_size_must_be_positive_multiples_of_32(size):
    with pytest.raises(ValueError):
        Qwen35Adapter(device="cpu", image_size=size)


def test_attach_reads_facts_from_tokenizer_and_config(adapter, components, base_models):
    tok = components[1]
    assert adapter.num_layers() == 4 and adapter.hidden_size() == 64
    assert adapter.llm_grid_hw == GRID_HW and adapter.dtype == torch.float32
    assert adapter.n_base_vocab == N_BASE_VOCAB and adapter.tied_word_embeddings is True
    for s in (IM_START, IM_END, END_OF_TEXT, THINK_END, IMAGE_PAD, VISION_START, VISION_END):
        assert adapter.token_id(s) == tok.convert_tokens_to_ids(s)
    assert adapter.token_id(IMAGE_PAD) == components[0].image_token_id
    untied = _adapter(components, base_models, tied=False)
    assert untied.tied_word_embeddings is False
    with pytest.raises(RuntimeError, match="已经挂过"):
        adapter.attach(copy.deepcopy(base_models[True]), tokenizer=tok, image_processor=components[2],
                       chat_template=components[3])


def test_attach_rejects_non_sdpa_attention(components):
    model = qwen35.build_tiny_model(components[0], tie_word_embeddings=True)
    model.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="sdpa"):
        qwen35.make_tiny_adapter(components, image_size=IMG_HW, model=model)


# ---------------------------------------------------------------------------
# build_inputs：区间、顺序、模板
# ---------------------------------------------------------------------------


def test_spans_instruction_first(adapter, images, components):
    tok = components[1]
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    ids = inp.tensors["input_ids"][0].tolist()
    # [<|im_start|>, user, \n, 指令×4, <|vs|>, pad×24, <|ve|>, <|vs|>, pad×24, <|ve|>, <|im_end|>, \n]
    assert len(ids) == 61
    assert (inp.instruction_span.start, inp.instruction_span.end) == (3, 7)
    assert [(i.span.start, i.span.end) for i in inp.images] == [(8, 32), (34, 58)]
    assert all(i.grid_hw == GRID_HW for i in inp.images)
    assert ids[3:7] == tok(INSTR_A, add_special_tokens=False)["input_ids"]
    assert ids[:3] == tok("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    assert [ids[7], ids[32], ids[33], ids[58]] == [adapter.token_id(VISION_START), adapter.token_id(VISION_END)] * 2
    assert set(ids[8:32]) == set(ids[34:58]) == {adapter.token_id(IMAGE_PAD)}
    assert ids[59:] == [adapter.token_id(IM_END), tok("\n", add_special_tokens=False)["input_ids"][0]]
    assert inp.order == "instruction_first" and inp.query_span is None


def test_spans_image_first(adapter, images, components):
    tok = components[1]
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    ids = inp.tensors["input_ids"][0].tolist()
    # [<|im_start|>, user, \n, <|vs|>, pad×24, <|ve|>, <|vs|>, pad×24, <|ve|>, 指令×4, <|im_end|>, \n]
    assert len(ids) == 61
    assert [(i.span.start, i.span.end) for i in inp.images] == [(4, 28), (30, 54)]
    assert (inp.instruction_span.start, inp.instruction_span.end) == (55, 59)
    assert ids[55:59] == tok(INSTR_A, add_special_tokens=False)["input_ids"]
    assert ids[59] == adapter.token_id(IM_END)


def test_tensors_grid_and_pixel_values(adapter, images):
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    t = inp.tensors
    ids = t["input_ids"]
    assert torch.equal(t["attention_mask"], torch.ones_like(ids))
    assert t["mm_token_type_ids"].dtype == torch.long
    assert torch.equal(t["mm_token_type_ids"], (ids == adapter.token_id(IMAGE_PAD)).long())
    assert t["mm_token_type_ids"].sum().item() == 2 * N_IMG_TOK
    assert t["image_grid_thw"].tolist() == [[1, 8, 12], [1, 8, 12]]  # patch 网格 = LLM 网格 ×2
    pv = t["pixel_values"]
    assert pv.shape == (2 * 8 * 12, 3 * 2 * 16 * 16) and pv.dtype == adapter.dtype and pv.device == adapter.device
    assert not torch.equal(pv[: 8 * 12], pv[8 * 12 :])  # 两张图内容不同


def test_resize_numpy_and_pil_agree(adapter, images):
    a_np = images[0]
    a_pil = Image.fromarray(a_np)
    x = adapter.build_inputs([a_np], INSTR_A, "image_first").tensors["pixel_values"]
    y = adapter.build_inputs([a_pil], INSTR_A, "image_first").tensors["pixel_values"]
    assert torch.equal(x, y)
    assert adapter.resize_image(a_np).size == (IMG_HW[1], IMG_HW[0])
    exact = np.asarray(adapter.resize_image(a_np))  # 已经是目标尺寸的图不再变化
    assert np.array_equal(np.asarray(adapter.resize_image(exact)), exact)


def test_prompt_stops_at_user_turn(adapter, images):
    ids = adapter.build_inputs(images, INSTR_A, "instruction_first").tensors["input_ids"][0].tolist()
    assert ids.count(adapter.token_id(IM_START)) == 1
    assert ids[-2] == adapter.token_id(IM_END)
    assert THINK_END not in adapter.tokenizer.decode(ids)


def test_assistant_headers_match_model_notes(adapter):
    dec = adapter.tokenizer.decode
    assert dec(list(adapter.assistant_header_ids(False))) == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert dec(list(adapter.assistant_header_ids(True))) == "<|im_start|>assistant\n<think>\n"
    with pytest.raises(TypeError):
        adapter.assistant_header_ids(1)


@pytest.mark.parametrize("order", ["instruction_first", "image_first"])
@pytest.mark.parametrize("thinking", [False, True])
def test_prompt_plus_header_equals_chat_template(adapter, images, order, thinking):
    inp = adapter.build_inputs(images, INSTR_A, order)
    full = inp.tensors["input_ids"][0].tolist() + list(adapter.assistant_header_ids(thinking))
    imgs = [{"type": "image"}] * len(images)
    text = {"type": "text", "text": INSTR_A}
    content = [text, *imgs] if order == "instruction_first" else [*imgs, text]
    ref = adapter.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )["input_ids"]
    assert _collapse_image_runs(full, adapter.token_id(IMAGE_PAD)) == list(ref)


def test_instruction_validation(adapter, images):
    with pytest.raises(ValueError, match="trim"):
        adapter.build_inputs(images, INSTR_A + " ", "image_first")  # 模板会 trim 整段内容
    with pytest.raises(ValueError, match="trim"):
        adapter.build_inputs(images, "\n" + INSTR_A, "instruction_first")
    with pytest.raises(ValueError, match="结构性"):
        adapter.build_inputs(images, "look <|image_pad|> here", "image_first")
    with pytest.raises(ValueError):
        adapter.build_inputs(images, INSTR_A, "image_last")
    with pytest.raises(TypeError):
        adapter.build_inputs(images[0], INSTR_A, "image_first")  # 单张图也要放进列表
    with pytest.raises(ValueError):
        adapter.build_inputs([np.zeros((10, 10), np.uint8)], INSTR_A, "image_first")


def test_empty_instruction_and_zero_images(adapter, images):
    for order in ("instruction_first", "image_first"):
        inp = adapter.build_inputs(images, "", order)
        assert len(inp.instruction_span) == 0
    inp = adapter.build_inputs([], INSTR_A, "instruction_first")
    assert inp.images == () and "pixel_values" not in inp.tensors
    L = inp.tensors["input_ids"].shape[1]
    assert torch.equal(adapter.position_ids(inp)[:, 0], torch.arange(L).expand(4, L))
    with torch.no_grad():
        hs = adapter.hidden_states(inp, [0, 4])
    assert hs.image[4].shape == (0, 64) and adapter.vision_features(inp).shape == (0, 64)
    res = adapter.generate(inp, enable_thinking=False, max_new_tokens=2)
    assert res.n_visual_tokens == 0 and 1 <= res.n_generated_tokens <= 2


# ---------------------------------------------------------------------------
# 隐藏状态：层号、仪器、自构位置
# ---------------------------------------------------------------------------


def test_instruction_pair_has_equal_token_length(components):
    tok = components[1]
    a = tok(INSTR_A, add_special_tokens=False)["input_ids"]
    b = tok(INSTR_B, add_special_tokens=False)["input_ids"]
    assert len(a) == len(b) == 4 and a != b


def test_image_first_bitwise_equal_instruction_first_differs(adapter, images):
    layers = _all_layers(adapter)
    got = {}
    for order in ("image_first", "instruction_first"):
        a = adapter.build_inputs(images, INSTR_A, order)
        b = adapter.build_inputs(images, INSTR_B, order)
        assert [i.span for i in a.images] == [i.span for i in b.images]
        with torch.no_grad():
            ha, hb = adapter.hidden_states(a, layers), adapter.hidden_states(b, layers)
        got[order] = [torch.equal(ha.image[l], hb.image[l]) for l in layers]
    assert got["image_first"] == [True] * len(layers)  # 精确为零的仪器
    assert got["instruction_first"] == [True] + [False] * (len(layers) - 1)


def test_layer_numbering_matches_decoder_outputs(adapter, images):
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    lm = adapter.model.model.language_model
    raw = {}
    hooks = [layer.register_forward_hook(lambda m, a, o, i=i: raw.__setitem__(i, o)) for i, layer in enumerate(lm.layers)]
    try:
        with torch.no_grad():
            hs = adapter.sequence_hidden_states(inp, _all_layers(adapter))
    finally:
        for h in hooks:
            h.remove()
    n = adapter.num_layers()
    with torch.no_grad():
        emb = adapter.model.get_input_embeddings()(inp.tensors["input_ids"])[0]
        normed = lm.norm(raw[n - 1])[0]
    text_pos = inp.instruction_span.start
    assert torch.equal(hs[0][text_pos], emb[text_pos])  # 层号 0：嵌入输出（文本位置就是查表值）
    for i in range(1, n):
        assert torch.equal(hs[i], raw[i - 1][0]), i  # 层号 i：第 i 个 decoder 层的输出
    assert torch.equal(hs[n], normed)  # 最后一个层号：最终归一化之后（HF 惯例）
    assert not torch.equal(hs[n], raw[n - 1][0])


@pytest.mark.parametrize("bad", [[-1], [5], [True], [1.0], [], [2, 2], "3"])
def test_layers_are_validated(adapter, images, bad):
    inp = adapter.build_inputs(images[:1], INSTR_A, "image_first")
    with pytest.raises((ValueError, TypeError)):
        adapter.hidden_states(inp, bad)


def test_hidden_states_shapes_and_order(adapter, images):
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    with torch.no_grad():
        hs = adapter.hidden_states(inp, [4, 0, 2])
        seq = adapter.sequence_hidden_states(inp, [2])
    assert hs.layers == (4, 0, 2) and hs.query is None
    assert hs.image[2].shape == (2 * N_IMG_TOK, 64)
    idx = torch.cat([torch.arange(i.span.start, i.span.end) for i in inp.images])
    assert torch.equal(hs.image[2], seq[2][idx])  # 多图按顺序拼接


@pytest.mark.parametrize("order", ["instruction_first", "image_first"])
def test_own_positions_and_merge_equal_model_path(adapter, images, order):
    inp = adapter.build_inputs(images, INSTR_A, order)
    pos = adapter.position_ids(inp)
    L = inp.tensors["input_ids"].shape[1]
    assert pos.shape == (4, 1, L) and torch.equal(pos[0, 0], torch.arange(L))
    ref_pos, _ = adapter.model.model.get_rope_index(
        inp.tensors["input_ids"], mm_token_type_ids=inp.tensors["mm_token_type_ids"],
        image_grid_thw=inp.tensors["image_grid_thw"], attention_mask=inp.tensors["attention_mask"])
    assert torch.equal(pos[1:], ref_pos)
    layers = _all_layers(adapter)
    with torch.no_grad():
        ours = adapter.sequence_hidden_states(inp, layers)
        auto = adapter.model(**inp.tensors, output_hidden_states=True, logits_to_keep=1).hidden_states
        given = adapter.model(**inp.tensors, position_ids=pos, output_hidden_states=True, logits_to_keep=1).hidden_states
    for l in layers:
        assert torch.equal(auto[l][0], given[l][0]), l  # 传自构 position_ids == 模型自己算
        assert torch.equal(ours[l], auto[l][0]), l  # 自己并入图像特征 + 自构位置 == 模型整条路径


def test_positions_follow_model_notes_rule(adapter, images):
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    p = adapter.position_ids(inp)[1:, 0]  # (3, L)：T, H, W
    assert p[:, :8].tolist() == [list(range(8))] * 3  # 文本 0..7
    first = p[:, 8:32]  # 第一张图 s = 8：T = 8，H = 8 + 行，W = 8 + 列
    assert set(first[0].tolist()) == {8}
    assert first[1].tolist() == [8 + r for r in range(4) for _ in range(6)]
    assert first[2].tolist() == [8 + c for _ in range(4) for c in range(6)]
    assert p[:, 32].tolist() == [14] * 3  # 图后从 s + max(h, w) = 8 + 6 接着编号
    assert p[:, 34].tolist() == [16, 16, 16]  # 第二张图 s = 16
    assert p[:, 58:].tolist() == [[22, 23, 24]] * 3


def test_vision_features_equal_layer0_image_positions(adapter, images):
    for order in ("instruction_first", "image_first"):
        inp = adapter.build_inputs(images, INSTR_A, order)
        with torch.no_grad():
            vf = adapter.vision_features(inp)
            h0 = adapter.hidden_states(inp, [0]).image[0]
        assert vf.shape == (2 * N_IMG_TOK, 64) and torch.equal(vf, h0)
        assert not torch.equal(vf[:N_IMG_TOK], vf[N_IMG_TOK:])


def test_vision_features_is_overridden_and_base_default_raises():
    assert Qwen35Adapter.vision_features is not ModelAdapter.vision_features

    class _Min(ModelAdapter):
        def load(self, model_id, dtype): ...
        def build_inputs(self, images, instruction, order): ...
        def generate(self, inputs, *, enable_thinking, allowed_token_ids=None, max_new_tokens, **kw): ...
        def hidden_states(self, inputs, layers, *, query_embeds=None): ...
        def inject_query_tokens(self, n, position_mode): ...
        def add_grid_tokens(self, names): ...
        def num_layers(self): return 1
        def hidden_size(self): return 1

    with pytest.raises(NotImplementedError):
        _Min().vision_features(None)
    sig = inspect.signature(ModelAdapter.hidden_states)
    assert sig.parameters["query_embeds"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["query_embeds"].default is None


# ---------------------------------------------------------------------------
# 查询 token
# ---------------------------------------------------------------------------


def test_query_positions_grid2d_and_seq1d(fresh, images):
    ad = fresh()
    img = images[:1]
    # [<|im_start|>, user, \n, 指令×4, <|vs|>, pad×24, <|ve|>, 查询×16, <|im_end|>, \n]
    ad.inject_query_tokens(16, "grid2d")
    inp = ad.build_inputs(img, INSTR_A, "instruction_first")
    assert (inp.query_span.start, inp.query_span.end) == (33, 49) and inp.query_position_mode == "grid2d"
    p = ad.position_ids(inp)[1:, 0]
    assert p[:, 32].tolist() == [14, 14, 14]  # 块前最后一个 token（<|vision_end|>）的位置 p = 14
    t0 = 15
    assert p[0, 33:49].tolist() == [t0] * 16
    assert p[1, 33:49].tolist() == [t0 + r for r in range(4) for _ in range(4)]
    assert p[2, 33:49].tolist() == [t0 + c for _ in range(4) for c in range(4)]
    assert p[:, 49:].tolist() == [[t0 + 4, t0 + 5]] * 3  # 块后从 t0 + q 接着编号

    ad.inject_query_tokens(16, "seq1d")
    inp = ad.build_inputs(img, INSTR_A, "instruction_first")
    p = ad.position_ids(inp)[1:, 0]
    assert p[:, 33:49].tolist() == [list(range(t0, t0 + 16))] * 3
    assert p[:, 49:].tolist() == [[t0 + 16, t0 + 17]] * 3

    ad.inject_query_tokens(9, "grid2d")  # image_first：块前是指令的最后一个 token
    inp = ad.build_inputs(img, INSTR_A, "image_first")
    qs = inp.query_span.start
    p = ad.position_ids(inp)[1:, 0]
    pre = p[0, qs - 1].item()
    assert p[:, qs - 1].tolist() == [pre] * 3
    assert p[1, qs : qs + 9].tolist() == [pre + 1 + r for r in range(3) for _ in range(3)]
    assert p[:, qs + 9].tolist() == [pre + 1 + 3] * 3


def test_query_causal_prefix_bitwise_equal_across_modes(fresh, images):
    ad = fresh(image_size=(256, 192))  # 每张图 8×6 = 48 个视觉 token，前缀长 107
    layers = _all_layers(ad)
    torch.manual_seed(7)
    qe = torch.randn(16, 64)
    seq = {}
    for mode in ("grid2d", "seq1d"):
        ad.inject_query_tokens(16, mode)
        inp = ad.build_inputs(images, INSTR_A, "instruction_first")
        with torch.no_grad():
            seq[mode] = ad.sequence_hidden_states(inp, layers, query_embeds=qe)
            hs = ad.hidden_states(inp, [0, ad.num_layers()], query_embeds=qe)
        assert hs.query[ad.num_layers()].shape == (16, 64)
    qs, qe_ = inp.query_span.start, inp.query_span.end
    assert qs == 107 and qs % 64 != 0  # 前缀跨过 GDN 的 chunk 边界（64），查询块与前缀的尾部在同一个 chunk 里
    for l in layers:
        assert torch.equal(seq["grid2d"][l][:qs], seq["seq1d"][l][:qs]), l
    last = ad.num_layers()  # 全注意力层之后两种位置确实不同
    assert not torch.equal(seq["grid2d"][last][qs:qe_], seq["seq1d"][last][qs:qe_])


def test_query_embeds_replace_placeholder_and_carry_grad(fresh, images):
    ad = fresh()
    ad.inject_query_tokens(4, "grid2d")
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    qe = torch.randn(4, 64, requires_grad=True)
    hs = ad.hidden_states(inp, [0, 4], query_embeds=qe)
    assert torch.equal(hs.query[0], qe.detach())
    hs.query[4].pow(2).sum().backward()
    assert qe.grad is not None and qe.grad.abs().sum() > 0  # E3a 的可学习查询从这里得到梯度
    with torch.no_grad():
        plain = ad.hidden_states(inp, [0])
    row = ad.model.get_input_embeddings().weight[ad.query_token_id]
    assert torch.equal(plain.query[0], row.expand(4, -1))  # 不给 query_embeds 时就是占位符行
    ex = ad.model.get_input_embeddings().weight[:N_BASE_VOCAB].double().mean(0).float()
    assert torch.equal(row, ex)  # 占位符行按 D9-3 初始化为基础行均值


def test_query_embeds_validation(fresh, adapter, images):
    ad = fresh()
    ad.inject_query_tokens(4, "seq1d")
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    with pytest.raises(ValueError, match="形状"):
        ad.hidden_states(inp, [1], query_embeds=torch.zeros(5, 64))
    with pytest.raises(AdapterDeviceError, match="float64"):
        ad.hidden_states(inp, [1], query_embeds=torch.zeros(4, 64, dtype=torch.float64))
    with pytest.raises(AdapterDeviceError, match="meta"):
        ad.hidden_states(inp, [1], query_embeds=torch.zeros(4, 64, device="meta"))
    plain = adapter.build_inputs(images[:1], INSTR_A, "image_first")
    with pytest.raises(ValueError, match="没有查询"):
        adapter.hidden_states(plain, [1], query_embeds=torch.zeros(4, 64))


def test_inject_query_tokens_validation_and_clear(fresh, images):
    ad = fresh()
    for n, mode in ((0, "seq1d"), (True, "seq1d"), (15, "grid2d"), (16, "grid3d"), (2.0, "seq1d")):
        with pytest.raises(ValueError):
            ad.inject_query_tokens(n, mode)
    assert ad.query_token_id is None and len(ad.tokenizer) == LEN_TOKENIZER
    ad.inject_query_tokens(16, "grid2d")
    assert ad.query_token_id == LEN_TOKENIZER  # 首个新增 id 落在预留行里
    assert ad.tokenizer(QUERY_PLACEHOLDER * 3, add_special_tokens=False)["input_ids"] == [ad.query_token_id] * 3
    assert ad.model.get_input_embeddings().weight.shape[0] == N_ROWS
    ad.inject_query_tokens(9, "seq1d")  # 再次登记只改个数与模式，不再加 token
    assert len(ad.tokenizer) == LEN_TOKENIZER + 1
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    assert len(inp.query_span) == 9 and inp.query_position_mode == "seq1d"
    with pytest.raises(ValueError, match="generate"):
        ad.generate(inp, enable_thinking=False, max_new_tokens=1)
    with pytest.raises(ValueError, match="结构性"):
        ad.build_inputs(images[:1], "a " + QUERY_PLACEHOLDER, "image_first")
    ad.clear_query_tokens()
    assert ad.build_inputs(images[:1], INSTR_A, "image_first").query_span is None


def test_query_placeholder_is_outside_grid_budget(fresh):
    ad = fresh()
    ad.inject_query_tokens(4, "seq1d")
    names = [f"<g{i}>" for i in range(MAX_NEW_GRID_TOKENS)]
    ids = ad.add_grid_tokens(names)
    assert ad.n_grid_tokens == MAX_NEW_GRID_TOKENS
    assert min(ids.values()) == LEN_TOKENIZER + 1
    assert ad.model.get_input_embeddings().weight.shape[0] == LEN_TOKENIZER + 1 + MAX_NEW_GRID_TOKENS
    with pytest.raises(ValueError, match="上限"):
        ad.add_grid_tokens(["<extra>"])


# ---------------------------------------------------------------------------
# 新增 grid token
# ---------------------------------------------------------------------------


def _forbid_resize(monkeypatch, ad):
    def boom(*a, **k):
        raise AssertionError("不许调用 resize_token_embeddings（D9-3）")

    monkeypatch.setattr(ad.model, "resize_token_embeddings", boom)


@pytest.mark.parametrize("tied", [True, False])
def test_add_grid_tokens_reuse_reserved_rows(fresh, monkeypatch, tied):
    ad = fresh(tied)
    _forbid_resize(monkeypatch, ad)
    emb = ad.model.get_input_embeddings().weight
    head = ad.model.get_output_embeddings().weight
    emb0, head0 = emb.detach().clone(), head.detach().clone()
    ids = ad.add_grid_tokens(["<g0>", "<g1>"])
    assert ids == {"<g0>": LEN_TOKENIZER, "<g1>": LEN_TOKENIZER + 1}  # 与 model_notes §5.1 一致
    assert emb.shape[0] == head.shape[0] == N_ROWS  # 不缩表、不扩表
    tok = ad.tokenizer
    assert tok("<g0>", add_special_tokens=False)["input_ids"] == [LEN_TOKENIZER]
    assert tok("<g1><g0><g1>", add_special_tokens=False)["input_ids"] == [LEN_TOKENIZER + 1, LEN_TOKENIZER, LEN_TOKENIZER + 1]
    assert tok("a<g0>b", add_special_tokens=False)["input_ids"][1] == LEN_TOKENIZER
    new = list(ids.values())
    emb_mean = emb0[:N_BASE_VOCAB].double().mean(0).float()
    assert torch.equal(emb[new], emb_mean.expand(2, -1))
    keep = torch.ones(N_ROWS, dtype=torch.bool)
    keep[new] = False
    assert torch.equal(emb[keep], emb0[keep])  # 其他行逐位不变
    if tied:
        assert head is emb
    else:
        head_mean = head0[:N_BASE_VOCAB].double().mean(0).float()
        assert torch.equal(head[new], head_mean.expand(2, -1))
        assert not torch.equal(head_mean, emb_mean)
        assert torch.equal(head[keep], head0[keep])


def test_grid_budget_is_cumulative(fresh):
    ad = fresh()
    ad.add_grid_tokens([f"<a{i}>" for i in range(200)])
    n_tok, rows = len(ad.tokenizer), ad.model.get_input_embeddings().weight.shape[0]
    with pytest.raises(ValueError, match="累计"):
        ad.add_grid_tokens([f"<b{i}>" for i in range(200)])
    assert (len(ad.tokenizer), ad.model.get_input_embeddings().weight.shape[0], ad.n_grid_tokens) == (n_tok, rows, 200)
    ad.add_grid_tokens([f"<c{i}>" for i in range(100)])
    assert ad.n_grid_tokens == MAX_NEW_GRID_TOKENS
    with pytest.raises(ValueError, match="上限"):
        ad.add_grid_tokens(["<d0>"])


@pytest.mark.parametrize("tied", [True, False])
def test_add_250_grows_to_exact_rows(fresh, monkeypatch, images, tied):
    """管理员补充规则：累计需要的行数超过现有行数时扩表到恰好需要的行数；新 token 的行 = 基础行均值。

    注意（已上报待裁）："原有 248320 行逐位不变"按字面不成立：其中 243 个预留行正是前 243 个新 token 的行，
    按 D9-3（预留行原有的值不用）要写成均值。这里钉住的解释是：已有 token 的行（前 len(tokenizer) 行）逐位不变，
    全部新 token 行 = 均值；扩表这一步本身把原有 248320 行逐位拷贝（见下一个测试）。
    """
    ad = fresh(tied)
    _forbid_resize(monkeypatch, ad)
    emb0 = ad.model.get_input_embeddings().weight.detach().clone()
    head0 = ad.model.get_output_embeddings().weight.detach().clone()
    ids = ad.add_grid_tokens([f"<g{i}>" for i in range(250)])
    emb = ad.model.get_input_embeddings().weight
    head = ad.model.get_output_embeddings().weight
    assert emb.shape == head.shape == (LEN_TOKENIZER + 250, 64) == (248327, 64)
    assert ad.model.config.text_config.vocab_size == 248327
    new = list(ids.values())
    assert new == list(range(LEN_TOKENIZER, 248327))
    assert torch.equal(emb[:LEN_TOKENIZER], emb0[:LEN_TOKENIZER])  # 已有 token 的行逐位不变
    emb_mean = emb0[:N_BASE_VOCAB].double().mean(0).float()
    assert torch.equal(emb[new], emb_mean.expand(250, -1))  # 复用的 243 个预留行与扩出的 7 行都 = 均值
    assert not torch.equal(emb[LEN_TOKENIZER:N_ROWS], emb0[LEN_TOKENIZER:N_ROWS])  # 预留行原值不用（D9-3）
    if tied:
        assert head is emb
    else:
        assert torch.equal(head[:LEN_TOKENIZER], head0[:LEN_TOKENIZER])
        assert torch.equal(head[new], head0[:N_BASE_VOCAB].double().mean(0).float().expand(250, -1))
    # 扩出来的 id 能在前向与限制生成里正常使用
    allowed = [ids["<g248>"], ids["<g249>"]]
    assert min(allowed) >= N_ROWS
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=3, allowed_token_ids=allowed)
    assert set(res.token_ids) <= set(allowed) and res.n_generated_tokens == 3


@pytest.mark.parametrize("tied", [True, False])
def test_grow_step_copies_all_existing_rows_bitwise(fresh, tied):
    """扩表这一步：原有 248320 行（含预留行）逐位拷贝，扩出的行 = 基础行均值；要求更少的行数时什么都不做。"""
    ad = fresh(tied)
    emb0 = ad.model.get_input_embeddings().weight.detach().clone()
    head0 = ad.model.get_output_embeddings().weight.detach().clone()
    ad._ensure_rows(N_ROWS + 7)
    emb = ad.model.get_input_embeddings().weight
    head = ad.model.get_output_embeddings().weight
    assert emb.shape == head.shape == (N_ROWS + 7, 64)
    assert torch.equal(emb[:N_ROWS], emb0)
    assert torch.equal(emb[N_ROWS:], emb0[:N_BASE_VOCAB].double().mean(0).float().expand(7, -1))
    if tied:
        assert head is emb
    else:
        assert torch.equal(head[:N_ROWS], head0)
        assert torch.equal(head[N_ROWS:], head0[:N_BASE_VOCAB].double().mean(0).float().expand(7, -1))
    ad._ensure_rows(N_ROWS)
    assert ad.model.get_input_embeddings().weight.shape[0] == N_ROWS + 7


def test_rows_never_shrink(fresh, monkeypatch):
    ad = fresh(False)
    _forbid_resize(monkeypatch, ad)
    ad.add_grid_tokens([f"<g{i}>" for i in range(250)])
    snap = ad.model.get_input_embeddings().weight.detach().clone()
    ad.add_grid_tokens([f"<h{i}>" for i in range(10)])
    emb = ad.model.get_input_embeddings().weight
    assert emb.shape[0] == ad.model.get_output_embeddings().weight.shape[0] == 248337
    assert torch.equal(emb[: snap.shape[0]], snap)
    ad2 = fresh(True)
    ad2.add_grid_tokens(["<g0>"])
    assert ad2.model.get_input_embeddings().weight.shape[0] == N_ROWS  # 行数富余时不动，更不会缩到 len(tokenizer)


def test_add_grid_tokens_rejects_existing_names(fresh):
    ad = fresh()
    for names in (["<think>"], ["<|im_end|>"], "<g0>", [], ["<g0>", "<g0>"], [""]):
        with pytest.raises((ValueError, TypeError)):
            ad.add_grid_tokens(names)
    ad.add_grid_tokens(["<g0>"])
    with pytest.raises(ValueError, match="已在词表"):
        ad.add_grid_tokens(["<g1>", "<g0>"])
    assert len(ad.tokenizer) == LEN_TOKENIZER + 1 and ad.n_grid_tokens == 1


def test_check_grid_token_names_counts_cumulatively():
    check_grid_token_names(["<a>"] * 1 + ["<b>"], n_existing=MAX_NEW_GRID_TOKENS - 2)
    with pytest.raises(ValueError, match="累计"):
        check_grid_token_names(["<a>", "<b>"], n_existing=MAX_NEW_GRID_TOKENS - 1)
    with pytest.raises(TypeError):
        check_grid_token_names("<a>")
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            check_grid_token_names(["<a>"], n_existing=bad)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


def test_allowed_token_ids_restrict_generation(fresh, images):
    ad = fresh()
    ids = list(ad.add_grid_tokens(["<g0>", "<g1>"]).values())
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=6, allowed_token_ids=ids)
    assert set(res.token_ids) <= set(ids) and res.n_generated_tokens == 6
    assert res.text == ad.tokenizer.decode(list(res.token_ids))  # grid token 是特殊 token，但 text 保留它们
    for s in (IM_END, END_OF_TEXT):  # 两个停止 token 都生效（id 从 tokenizer 查）
        stop = ad.token_id(s)
        res = ad.generate(inp, enable_thinking=False, max_new_tokens=6, allowed_token_ids=[stop])
        assert res.token_ids == (stop,) and res.text == ""


def test_allowed_token_ids_validation(adapter, images):
    inp = adapter.build_inputs(images[:1], INSTR_A, "image_first")
    for bad in ([], [N_ROWS], [-1], [5, 5]):
        with pytest.raises(ValueError):
            adapter.generate(inp, enable_thinking=False, max_new_tokens=1, allowed_token_ids=bad)


def test_generate_requires_explicit_keywords(adapter, images):
    inp = adapter.build_inputs(images[:1], INSTR_A, "image_first")
    with pytest.raises(TypeError):
        adapter.generate(inp, max_new_tokens=2)
    with pytest.raises(TypeError):
        adapter.generate(inp, enable_thinking=False)
    with pytest.raises(TypeError):
        adapter.generate(inp, enable_thinking=None, max_new_tokens=2)
    with pytest.raises(ValueError):
        adapter.generate(inp, enable_thinking=False, max_new_tokens=0)
    with pytest.raises(ValueError, match="覆盖"):
        adapter.generate(inp, enable_thinking=False, max_new_tokens=1, eos_token_id=1)


@pytest.mark.parametrize("thinking", [False, True])
def test_generation_result_fields(adapter, images, thinking):
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    res = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=4)
    n_header = len(adapter.assistant_header_ids(thinking))
    assert res.n_prompt_tokens == inp.tensors["input_ids"].shape[1] + n_header
    assert res.n_visual_tokens == 2 * N_IMG_TOK
    assert 1 <= res.n_generated_tokens <= 4 and res.wall_time_s > 0
    assert res.raw_text == adapter.tokenizer.decode(list(res.token_ids), skip_special_tokens=False)
    if thinking and adapter.token_id(THINK_END) not in res.token_ids:
        assert res.thinking_unclosed and res.text == ""
    stop = adapter.token_id(IM_END)
    forced = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=3, allowed_token_ids=[stop])
    assert forced.token_ids == (stop,) and forced.text == "" and forced.thinking_unclosed is thinking


def test_split_answer_rules(adapter, components):
    tok = components[1]
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    end, stop = adapter.token_id(THINK_END), adapter.token_id(IM_END)
    think, ans = enc("Let me look."), enc("Blue.")
    nl = enc("\n\n")
    text, raw, unclosed = adapter.split_answer(think + [end] + nl + ans + [stop], enable_thinking=True)
    assert (text, unclosed) == ("Blue.", False)
    assert raw == "Let me look.</think>\n\nBlue.<|im_end|>" and "Let me look." not in text
    text, _, _ = adapter.split_answer(think + [end] + think + [end] + ans, enable_thinking=True)
    assert text == "Blue."  # 取最后一个 </think> 之后
    text, raw, unclosed = adapter.split_answer(think, enable_thinking=True)
    assert (text, unclosed) == ("", True) and raw == "Let me look."
    text, _, unclosed = adapter.split_answer(ans + [stop], enable_thinking=False)
    assert (text, unclosed) == ("Blue.", False)
    grid = enc("<|box_start|>") + ans  # 特殊 token 保留在 text 里，只去掉末尾的停止 token
    assert adapter.split_answer(grid + [adapter.token_id(END_OF_TEXT)], enable_thinking=False)[0] == "<|box_start|>Blue."


def test_generation_result_new_fields():
    r = GenerationResult("x", (7,), 1, 3, 0, 0.0)
    assert r.raw_text is None and r.thinking_unclosed is False
    with pytest.raises(ValueError, match="未闭合"):
        GenerationResult("x", (7,), 1, 3, 0, 0.0, raw_text="x", thinking_unclosed=True)


# ---------------------------------------------------------------------------
# 入口断言（meta 设备做可观测代理）与 dtype
# ---------------------------------------------------------------------------


def test_entry_assertion_when_model_moved_to_meta(fresh, images):
    ad = fresh()
    inp = ad.build_inputs(images, INSTR_A, "image_first")
    ad.model.to("meta")
    calls = [
        lambda: ad.hidden_states(inp, [1]),
        lambda: ad.sequence_hidden_states(inp, [1]),
        lambda: ad.vision_features(inp),
        lambda: ad.generate(inp, enable_thinking=False, max_new_tokens=1),
    ]
    for call in calls:
        with pytest.raises(AdapterDeviceError) as e:
            call()
        assert "meta" in str(e.value) and "cpu" in str(e.value)


def test_entry_assertion_when_input_on_meta(adapter, images):
    for key in ("input_ids", "pixel_values", "attention_mask"):
        inp = adapter.build_inputs(images, INSTR_A, "image_first")
        inp.tensors[key] = inp.tensors[key].to("meta")
        with pytest.raises(AdapterDeviceError) as e:
            adapter.hidden_states(inp, [1])
        assert key in str(e.value) and "meta" in str(e.value) and "cpu" in str(e.value)


def test_entry_assertion_dtype(adapter, images):
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    inp.tensors["pixel_values"] = inp.tensors["pixel_values"].to(torch.bfloat16)
    for call in (lambda: adapter.vision_features(inp), lambda: adapter.generate(inp, enable_thinking=False, max_new_tokens=1)):
        with pytest.raises(AdapterDeviceError) as e:
            call()
        assert "bfloat16" in str(e.value) and "float32" in str(e.value)


def test_bf16_model(components, images):
    model = qwen35.build_tiny_model(components[0], tie_word_embeddings=True, dtype=torch.bfloat16)
    ad = qwen35.make_tiny_adapter(components, image_size=IMG_HW, model=model)
    assert ad.dtype == torch.bfloat16
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    assert inp.tensors["pixel_values"].dtype == torch.bfloat16
    with torch.no_grad():
        hs = ad.hidden_states(inp, [0, 4])
        vf = ad.vision_features(inp)
    assert hs.image[4].dtype == torch.bfloat16 and torch.equal(vf, hs.image[0])


# ---------------------------------------------------------------------------
# 冒烟脚本的 CPU 彩排（端到端，与服务器同一流程）
# ---------------------------------------------------------------------------


def test_smoke_script_tiny(tmp_path):
    out = tmp_path / "smoke_tiny.json"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/smoke_adapter.py"), "--tiny", "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert out.exists(), proc.stderr[-3000:]
    report = json.loads(out.read_text())
    assert proc.returncode == 0 and report["all_ok"] is True, json.dumps(report.get("ok"), ensure_ascii=False)
    assert list(report["ok"]) == [
        "0_facts", "1_build_generate", "2_task_modulation", "3_grid_tokens", "4_query_tokens",
        "5_position_ids", "6_vision_features", "7_thinking", "8_device_audit",
    ]
