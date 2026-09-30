"""Gemma 4 适配器（DEV-0004）的 CPU 测试。覆盖面照 tests/test_adapter_qwen35.py。

模型用本地构造的小型随机 Gemma 4：取固定 revision 的真实 config，缩到 4 层（sliding / full 交替，后 2 层 KV 共享）、
hidden 64、PLE 维度 8、视觉塔 1 层；tokenizer / 图像处理器（PIL 后端）/ 聊天模板 / generation_config 从固定 revision
下载（小文件，不含权重）。取不到就失败并说明原因，不静默跳过。fixture 的两张图一横一竖（H≠W，网格互为转置），
内容不同；两条指令内容不同、token 长度相同。
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import gemma4, registry
from src.adapters.base import ModelAdapter
from src.adapters.gemma4 import (
    BOI,
    BOS,
    CHANNEL_CLOSE,
    CHANNEL_OPEN,
    EOI,
    EOS,
    IMAGE_TOKEN,
    PAD,
    THINK,
    TURN_CLOSE,
    TURN_OPEN,
    AdapterDeviceError,
    Gemma4Adapter,
    box2d_to_xyxy,
    collapse_image_blocks,
    grid_from_position_ids,
)
from src.adapters.qwen35 import AdapterDeviceError as QwenAdapterDeviceError
from src.constants import MAX_NEW_GRID_TOKENS
from src.executors.geometry import box_iou

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_P4 = REPO_ROOT / "experiments/P0/0001_env_probe/raw/p4/google__gemma-4-E2B-it"
RAW_GEMMA = REPO_ROOT / "experiments/P0/0002_family_probe/raw/gemma4-e2b-it"

MST = gemma4.TINY_MAX_SOFT_TOKENS  # 70
INSTR_A = "Describe the cat."
INSTR_B = "Describe the sofa."
GRID_A, GRID_B = (6, 10), (10, 6)  # 100×150 横图、150×100 竖图在 70 档下的网格
N_IMG_TOK = 60

# E2B-it 固定 revision 的词表事实（docs/model_notes.md §6.1），测试里钉住；适配器本身不写死它们
N_ROWS = 262144
N_UNUSED = 6227
FIRST_UNUSED, LAST_UNUSED = 6, 262143
UNUSED_RUNS = ((6, 45), (53, 97), (99, 99), (102, 104), (256001, 258879), (258885, 262143))  # 闭区间（§6.1）
UNUSED_IDS = frozenset(i for a, b in UNUSED_RUNS for i in range(a, b + 1))
SPECIAL_IDS = (0, 1, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 106,
               255999, 256000, 258880, 258881, 258882, 258883, 258884)
N_BASE = N_ROWS - N_UNUSED - len(SPECIAL_IDS)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def components():
    entry = registry.get_model(gemma4.TINY_SOURCE_KEY)
    try:
        return gemma4.load_pinned_components(entry.key)
    except Exception as e:  # noqa: BLE001 — 取不到小文件时给出原因后失败，不跳过
        pytest.fail(
            f"取不到 {entry.repo}@{entry.revision} 的 config/tokenizer/处理器/模板/generation_config 小文件：{e!r}。"
            "测试需要联网下载一次（之后走本地 HF 缓存）。",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def base_model(components):
    return gemma4.build_tiny_model(components.config, generation_config=components.generation_config)


def _adapter(components, base_model, *, fresh_tokenizer=False, max_soft_tokens=MST, model=None):
    comps = components
    if fresh_tokenizer:
        comps = comps._replace(tokenizer=copy.deepcopy(comps.tokenizer))
    m = model if model is not None else copy.deepcopy(base_model)
    return gemma4.make_tiny_adapter(comps, max_soft_tokens=max_soft_tokens, model=m)


@pytest.fixture(scope="module")
def adapter(components, base_model):
    """只读共享：不许在它上面加 token 或注入查询（用例结束时检查）。"""
    ad = _adapter(components, base_model)
    yield ad
    assert ad.n_grid_tokens == 0 and ad.query_token_id is None, "共享适配器被改动过"
    assert len(ad.tokenizer) == N_ROWS and len(ad.tokenizer.added_tokens_decoder) == len(SPECIAL_IDS)


@pytest.fixture
def fresh(components, base_model):
    """每次一个独立的适配器（tokenizer 副本 + 模型副本），可以加 token、注入查询。"""
    return lambda max_soft_tokens=MST: _adapter(components, base_model, fresh_tokenizer=True,
                                                max_soft_tokens=max_soft_tokens)


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


def _marks(ad):
    return {"boi": ad.token_id(BOI), "image": ad.token_id(IMAGE_TOKEN), "eoi": ad.token_id(EOI)}


def _content(n_images, instruction, order):
    imgs = [{"type": "image"}] * n_images
    text = {"type": "text", "text": instruction}
    return [text, *imgs] if order == "instruction_first" else [*imgs, text]


def _base_mean(weight):
    """测试里独立算的基础行均值：全部行去掉 model_notes §6.1 登记的 <unusedN> 区间与 24 个特殊 token（不看适配器）。"""
    keep = torch.tensor(sorted(set(range(weight.shape[0])) - UNUSED_IDS - set(SPECIAL_IDS)))
    assert keep.numel() == N_BASE
    return weight.detach()[keep].double().mean(0).to(weight.dtype)


# ---------------------------------------------------------------------------
# 注册表、固定 revision 的小文件与 model_notes 的事实
# ---------------------------------------------------------------------------


def test_registry_gemma_keys_and_load_rejects_other_families():
    assert registry.keys_for_family(registry.FAMILY_GEMMA4) == ("gemma-4-e2b-it", "gemma-4-e4b-it")
    ad = Gemma4Adapter(device="cpu", max_soft_tokens=MST)
    with pytest.raises(KeyError):
        ad.load("google/gemma-4-E2B-it", torch.float32)  # 裸仓库名不行：必须走注册表的 revision
    with pytest.raises(KeyError):
        ad.load("gemma4-e2b-it", torch.float32)  # 任务书里的写法不是注册键（注册键是 gemma-4-e2b-it）
    with pytest.raises(ValueError, match="qwen3.5"):
        ad.load("qwen3.5-0.8b", torch.float32)
    with pytest.raises(TypeError):
        ad.load("gemma-4-e2b-it", "bf16")


def test_pinned_files_match_run0001_copies(components):
    config, tok, _, template, gen = components
    assert template == (RAW_P4 / "chat_template.jinja").read_text()
    raw = json.loads((RAW_P4 / "config.json").read_text())
    for key in ("image_token_id", "boi_token_id", "eoi_token_id", "tie_word_embeddings"):
        assert getattr(config, key) == raw[key], key
    tc, rtc = config.text_config, raw["text_config"]
    for key in ("vocab_size", "vocab_size_per_layer_input", "hidden_size_per_layer_input", "num_kv_shared_layers",
                "num_hidden_layers", "use_bidirectional_attention", "pad_token_id"):
        assert getattr(tc, key) == rtc[key], key
    assert list(tc.layer_types) == rtc["layer_types"]
    assert tc.num_hidden_layers + 1 == 36  # output_hidden_states 36 个张量（§6.1）
    raw_tok = json.loads((RAW_P4 / "tokenizer_config.json").read_text())
    assert tok.response_template["fields"]["thinking"] == raw_tok["response_template"]["fields"]["thinking"]
    assert raw_tok["response_template"]["fields"]["thinking"]["open"] == gemma4.THOUGHT_OPEN
    assert raw_tok["response_template"]["fields"]["thinking"]["close"] == CHANNEL_CLOSE
    assert (raw_tok["soc_token"], raw_tok["eoc_token"], raw_tok["eot_token"]) == (CHANNEL_OPEN, CHANNEL_CLOSE, TURN_CLOSE)
    raw_gen = json.loads((RAW_P4 / "generation_config.json").read_text())
    assert gen.do_sample is True and raw_gen["do_sample"] is True and gen.top_k == raw_gen["top_k"] == 64
    assert len(tok) == tok.vocab_size == N_ROWS


def test_tokenizer_facts_match_model_notes(components):
    tok = components.tokenizer
    assert sorted(tok.added_tokens_decoder) == list(SPECIAL_IDS)
    unused = sorted(i for s, i in tok.get_vocab().items() if gemma4.UNUSED_TOKEN_RE.fullmatch(s))
    assert (len(unused), unused[0], unused[-1]) == (N_UNUSED, FIRST_UNUSED, LAST_UNUSED)
    assert set(unused) == UNUSED_IDS  # 与 §6.1 登记的 6 段 id 逐个相同
    assert tok("<unused0>", add_special_tokens=False)["input_ids"] != [FIRST_UNUSED]  # 注册前会被切开（§6.1）
    for s, i in ((TURN_CLOSE, 106), (EOS, 1), (CHANNEL_OPEN, 100), (CHANNEL_CLOSE, 101), (IMAGE_TOKEN, 258880)):
        assert tok.convert_tokens_to_ids(s) == i  # 与 §6.1 的表一致


def test_max_soft_tokens_choices_match_notes_and_source():
    from transformers.models.gemma4.image_processing_pil_gemma4 import _SUPPORTED_SOFT_TOKENS

    assert gemma4.MAX_SOFT_TOKENS_CHOICES == tuple(_SUPPORTED_SOFT_TOKENS) == (70, 140, 280, 560, 1120)


@pytest.mark.parametrize("bad", [0, 100, 279, 2240, 280.0, True, "280", None])
def test_max_soft_tokens_must_be_a_supported_budget(bad):
    with pytest.raises(ValueError, match="max_soft_tokens"):
        Gemma4Adapter(device="cpu", max_soft_tokens=bad)


@pytest.mark.parametrize("good", [70, 140, 280, 560, 1120, np.int64(280)])
def test_max_soft_tokens_accepts_the_five_budgets(good):
    assert Gemma4Adapter(device="cpu", max_soft_tokens=good).max_soft_tokens == int(good)


# ---------------------------------------------------------------------------
# 小模型的结构与挂载
# ---------------------------------------------------------------------------


def test_tiny_config_has_sliding_full_and_kv_sharing(base_model, components):
    tc = base_model.config.text_config
    assert tc.num_hidden_layers == 4 and set(tc.layer_types) == {"sliding_attention", "full_attention"}
    layers = base_model.model.language_model.layers
    shared = [i for i, layer in enumerate(layers) if layer.self_attn.is_kv_shared_layer]
    assert shared == [2, 3] and {layers[i].self_attn.layer_type for i in shared} == {"sliding_attention", "full_attention"}
    assert all(not hasattr(layers[i].self_attn, "k_proj") for i in shared)
    assert tc.hidden_size_per_layer_input == 8 < components.config.text_config.hidden_size_per_layer_input
    assert base_model.model.vision_tower.config.num_hidden_layers == 1
    assert tc.vocab_size == N_ROWS and tc.sliding_window == components.config.text_config.sliding_window
    assert base_model.model.audio_tower is None
    assert base_model.config.vision_config.patch_size == 16 and base_model.config.vision_config.pooling_kernel_size == 3


def test_tiny_from_e4b_config_attaches_and_matches_model_path(components, images):
    """E4B（42 层、18 层 KV 共享、不用 double-wide MLP）的固定 revision config（RUN-0001 的副本，不联网）缩出来的小模型：
    挂载的各项断言照样成立，适配器路径与 HF 原生路径逐位相同。模板、processor、tokenizer_config 两个尺寸逐字相同（§3.2）。"""
    raw = json.loads((RAW_P4.parent / "google__gemma-4-E4B-it/config.json").read_text())
    assert raw["text_config"]["use_double_wide_mlp"] is False and raw["text_config"]["num_kv_shared_layers"] == 18
    e4b = type(components.config).from_dict(raw)
    model = gemma4.build_tiny_model(e4b)
    assert model.config.text_config.use_double_wide_mlp is False
    ad = gemma4.make_tiny_adapter(components, model=model)
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    with torch.no_grad():
        ours = ad.sequence_hidden_states(inp, _all_layers(ad))
        auto = ad.model(**inp.tensors, output_hidden_states=True, logits_to_keep=1, use_cache=False).hidden_states
        vf = ad.vision_features(inp)
    assert all(torch.equal(ours[l], auto[l][0]) for l in _all_layers(ad))
    assert torch.equal(vf, ours[0][torch.cat([torch.arange(i.span.start, i.span.end) for i in inp.images])])


def test_attach_reads_facts_from_tokenizer_and_config(adapter, components, base_model):
    tok = components.tokenizer
    assert adapter.num_layers() == 4 and adapter.hidden_size() == 64 and adapter.dtype == torch.float32
    assert adapter.tied_word_embeddings is True and adapter.n_base_vocab == N_BASE
    assert adapter.unused_token_ids[0] == FIRST_UNUSED and adapter.unused_token_ids[-1] == LAST_UNUSED
    for s in (BOS, EOS, PAD, TURN_OPEN, TURN_CLOSE, IMAGE_TOKEN, BOI, EOI, THINK, CHANNEL_OPEN, CHANNEL_CLOSE):
        assert adapter.token_id(s) == tok.convert_tokens_to_ids(s)
    assert adapter.token_id(IMAGE_TOKEN) == components.config.image_token_id
    with pytest.raises(RuntimeError, match="已经挂过"):
        adapter.attach(copy.deepcopy(base_model), tokenizer=tok, image_processor=components.image_processor,
                       chat_template=components.chat_template)
    assert AdapterDeviceError is QwenAdapterDeviceError  # 跨家族同一种入口断言异常


def test_attach_rejects_non_sdpa_untied_and_mismatched_processor(components, base_model):
    m = copy.deepcopy(base_model)
    m.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="sdpa"):
        gemma4.make_tiny_adapter(components, model=m)
    m = copy.deepcopy(base_model)
    m.lm_head.weight = torch.nn.Parameter(m.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="共享"):
        gemma4.make_tiny_adapter(components, model=m)
    ip = copy.deepcopy(components.image_processor)
    ip.pooling_kernel_size = 2
    with pytest.raises(ValueError, match="pooling_kernel_size"):
        gemma4.make_tiny_adapter(components._replace(image_processor=ip), model=copy.deepcopy(base_model))


def test_attach_rejects_tokenizer_with_registered_unused(fresh, components, base_model):
    ad = fresh()
    ad.add_grid_tokens(["<g0>"])
    with pytest.raises(ValueError, match="注册过"):
        gemma4.make_tiny_adapter(components._replace(tokenizer=ad.tokenizer), model=copy.deepcopy(base_model))


# ---------------------------------------------------------------------------
# build_inputs：区间、顺序、网格、模板
# ---------------------------------------------------------------------------


def test_spans_instruction_first(adapter, images, components):
    tok = components.tokenizer
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    ids = inp.tensors["input_ids"][0].tolist()
    # [<bos>, <|turn>, user, \n, 指令×4, <|image>, 图×60, <image|>, <|image>, 图×60, <image|>, <turn|>, \n]
    assert len(ids) == 134
    assert (inp.instruction_span.start, inp.instruction_span.end) == (4, 8)
    assert [(i.span.start, i.span.end) for i in inp.images] == [(9, 69), (71, 131)]
    assert [i.grid_hw for i in inp.images] == [GRID_A, GRID_B]
    assert ids[4:8] == tok(INSTR_A, add_special_tokens=False)["input_ids"]
    assert ids[:4] == tok("<bos><|turn>user\n", add_special_tokens=False)["input_ids"]
    boi, eoi = adapter.token_id(BOI), adapter.token_id(EOI)
    assert [ids[8], ids[69], ids[70], ids[131]] == [boi, eoi, boi, eoi]
    assert set(ids[9:69]) == set(ids[71:131]) == {adapter.token_id(IMAGE_TOKEN)}
    assert ids[132:] == [adapter.token_id(TURN_CLOSE), tok("\n", add_special_tokens=False)["input_ids"][0]]
    assert inp.order == "instruction_first" and inp.query_span is None


def test_spans_image_first(adapter, images, components):
    tok = components.tokenizer
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    ids = inp.tensors["input_ids"][0].tolist()
    # [<bos>, <|turn>, user, \n, <|image>, 图×60, <image|>, <|image>, 图×60, <image|>, 指令×4, <turn|>, \n]
    assert len(ids) == 134
    assert [(i.span.start, i.span.end) for i in inp.images] == [(5, 65), (67, 127)]
    assert (inp.instruction_span.start, inp.instruction_span.end) == (128, 132)
    assert ids[128:132] == tok(INSTR_A, add_special_tokens=False)["input_ids"]
    assert ids[132] == adapter.token_id(TURN_CLOSE)


def test_grid_from_image_position_ids_x_is_dim0(adapter, images):
    """横图（H=100, W=150）→ 列 patch 多于行 patch：第 0 维是列 x、第 1 维是行 y（§6.3），网格 (h, w) = (6, 10)。"""
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    pos = inp.tensors["image_position_ids"]
    valid = (pos[0] >= 0).all(-1)
    assert int(pos[0][valid][:, 0].max()) + 1 == 30 and int(pos[0][valid][:, 1].max()) + 1 == 18  # x: 30 列，y: 18 行
    assert [i.grid_hw for i in inp.images] == [GRID_A, GRID_B]  # 竖图正好转置
    assert pos[0][:3].tolist() == [[0, 0], [1, 0], [2, 0]]  # x 变化最快（与 §6.4 的实测前几行相同）
    from transformers.models.gemma4.image_processing_pil_gemma4 import get_aspect_ratio_preserving_size

    th, tw = get_aspect_ratio_preserving_size(100, 150, 16, MST * 9, 3)
    assert (th // 48, tw // 48) == GRID_A


def test_grid_from_position_ids_validation():
    k = 3
    xy = torch.stack([torch.arange(6).repeat(3), torch.arange(3).repeat_interleave(6)], dim=1)  # 3 行 6 列 patch
    pos = torch.cat([xy, torch.full((9, 2), -1)])
    assert grid_from_position_ids(pos, k) == (1, 2)
    bad_half = pos.clone()
    bad_half[-1, 0] = 0
    swapped = torch.cat([xy.flip(1), torch.full((9, 2), -1)])  # 当成 (y, x) 读就不是 x 最快的网格
    hole = torch.cat([torch.full((1, 2), -1), xy, torch.full((8, 2), -1)])
    for bad in (bad_half, swapped, hole):
        with pytest.raises(RuntimeError):
            grid_from_position_ids(bad, k)
    with pytest.raises(RuntimeError, match="倍数"):
        grid_from_position_ids(pos, 4)


def test_pooled_token_order_is_row_major():
    """transformers 的 pooler：输出第 r·w + c 个 token 是第 r 行、第 c 列的 3×3 patch 均值（ImageTokenInfo 的行优先约定依赖它）。"""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler

    n_x, n_y, k = 12, 6, 3
    xy = torch.stack([torch.arange(n_x).repeat(n_y), torch.arange(n_y).repeat_interleave(n_x)], dim=1)
    feats = xy.float().unsqueeze(0)  # 每个 patch 的特征就是它的 (x, y)
    out, mask = Gemma4VisionPooler._avg_pool_by_positions(None, feats, xy.unsqueeze(0), (n_x // k) * (n_y // k))
    w = n_x // k
    for j in range(out.shape[1]):
        r, c = divmod(j, w)
        assert out[0, j].tolist() == pytest.approx([k * c + 1.0, k * r + 1.0], abs=1e-5), j  # 权重是 1/9，有舍入
    assert bool(mask.all())


@pytest.mark.parametrize("budget,grid", [(70, (7, 9)), (140, (10, 13)), (280, (14, 19)), (560, (20, 27)),
                                         (1120, (28, 38))])
def test_budget_to_grid_matches_model_notes_table(components, base_model, budget, grid):
    """model_notes §6.3 表：640×480 在五个档位下的 LLM 网格（服务器实测）。"""
    ad = _adapter(components, base_model, max_soft_tokens=budget, model=base_model)
    img = np.zeros((480, 640, 3), np.uint8)
    inp = ad.build_inputs([img], INSTR_A, "image_first")
    assert inp.images[0].grid_hw == grid and len(inp.images[0].span) == grid[0] * grid[1]
    assert tuple(inp.tensors["pixel_values"].shape) == (1, budget * 9, 768)


def test_tensors_and_pixel_values(adapter, images):
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    t = inp.tensors
    ids = t["input_ids"]
    assert set(t) == {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_position_ids"}
    assert torch.equal(t["attention_mask"], torch.ones_like(ids))
    assert t["mm_token_type_ids"].dtype == torch.long
    assert torch.equal(t["mm_token_type_ids"], (ids == adapter.token_id(IMAGE_TOKEN)).long())
    assert t["mm_token_type_ids"].sum().item() == 2 * N_IMG_TOK
    pv = t["pixel_values"]
    assert pv.shape == (2, MST * 9, 3 * 16 * 16) and pv.dtype == adapter.dtype and pv.device == adapter.device
    assert t["image_position_ids"].dtype == torch.long and t["image_position_ids"].shape == (2, MST * 9, 2)
    assert not torch.equal(pv[0], pv[1])


def test_numpy_and_pil_agree(adapter, images):
    a_np = images[0]
    x = adapter.build_inputs([a_np], INSTR_A, "image_first").tensors["pixel_values"]
    y = adapter.build_inputs([Image.fromarray(a_np)], INSTR_A, "image_first").tensors["pixel_values"]
    assert torch.equal(x, y)
    assert adapter.prepare_image(Image.fromarray(a_np).convert("L")).mode == "RGB"


def test_prompt_stops_at_user_turn(adapter, images):
    ids = adapter.build_inputs(images, INSTR_A, "instruction_first").tensors["input_ids"][0].tolist()
    assert ids.count(adapter.token_id(TURN_OPEN)) == 1 and ids[0] == adapter.token_id(BOS)
    assert ids[-2] == adapter.token_id(TURN_CLOSE)
    assert adapter.token_id(THINK) not in ids


def test_thinking_prefix_and_header_match_model_notes(adapter):
    dec = adapter.tokenizer.decode
    assert adapter.thinking_prefix_ids(False) == ()
    assert dec(list(adapter.thinking_prefix_ids(True))) == "<|turn>system\n<|think|>\n<turn|>\n"
    assert dec(list(adapter.assistant_header_ids(False))) == "<|turn>model\n"
    assert adapter.assistant_header_ids(True) == adapter.assistant_header_ids(False)
    with pytest.raises(TypeError):
        adapter.thinking_prefix_ids(1)
    with pytest.raises(TypeError):
        adapter.assistant_header_ids(None)


@pytest.mark.parametrize("order", ["instruction_first", "image_first"])
@pytest.mark.parametrize("thinking", [False, True])
def test_prompt_equals_chat_template(adapter, images, order, thinking):
    inp = adapter.build_inputs(images, INSTR_A, order)
    full = adapter.prompt_ids(inp, enable_thinking=thinking)[0].tolist()
    ref = adapter.tokenizer.apply_chat_template(
        [{"role": "user", "content": _content(len(images), INSTR_A, order)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=thinking,
    )["input_ids"]
    assert collapse_image_blocks(full, **_marks(adapter)) == list(ref)


def test_prompt_renders_match_run0002_facts(adapter, images):
    """RUN-0002 的服务器实测渲染（facts.json）：纯文本与带图两种、思考开与关。"""
    renders = json.loads((RAW_GEMMA / "facts.json").read_text())["chat_template_renders"]
    dec = adapter.tokenizer.decode
    for thinking in (False, True):
        key = f"enable_thinking={thinking}"
        txt = adapter.build_inputs([], "What is 2+3?", "instruction_first")
        assert dec(adapter.prompt_ids(txt, enable_thinking=thinking)[0].tolist()) == renders[f"{key} | text"]["text"]
        img = adapter.build_inputs(images[:1], "Describe the image.", "image_first")
        ids = collapse_image_blocks(adapter.prompt_ids(img, enable_thinking=thinking)[0].tolist(), **_marks(adapter))
        assert dec(ids) == renders[f"{key} | image"]["text"]
    # 默认（不传开关）与关闭相同（§6.2），适配器总是显式传
    assert renders["default(no kwarg) | text"]["text"] == renders["enable_thinking=False | text"]["text"]


def test_collapse_image_blocks_rules():
    m = {"boi": 5, "image": 7, "eoi": 6}
    assert collapse_image_blocks([1, 5, 7, 7, 7, 6, 2, 5, 7, 6], **m) == [1, 7, 2, 7]
    for bad in ([5, 6], [5, 7, 7], [5, 7, 8]):
        with pytest.raises(ValueError):
            collapse_image_blocks(bad, **m)


def test_instruction_validation(adapter, images):
    with pytest.raises(ValueError, match="trim"):
        adapter.build_inputs(images, INSTR_A + " ", "image_first")  # 模板会 trim 每段文本
    with pytest.raises(ValueError, match="trim"):
        adapter.build_inputs(images, "\n" + INSTR_A, "instruction_first")
    for s in ("look <|image|> here", "a <turn|> b", "x<|channel>y", "<bos>hi", "<|video|>"):
        with pytest.raises(ValueError, match="结构性"):
            adapter.build_inputs(images, s, "image_first")
    with pytest.raises(ValueError):
        adapter.build_inputs(images, INSTR_A, "image_last")
    with pytest.raises(TypeError):
        adapter.build_inputs(images[0], INSTR_A, "image_first")  # 单张图也要放进列表
    with pytest.raises(ValueError):
        adapter.build_inputs([np.zeros((10, 10), np.uint8)], INSTR_A, "image_first")
    with pytest.raises(TypeError):
        adapter.build_inputs(images, 3, "image_first")


def test_empty_instruction_and_zero_images(adapter, images):
    for order in ("instruction_first", "image_first"):
        inp = adapter.build_inputs(images, "", order)
        assert len(inp.instruction_span) == 0
    inp = adapter.build_inputs([], INSTR_A, "instruction_first")
    assert inp.images == () and "pixel_values" not in inp.tensors and "image_position_ids" not in inp.tensors
    L = inp.tensors["input_ids"].shape[1]
    assert torch.equal(adapter.position_ids(inp), torch.arange(L).unsqueeze(0))
    with torch.no_grad():
        hs = adapter.hidden_states(inp, [0, 4])
    assert hs.image[4].shape == (0, 64) and adapter.vision_features(inp).shape == (0, 64)
    res = adapter.generate(inp, enable_thinking=False, max_new_tokens=2)
    assert res.n_visual_tokens == 0 and 1 <= res.n_generated_tokens <= 2


# ---------------------------------------------------------------------------
# 隐藏状态：层号、仪器、前向路径
# ---------------------------------------------------------------------------


def test_instruction_pair_has_equal_token_length(components):
    tok = components.tokenizer
    a = tok(INSTR_A, add_special_tokens=False)["input_ids"]
    b = tok(INSTR_B, add_special_tokens=False)["input_ids"]
    assert len(a) == len(b) == 4 and a != b  # 与 §6.6 的 [4, 4] 一致


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
    assert got["image_first"] == [True] * len(layers)  # 精确为零的仪器（PLE 在图像位置与指令无关）
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
    ids = inp.tensors["input_ids"][0]
    with torch.no_grad():
        emb = lm.embed_tokens.weight[ids] * lm.embed_tokens.embed_scale
        normed = lm.norm(raw[n - 1])[0]
    text_pos = inp.instruction_span.start
    assert torch.equal(hs[0][text_pos], emb[text_pos])  # 层号 0：嵌入输出（文本位置是查表值 × embed_scale）
    for i in range(1, n):
        assert torch.equal(hs[i], raw[i - 1][0]), i  # 层号 i：第 i 个 decoder 层的输出
    assert torch.equal(hs[n], normed)  # 最后一个层号：最终归一化之后（HF 惯例，D15-2）
    assert not torch.equal(hs[n], raw[n - 1][0])
    assert len(hs) == n + 1


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
def test_own_path_equals_model_path(adapter, images, order):
    """适配器的路径（inputs_embeds + per_layer_inputs + 自构一维位置）与 HF 原生路径（input_ids + pixel_values）逐位相同。"""
    inp = adapter.build_inputs(images, INSTR_A, order)
    t = inp.tensors
    L = t["input_ids"].shape[1]
    pos = adapter.position_ids(inp)
    assert pos.shape == (1, L) and torch.equal(pos[0], torch.arange(L))
    layers = _all_layers(adapter)
    with torch.no_grad():
        ours = adapter.sequence_hidden_states(inp, layers)
        auto = adapter.model(**t, output_hidden_states=True, logits_to_keep=1, use_cache=False)
        given = adapter.model(**t, position_ids=pos, output_hidden_states=True, logits_to_keep=1, use_cache=False)
    assert len(auto.hidden_states) == adapter.num_layers() + 1
    for l in layers:
        assert torch.equal(auto.hidden_states[l][0], given.hidden_states[l][0]), l
        assert torch.equal(ours[l], auto.hidden_states[l][0]), l


def test_own_path_equals_model_path_with_active_sliding_window(components, images):
    """滑动窗口（真值 512）比彩排序列短得多；改成 16 让截断真的发生，两条路径仍逐位相同，且确实不同于全窗口。"""
    small = gemma4.build_tiny_model(components.config, sliding_window=16)
    ad = gemma4.make_tiny_adapter(components, model=small)
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    with torch.no_grad():
        ours = ad.sequence_hidden_states(inp, _all_layers(ad))
        auto = ad.model(**inp.tensors, output_hidden_states=True, logits_to_keep=1, use_cache=False).hidden_states
        wide = gemma4.make_tiny_adapter(components, model=gemma4.build_tiny_model(components.config))
        full = wide.sequence_hidden_states(wide.build_inputs(images, INSTR_A, "instruction_first"), [1])
    for l in _all_layers(ad):
        assert torch.equal(ours[l], auto[l][0]), l
    assert not torch.equal(ours[1][-1], full[1][-1])  # 同样的权重（同 seed），窗口不同，结果不同


# ---------------------------------------------------------------------------
# vision_features 与第 0 层的精确关系（按源码；模块 docstring）
# ---------------------------------------------------------------------------


def test_vision_features_equal_layer0_image_positions(adapter, images):
    for order in ("instruction_first", "image_first"):
        inp = adapter.build_inputs(images, INSTR_A, order)
        with torch.no_grad():
            vf = adapter.vision_features(inp)
            h0 = adapter.hidden_states(inp, [0]).image[0]
            direct = torch.cat(adapter.model.model.get_image_features(
                inp.tensors["pixel_values"], inp.tensors["image_position_ids"], return_dict=True).pooler_output)
        assert vf.shape == (2 * N_IMG_TOK, 64) and torch.equal(vf, h0)  # 逐位相同：视觉特征不乘 embed_scale
        assert torch.equal(vf, direct)  # 就是 embed_vision 的输出
        assert not torch.equal(vf[:N_IMG_TOK], vf[N_IMG_TOK:])
    scale = float(adapter.model.model.language_model.embed_tokens.embed_scale)
    assert scale == math.sqrt(64) and not torch.equal(vf * scale, h0)  # embed_scale ≠ 1，若误乘必被抓住


def test_embed_scale_applies_to_text_positions_only(adapter, images):
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    ids = inp.tensors["input_ids"][0]
    img = ids == adapter.token_id(IMAGE_TOKEN)
    emb = adapter.model.get_input_embeddings()
    with torch.no_grad():
        seq0 = adapter.sequence_hidden_states(inp, [0])[0]
    assert torch.equal(seq0[~img], emb.weight[ids[~img]] * emb.embed_scale)
    assert not torch.equal(seq0[img], emb.weight[ids[img]] * emb.embed_scale)


def test_ple_at_image_positions_uses_pad_row_and_skips_layer0(adapter, images):
    """PLE：图像位置的 per-layer input = 上下文投影(视觉特征) 与 pad 行的 token 身份合成；它从 decoder 层内部进入，不改第 0 层。"""
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    lm = adapter.model.model.language_model
    ids = inp.tensors["input_ids"]
    img = ids[0] == adapter.token_id(IMAGE_TOKEN)
    got = {}

    def grab(mod, args, kwargs):
        got["ple"] = args[1]

    h = lm.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    try:
        with torch.no_grad():
            seq0 = adapter.sequence_hidden_states(inp, [0])[0]
    finally:
        h.remove()
    with torch.no_grad():
        pad_ids = torch.where(ids == adapter.token_id(IMAGE_TOKEN), adapter.token_id(PAD), ids)
        want = lm.project_per_layer_inputs(seq0[None], lm.get_per_layer_inputs(pad_ids, None))
        alt = lm.project_per_layer_inputs(seq0[None], lm.get_per_layer_inputs(ids, None))
    assert got["ple"].shape == (1, ids.shape[1], 8)
    assert torch.equal(got["ple"][0], want[0, :, 0])
    assert not torch.equal(got["ple"][0][img], alt[0, :, 0][img])  # 若误用 <|image|> 的 PLE 行必被抓住
    assert torch.equal(got["ple"][0][~img], alt[0, :, 0][~img])


def test_vision_features_is_abstract_in_base_and_implemented_here():
    assert "vision_features" in ModelAdapter.__abstractmethods__
    assert "vision_features" not in Gemma4Adapter.__abstractmethods__
    sig = inspect.signature(Gemma4Adapter.hidden_states)
    assert sig.parameters["query_embeds"].kind is inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# 查询 token（只支持 seq1d）
# ---------------------------------------------------------------------------


def test_grid2d_raises_and_leaves_state_unchanged(fresh, images):
    ad = fresh()
    with pytest.raises(ValueError, match="seq1d"):
        ad.inject_query_tokens(16, "grid2d")
    assert ad.query_token_id is None and len(ad.tokenizer.added_tokens_decoder) == len(SPECIAL_IDS)
    assert ad.build_inputs(images[:1], INSTR_A, "image_first").query_span is None
    ad.inject_query_tokens(16, "seq1d")
    with pytest.raises(ValueError, match="seq1d"):
        ad.inject_query_tokens(9, "grid2d")
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    assert len(inp.query_span) == 16 and inp.query_position_mode == "seq1d"  # 报错前的登记不受影响


def test_query_positions_seq1d(fresh, images):
    ad = fresh()
    ad.inject_query_tokens(16, "seq1d")
    assert ad.query_token_id == LAST_UNUSED and ad.query_token_surface == "<unused6226>"
    inp = ad.build_inputs(images[:1], INSTR_A, "instruction_first")
    # [<bos>, <|turn>, user, \n, 指令×4, <|image>, 图×60, <image|>, 查询×16, <turn|>, \n]
    assert (inp.query_span.start, inp.query_span.end) == (70, 86) and inp.query_position_mode == "seq1d"
    ids = inp.tensors["input_ids"][0].tolist()
    assert ids[69] == ad.token_id(EOI) and ids[86] == ad.token_id(TURN_CLOSE) and len(ids) == 88
    assert torch.equal(ad.position_ids(inp), torch.arange(88).unsqueeze(0))
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    assert inp.query_span.start == inp.instruction_span.end  # image_first：块紧跟指令


def test_query_causal_prefix_bitwise_equal(fresh, images):
    """两组不同的 query_embeds：查询块之前各层逐位相同（因果），查询位置不同。"""
    ad = fresh()
    layers = _all_layers(ad)
    ad.inject_query_tokens(16, "seq1d")
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    torch.manual_seed(7)
    qa, qb = torch.randn(16, 64), torch.randn(16, 64)
    with torch.no_grad():
        sa = ad.sequence_hidden_states(inp, layers, query_embeds=qa)
        sb = ad.sequence_hidden_states(inp, layers, query_embeds=qb)
        plain = ad.sequence_hidden_states(inp, layers)
    qs, qe = inp.query_span.start, inp.query_span.end
    for l in layers:
        assert torch.equal(sa[l][:qs], sb[l][:qs]) and torch.equal(sa[l][:qs], plain[l][:qs]), l
        assert not torch.equal(sa[l][qs:qe], sb[l][qs:qe]), l
    assert not torch.equal(sa[4][qe:], sb[4][qe:])  # 块之后的 <turn|>\n 看得见查询


def test_query_embeds_replace_placeholder_and_carry_grad(fresh, images):
    ad = fresh()
    ad.inject_query_tokens(4, "seq1d")
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    qe = torch.randn(4, 64, requires_grad=True)
    hs = ad.hidden_states(inp, [0, 4], query_embeds=qe)
    assert torch.equal(hs.query[0], qe.detach())
    hs.query[4].pow(2).sum().backward()
    assert qe.grad is not None and qe.grad.abs().sum() > 0  # E3a 的可学习查询从这里得到梯度
    with torch.no_grad():
        plain = ad.hidden_states(inp, [0])
    emb = ad.model.get_input_embeddings()
    row = emb.weight[ad.query_token_id]
    assert torch.equal(plain.query[0], (row * emb.embed_scale).expand(4, -1))  # 不给 query_embeds 时：占位符行 × embed_scale
    assert torch.equal(row, _base_mean(emb.weight))  # 占位符行按 D13-1 初始化为基础行均值
    ple = ad.model.get_per_layer_input_embeddings().weight
    assert torch.equal(ple[ad.query_token_id], _base_mean(ple))


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
    for n, mode in ((0, "seq1d"), (True, "seq1d"), (16, "grid3d"), (2.0, "seq1d")):
        with pytest.raises(ValueError):
            ad.inject_query_tokens(n, mode)
    assert ad.query_token_id is None
    ad.inject_query_tokens(16, "seq1d")
    ad.inject_query_tokens(9, "seq1d")  # 再次登记只改个数，不再注册 token
    assert len(ad.tokenizer) == N_ROWS and len(ad.tokenizer.added_tokens_decoder) == len(SPECIAL_IDS) + 1
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    assert len(inp.query_span) == 9
    with pytest.raises(ValueError, match="generate"):
        ad.generate(inp, enable_thinking=False, max_new_tokens=1)
    with pytest.raises(ValueError, match="结构性"):
        ad.build_inputs(images[:1], "a " + ad.query_token_surface, "image_first")
    ad.clear_query_tokens()
    assert ad.build_inputs(images[:1], INSTR_A, "image_first").query_span is None


def test_query_placeholder_is_outside_grid_budget(fresh):
    ad = fresh()
    ad.inject_query_tokens(4, "seq1d")
    ids = ad.add_grid_tokens([f"<g{i}>" for i in range(MAX_NEW_GRID_TOKENS)])
    assert ad.n_grid_tokens == MAX_NEW_GRID_TOKENS
    assert ad.query_token_id not in ids.values()
    assert sorted(ids.values()) == list(ad.unused_token_ids[:MAX_NEW_GRID_TOKENS])  # 从最小的 <unusedN> 起
    assert ad.model.get_input_embeddings().weight.shape[0] == N_ROWS
    with pytest.raises(ValueError, match="上限"):
        ad.add_grid_tokens(["<extra>"])


# ---------------------------------------------------------------------------
# 新增 grid token（D13-1：复用 <unusedN>，初始化主嵌入行与 PLE 行）
# ---------------------------------------------------------------------------


def _forbid_resize(monkeypatch, ad):
    def boom(*a, **k):
        raise AssertionError("不许调用 resize_token_embeddings（D13-1）")

    monkeypatch.setattr(ad.model, "resize_token_embeddings", boom)


def test_add_grid_tokens_reuse_unused_rows(fresh, monkeypatch):
    ad = fresh()
    _forbid_resize(monkeypatch, ad)
    m = ad.model
    emb = m.get_input_embeddings().weight
    ple = m.get_per_layer_input_embeddings().weight
    emb0, ple0 = emb.detach().clone(), ple.detach().clone()
    ids = ad.add_grid_tokens(["<g0>", "<g1>"])
    assert ids == {"<g0>": FIRST_UNUSED, "<g1>": FIRST_UNUSED + 1}  # 按 id 从小到大：<unused0>=6、<unused1>=7
    assert ad.grid_token_surfaces == {"<g0>": "<unused0>", "<g1>": "<unused1>"}
    assert (emb.shape[0], m.get_output_embeddings().weight.shape[0], ple.shape[0]) == (N_ROWS,) * 3  # 不 resize
    assert m.config.text_config.vocab_size == N_ROWS and len(ad.tokenizer) == N_ROWS
    assert m.get_output_embeddings().weight is emb  # 仍共享
    tok = ad.tokenizer
    assert tok("<unused0>", add_special_tokens=False)["input_ids"] == [6]
    assert tok("<unused1><unused0><unused1>", add_special_tokens=False)["input_ids"] == [7, 6, 7]
    assert tok("a<unused0>b", add_special_tokens=False)["input_ids"][1] == 6
    assert tok.decode([6, 7]) == "<unused0><unused1>"
    new = list(ids.values())
    assert torch.equal(emb[new], _base_mean(emb0).expand(2, -1))
    assert torch.equal(ple[new], _base_mean(ple0).expand(2, -1))
    assert not torch.equal(_base_mean(emb0), emb0.double().mean(0).float())  # 均值确实排除了 <unusedN> 与特殊 token
    keep = torch.ones(N_ROWS, dtype=torch.bool)
    keep[new] = False
    assert torch.equal(emb[keep], emb0[keep]) and torch.equal(ple[keep], ple0[keep])  # 其他行逐位不变
    assert not torch.equal(emb0[new], emb[new])  # <unusedN> 原有的值不用


def test_grid_budget_is_cumulative(fresh):
    ad = fresh()
    ad.add_grid_tokens([f"<a{i}>" for i in range(200)])
    n_reg = len(ad.tokenizer.added_tokens_decoder)
    with pytest.raises(ValueError, match="累计"):
        ad.add_grid_tokens([f"<b{i}>" for i in range(200)])
    assert (len(ad.tokenizer.added_tokens_decoder), ad.n_grid_tokens) == (n_reg, 200)
    ad.add_grid_tokens([f"<c{i}>" for i in range(100)])
    assert ad.n_grid_tokens == MAX_NEW_GRID_TOKENS
    with pytest.raises(ValueError, match="上限"):
        ad.add_grid_tokens(["<d0>"])
    assert sorted(ad.grid_token_ids.values()) == list(ad.unused_token_ids[:MAX_NEW_GRID_TOKENS])


def test_add_grid_tokens_rejects_bad_names(fresh):
    ad = fresh()
    for names in (["<unused0>"], ["<|image|>"], ["<eos>"], "<g0>", [], ["<g0>", "<g0>"], [""]):
        with pytest.raises((ValueError, TypeError)):
            ad.add_grid_tokens(names)
    ad.add_grid_tokens(["<g0>"])
    with pytest.raises(ValueError, match="已登记"):
        ad.add_grid_tokens(["<g1>", "<g0>"])
    assert ad.n_grid_tokens == 1 and ad.grid_token_ids == {"<g0>": FIRST_UNUSED}


def test_grid_tokens_usable_in_forward_and_generation(fresh, images):
    ad = fresh()
    ids = ad.add_grid_tokens(["<g0>", "<g1>"])
    inp = ad.build_inputs(images[:1], "grid " + ad.grid_token_surfaces["<g0>"], "image_first")  # 指令里可以出现 grid token
    assert ids["<g0>"] in inp.tensors["input_ids"][0].tolist()
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=3, allowed_token_ids=list(ids.values()))
    assert set(res.token_ids) <= set(ids.values()) and res.n_generated_tokens == 3
    assert res.text == ad.tokenizer.decode(list(res.token_ids))  # text 保留 grid token 的 <unusedN> 字样


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


def test_allowed_token_ids_restrict_generation_and_both_stop_tokens(fresh, images):
    ad = fresh()
    ids = list(ad.add_grid_tokens(["<g0>", "<g1>"]).values())
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=6, allowed_token_ids=ids)
    assert set(res.token_ids) <= set(ids) and res.n_generated_tokens == 6
    for s in (TURN_CLOSE, EOS):  # 两个停止 token 都生效（id 从 tokenizer 查）
        stop = ad.token_id(s)
        res = ad.generate(inp, enable_thinking=False, max_new_tokens=6, allowed_token_ids=[stop])
        assert res.token_ids == (stop,) and res.text == ""
    other = ad.tokenizer.convert_tokens_to_ids("<|tool_response>")  # generation_config 的第三个 eos 不在 D13-3 里
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=3, allowed_token_ids=[other])
    assert res.token_ids == (other,) * 3


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
    for kw in ("eos_token_id", "image_position_ids", "per_layer_inputs"):
        with pytest.raises(ValueError, match="覆盖"):
            adapter.generate(inp, enable_thinking=False, max_new_tokens=1, **{kw: 1})


def test_greedy_despite_repo_sampling_config(adapter, images):
    """仓库 generation_config 默认采样（do_sample=True, top_k=64，小模型也带着它）；适配器显式 greedy：
    换随机种子结果不变、等于逐步 argmax；不显式关掉时 HF 默认确实在采样（换种子结果不同）。"""
    assert adapter.model.generation_config.do_sample is True and adapter.model.generation_config.top_k == 64
    inp = adapter.build_inputs(images[:1], INSTR_A, "image_first")
    outs = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        outs.append(adapter.generate(inp, enable_thinking=False, max_new_tokens=6).token_ids)
    assert outs[0] == outs[1]
    ids = adapter.prompt_ids(inp, enable_thinking=False)
    t = inp.tensors
    cur, manual = ids, []
    with torch.no_grad():
        for _ in range(len(outs[0])):
            lg = adapter.model(input_ids=cur, attention_mask=torch.ones_like(cur), pixel_values=t["pixel_values"],
                               image_position_ids=t["image_position_ids"], use_cache=False).logits[0, -1]
            manual.append(int(lg.argmax()))
            cur = torch.cat([cur, torch.tensor([[manual[-1]]])], dim=1)
    assert tuple(manual) == outs[0]
    sampled = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        out = adapter.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=t["pixel_values"],
                                     image_position_ids=t["image_position_ids"], max_new_tokens=6)
        sampled.append(out[0, ids.shape[1]:].tolist())
    assert sampled[0] != sampled[1]


@pytest.mark.parametrize("thinking", [False, True])
def test_generation_result_fields(adapter, images, thinking):
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    res = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=4)
    n_extra = len(adapter.thinking_prefix_ids(thinking)) + len(adapter.assistant_header_ids(thinking))
    assert res.n_prompt_tokens == inp.tensors["input_ids"].shape[1] + n_extra
    assert n_extra == (10 if thinking else 3)
    assert res.n_visual_tokens == 2 * N_IMG_TOK
    assert 1 <= res.n_generated_tokens <= 4 and res.wall_time_s > 0
    assert res.raw_text == adapter.tokenizer.decode(list(res.token_ids), skip_special_tokens=False)
    stop = adapter.token_id(TURN_CLOSE)
    forced = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=3, allowed_token_ids=[stop])
    assert forced.token_ids == (stop,) and forced.text == "" and forced.thinking_unclosed is False
    opened = adapter.token_id(CHANNEL_OPEN)
    forced = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=2, allowed_token_ids=[opened])
    assert forced.thinking_unclosed is True and forced.text == ""  # 思考段开了没关


def test_split_answer_rules(adapter, components):
    tok = components.tokenizer
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    op, cl, stop = adapter.token_id(CHANNEL_OPEN), adapter.token_id(CHANNEL_CLOSE), adapter.token_id(TURN_CLOSE)
    thought = enc("thought\nLet me look.")
    ans = enc("Blue.")
    text, raw, unclosed = adapter.split_answer([op] + thought + [cl] + ans + [stop], enable_thinking=True)
    assert (text, unclosed) == ("Blue.", False)
    assert raw == "<|channel>thought\nLet me look.<channel|>Blue.<turn|>" and "Let me look." not in text
    text, _, _ = adapter.split_answer([op] + thought + [cl] + enc("mid") + [op] + thought + [cl] + ans, enable_thinking=True)
    assert text == "Blue."  # 取最后一个 <channel|> 之后
    text, raw, unclosed = adapter.split_answer([op] + thought, enable_thinking=True)
    assert (text, unclosed) == ("", True) and raw == "<|channel>thought\nLet me look."
    text, _, unclosed = adapter.split_answer([op] + thought + [cl] + ans + [op] + thought, enable_thinking=True)
    assert (text, unclosed) == ("", True)  # 最后又开了一个没关
    for flag in (False, True):  # 没有思考段就取全部（与开关无关）
        text, _, unclosed = adapter.split_answer(ans + [stop], enable_thinking=flag)
        assert (text, unclosed) == ("Blue.", False)
    text, _, _ = adapter.split_answer(enc("\nBlue.\n") + [adapter.token_id(EOS)], enable_thinking=False)
    assert text == "Blue."  # 末尾停止 token 去掉，首尾换行去掉
    kept = enc("<|image>") + ans + [stop]
    assert adapter.split_answer(kept, enable_thinking=False)[0] == "<|image>Blue."  # 其他特殊 token 保留
    with pytest.raises(TypeError):
        adapter.split_answer(ans, enable_thinking=None)


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


def test_entry_assertion_when_submodule_on_meta(fresh, images):
    """只有视觉投影或 PLE 表在别的设备上（嵌入层还在 cpu）：同样在进算子前报出模块名。"""
    for name in ("embed_vision", "embed_tokens_per_layer"):
        ad = fresh()
        inp = ad.build_inputs(images, INSTR_A, "image_first")
        mod = ad.model.model.embed_vision if name == "embed_vision" else ad.model.get_per_layer_input_embeddings()
        mod.to("meta")
        with pytest.raises(AdapterDeviceError, match=name):
            ad.hidden_states(inp, [1])


def test_entry_assertion_when_input_on_meta(adapter, images):
    for key in ("input_ids", "pixel_values", "attention_mask", "image_position_ids"):
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
    model = gemma4.build_tiny_model(components.config, dtype=torch.bfloat16)
    ad = gemma4.make_tiny_adapter(components, model=model)
    assert ad.dtype == torch.bfloat16
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    assert inp.tensors["pixel_values"].dtype == torch.bfloat16
    with torch.no_grad():
        hs = ad.hidden_states(inp, [0, 4])
        vf = ad.vision_features(inp)
        auto = ad.model(**inp.tensors, output_hidden_states=True, logits_to_keep=1, use_cache=False).hidden_states
    assert hs.image[4].dtype == torch.bfloat16 and torch.equal(vf, hs.image[0])
    idx = torch.cat([torch.arange(i.span.start, i.span.end) for i in inp.images])
    assert torch.equal(hs.image[4], auto[4][0][idx])


# ---------------------------------------------------------------------------
# 坐标：box_2d（y 在前，0–1000）→ 内部绝对像素 xyxy（D13-4，model_notes §6.5）
# ---------------------------------------------------------------------------

# §6.5 的零样本实测输出与 COCO GT（像素 xyxy）
_CASES = [
    ((640, 480), [107, 13, 987, 496], [17.5, 54.4, 318.8, 469.4]),  # 000000039769.jpg，左边的猫
    ((640, 426), [394, 13, 621, 244], [7.0, 167.8, 156.3, 262.6]),  # 000000000139.jpg，电视
]


@pytest.mark.parametrize("wh,box,gt", _CASES)
def test_box2d_to_xyxy_on_model_notes_samples(wh, box, gt):
    w, h = wh
    xyxy = box2d_to_xyxy(box, width=w, height=h)
    y1, x1, y2, x2 = box
    np.testing.assert_allclose(xyxy, [x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h], rtol=0, atol=1e-9)
    assert box_iou(xyxy, gt) > 0.9  # y 在前的读法与 GT 吻合
    x_first = [box[0] / 1000 * w, box[1] / 1000 * h, box[2] / 1000 * w, box[3] / 1000 * h]  # 误当 x 在前
    assert box_iou(x_first, gt) < 0.5


def test_box2d_to_xyxy_inverts_gt_table_and_accepts_batches():
    """§6.5 表：GT ×1000、y 在前的整数值换回像素。误差不超过：千分位取整的半格 + 表里 GT 像素值显示到 0.1 的取整（0.05）。"""
    table = [((640, 480), [113, 27, 978, 498], [17.5, 54.4, 318.8, 469.4]),
             ((640, 480), [53, 543, 770, 1000], [347.5, 25.6, 640.0, 369.4]),
             ((640, 426), [394, 11, 617, 244], [7.0, 167.8, 156.3, 262.6]),
             ((640, 426), [491, 871, 676, 998], [557.2, 209.2, 638.6, 287.9])]
    for (w, h), box, gt in table:
        got = box2d_to_xyxy(box, width=w, height=h)
        assert np.all(np.abs(got - np.asarray(gt)) <= np.array([w, h, w, h]) * 0.5 / 1000 + 0.05 + 1e-9)
    batch = box2d_to_xyxy(np.array([t[1] for t in table[:2]]), width=640, height=480)
    assert batch.shape == (2, 4) and np.array_equal(batch[1], box2d_to_xyxy(table[1][1], width=640, height=480))


def test_box2d_to_xyxy_validation():
    np.testing.assert_allclose(box2d_to_xyxy([0, 0, 1000, 1000], width=640, height=480), [0, 0, 640, 480], atol=1e-9)
    np.testing.assert_allclose(box2d_to_xyxy([500, 250, 400, 1100], width=100, height=10), [25, 5, 110, 4],
                               atol=1e-9)  # 不裁剪、不交换
    for bad in ([1, 2, 3], [1, 2, 3, 4, 5], [1, 2, float("nan"), 4], [True, 0, 1, 1], "1234", ["a", 0, 1, 1]):
        with pytest.raises((ValueError, TypeError)):
            box2d_to_xyxy(bad, width=640, height=480)
    with pytest.raises(ValueError):
        box2d_to_xyxy([0, 0, 1, 1], width=0, height=480)


# ---------------------------------------------------------------------------
# 冒烟脚本的 CPU 彩排（端到端，与服务器同一流程）
# ---------------------------------------------------------------------------


class _FakeGemma4Processor:
    """开发机没有 torchvision，官方 Gemma4Processor 连 import 都不行。这里按它的源码写一个最小替身，只为让冒烟的
    "与官方 processor 比对"那条代码路径在开发机上跑一遍（真比对在服务器上）：按样本嵌套传图（不嵌套就报错，
    同 validate_inputs），每个 <|image|> 换成 <|image> + n×<|image|> + <image|>（replace_image_token），
    mm_token_type_ids 只给 <|image|> 记 1（create_mm_token_type_ids）。"""

    def __init__(self, tokenizer, image_processor):
        self.tok, self.ip = tokenizer, image_processor

    def apply_chat_template(self, messages, **kw):
        return self.tok.apply_chat_template(messages, **kw)

    def __call__(self, *, text, images, max_soft_tokens, return_tensors):
        assert return_tensors == "pt"
        if len(images) != len(text) or not all(isinstance(x, list) for x in images):
            raise ValueError("Received inconsistently sized batches of images and text")
        (sample_text,), (sample_images,) = text, images
        enc = self.ip(images=sample_images, max_soft_tokens=max_soft_tokens, return_tensors="pt")
        parts = sample_text.split(IMAGE_TOKEN)
        out = parts[0]
        for n, part in zip(enc["num_soft_tokens_per_image"].tolist(), parts[1:]):
            out += BOI + IMAGE_TOKEN * n + EOI + part
        ids = torch.tensor([self.tok(out)["input_ids"]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": (ids == self.tok.convert_tokens_to_ids(IMAGE_TOKEN)).long(),
                "pixel_values": enc["pixel_values"], "image_position_ids": enc["image_position_ids"]}


def test_smoke_official_compare_path_runs_with_nested_images(adapter, images):
    import types

    ctx = {"args": types.SimpleNamespace(max_new_tokens=2), "img": images[0], "img2": images[1],
           "smoke": types.SimpleNamespace(INSTR_A=INSTR_A)}
    adapter.hf_processor = _FakeGemma4Processor(adapter.tokenizer, adapter.image_processor)
    try:
        r = gemma4.smoke_check_1_build_generate(adapter, ctx)  # 比对代码若不按样本嵌套传图，替身会像官方一样报错
        assert r["ok"] and r["official_processor_compared"] and r["official_two_images"]["ok"]
        assert all(c["official_processor"]["ok"] for c in r["cases"]) and len(r["cases"]) == 4
        inp = adapter.build_inputs(images, INSTR_A, "image_first")  # 两张不同宽高比的图 + 思考开启
        assert gemma4._smoke_compare_official(adapter, images, INSTR_A, "image_first", True, inp)["ok"]
    finally:
        adapter.hf_processor = None


def test_smoke_script_tiny_gemma(tmp_path):
    out = tmp_path / "smoke_gemma_tiny.json"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/smoke_adapter.py"), "--model", "gemma-4-e2b-it", "--tiny",
         "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert out.exists(), proc.stderr[-3000:]
    report = json.loads(out.read_text())
    assert proc.returncode == 0 and report["all_ok"] is True, json.dumps(report.get("ok"), ensure_ascii=False)
    assert list(report["ok"]) == [
        "0_facts", "1_build_generate", "2_task_modulation", "3_grid_tokens", "4_query_tokens",
        "5_position_ids", "6_vision_features", "7_thinking", "8_device_audit",
    ]
    checks = report["checks"]
    assert checks["0_facts"]["model_key"] == "tiny(gemma-4-e2b-it)" and checks["0_facts"]["max_soft_tokens"] == MST
    assert checks["4_query_tokens"]["grid2d"]["raised"] is True
    assert checks["6_vision_features"]["image_first"]["bitwise_equal_layer0_image_positions"] is True
    assert checks["2_task_modulation"]["image_first"]["equal_layers"] == "5/5"
    assert checks["2_task_modulation"]["instruction_first"]["equal_layers"] == "1/5"
