"""InternVL3.5 适配器（DEV-0005）的 CPU 测试。

模型用本地构造的小型随机 InternVL：取固定 revision 的真实 config，LLM 缩到 4 层全注意力、hidden 64，InternViT 缩到 1 层、
hidden 64；tile 边长 448、patch 14、每 tile 256 token 保持真实值。tokenizer / 图像处理器（PIL 后端）/ 聊天模板 / README
从固定 revision 下载（小文件，不含权重）。取不到就失败并说明原因，不静默跳过。
事实的比对对象是 RUN-0002 的原始证据（experiments/P0/0002_family_probe/raw/internvl3.5-2b-hf/）与 docs/model_notes.md §7。
fixture 的图 H≠W、两张图内容不同；两条指令内容不同、token 长度相同。
"""

from __future__ import annotations

import ast
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
from huggingface_hub import hf_hub_download
from PIL import Image

from src.adapters import internvl35, registry
from src.adapters.base import ORDERS, GenerationResult, ModelAdapter, ModelInputs
from src.adapters.internvl35 import (
    END_OF_TEXT,
    IM_END,
    IM_START,
    IMG_CONTEXT,
    IMG_END,
    IMG_START,
    QUERY_PLACEHOLDER,
    R1_SYSTEM_PROMPT,
    THINK_END,
    THINK_START,
    VIDEO,
    AdapterDeviceError,
    BoxParseFailure,
    InternVL35Adapter,
    parse_boxes,
    parse_one_box,
    tiles_of,
)
from src.constants import BOX_ACC_IOU_THRESHOLD, MAX_NEW_GRID_TOKENS
from src.executors.geometry import abs_to_norm1000, box_iou

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "experiments/P0/0002_family_probe/raw/internvl3.5-2b-hf"
FACTS = json.loads((RAW / "facts.json").read_text())
BOX_CASES = json.loads((RAW / "box_zeroshot.json").read_text())["cases"]
MODEL_NOTES = (REPO_ROOT / "docs/model_notes.md").read_text()

INSTR_A = "Describe the cat."
INSTR_B = "Describe the sofa."
SINGLE = {"crop_to_patches": False, "max_patches": 1}
MULTI = {"crop_to_patches": True, "max_patches": 6}

# 2B-HF 固定 revision 的词表事实（model_notes §7.1，取自 RUN-0002 的 facts.json），测试里钉住；适配器本身不写死它们
LEN_TOKENIZER = FACTS["vocab"]["len_tokenizer"]  # 151679
N_BASE_VOCAB = FACTS["vocab"]["tokenizer_vocab_size_base"]  # 151643
N_ROWS = FACTS["vocab"]["embed_rows"]  # 151936 = len(tokenizer) + 257 个预留行
N_LAYERS_TINY = internvl35.TINY_TEXT["num_hidden_layers"]
HIDDEN_TINY = internvl35.TINY_TEXT["hidden_size"]


def _section(start: str, end: str) -> str:
    i = MODEL_NOTES.index(start)
    return MODEL_NOTES[i : MODEL_NOTES.index(end, i)]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def components():
    entry = registry.get_model(internvl35.TINY_SOURCE_KEY)
    try:
        return internvl35.load_pinned_components(entry.key)
    except Exception as e:  # noqa: BLE001 — 取不到小文件时给出原因后失败，不跳过
        pytest.fail(
            f"取不到 {entry.repo}@{entry.revision} 的 config/tokenizer/处理器/模板小文件：{e!r}。"
            "测试需要联网下载一次（之后走本地 HF 缓存）。",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def ref_proc(components):
    """官方 InternVLProcessor 类（开发机上绕过视频处理器的类型检查），比对用。"""
    return internvl35.reference_processor(components)


@pytest.fixture(scope="module")
def base_models(components):
    return {tied: internvl35.build_tiny_model(components[0], tie_word_embeddings=tied) for tied in (False, True)}


def _adapter(components, base_models, *, tied=False, fresh_tokenizer=False, policy=SINGLE, model=None, share=False):
    config, tok, ip, template = components
    if fresh_tokenizer:
        tok = copy.deepcopy(tok)
    if model is None:
        model = base_models[tied] if share else copy.deepcopy(base_models[tied])
    return internvl35.make_tiny_adapter((config, tok, ip, template), **policy, model=model)


@pytest.fixture(scope="module")
def adapter(components, base_models):
    """只读共享、单 tile：不许在它上面加 token 或注入查询（用例结束时检查）。"""
    ad = _adapter(components, base_models, share=True)
    yield ad
    assert ad.n_grid_tokens == 0 and ad.query_token_id is None, "共享适配器被改动过"
    assert len(ad.tokenizer) == LEN_TOKENIZER
    assert ad.model.get_input_embeddings().weight.shape[0] == N_ROWS


@pytest.fixture(scope="module")
def multi(components, base_models):
    """只读共享、多 tile 策略（crop_to_patches=True, max_patches=6），与 adapter 共用同一个模型。"""
    return _adapter(components, base_models, share=True, policy=MULTI)


@pytest.fixture
def fresh(components, base_models):
    """每次一个独立的适配器（tokenizer 副本 + 模型副本），可以加 token、注入查询。"""
    return lambda tied=False, policy=SINGLE: _adapter(
        components, base_models, tied=tied, fresh_tokenizer=True, policy=policy
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


@pytest.fixture(scope="module")
def wide_image():
    """900×600（W×H，宽高比 1.5）：max_patches=6 时切成 3×2 画布 + 缩略图 = 7 个 tile。"""
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (600, 900, 3), dtype=np.uint8)
    a[100:300, 200:500, 2] = 255
    return Image.fromarray(a)


def _all_layers(ad):
    return list(range(ad.num_layers() + 1))


def _enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def _collapse(ad, ids):
    """每段 <img> <IMG_CONTEXT>×k </img> 收成一个 <IMG_CONTEXT>（测试自己的实现，不用适配器模块里的）。"""
    s, c, e = ad.token_id(IMG_START), ad.token_id(IMG_CONTEXT), ad.token_id(IMG_END)
    out, k = [], 0
    ids = list(ids)
    while k < len(ids):
        if ids[k] == s:
            j = ids.index(e, k)
            assert j > k + 1 and set(ids[k + 1 : j]) == {c}
            out.append(c)
            k = j + 1
        else:
            out.append(ids[k])
            k += 1
    return out


def _messages(n_images, instruction, order, thinking):
    imgs = [{"type": "image"}] * n_images
    text = {"type": "text", "text": instruction}
    user = {"role": "user", "content": [text, *imgs] if order == "instruction_first" else [*imgs, text]}
    system = {"role": "system", "content": [{"type": "text", "text": R1_SYSTEM_PROMPT}]}
    return [system, user] if thinking else [user]


# ---------------------------------------------------------------------------
# 注册表、固定 revision 与 RUN-0002 的事实
# ---------------------------------------------------------------------------


def test_tiny_source_is_the_registered_2b_hf():
    e = registry.get_model(internvl35.TINY_SOURCE_KEY)
    assert e.family == registry.FAMILY_INTERNVL35 and e.repo == FACTS["repo"] and e.revision == FACTS["revision"]
    assert registry.keys_for_family(registry.FAMILY_INTERNVL35) == ("internvl3.5-2b-hf", "internvl3.5-4b-hf")


def test_load_accepts_only_registered_internvl_keys():
    ad = InternVL35Adapter(device="cpu", **SINGLE)
    with pytest.raises(KeyError):
        ad.load("OpenGVLab/InternVL3_5-2B-HF", torch.float32)  # 裸仓库名不行：必须走注册表的 revision
    with pytest.raises(ValueError, match="qwen3.5"):
        ad.load("qwen3.5-0.8b", torch.float32)
    with pytest.raises(TypeError):
        ad.load("internvl3.5-2b-hf", "bf16")


def test_load_wiring_uses_registry_revision_and_official_processor(monkeypatch, ref_proc, base_models):
    """开发机没有权重：把两个 from_pretrained 换成桩，核对 load() 的接线——都带注册表的 revision，模型用 sdpa、调用方给的
    dtype 与构造时的设备；官方处理器留在 hf_processor 上，挂载用的就是它的 tokenizer / 图像处理器 / 模板。"""
    import transformers

    calls = {}

    def fake_processor(repo, **kw):
        calls["processor"] = (repo, kw)
        return proc

    def fake_model(repo, **kw):
        calls["model"] = (repo, kw)
        return copy.deepcopy(base_models[False])

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_processor)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", fake_model)
    e = registry.get_model("internvl3.5-2b-hf")
    proc = ref_proc
    ad = InternVL35Adapter(device="cpu", **SINGLE)
    ad.load(e.key, torch.float32)
    assert calls["processor"] == (e.repo, {"revision": e.revision})
    assert calls["model"] == (e.repo, {"revision": e.revision, "dtype": torch.float32, "attn_implementation": "sdpa",
                                       "device_map": "cpu"})
    assert ad.hf_processor is ref_proc and ad.model_key == e.key
    assert ad.tokenizer is ref_proc.tokenizer and ad.image_processor is ref_proc.image_processor
    assert ad.chat_template == ref_proc.chat_template
    with pytest.raises(AdapterDeviceError, match="bfloat16"):
        InternVL35Adapter(device="cpu", **SINGLE).load(e.key, torch.bfloat16)  # 桩给的是 float32：dtype 对不上就报错
    proc = copy.copy(ref_proc)
    proc.image_seq_length = 128
    with pytest.raises(ValueError, match="image_seq_length"):
        InternVL35Adapter(device="cpu", **SINGLE).load(e.key, torch.float32)


def test_pinned_files_match_run0002_facts(components):
    config, tok, ip, template = components
    tc, vc = config.text_config, config.vision_config
    assert config.image_token_id == FACTS["named_mm_tokens"]["image_token"][1]
    assert tc.vocab_size == FACTS["vocab"]["config_vocab_size"] == N_ROWS
    assert tc.num_hidden_layers == FACTS["n_layers"] and tc.hidden_size == FACTS["text_cfg"]["hidden_size"]
    assert list(tc.layer_types) == [x["layer_type"] for x in FACTS["layers"]]
    # §7.1：config 顶层写 tie、text_config 写不 tie；加载真实权重时 transformers 因两者数值不同而不绑定
    assert (config.tie_word_embeddings, tc.tie_word_embeddings) == (True, FACTS["text_cfg"]["tie_word_embeddings"])
    assert list(vc.image_size) == FACTS["vision"]["image_size"] and list(vc.patch_size) == FACTS["vision"]["patch_size"]
    assert config.downsample_ratio == FACTS["vision"]["downsample_ratio"]
    assert config.vision_feature_layer == FACTS["vision"]["vision_feature_layer"]
    assert len(tok) == LEN_TOKENIZER and tok.vocab_size == N_BASE_VOCAB
    added = {d["content"]: d["id"] for d in FACTS["added_special_tokens(non-unused)"]}
    assert len(added) == FACTS["n_added_tokens_decoder"]
    assert {t: tok.convert_tokens_to_ids(t) for t in added} == added
    e = registry.get_model(internvl35.TINY_SOURCE_KEY)
    assert template == Path(hf_hub_download(e.repo, "chat_template.jinja", revision=e.revision)).read_text()
    attrs = FACTS["image_processor_attrs"]
    assert (ip.crop_to_patches, ip.min_patches, ip.max_patches) == (
        attrs["crop_to_patches"], attrs["min_patches"], attrs["max_patches"])


def test_adapter_constants_match_model_notes():
    v = FACTS["vision"]
    assert internvl35.TOKENS_PER_TILE == v["tokens_per_tile"] == 256
    assert internvl35.TILE_PX == v["image_size"][0] == v["image_size"][1]
    assert internvl35.TILE_GRID_HW[0] * internvl35.TILE_GRID_HW[1] == internvl35.TOKENS_PER_TILE
    assert internvl35.TILE_PX // internvl35.TILE_GRID_HW[0] == v["px_per_llm_token_side"]
    assert internvl35.MIN_PATCHES == FACTS["image_processor_attrs"]["min_patches"]
    assert internvl35.STOP_TOKENS == (IM_END,) and FACTS["eos_tokens"] == {str(FACTS["text_cfg"]["eos_token_id"]): IM_END}


def test_processor_default_is_the_opposite_of_preprocessor_config(components, ref_proc, adapter):
    """D13-5 的陷阱在开发机上复现：preprocessor_config 写 False，处理器调用时的默认却是 True（§7.3 的 default 行）。"""
    from transformers.models.internvl.processing_internvl import InternVLProcessorKwargs

    tok, ip = components[1], components[2]
    assert ip.crop_to_patches is False
    assert InternVLProcessorKwargs._defaults["images_kwargs"]["crop_to_patches"] is True
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8))
    text = ref_proc.apply_chat_template(_messages(1, "hi", "image_first", False), tokenize=False,
                                        add_generation_prompt=True)
    default = ref_proc(text=[text], images=[img], return_tensors="pt")  # 不传参：就是"依赖默认值"
    n_default = int((default["input_ids"] == tok.convert_tokens_to_ids(IMG_CONTEXT)).sum())
    table = FACTS["image_tokenization"]
    assert n_default == table["640x480 default"]["n_image_tokens"] == 3328
    ours = adapter.build_inputs([img], "hi", "image_first")
    assert len(ours.images[0].span) == table["640x480 crop_to_patches=False max_patches=12"]["n_image_tokens"] == 256


def test_chat_template_has_no_thinking_switch(adapter):
    """§7.2：模板只用 messages 与 add_generation_prompt；传 enable_thinking 不起作用。"""
    assert sorted(FACTS["template_variables"]) == ["add_generation_prompt", "messages"]
    msgs = _messages(1, INSTR_A, "image_first", False)
    tok = adapter.tokenizer
    plain = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for flag in (False, True):
        assert tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=flag) == plain


# ---------------------------------------------------------------------------
# R1 system prompt（思考开关，D13-2）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["internvl3.5-2b-hf", "internvl3.5-4b-hf"])
def test_r1_system_prompt_is_verbatim_from_card(key):
    e = registry.get_model(key)
    card = Path(hf_hub_download(e.repo, "README.md", revision=e.revision)).read_text()
    m = re.search(r'R1_SYSTEM_PROMPT = """(.*?)"""\.strip\(\)', card, flags=re.S)
    assert m is not None
    assert m.group(1).strip() == R1_SYSTEM_PROMPT  # card 写的是 """…""".strip()
    src = registry.get_model(internvl35.TINY_SOURCE_KEY)
    assert f"{src.repo}/blob/{src.revision}/README.md" in internvl35.R1_SYSTEM_PROMPT_SOURCE


def _model_notes_renders():
    sec = _section("### 7.2 ", "### 7.3 ")
    rows = {}
    for m in re.finditer(r"^\| (default\(no kwarg\)|system=R1_SYSTEM_PROMPT \(card thinking mode\)) \\\| (text|image) "
                         r"\| `(.*)` \|", sec, flags=re.M):
        rows[f"{m.group(1)} | {m.group(2)}"] = ast.literal_eval(m.group(3).replace("\\|", "|"))
    return rows


@pytest.mark.parametrize("kind", ["text", "image"])
@pytest.mark.parametrize("thinking", [False, True])
def test_prompt_render_matches_run0002_and_model_notes(adapter, kind, thinking):
    """开思考 = 加 R1 system 轮，关 = 不加；完整提示与 RUN-0002 在服务器上渲染的逐字相同（§7.2 表的四行）。"""
    key = f"{'system=R1_SYSTEM_PROMPT (card thinking mode)' if thinking else 'default(no kwarg)'} | {kind}"
    want = FACTS["chat_template_renders"][key]["text"]
    if kind == "text":
        inp = adapter.build_inputs([], "What is 2+3?", "image_first")
    else:
        inp = adapter.build_inputs([np.full((64, 96, 3), 128, np.uint8)], "Describe the image.", "image_first")
    ids = _collapse(adapter, adapter.prompt_ids(inp, thinking))
    got = adapter.tokenizer.decode(ids, skip_special_tokens=False)
    assert got == want and ids == _enc(adapter.tokenizer, want)
    row = _model_notes_renders()[key]  # R1 两行在 model_notes 里只截取了末段，以 "…" 开头
    assert got.endswith(row[1:]) if row.startswith("…") else got == row
    assert (R1_SYSTEM_PROMPT in got) is thinking


def test_system_prefix_and_header(adapter):
    dec = lambda ids: adapter.tokenizer.decode(list(ids), skip_special_tokens=False)  # noqa: E731
    assert adapter.system_prefix_ids(False) == ()
    assert dec(adapter.system_prefix_ids(True)) == f"{IM_START}system\n{R1_SYSTEM_PROMPT}{IM_END}\n"
    for flag in (False, True):
        assert dec(adapter.assistant_header_ids(flag)) == f"{IM_START}assistant\n"
    with pytest.raises(TypeError):
        adapter.system_prefix_ids(1)
    with pytest.raises(TypeError):
        adapter.assistant_header_ids(None)


# ---------------------------------------------------------------------------
# 构造与挂载
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    dict(crop_to_patches=1, max_patches=1), dict(crop_to_patches=None, max_patches=4),
    dict(crop_to_patches=True, max_patches=0), dict(crop_to_patches=True, max_patches=True),
    dict(crop_to_patches=True, max_patches=2.0), dict(crop_to_patches=False, max_patches=-3),
])
def test_constructor_validates_tile_policy(kw):
    with pytest.raises((TypeError, ValueError)):
        InternVL35Adapter(device="cpu", **kw)


def test_constructor_requires_an_explicit_tile_policy():
    with pytest.raises(TypeError):
        InternVL35Adapter(device="cpu")  # 没有默认值：tile 策略必须显式给出（D13-5）
    with pytest.raises(TypeError):
        InternVL35Adapter(device="cpu", crop_to_patches=False)


def test_attach_reads_facts_from_tokenizer_and_config(adapter, multi, components):
    tok, config = components[1], components[0]
    assert adapter.num_layers() == N_LAYERS_TINY and adapter.hidden_size() == HIDDEN_TINY
    assert adapter.dtype == torch.float32 and adapter.n_base_vocab == N_BASE_VOCAB
    for s in (IM_START, IM_END, END_OF_TEXT, THINK_START, THINK_END, IMG_CONTEXT, IMG_START, IMG_END, VIDEO):
        assert adapter.token_id(s) == tok.convert_tokens_to_ids(s)
    assert adapter.token_id(IMG_CONTEXT) == config.image_token_id
    assert adapter.tile_policy == {"crop_to_patches": False, "max_patches": 1, "min_patches": 1, "tile_px": 448}
    assert adapter.always_single_tile and not multi.always_single_tile
    assert multi.processor_kwargs() == {"crop_to_patches": True, "max_patches": 6, "min_patches": 1,
                                        "size": {"height": 448, "width": 448}}


def test_attach_uses_runtime_tying_not_config(components, base_models):
    """真实 2B：config 顶层写 tie，但运行时 lm_head 与嵌入是两个参数（§7.1）。适配器以运行时为准，只记录 config。"""
    m = copy.deepcopy(base_models[False])
    m.config.tie_word_embeddings = True
    ad = _adapter(components, base_models, model=m)
    assert ad.tied_word_embeddings is False
    assert ad.config_tie_word_embeddings == {"top": True, "text": False}
    tied = _adapter(components, base_models, tied=True)
    assert tied.tied_word_embeddings is True and tied.model.get_output_embeddings().weight is tied.model.get_input_embeddings().weight


def test_attach_rejections(components, base_models, adapter):
    model = copy.deepcopy(base_models[False])
    model.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="sdpa"):
        _adapter(components, base_models, model=model)
    with pytest.raises(RuntimeError, match="已经挂过"):
        adapter.attach(base_models[False], tokenizer=components[1], image_processor=components[2],
                       chat_template=components[3])
    with pytest.raises(TypeError, match="InternVLForConditionalGeneration"):
        InternVL35Adapter(device="cpu", **SINGLE).attach(torch.nn.Linear(2, 2), tokenizer=components[1],
                                                         image_processor=components[2], chat_template=components[3])


# ---------------------------------------------------------------------------
# build_inputs：区间、顺序、tile、与官方处理器一致
# ---------------------------------------------------------------------------


def test_spans_instruction_first(adapter, images, components):
    tok = components[1]
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    ids = inp.tensors["input_ids"][0].tolist()
    # [<|im_start|>, user, \n, 指令×4, <img>, ctx×256, </img>, \n, <img>, ctx×256, </img>, \n, <|im_end|>, \n]
    assert len(ids) == 527
    assert (inp.instruction_span.start, inp.instruction_span.end) == (3, 7)
    assert [(i.span.start, i.span.end) for i in inp.images] == [(8, 264), (267, 523)]
    assert all(i.grid_hw == (16, 16) and tiles_of(i) == 1 for i in inp.images)
    nl = _enc(tok, "\n")[0]
    assert ids[:3] == _enc(tok, "<|im_start|>user\n") and ids[3:7] == _enc(tok, INSTR_A)
    assert [ids[7], ids[264], ids[266], ids[523]] == [adapter.token_id(IMG_START), adapter.token_id(IMG_END)] * 2
    assert ids[265] == ids[524] == nl
    assert set(ids[8:264]) == set(ids[267:523]) == {adapter.token_id(IMG_CONTEXT)}
    assert ids[525:] == [adapter.token_id(IM_END), nl]
    assert inp.order == "instruction_first" and inp.query_span is None


def test_spans_image_first(adapter, images, components):
    tok = components[1]
    inp = adapter.build_inputs(images, INSTR_A, "image_first")
    ids = inp.tensors["input_ids"][0].tolist()
    # [<|im_start|>, user, \n, <img>, ctx×256, </img>, \n, <img>, ctx×256, </img>, \n, 指令×4, <|im_end|>, \n]
    assert len(ids) == 527
    assert [(i.span.start, i.span.end) for i in inp.images] == [(4, 260), (263, 519)]
    assert (inp.instruction_span.start, inp.instruction_span.end) == (521, 525)
    assert ids[521:525] == _enc(tok, INSTR_A) and ids[525] == adapter.token_id(IM_END)


def test_tensors_and_pixel_values(adapter, images):
    inp = adapter.build_inputs(images, INSTR_A, "instruction_first")
    t = inp.tensors
    assert sorted(t) == ["attention_mask", "input_ids", "pixel_values"]  # 与官方处理器的输出键相同（§7.3）
    assert torch.equal(t["attention_mask"], torch.ones_like(t["input_ids"]))
    pv = t["pixel_values"]
    assert pv.shape == (2, 3, 448, 448) and pv.dtype == adapter.dtype and pv.device == adapter.device
    assert not torch.equal(pv[0], pv[1])


def test_image_processor_gets_the_tile_policy_explicitly_every_call(fresh, images, monkeypatch):
    ad = fresh(policy=MULTI)
    calls = []
    cls = type(ad.image_processor)
    orig = cls.__call__

    def spy(self, *a, **k):
        calls.append(k)
        return orig(self, *a, **k)

    monkeypatch.setattr(cls, "__call__", spy)
    ad.build_inputs(images, INSTR_A, "image_first")
    ad.build_inputs(images[:1], INSTR_A, "instruction_first")
    assert len(calls) == 2
    want = {"crop_to_patches": True, "max_patches": 6, "min_patches": 1, "size": {"height": 448, "width": 448}}
    for k in calls:
        assert {key: k.get(key) for key in want} == want


@pytest.mark.parametrize("policy", [SINGLE, MULTI], ids=["single", "multi"])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("thinking", [False, True])
def test_matches_official_processor_class(components, base_models, ref_proc, images, wide_image, policy, order,
                                          thinking):
    """完整提示与像素和官方 InternVLProcessor（同一个 PIL 图像处理器、同样显式传的 tile 参数）逐项相同。"""
    ad = _adapter(components, base_models, share=True, policy=policy)
    imgs = [wide_image, images[1]]
    inp = ad.build_inputs(imgs, INSTR_A, order)
    pil = [ad.to_rgb(im) for im in imgs]
    text = ref_proc.apply_chat_template(_messages(2, INSTR_A, order, thinking), tokenize=False,
                                        add_generation_prompt=True)
    enc = ref_proc(text=[text], images=pil, return_tensors="pt", crop_to_patches=policy["crop_to_patches"],
                   max_patches=policy["max_patches"], min_patches=1, size={"height": 448, "width": 448})
    assert sorted(enc.keys()) == ["attention_mask", "input_ids", "pixel_values"]
    assert enc["input_ids"][0].tolist() == ad.prompt_ids(inp, thinking)
    assert torch.equal(enc["pixel_values"], inp.tensors["pixel_values"])
    n_tiles = [tiles_of(i) for i in inp.images]
    assert n_tiles == ([1, 1] if policy is SINGLE else [7, 7])  # 900×600 → 3×2、100×150 → 2×3，各加缩略图


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("thinking", [False, True])
def test_prompt_equals_chat_template(adapter, images, order, thinking):
    inp = adapter.build_inputs(images, INSTR_A, order)
    full = adapter.prompt_ids(inp, thinking)
    ref = adapter.tokenizer.apply_chat_template(_messages(2, INSTR_A, order, thinking), tokenize=True,
                                                add_generation_prompt=True)["input_ids"]
    assert _collapse(adapter, full) == list(ref)


def _tile_table():
    sec = _section("### 7.3 ", "### 7.4 ")
    pat = (r"^\| (\d+)x(\d+) crop_to_patches=(True|False) max_patches=(\d+) \| (\d+) \| (\d+) "
           r"\| \[(\d+), 3, 448, 448\] \|$")
    return [(int(w), int(h), c == "True", int(mp), int(nt), int(nti)) for w, h, c, mp, nt, nti, _ in
            re.findall(pat, sec, flags=re.M)]


def test_tile_counts_match_model_notes_table(components, base_models):
    """§7.3 表里每一行显式参数（6 种图尺寸 × 5 种策略）：视觉 token 数与 tile 数和服务器实测逐项相同。"""
    rows = _tile_table()
    assert len(rows) == 30
    ads = {(c, mp): _adapter(components, base_models, share=True, policy={"crop_to_patches": c, "max_patches": mp})
           for _, _, c, mp, _, _ in rows}
    rng = np.random.default_rng(0)
    for w, h, crop, mp, n_tok, n_tiles in rows:
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        inp = ads[(crop, mp)].build_inputs([img], "hi", "image_first")
        (info,) = inp.images
        assert (len(info.span), tiles_of(info), inp.tensors["pixel_values"].shape[0]) == (n_tok, n_tiles, n_tiles), (w, h, crop, mp)
        assert info.grid_hw == (16 * n_tiles, 16)


def test_multi_tile_order_is_canvas_row_major_then_thumbnail(multi, adapter, components):
    """grid_hw = (16·n, 16) 的含义：第 t 个 16 行就是处理器的第 t 个 tile——画布 tile 按行主序，最后一个是缩略图。"""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    arr = np.zeros((896, 1344, 3), np.uint8)  # 恰好 3×2 个 448 的块，每块一种颜色
    for k, col in enumerate(colors):
        r, c = divmod(k, 3)
        arr[448 * r : 448 * (r + 1), 448 * c : 448 * (c + 1)] = col
    inp = multi.build_inputs([arr], "hi", "image_first")
    (info,) = inp.images
    assert tiles_of(info) == 7 and info.grid_hw == (112, 16) and len(info.span) == 7 * 256
    pv = inp.tensors["pixel_values"]
    ip = components[2]
    mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=torch.float32).view(3, 1, 1)
    for k, col in enumerate(colors):
        want = (torch.tensor(col, dtype=torch.float32).view(3, 1, 1) / 255 - mean) / std
        assert torch.allclose(pv[k], want.expand(3, 448, 448), atol=1e-5), k
    thumb = adapter.build_inputs([arr], "hi", "image_first").tensors["pixel_values"][0]  # 整图直接缩到 448×448
    assert torch.equal(pv[6], thumb)


def test_token_grid_is_row_major_over_2x2_patches(base_models):
    """单 tile 的 (16, 16)：第 k 个视觉 token 覆盖第 k//16 行、第 k%16 列的 2×2 个 patch（走 HF 的 get_image_features，
    视觉塔换成输出 patch 编号的桩、投影换成恒等，CLS 应被丢掉）。"""
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    m = copy.deepcopy(base_models[False])
    inner = m.model
    vh = m.config.vision_config.hidden_size
    side = internvl35.TILE_PX // m.config.vision_config.patch_size[0]  # 每边 32 个 patch
    x = torch.zeros(1, 1 + side * side, vh)
    x[0, 0, 0] = -1.0
    x[0, 1:, 0] = torch.arange(side * side, dtype=torch.float32)

    class _Stub(torch.nn.Module):
        def forward(self, pixel_values=None, **kwargs):
            return BaseModelOutputWithPooling(last_hidden_state=x)

    inner.vision_tower = _Stub()
    inner.multi_modal_projector = torch.nn.Identity()
    out = inner.get_image_features(pixel_values=torch.zeros(1, 3, 448, 448), vision_feature_layer=-1,
                                   vision_feature_select_strategy="default", return_dict=True).pooler_output
    gh, gw = internvl35.TILE_GRID_HW
    assert out.shape == (1, gh * gw, 4 * vh)
    for k in range(gh * gw):
        r, c = divmod(k, gw)
        got = sorted(int(v) for v in out[0, k].view(4, vh)[:, 0])
        assert got == sorted(side * (2 * r + dy) + 2 * c + dx for dy in (0, 1) for dx in (0, 1)), k


def test_tile_features_are_tile_major(multi, wide_image):
    """多 tile：图像段里第 t 个 256 行来自 pixel_values[t]（按 tile 分别过视觉塔再比，允许浮点误差）。"""
    inp = multi.build_inputs([wide_image], INSTR_A, "image_first")
    pv = inp.tensors["pixel_values"]
    (info,) = inp.images
    with torch.no_grad():
        seq0 = multi.sequence_hidden_states(inp, [0])[0][info.span.start : info.span.end]
        assert not torch.allclose(seq0[:256], seq0[256:512], rtol=1e-4, atol=1e-5)  # 相邻 tile 的特征确实可区分
        for t in range(pv.shape[0]):
            one = multi.model.model.get_image_features(pixel_values=pv[t : t + 1], vision_feature_layer=-1,
                                                       vision_feature_select_strategy="default",
                                                       return_dict=True).pooler_output[0]
            # 批大小不同，矩阵乘的分块可能不同（跨机器尤其如此），所以不要求逐位；次序错了差的是 O(1)
            torch.testing.assert_close(seq0[256 * t : 256 * (t + 1)], one, rtol=1e-4, atol=1e-5)


def test_probe_methods_refuse_multi_tile(multi, fresh, wide_image):
    inp = multi.build_inputs([wide_image], INSTR_A, "image_first")
    with pytest.raises(ValueError, match="tile"):
        multi.hidden_states(inp, [0])
    with pytest.raises(ValueError, match="tile"):
        multi.vision_features(inp)
    with torch.no_grad():
        assert multi.sequence_hidden_states(inp, [0])[0].shape[0] == inp.tensors["input_ids"].shape[1]  # 诊断用：接受
    res = multi.generate(inp, enable_thinking=False, max_new_tokens=2)  # 生成：接受
    assert res.n_visual_tokens == 7 * 256
    ad = fresh(policy=MULTI)
    ad.inject_query_tokens(4, "seq1d")
    with pytest.raises(ValueError, match="单 tile"):
        ad.build_inputs([wide_image], INSTR_A, "image_first")
    square = np.full((448, 448, 3), 90, np.uint8)  # 同一策略下，448×448 的图只切出 1 个 tile（§7.3），查询与探针都可用
    ok = ad.build_inputs([square], INSTR_A, "image_first")
    assert tiles_of(ok.images[0]) == 1 and len(ok.query_span) == 4
    with torch.no_grad():  # DEV-0020 第 8 条：查询嵌入必须显式给
        assert ad.hidden_states(ok, [0], query_embeds=torch.zeros(4, HIDDEN_TINY)).query[0].shape == (4, HIDDEN_TINY)


def test_numpy_and_pil_agree(adapter, images):
    a = images[0]
    x = adapter.build_inputs([a], INSTR_A, "image_first").tensors["pixel_values"]
    y = adapter.build_inputs([Image.fromarray(a)], INSTR_A, "image_first").tensors["pixel_values"]
    assert torch.equal(x, y)
    assert adapter.to_rgb(Image.fromarray(a).convert("RGBA")).mode == "RGB"
    assert adapter.to_rgb(a).size == (150, 100)  # 不缩放：缩放与切 tile 交给图像处理器


def test_instruction_validation(adapter, images):
    for bad in ("look <IMG_CONTEXT> here", "a <img> b", "x</img>", "stop<|im_end|>", "a <video>", "<|im_start|>"):
        with pytest.raises(ValueError, match="结构性"):
            adapter.build_inputs(images, bad, "image_first")
    with pytest.raises(ValueError, match="合并"):
        adapter.build_inputs(images, "\n" + INSTR_A, "image_first")  # 与前面 </img> 后的换行合成一个 token
    with pytest.raises(ValueError):
        adapter.build_inputs(images, INSTR_A, "image_last")
    with pytest.raises(TypeError):
        adapter.build_inputs(images[0], INSTR_A, "image_first")  # 单张图也要放进列表
    with pytest.raises(ValueError):
        adapter.build_inputs([np.zeros((10, 10), np.uint8)], INSTR_A, "image_first")
    with pytest.raises(TypeError):
        adapter.build_inputs(images, 3, "image_first")


def test_empty_instruction_and_zero_images(adapter, images):
    for order in ORDERS:
        inp = adapter.build_inputs(images, "", order)
        assert len(inp.instruction_span) == 0
    inp = adapter.build_inputs([], INSTR_A, "instruction_first")
    assert inp.images == () and "pixel_values" not in inp.tensors
    L = inp.tensors["input_ids"].shape[1]
    assert torch.equal(adapter.position_ids(inp), torch.arange(L).unsqueeze(0))
    with torch.no_grad():
        hs = adapter.hidden_states(inp, [0, N_LAYERS_TINY])
    assert hs.image[N_LAYERS_TINY].shape == (0, HIDDEN_TINY)
    assert adapter.vision_features(inp).shape == (0, HIDDEN_TINY)
    res = adapter.generate(inp, enable_thinking=False, max_new_tokens=2)
    assert res.n_visual_tokens == 0 and 1 <= res.n_generated_tokens <= 2


# ---------------------------------------------------------------------------
# 隐藏状态：层号、仪器、一维位置
# ---------------------------------------------------------------------------


def test_instruction_pair_has_equal_token_length(components):
    tok = components[1]
    a, b = _enc(tok, INSTR_A), _enc(tok, INSTR_B)
    assert len(a) == len(b) == 4 and a != b  # 与 §7.6 的 token 长度 [4, 4] 相同


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
    assert torch.equal(hs[n], normed)  # 最后一个层号：最终归一化之后（HF 惯例，D15-2）
    assert not torch.equal(hs[n], raw[n - 1][0])


@pytest.mark.parametrize("bad", [[-1], [N_LAYERS_TINY + 1], [True], [1.0], [], [2, 2], "3"])
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
    assert hs.image[2].shape == (2 * 256, HIDDEN_TINY)
    idx = torch.cat([torch.arange(i.span.start, i.span.end) for i in inp.images])
    assert torch.equal(hs.image[2], seq[2][idx])  # 多图按顺序拼接


@pytest.mark.parametrize("order", ORDERS)
def test_adapter_path_equals_model_path(adapter, multi, images, wide_image, order):
    """一维位置 0..L-1；传它与让模型自己算相同；适配器自己并入视觉特征 = HF 原生路径（单 tile 与多 tile）。"""
    for ad, imgs in ((adapter, images), (multi, [wide_image, images[1]])):
        inp = ad.build_inputs(imgs, INSTR_A, order)
        pos = ad.position_ids(inp)
        L = inp.tensors["input_ids"].shape[1]
        assert torch.equal(pos, torch.arange(L).unsqueeze(0))
        layers = _all_layers(ad)
        kw = dict(output_hidden_states=True, logits_to_keep=1, use_cache=False)
        with torch.no_grad():
            ours = ad.sequence_hidden_states(inp, layers)
            auto = ad.model(**inp.tensors, **kw).hidden_states
            given = ad.model(**inp.tensors, position_ids=pos, **kw).hidden_states
        for l in layers:
            assert torch.equal(auto[l][0], given[l][0]), l
            assert torch.equal(ours[l], auto[l][0]), l


def test_vision_features_equal_layer0_image_positions(adapter, images):
    for order in ORDERS:
        inp = adapter.build_inputs(images, INSTR_A, order)
        with torch.no_grad():
            vf = adapter.vision_features(inp)
            h0 = adapter.hidden_states(inp, [0]).image[0]
        assert vf.shape == (2 * 256, HIDDEN_TINY) and torch.equal(vf, h0)
        assert not torch.equal(vf[:256], vf[256:])


def test_vision_features_is_a_real_override():
    assert InternVL35Adapter.vision_features is not ModelAdapter.vision_features
    sig = inspect.signature(InternVL35Adapter.generate)
    for p in ("enable_thinking", "max_new_tokens"):
        assert sig.parameters[p].kind is inspect.Parameter.KEYWORD_ONLY and sig.parameters[p].default is inspect._empty


# ---------------------------------------------------------------------------
# 查询 token（一维位置；grid2d 报错）
# ---------------------------------------------------------------------------


def test_query_tokens_seq1d_placement_and_positions(fresh, images):
    ad = fresh()
    ad.inject_query_tokens(16, "seq1d")
    for order in ORDERS:
        # instruction_first：[…, 指令×4, <img>, ctx×256, </img>, \n, 查询×16, <|im_end|>, \n]
        # image_first：      […, <img>, ctx×256, </img>, \n, 指令×4, 查询×16, <|im_end|>, \n]
        inp = ad.build_inputs(images[:1], INSTR_A, order)
        assert (inp.query_span.start, inp.query_span.end) == (266, 282) and inp.query_position_mode == "seq1d"
        ids = inp.tensors["input_ids"][0].tolist()
        assert ids[266:282] == [ad.query_token_id] * 16 and ids[282] == ad.token_id(IM_END) and len(ids) == 284
        assert torch.equal(ad.position_ids(inp), torch.arange(284).unsqueeze(0))


def test_grid2d_is_rejected_without_side_effects(fresh):
    ad = fresh()
    with pytest.raises(ValueError, match="grid2d"):
        ad.inject_query_tokens(16, "grid2d")
    assert ad.query_token_id is None and len(ad.tokenizer) == LEN_TOKENIZER
    ad.inject_query_tokens(4, "seq1d")
    with pytest.raises(ValueError, match="grid2d"):
        ad.inject_query_tokens(16, "grid2d")
    inp = ad.build_inputs([], INSTR_A, "image_first")
    assert len(inp.query_span) == 4 and inp.query_position_mode == "seq1d"  # 被拒的调用不改变已登记的查询


def test_query_prefix_bitwise_equal_across_query_embeds(fresh, images):
    """精确为零的仪器：换一组 query_embeds，查询块之前各层逐位不变（因果注意力），查询位置变。"""
    ad = fresh()
    ad.inject_query_tokens(16, "seq1d")
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    layers = _all_layers(ad)
    g = torch.Generator().manual_seed(3)
    qa, qb = torch.randn(16, HIDDEN_TINY, generator=g), torch.randn(16, HIDDEN_TINY, generator=g)
    with torch.no_grad():
        sa = ad.sequence_hidden_states(inp, layers, query_embeds=qa)
        sb = ad.sequence_hidden_states(inp, layers, query_embeds=qb)
    qs, qe = inp.query_span.start, inp.query_span.end
    for l in layers:
        assert torch.equal(sa[l][:qs], sb[l][:qs]), l
        assert not torch.equal(sa[l][qs:qe], sb[l][qs:qe]), l


def test_query_embeds_replace_placeholder_and_carry_grad(fresh, images):
    ad = fresh()
    ad.inject_query_tokens(4, "seq1d")
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    qe = torch.randn(4, HIDDEN_TINY, requires_grad=True)
    hs = ad.hidden_states(inp, [0, N_LAYERS_TINY], query_embeds=qe)
    assert torch.equal(hs.query[0], qe.detach())
    hs.query[N_LAYERS_TINY].pow(2).sum().backward()
    assert qe.grad is not None and qe.grad.abs().sum() > 0  # E3a 的可学习查询从这里得到梯度
    with pytest.raises(ValueError, match="query_embeds"):  # DEV-0020 第 8 条：不给 query_embeds 报错（原来静默用占位符行）
        ad.hidden_states(inp, [0])
    emb = ad.model.get_input_embeddings().weight
    head = ad.model.get_output_embeddings().weight
    row = emb[ad.query_token_id]
    assert torch.equal(row, emb[:N_BASE_VOCAB].double().mean(0).float())  # 占位符行 = 基础行均值（D13-1）
    assert torch.equal(head[ad.query_token_id], head[:N_BASE_VOCAB].double().mean(0).float())  # lm_head 行各用各的均值


def test_query_embeds_validation(fresh, adapter, images):
    ad = fresh()
    ad.inject_query_tokens(4, "seq1d")
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    with pytest.raises(ValueError, match="形状"):
        ad.hidden_states(inp, [1], query_embeds=torch.zeros(5, HIDDEN_TINY))
    with pytest.raises(AdapterDeviceError, match="float64"):
        ad.hidden_states(inp, [1], query_embeds=torch.zeros(4, HIDDEN_TINY, dtype=torch.float64))
    with pytest.raises(AdapterDeviceError, match="meta"):
        ad.hidden_states(inp, [1], query_embeds=torch.zeros(4, HIDDEN_TINY, device="meta"))
    plain = adapter.build_inputs(images[:1], INSTR_A, "image_first")
    with pytest.raises(ValueError, match="没有查询"):
        adapter.hidden_states(plain, [1], query_embeds=torch.zeros(4, HIDDEN_TINY))


def test_inject_query_tokens_validation_and_clear(fresh, images):
    ad = fresh()
    for n, mode in ((0, "seq1d"), (True, "seq1d"), (2.0, "seq1d"), (16, "grid3d"), (16, "grid2d")):
        with pytest.raises(ValueError):
            ad.inject_query_tokens(n, mode)
    assert ad.query_token_id is None and len(ad.tokenizer) == LEN_TOKENIZER
    ad.inject_query_tokens(16, "seq1d")
    assert ad.query_token_id == N_ROWS - 1  # DEV-0020：固定取预留区末行，平时不登记进 tokenizer
    assert len(ad.tokenizer) == LEN_TOKENIZER
    assert ad.model.get_input_embeddings().weight.shape[0] == N_ROWS
    ad.inject_query_tokens(9, "seq1d")  # 再次登记只改个数，不再加 token
    assert len(ad.tokenizer) == LEN_TOKENIZER
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    assert len(inp.query_span) == 9
    with pytest.raises(ValueError, match="generate"):
        ad.generate(inp, enable_thinking=False, max_new_tokens=1)
    with pytest.raises(ValueError, match="结构性"):
        ad.build_inputs(images[:1], "a " + QUERY_PLACEHOLDER, "image_first")
    ad.clear_query_tokens()
    assert ad.build_inputs(images[:1], INSTR_A, "image_first").query_span is None


def test_query_placeholder_is_outside_grid_budget(fresh):
    ad = fresh()
    ad.inject_query_tokens(4, "seq1d")
    ids = ad.add_grid_tokens([f"<g{i}>" for i in range(MAX_NEW_GRID_TOKENS)])
    assert ad.n_grid_tokens == MAX_NEW_GRID_TOKENS and min(ids.values()) == LEN_TOKENIZER  # DEV-0020：跳过末行
    assert ad.query_token_id == N_ROWS - 1 and ad.query_token_id not in ids.values()
    rows = ad.model.get_input_embeddings().weight.shape[0]
    assert rows == ad.model.get_output_embeddings().weight.shape[0] == LEN_TOKENIZER + 1 + MAX_NEW_GRID_TOKENS
    with pytest.raises(ValueError, match="上限"):
        ad.add_grid_tokens(["<extra>"])


# ---------------------------------------------------------------------------
# 新增 grid token（D13-1：257 个预留行、只增不缩、嵌入与 lm_head 各用各的均值）
# ---------------------------------------------------------------------------


def _forbid_resize(monkeypatch, ad):
    def boom(*a, **k):
        raise AssertionError("不许调用 resize_token_embeddings（D13-1）")

    monkeypatch.setattr(ad.model, "resize_token_embeddings", boom)


@pytest.mark.parametrize("tied", [False, True])
def test_add_grid_tokens_reuse_reserved_rows(fresh, monkeypatch, tied):
    ad = fresh(tied)
    _forbid_resize(monkeypatch, ad)
    assert N_ROWS - LEN_TOKENIZER == FACTS["vocab"]["n_rows_beyond_tokenizer"] == 257
    emb = ad.model.get_input_embeddings().weight
    head = ad.model.get_output_embeddings().weight
    emb0, head0 = emb.detach().clone(), head.detach().clone()
    ids = ad.add_grid_tokens(["<g0>", "<g1>"])
    t = FACTS["new_token_test"]
    assert ids == {"<g0>": t["id_<g0>"], "<g1>": t["id_<g1>"]} == {"<g0>": LEN_TOKENIZER, "<g1>": LEN_TOKENIZER + 1}
    assert emb.shape[0] == head.shape[0] == N_ROWS  # 不缩表、不扩表
    tok = ad.tokenizer
    for s, info in t["tokenization"].items():  # §7.1 的分词表：新 token 都是单个 token
        assert tok(s, add_special_tokens=False)["input_ids"] == info["ids"], s
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
        ad.add_grid_tokens([f"<b{i}>" for i in range(101)])
    assert (len(ad.tokenizer), ad.model.get_input_embeddings().weight.shape[0], ad.n_grid_tokens) == (n_tok, rows, 200)
    ad.add_grid_tokens([f"<c{i}>" for i in range(100)])
    assert ad.n_grid_tokens == MAX_NEW_GRID_TOKENS
    with pytest.raises(ValueError, match="上限"):
        ad.add_grid_tokens(["<d0>"])


@pytest.mark.parametrize("tied", [False, True])
def test_add_300_grows_to_exact_rows(fresh, monkeypatch, images, tied):
    """累计超过 257 个预留行时扩到恰好需要的行数：已有 token 的行逐位不变，全部新 token 行（复用的预留行与扩出的行）= 均值。"""
    ad = fresh(tied)
    _forbid_resize(monkeypatch, ad)
    emb0 = ad.model.get_input_embeddings().weight.detach().clone()
    head0 = ad.model.get_output_embeddings().weight.detach().clone()
    ids = ad.add_grid_tokens([f"<g{i}>" for i in range(MAX_NEW_GRID_TOKENS)])
    emb = ad.model.get_input_embeddings().weight
    head = ad.model.get_output_embeddings().weight
    want_rows = LEN_TOKENIZER + MAX_NEW_GRID_TOKENS + 1  # DEV-0020：预留区末行固定留给查询占位符，grid 跳过它
    assert emb.shape == head.shape == (want_rows, HIDDEN_TINY) and want_rows > N_ROWS
    assert ad.model.config.text_config.vocab_size == want_rows
    new = list(ids.values())
    assert new == list(range(LEN_TOKENIZER, N_ROWS - 1)) + list(range(N_ROWS, want_rows))
    assert torch.equal(emb[:LEN_TOKENIZER], emb0[:LEN_TOKENIZER])
    emb_mean = emb0[:N_BASE_VOCAB].double().mean(0).float()
    assert torch.equal(emb[new], emb_mean.expand(len(new), -1))
    assert not torch.equal(emb[LEN_TOKENIZER:N_ROWS], emb0[LEN_TOKENIZER:N_ROWS])  # 预留行原值不用
    if tied:
        assert head is emb
    else:
        assert torch.equal(head[:LEN_TOKENIZER], head0[:LEN_TOKENIZER])
        assert torch.equal(head[new], head0[:N_BASE_VOCAB].double().mean(0).float().expand(len(new), -1))
    allowed = [ids[f"<g{MAX_NEW_GRID_TOKENS - 2}>"], ids[f"<g{MAX_NEW_GRID_TOKENS - 1}>"]]
    assert min(allowed) >= N_ROWS  # 扩出来的 id 能在前向与限制生成里正常使用
    inp = ad.build_inputs(images[:1], INSTR_A, "image_first")
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=3, allowed_token_ids=allowed)
    assert set(res.token_ids) <= set(allowed) and res.n_generated_tokens == 3


@pytest.mark.parametrize("tied", [False, True])
def test_grow_step_copies_all_existing_rows_bitwise(fresh, tied):
    ad = fresh(tied)
    emb0 = ad.model.get_input_embeddings().weight.detach().clone()
    head0 = ad.model.get_output_embeddings().weight.detach().clone()
    ad._ensure_rows(N_ROWS + 7)
    emb = ad.model.get_input_embeddings().weight
    head = ad.model.get_output_embeddings().weight
    assert emb.shape == head.shape == (N_ROWS + 7, HIDDEN_TINY)
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
    ad = fresh()
    _forbid_resize(monkeypatch, ad)
    ad.add_grid_tokens([f"<g{i}>" for i in range(260)])
    snap = ad.model.get_input_embeddings().weight.detach().clone()
    ad.add_grid_tokens([f"<h{i}>" for i in range(10)])
    emb = ad.model.get_input_embeddings().weight
    assert emb.shape[0] == ad.model.get_output_embeddings().weight.shape[0] == LEN_TOKENIZER + 271  # 含占位符末行
    assert torch.equal(emb[: snap.shape[0]], snap)
    ad2 = fresh()
    ad2.add_grid_tokens(["<g0>"])
    assert ad2.model.get_input_embeddings().weight.shape[0] == N_ROWS  # 行数富余时不动，更不会缩到 len(tokenizer)


def test_add_grid_tokens_rejects_existing_names(fresh):
    ad = fresh()
    for names in (["<think>"], ["<|im_end|>"], ["<IMG_CONTEXT>"], "<g0>", [], ["<g0>", "<g0>"], [""]):
        with pytest.raises((ValueError, TypeError)):
            ad.add_grid_tokens(names)
    ad.add_grid_tokens(["<g0>"])
    with pytest.raises(ValueError, match="已在词表"):
        ad.add_grid_tokens(["<g1>", "<g0>"])
    assert len(ad.tokenizer) == LEN_TOKENIZER + 1 and ad.n_grid_tokens == 1


# ---------------------------------------------------------------------------
# 生成与思考段
# ---------------------------------------------------------------------------


def test_allowed_token_ids_restrict_generation(fresh, images):
    ad = fresh()
    ids = list(ad.add_grid_tokens(["<g0>", "<g1>"]).values())
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=6, allowed_token_ids=ids)
    assert set(res.token_ids) <= set(ids) and res.n_generated_tokens == 6
    assert res.text == ad.tokenizer.decode(list(res.token_ids))  # grid token 是特殊 token，但 text 保留它们


def test_only_im_end_stops_generation(adapter, images):
    """D13-3：InternVL 的停止 token 只有 <|im_end|>；<|endoftext|> 是 pad，不停。"""
    inp = adapter.build_inputs(images[:1], INSTR_A, "image_first")
    stop, eot = adapter.token_id(IM_END), adapter.token_id(END_OF_TEXT)
    res = adapter.generate(inp, enable_thinking=False, max_new_tokens=3, allowed_token_ids=[stop])
    assert res.token_ids == (stop,) and res.text == ""
    res = adapter.generate(inp, enable_thinking=False, max_new_tokens=3, allowed_token_ids=[eot])
    assert res.token_ids == (eot,) * 3 and res.text == END_OF_TEXT * 3


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
    n_sys = len(adapter.system_prefix_ids(thinking))
    assert (n_sys > 0) is thinking
    assert res.n_prompt_tokens == n_sys + inp.tensors["input_ids"].shape[1] + len(adapter.assistant_header_ids(thinking))
    assert res.n_visual_tokens == 2 * 256
    assert 1 <= res.n_generated_tokens <= 4 and res.wall_time_s > 0
    assert res.raw_text == adapter.tokenizer.decode(list(res.token_ids), skip_special_tokens=False)
    assert (res.text, res.raw_text, res.thinking_unclosed) == adapter.split_answer(res.token_ids, enable_thinking=thinking)
    stop, start, end = adapter.token_id(IM_END), adapter.token_id(THINK_START), adapter.token_id(THINK_END)
    forced = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=3, allowed_token_ids=[stop])
    assert forced.token_ids == (stop,) and forced.text == "" and forced.thinking_unclosed is False  # 没写思考段
    opened = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=3, allowed_token_ids=[start])
    assert opened.token_ids == (start,) * 3 and opened.text == "" and opened.thinking_unclosed is True  # 开了没关
    closed = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=2, allowed_token_ids=[end])
    assert closed.token_ids == (end,) * 2 and closed.text == "" and closed.thinking_unclosed is False


@pytest.mark.parametrize("thinking", [False, True])
def test_greedy_generate_matches_full_forward(adapter, images, thinking):
    """generate（带 KV cache、pixel_values 只在预填充时用）逐步等于对完整提示做前向后取最大 logit。"""
    inp = adapter.build_inputs(images[:1], INSTR_A, "instruction_first")
    res = adapter.generate(inp, enable_thinking=thinking, max_new_tokens=3)
    ids = adapter.prompt_ids(inp, thinking)
    for tok_id in res.token_ids:
        x = torch.tensor([ids])
        with torch.no_grad():
            logits = adapter.model(input_ids=x, attention_mask=torch.ones_like(x),
                                   pixel_values=inp.tensors["pixel_values"], use_cache=False,
                                   logits_to_keep=1).logits[0, -1]
        assert logits[tok_id] >= logits.max() - 1e-4
        ids.append(tok_id)


def test_think_tokens_are_single_and_not_special(components):
    """§7.1：<think>/</think> 是单个、非 special 的 token，所以 skip_special_tokens=True 也删不掉——剥离必须按 id 做。"""
    tok = components[1]
    added = {d["content"]: d for d in FACTS["added_special_tokens(non-unused)"]}
    for s in (THINK_START, THINK_END):
        assert added[s]["special"] is False
        assert _enc(tok, s) == [added[s]["id"]]
    s = "<think>\nx\n</think>\n\ny"
    assert tok.decode(_enc(tok, s), skip_special_tokens=True) == s


def test_split_answer_r1_format(adapter, components):
    """R1 prompt 要求的写法：思考段包在 <think> 与 </think> 里，答案与思考段之间隔一个换行（按真实 tokenizer 实测分词）。"""
    tok = components[1]
    start, end, stop = adapter.token_id(THINK_START), adapter.token_id(THINK_END), adapter.token_id(IM_END)
    body = "<think>\nThe image is mostly blue.\n</think>"
    for sep in ("\n", "\n\n"):
        ids = _enc(tok, body + sep + "Blue.") + [stop]
        assert ids[0] == start and ids.count(end) == 1
        text, raw, unclosed = adapter.split_answer(ids, enable_thinking=True)
        assert (text, unclosed) == ("Blue.", False) and raw == body + sep + "Blue.<|im_end|>"
    text, _, _ = adapter.split_answer(_enc(tok, "<think>a</think><think>b</think>\nC"), enable_thinking=True)
    assert text == "C"  # 取最后一个 </think> 之后
    text, raw, unclosed = adapter.split_answer(_enc(tok, "<think>\nstill going"), enable_thinking=True)
    assert (text, unclosed) == ("", True) and raw == "<think>\nstill going"  # max_new_tokens 用完
    text, raw, unclosed = adapter.split_answer(_enc(tok, "</think>\nBlue."), enable_thinking=True)
    assert (text, unclosed) == ("Blue.", False)  # 只有 </think>：取它之后


def test_split_answer_unclosed_only_when_think_opened(adapter, components):
    """第 1 轮审查（与 Gemma 同一条规则）：只有写了 <think>、之后没有 </think> 才算未闭合；模型没写思考段就直接作答时
    取全文，不论 enable_thinking；规则只看输出。"""
    tok = components[1]
    stop = adapter.token_id(IM_END)
    for thinking in (False, True):
        text, raw, unclosed = adapter.split_answer(_enc(tok, "Blue.") + [stop], enable_thinking=thinking)
        assert (text, unclosed) == ("Blue.", False) and raw == "Blue.<|im_end|>"  # 没写思考段：取全文
        text, _, unclosed = adapter.split_answer(_enc(tok, "<think>a</think>\nB<think>c"), enable_thinking=thinking)
        assert (text, unclosed) == ("", True)  # 最后一个 <think> 在最后一个 </think> 之后：又开了一段没关
        text, _, unclosed = adapter.split_answer(_enc(tok, "<think>a"), enable_thinking=thinking)
        assert (text, unclosed) == ("", True)
        assert adapter.split_answer([], enable_thinking=thinking) == ("", "", False)
    with pytest.raises(TypeError):
        adapter.split_answer([stop], enable_thinking=None)


def test_split_answer_keeps_special_tokens(adapter, components):
    tok = components[1]
    stop = adapter.token_id(IM_END)
    box = "<ref>the cat</ref><box>[[1, 2, 3, 4]]</box>"
    assert adapter.split_answer(_enc(tok, box) + [stop], enable_thinking=False)[0] == box  # 特殊 token 保留
    eot = adapter.token_id(END_OF_TEXT)
    assert adapter.split_answer(_enc(tok, "x") + [eot], enable_thinking=False)[0] == "x" + END_OF_TEXT  # 不是停止 token


def test_generation_result_contract():
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


def test_entry_assertion_when_submodule_on_meta(fresh, images):
    ad = fresh()
    inp = ad.build_inputs(images, INSTR_A, "image_first")
    ad.model.model.multi_modal_projector.to("meta")
    with pytest.raises(AdapterDeviceError, match="multi_modal_projector"):
        ad.vision_features(inp)


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
    model = internvl35.build_tiny_model(components[0], dtype=torch.bfloat16)
    ad = internvl35.make_tiny_adapter(components, **SINGLE, model=model)
    assert ad.dtype == torch.bfloat16
    inp = ad.build_inputs(images, INSTR_A, "instruction_first")
    assert inp.tensors["pixel_values"].dtype == torch.bfloat16
    with torch.no_grad():
        hs = ad.hidden_states(inp, [0, N_LAYERS_TINY])
        vf = ad.vision_features(inp)
    assert hs.image[N_LAYERS_TINY].dtype == torch.bfloat16 and torch.equal(vf, hs.image[0])


# ---------------------------------------------------------------------------
# 原生坐标 <box>[[x1, y1, x2, y2]]</box> → 内部绝对像素 xyxy（D13-4、§7.5）
# ---------------------------------------------------------------------------


def _gt_table():
    sec = _section("### 7.5 ", "### 7.6 ")
    pat = r"^\| (\S+\.jpg) \((\d+)×(\d+)\) \| (\w+) \| \[([^\]]+)\] \| \[([^\]]+)\] \|$"
    rows = []
    for img, w, h, obj, xyxy, k1000 in re.findall(pat, sec, flags=re.M):
        rows.append((img, int(w), int(h), obj, [float(v) for v in xyxy.split(",")], [int(v) for v in k1000.split(",")]))
    return rows


def test_gt_table_uses_x_first_per_mille():
    """§7.5 GT 表的两列互相印证"x 在前、按原图宽高归一到 0–1000"。表里的像素值只保留一位小数，
    千分值是用原始精度算的（例如 tv 的 y2：262.6/426·1000 = 616.4，表里是 617），所以只要求差 < 1。"""
    rows = _gt_table()
    assert len(rows) == 4
    for _, w, h, _, xyxy, k1000 in rows:
        assert np.abs(abs_to_norm1000(xyxy, width=w, height=h) - np.asarray(k1000)).max() < 1.0


def test_parse_box_on_run0002_zero_shot_outputs():
    """§7.5 的两条实测输出：x 在前的读法与 GT 的 IoU ≥ 0.5；把轴读反（y 在前）则 < 0.5，钉住轴序。"""
    gt = _gt_table()
    assert len(BOX_CASES) == 2
    for case in BOX_CASES:
        w, h = case["image_wh"]
        out = case["raw_output_with_special"]
        p = parse_one_box(out, width=w, height=h)
        assert p.failure is None
        q = [int(v) for v in re.search(r"\[\[([^\]]+)\]\]", out).group(1).split(",")]
        assert p.box == pytest.approx((q[0] * w / 1000, q[1] * h / 1000, q[2] * w / 1000, q[3] * h / 1000), abs=1e-9)
        boxes = [xyxy for img, gw, gh, _, xyxy, _ in gt if img == case["image"] and (gw, gh) == (w, h)]
        assert boxes
        assert max(box_iou(p.box, g) for g in boxes) >= BOX_ACC_IOU_THRESHOLD
        swapped = (q[1] * w / 1000, q[0] * h / 1000, q[3] * w / 1000, q[2] * h / 1000)
        assert max(box_iou(swapped, g) for g in boxes) < BOX_ACC_IOU_THRESHOLD


@pytest.mark.parametrize("text, n", [
    ("<box>[[1, 2, 3, 4], [5, 6, 7, 8]]</box>", 2),
    ("<ref>a</ref><box>[[1,2,3,4]]</box> and <ref>b</ref><box>[[5,6,7,8]]</box>", 2),
    ("<box>\n[ [0, 0, 1000, 1000] ]\n</box>", 1),
    ("<box>[[10, 10, 10, 20]]</box>", 1),  # 退化框照样返回
    ("<ref>x</ref><box>[[1, 2, 3, 4]]</box><|im_end|>", 1),  # 标签外的文字与停止 token 忽略
])
def test_parse_boxes_accepts(text, n):
    p = parse_boxes(text, width=200, height=100)
    assert p.failure is None and len(p.boxes) == n
    assert all(len(b) == 4 and all(isinstance(v, float) for v in b) for b in p.boxes)


def test_parse_boxes_order_and_scale():
    p = parse_boxes("<box>[[0, 0, 1000, 1000]]</box><box>[[500, 250, 750, 1000]]</box>", width=640, height=480)
    assert p.boxes == ((0.0, 0.0, 640.0, 480.0), (320.0, 120.0, 480.0, 480.0))


@pytest.mark.parametrize("text, failure", [
    ("no box here", BoxParseFailure.NO_BOX),
    ("[[24, 114, 492, 984]]", BoxParseFailure.NO_BOX),  # 裸列表，没有标签：不猜
    ("<box>[[1, 2, 3, 4]]", BoxParseFailure.UNBALANCED_TAGS),
    ("</box>[[1, 2, 3, 4]]<box>", BoxParseFailure.UNBALANCED_TAGS),
    ("<box><box>[[1, 2, 3, 4]]</box></box>", BoxParseFailure.UNBALANCED_TAGS),
    ("<box>[1, 2, 3, 4]</box>", BoxParseFailure.MALFORMED),  # 单层列表
    ("<box>[[1, 2, 3]]</box>", BoxParseFailure.MALFORMED),
    ("<box>[[1, 2, 3, 4, 5]]</box>", BoxParseFailure.MALFORMED),
    ("<box>[[1.5, 2, 3, 4]]</box>", BoxParseFailure.MALFORMED),
    ("<box>[[-1, 2, 3, 4]]</box>", BoxParseFailure.MALFORMED),
    ("<box>[[１, 2, 3, 4]]</box>", BoxParseFailure.MALFORMED),  # 全角数字
    ("<box>[]</box>", BoxParseFailure.MALFORMED),
    ("<box>[[1, 2, 3, 4]] cat</box>", BoxParseFailure.MALFORMED),
    ("<box>[[1, 2, 3, 4]]</box><box>oops</box>", BoxParseFailure.MALFORMED),  # 一个坏了整条判失败
    ("<box>[[0, 0, 1001, 5]]</box>", BoxParseFailure.OUT_OF_RANGE),
    ("<box>[[500, 0, 100, 5]]</box>", BoxParseFailure.INVERTED),
    ("<box>[[0, 50, 100, 5]]</box>", BoxParseFailure.INVERTED),
])
def test_parse_boxes_failures(text, failure):
    assert parse_boxes(text, width=640, height=480) == (None, None, failure)  # DEV-0006：BoxParse 增加 labels 字段
    assert parse_one_box(text, width=640, height=480) == (None, failure)


def test_parse_one_box_refuses_multiple():
    assert parse_one_box("<box>[[1, 2, 3, 4], [5, 6, 7, 8]]</box>", width=10, height=10) == (
        None, BoxParseFailure.MULTIPLE_BOXES)
    assert parse_one_box("<box>[[1, 2, 3, 4]]</box><box>[[1, 2, 3, 4]]</box>", width=10, height=10).failure is (
        BoxParseFailure.MULTIPLE_BOXES)
    one = parse_one_box("<box>[[100, 200, 300, 400]]</box>", width=10, height=10)
    assert one.failure is None and one.box == pytest.approx((1.0, 2.0, 3.0, 4.0))


def test_parse_boxes_validates_arguments():
    with pytest.raises(ValueError):
        parse_boxes("<box>[[1, 2, 3, 4]]</box>", width=0, height=10)
    with pytest.raises(ValueError):
        parse_boxes("no box", width=10, height=-1)
    with pytest.raises(TypeError):
        parse_boxes(None, width=10, height=10)


# ---------------------------------------------------------------------------
# 冒烟脚本的 CPU 彩排（端到端，与服务器同一流程）
# ---------------------------------------------------------------------------


def test_smoke_script_tiny_internvl(tmp_path):
    out = tmp_path / "smoke_internvl_tiny.json"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/smoke_adapter.py"), "--model", "internvl3.5-2b-hf", "--tiny",
         "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert out.exists(), proc.stderr[-3000:]
    report = json.loads(out.read_text())
    assert proc.returncode == 0 and report["all_ok"] is True, json.dumps(report.get("ok"), ensure_ascii=False)
    assert list(report["ok"]) == [
        "0_facts", "1_build_generate", "2_task_modulation", "3_grid_tokens", "4_query_tokens",
        "5_position_ids", "6_vision_features", "7_thinking", "8_device_audit", "9_multi_tile",
    ]
    checks = report["checks"]
    assert checks["0_facts"]["tile_policy"] == {"crop_to_patches": False, "max_patches": 1, "min_patches": 1,
                                                "tile_px": 448}
    assert checks["4_query_tokens"]["grid2d"]["rejected"] is True
    assert checks["9_multi_tile"]["n_tiles"] == 7 and checks["9_multi_tile"]["grid_hw"] == [112, 16]
    assert all(c["processor_compare"]["ok"] for c in checks["1_build_generate"]["cases"])
    think = checks["7_thinking"]  # 随机小模型写不写 <think> 取决于权重，这里只核对情形与标记一致
    assert think["case"] in ("closed", "unclosed", "no_think")
    assert think["thinking_unclosed"] is (think["case"] == "unclosed")
    assert (think["text"] == "") if think["case"] == "unclosed" else think["system_turn_in_prompt"]


def test_smoke_inputs_are_modelinputs(adapter, images):
    """冒烟与测试都依赖的约定：build_inputs 返回 base 的 ModelInputs，order 原样保留。"""
    for order in ORDERS:
        inp = adapter.build_inputs(images[:1], INSTR_A, order)
        assert isinstance(inp, ModelInputs) and inp.order == order
