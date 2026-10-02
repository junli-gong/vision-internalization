"""DEV-0020 第 9 条：类别名在提示里的 token 区间（E3a 开放词汇头）。CPU 小模型，三个家族的真实 tokenizer。

拼法（training.CLASS_NAME_SEPARATOR）：指令里恰好出现一次 " " + ", ".join(names)。区间是 " 名字" 这一段的 token（前导空格并进
第一个 token），解码回来就是 " 名字"；多 token 的类别名（"traffic light"、"potted plant"、"hair drier"）也对。
training_forward 的 HiddenRequest.spans 用这些区间读隐藏状态：第 0 层就是这些 token 的嵌入查表（开思考时下标偏移也对）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy
from src.adapters.training import CLASS_NAME_SEPARATOR, HiddenRequest, TrainingInputError, join_class_names

FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=False, max_patches=1),
}
NAMES = ["cat", "traffic light", "potted plant", "dog", "hair drier"]


@pytest.fixture(scope="module", params=FAMILY_KEYS)
def ad(request):
    key = request.param
    try:
        return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True)
    except OSError as e:
        pytest.fail(f"取不到 {key} 的小文件：{e!r}", pytrace=False)


@pytest.fixture(scope="module")
def image():
    return Image.fromarray(np.random.default_rng(0).integers(0, 256, (100, 150, 3), dtype=np.uint8))


def _instruction(names):
    return "Segment these classes: " + join_class_names(names) + "."


@pytest.mark.parametrize("order", ["instruction_first", "image_first"])
def test_spans_decode_to_names(ad, image, order):
    inp = ad.build_inputs([image], _instruction(NAMES), order)
    spans = ad.class_name_spans(inp, NAMES)
    ids = inp.tensors["input_ids"][0].tolist()
    assert [ad.tokenizer.decode(ids[s.start: s.end]) for s in spans] == [" " + n for n in NAMES]
    assert all(inp.instruction_span.start <= s.start < s.end <= inp.instruction_span.end for s in spans)
    assert all(a.end < b.start for a, b in zip(spans, spans[1:]))  # 中间隔着 ","
    assert max(len(s) for s in spans) >= 2  # 前提：确实有多 token 的类别名
    # 每个区间的 token 与单独编码 " 名字" 相同
    for s, n in zip(spans, NAMES):
        assert ids[s.start: s.end] == ad.tokenizer(" " + n, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize("thinking", [False, True])
def test_spans_feed_training_forward(ad, image, thinking):
    inp = ad.build_inputs([image], _instruction(NAMES), "instruction_first")
    spans = ad.class_name_spans(inp, NAMES)
    req = HiddenRequest(layers=(0, ad.num_layers()), spans=({n: s for n, s in zip(NAMES, spans)},))
    with torch.no_grad():
        tf = ad.training_forward([inp], [[]], enable_thinking=thinking, return_hidden=req)
        for n, s in zip(NAMES, spans):
            want = ad.model.get_input_embeddings()(inp.tensors["input_ids"][:, s.start: s.end])[0]
            assert torch.equal(tf.hidden[n][0][0], want), n
            assert tuple(tf.hidden[n][ad.num_layers()][0].shape) == (len(s), ad.hidden_size())


def test_errors(ad, image):
    inp = ad.build_inputs([image], _instruction(NAMES), "instruction_first")
    with pytest.raises(TrainingInputError, match="恰好出现一次"):
        ad.class_name_spans(inp, ["cat", "dog"])  # 顺序 / 内容与指令里的列表不同
    twice = ad.build_inputs([image], _instruction(["cat"]) + " Also: cat", "instruction_first")
    with pytest.raises(TrainingInputError, match="恰好出现一次"):
        ad.class_name_spans(twice, ["cat"])
    for bad in ([], ["a, b"], [" cat"], ["cat", "cat"], "cat", [""]):
        with pytest.raises(TrainingInputError):
            ad.class_name_spans(inp, bad)
    assert CLASS_NAME_SEPARATOR == ", " and join_class_names(["a", "b c"]) == "a, b c"
