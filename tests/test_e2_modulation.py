"""DEV-0023：E2 的两种输入顺序（简报 §12 因果注意力；思路同冒烟的 2_task_modulation）。

- 图在前（深度、语义分割、法向）：图像 token 看不到后面的文字，换一句提示，全部层在图像位置的特征逐位不变；
- 指令在前（指代分割）：换一个指代句（token 数相同，图像位置不动），第 0 层（嵌入）逐位不变，第 1 层起每一层都变。
走的是 E2 自己的路径（FeatureExtractor：配置里的顺序与模板 → build_inputs → hidden_states → 网格）。
"""

from __future__ import annotations

import dataclasses
import copy

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters.factory import make_adapter
from src.data.records import Sample
from src.e2.config import load_config
from src.e2.features import FeatureExtractor

FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
EXPRESSIONS = ("the cat", "the dog")  # 三个家族的分词器里都是同样多的 token（下面核对）


@pytest.fixture(scope="module", params=FAMILIES)
def adapter(request):
    return make_adapter(request.param, device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)


@pytest.fixture()
def sample(tmp_path):
    p = tmp_path / "img.png"
    Image.fromarray(np.random.default_rng(0).uniform(0, 255, size=(96, 128, 3)).astype(np.uint8)).save(p)

    def make(expr: str | None = None) -> Sample:
        return Sample(id=f"s/{expr}", task="refexp" if expr else "depth", images=(str(p),), image_keys=("k",),
                      fields={"expression": expr} if expr else {}, gt={}, source="t", split="t")

    return make


def _with_prompt(cfg, prompt: str):
    doc = copy.deepcopy(dict(cfg.doc))
    doc["inputs"] = {**doc["inputs"], "dense_prompt": prompt}
    return dataclasses.replace(cfg, doc=doc)


def test_image_first_features_do_not_depend_on_the_prompt(adapter, sample):
    """同样长的另一句：逐位不变。长度不同的一句：只差浮点舍入（Gemma 的小模型在 CPU 上序列长 286 对 278 时差 1e-5 量级，
    矩阵乘的分块随序列长度变；Qwen、InternVL 逐位相同）。E2 全程只用配置里那一句，不受影响。"""
    cfg = load_config()
    assert cfg.order("depth") == cfg.order("semseg") == cfg.order("normals") == "image_first"
    layers = list(range(adapter.num_layers() + 1))
    feats = []
    for prompt in (cfg.dense_prompt, "Describe the photo.", "What is the depth of every pixel here? Answer carefully."):
        ex = FeatureExtractor(adapter, _with_prompt(cfg, prompt), "depth", layers, [])
        raw, inp = ex.raw(sample(), phase="test")
        assert inp.order == "image_first" and inp.images[0].span.end <= inp.instruction_span.start
        feats.append((raw, inp))
    (r0, i0), (r1, i1), (r2, i2) = feats
    assert i0.tensors["input_ids"].shape == i1.tensors["input_ids"].shape
    assert i2.tensors["input_ids"].shape[1] > i0.tensors["input_ids"].shape[1]
    for l in layers:
        assert torch.equal(r0[l], r1[l]), f"图在前时第 {l} 层随提示变化"
        assert torch.allclose(r0[l], r2[l], rtol=1e-4, atol=1e-4), f"图在前时第 {l} 层随提示变化（超出浮点舍入）"


def test_instruction_first_features_are_modulated_by_the_expression(adapter, sample):
    cfg = load_config()
    assert cfg.order("refseg") == "instruction_first"
    layers = list(range(adapter.num_layers() + 1))
    ex = FeatureExtractor(adapter, cfg, "refseg", layers, [])
    out = []
    for e in EXPRESSIONS:
        raw, inp = ex.raw(sample(e), phase="test")
        assert inp.order == "instruction_first" and inp.instruction_span.end <= inp.images[0].span.start
        out.append((raw, inp))
    (ra, ia), (rb, ib) = out
    assert len(ia.instruction_span) == len(ib.instruction_span) and ia.images[0].span == ib.images[0].span
    assert torch.equal(ra[0], rb[0]), "第 0 层是嵌入输出，图像位置不该随指令变化"
    changed = [l for l in layers[1:] if not torch.equal(ra[l], rb[l])]
    assert changed == layers[1:], f"指令在前时这些层没有被指令调制：{sorted(set(layers[1:]) - set(changed))}"
    assert cfg.instruction("refseg", "  the cat ") == cfg.refseg_template.format(expression="the cat")
    with pytest.raises(ValueError):
        cfg.instruction("depth", "the cat")
