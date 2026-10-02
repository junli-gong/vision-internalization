"""DEV-0024 第 3 条 I1：hidden_states 的可选层键 PRE_NORM_LAST（最后一个 decoder 层的原始输出，最终归一化之前）。

三个家族（小型随机模型，CPU fp32）：
- 它等于独立挂在最后一个 decoder 层上的钩子抓到的原始输出（图像与查询位置），且 norm(它) 等于层号 N 的读数，而它本身 ≠ 层号 N；
- 与整数层号混用时顺序、键原样保留，整数层的读数与只给整数时逐位相同；只给 PRE_NORM_LAST 也可以；
- 钩子用完即摘；非法的层键照旧报错；只收整数的接口（sequence_hidden_states）照旧拒绝它。
另：base.ModelAdapter.trainable_token_rows 的桩（D45-5）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters.base import PRE_NORM_LAST, ModelAdapter, capture_output, captured_output, resolve_layer_keys
from src.adapters.factory import make_adapter

FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")


@pytest.fixture(scope="module", params=FAMILIES)
def ad(request):
    return make_adapter(request.param, device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)


def _img(w: int = 640, h: int = 480, seed: int = 0) -> Image.Image:
    return Image.fromarray(np.random.default_rng(seed).uniform(0, 255, size=(h, w, 3)).astype(np.uint8))


def _raw_last_and_norm(ad, inputs, layers, **kw):
    """独立核对：在最后一个 decoder 层与最终 norm 上各挂一个钩子（不经过适配器的 capture_output）。"""
    lm = ad.model.model.language_model
    cap = {}
    h1 = lm.layers[ad.num_layers() - 1].register_forward_hook(
        lambda m, a, o: cap.__setitem__("raw", (o[0] if isinstance(o, tuple) else o).detach().clone()))
    h2 = lm.norm.register_forward_hook(lambda m, a, o: cap.__setitem__("norm", o.detach().clone()))
    try:
        with torch.no_grad():
            hs = ad.hidden_states(inputs, layers, **kw)
    finally:
        h1.remove()
        h2.remove()
    return hs, cap, lm.norm


@pytest.mark.parametrize("order", ["image_first", "instruction_first"])
def test_pre_norm_last_is_raw_last_layer_output_and_norm_gives_layer_n(ad, order):
    inp = ad.build_inputs([_img()], "Describe this image.", order)
    N = ad.num_layers()
    last = ad.model.model.language_model.layers[N - 1]
    hooks_before = dict(last._forward_hooks)  # transformers 自己的 output-capturing 钩子
    hs, cap, norm = _raw_last_and_norm(ad, inp, [PRE_NORM_LAST, 0, N])
    assert hs.layers == (PRE_NORM_LAST, 0, N) and list(hs.image) == [PRE_NORM_LAST, 0, N]
    s, e = inp.images[0].span.start, inp.images[0].span.end
    pre = hs.image[PRE_NORM_LAST]
    assert pre.shape == hs.image[N].shape == (e - s, ad.hidden_size())
    assert torch.equal(pre, cap["raw"][0, s:e])  # 最后一个 decoder 层的原始输出
    assert torch.equal(hs.image[N], cap["norm"][0, s:e])  # 层号 N = 最终 norm 的输出
    with torch.no_grad():
        assert torch.equal(norm(pre[None])[0], hs.image[N])  # norm(PRE_NORM_LAST) = 层号 N
    assert not torch.equal(pre, hs.image[N])  # 归一化之前与之后不是同一种量
    with torch.no_grad():
        ints = ad.hidden_states(inp, [0, N])
    assert ints.layers == (0, N) and all(torch.equal(ints.image[l], hs.image[l]) for l in (0, N))
    # 适配器的钩子用完即摘（剩下的只有 transformers 首次前向时自己装的 output-capturing 钩子）
    left = [h.__qualname__ for h in last._forward_hooks.values()]
    assert not any("capture_output" in q for q in left) and set(hooks_before) <= set(last._forward_hooks), left


def test_pre_norm_last_alone_and_on_query_positions(ad):
    with torch.no_grad():
        only = ad.hidden_states(ad.build_inputs([_img(seed=1)], "Describe this image.", "image_first"), [PRE_NORM_LAST])
    assert only.layers == (PRE_NORM_LAST,) and set(only.image) == {PRE_NORM_LAST}
    q = make_adapter(ad.model_key.removeprefix("tiny(").removesuffix(")"), device="cpu", dtype=torch.float32, tiny=True,
                     init_std=0.05)
    q.inject_query_tokens(4, "seq1d")
    inp = q.build_inputs([_img(seed=2)], "Describe this image.", "image_first")
    qe = torch.randn(4, q.hidden_size(), generator=torch.Generator().manual_seed(0)) * 0.05
    N = q.num_layers()
    hs, cap, _ = _raw_last_and_norm(q, inp, [N, PRE_NORM_LAST], query_embeds=qe)
    qs, qe_ = inp.query_span.start, inp.query_span.end
    assert set(hs.query) == {N, PRE_NORM_LAST}
    assert torch.equal(hs.query[PRE_NORM_LAST], cap["raw"][0, qs:qe_]) and torch.equal(hs.query[N], cap["norm"][0, qs:qe_])


def test_layer_key_errors(ad):
    inp = ad.build_inputs([_img(seed=3)], "Describe this image.", "image_first")
    N = ad.num_layers()
    with pytest.raises(ValueError, match="出现了 2 次"):
        ad.hidden_states(inp, [PRE_NORM_LAST, 1, PRE_NORM_LAST])
    with pytest.raises(TypeError):
        ad.hidden_states(inp, [1, "pre_norm"])  # 别的字符串照旧是"层号必须是整数"
    with pytest.raises(TypeError):
        ad.hidden_states(inp, PRE_NORM_LAST)  # 单个字符串不是序列
    with pytest.raises(ValueError, match="越界"):
        ad.hidden_states(inp, [PRE_NORM_LAST, N + 1])
    with pytest.raises(ValueError, match="为空"):
        ad.hidden_states(inp, [])
    with pytest.raises(TypeError):
        ad.sequence_hidden_states(inp, [PRE_NORM_LAST])  # 只收整数的接口照旧拒绝


def test_resolve_layer_keys_keeps_integer_path_unchanged():
    seen = []

    def check(xs):
        seen.append(list(xs))
        return [int(x) for x in xs]

    assert resolve_layer_keys([3, 1], check) == ((3, 1), [3, 1], False) and seen == [[3, 1]]
    assert resolve_layer_keys([PRE_NORM_LAST], check) == ((PRE_NORM_LAST,), [], True) and len(seen) == 1  # 不调 check
    assert resolve_layer_keys([2, PRE_NORM_LAST, np.int64(0)], check) == ((2, PRE_NORM_LAST, 0), [2, 0], True)


def test_captured_output_requires_exactly_one_call():
    m = torch.nn.Linear(2, 2)
    with capture_output(m) as cap:
        m(torch.zeros(1, 2))
        y = m(torch.ones(1, 2))
    assert len(m._forward_hooks) == 0 and torch.equal(cap["out"], y)
    with pytest.raises(RuntimeError, match="2 次"):
        captured_output(cap, what="x")
    with capture_output(m) as cap:
        pass
    with pytest.raises(RuntimeError, match="0 次"):
        captured_output(cap, what="x")


def test_base_trainable_token_rows_stub():
    """D45-5：基类的 trainable_token_rows 是占位实现（NotImplementedError），三个家族的实现来自 CheckpointSupport。"""
    from src.adapters.checkpoint import CheckpointSupport
    from src.adapters.gemma4 import Gemma4Adapter
    from src.adapters.internvl35 import InternVL35Adapter
    from src.adapters.qwen35 import Qwen35Adapter

    assert "trainable_token_rows" in ModelAdapter.__dict__ and "trainable_token_rows" not in ModelAdapter.__abstractmethods__
    with pytest.raises(NotImplementedError, match="trainable_token_rows"):
        ModelAdapter.trainable_token_rows(object.__new__(Qwen35Adapter), ["<g0>"])
    for cls in (Qwen35Adapter, Gemma4Adapter, InternVL35Adapter):
        assert cls.trainable_token_rows is CheckpointSupport.trainable_token_rows
