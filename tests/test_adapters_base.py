import re
from pathlib import Path

import pytest

from src.adapters import base
from src.adapters.base import (
    GenerationResult,
    HiddenStates,
    ImageTokenInfo,
    ModelAdapter,
    ModelInputs,
    Span,
    check_grid_token_names,
)
from src.constants import MAX_NEW_GRID_TOKENS
from src.parsing import BoxParse, ParseFailure


class _Dummy(ModelAdapter):
    """签名与基类一致的最小实现，只为测试调用约束。"""

    def load(self, model_id, dtype):
        pass

    def build_inputs(self, images, instruction, order):
        return ModelInputs({}, order, (), Span(0, 0))

    def generate(self, inputs, *, enable_thinking, allowed_token_ids=None, max_new_tokens, **gen_kwargs):
        self.calls = getattr(self, "calls", []) + [("generate", 1, enable_thinking, max_new_tokens, gen_kwargs)]
        return GenerationResult("x", (7,), 1, 3, 0, 0.0)

    def generate_batch(self, inputs_list, *, enable_thinking, max_new_tokens, allowed_token_ids=None, **gen_kwargs):
        self.calls = getattr(self, "calls", []) + [("generate_batch", len(inputs_list), enable_thinking, max_new_tokens,
                                                    gen_kwargs)]
        return [GenerationResult("x", (7,), 1, 3, 0, 0.0) for _ in inputs_list]

    def option_logits(self, inputs_list, options, *, enable_thinking):  # DEV-0016
        self.calls = getattr(self, "calls", []) + [("option_logits", len(inputs_list), enable_thinking)]
        return []

    def hidden_states(self, inputs, layers):
        return HiddenStates(tuple(layers), {l: None for l in layers})

    def vision_features(self, inputs):
        return None

    def inject_query_tokens(self, n, position_mode):
        pass

    def add_grid_tokens(self, names):
        check_grid_token_names(names)
        return {n: i for i, n in enumerate(names)}

    def num_layers(self):
        return 2

    def hidden_size(self):
        return 8

    def parse_boxes(self, text, *, width, height):  # DEV-0006：parse_boxes 成为抽象方法
        return BoxParse(None, None, ParseFailure.NO_BOX)


def _inputs():
    return ModelInputs({}, "instruction_first", (ImageTokenInfo(Span(5, 17), (3, 4)),), Span(0, 5))


def test_generate_requires_enable_thinking_and_max_new_tokens():
    m = _Dummy()
    with pytest.raises(TypeError):
        m.generate(_inputs(), max_new_tokens=4)
    with pytest.raises(TypeError):
        m.generate(_inputs(), enable_thinking=False)
    assert m.generate(_inputs(), enable_thinking=False, max_new_tokens=4).text == "x"


def test_subclass_cannot_default_enable_thinking():
    with pytest.raises(TypeError, match="enable_thinking"):

        class _Bad(_Dummy):
            def generate(self, inputs, *, enable_thinking=False, allowed_token_ids=None, max_new_tokens, **kw):
                pass


def test_subclass_cannot_default_max_new_tokens():
    with pytest.raises(TypeError, match="max_new_tokens"):

        class _Bad(_Dummy):
            def generate(self, inputs, *, enable_thinking, allowed_token_ids=None, max_new_tokens=64, **kw):
                pass


def test_subclass_cannot_make_enable_thinking_positional():
    with pytest.raises(TypeError, match="enable_thinking"):

        class _Bad(_Dummy):
            def generate(self, inputs, enable_thinking, *, allowed_token_ids=None, max_new_tokens, **kw):
                pass


def test_subclass_cannot_default_layers():
    with pytest.raises(TypeError, match="layers"):

        class _Bad(_Dummy):
            def hidden_states(self, inputs, layers=(3,)):
                pass


def test_abstract_base_not_instantiable():
    with pytest.raises(TypeError):
        ModelAdapter()


def test_vision_features_is_abstract():
    """D15-2：vision_features 是抽象方法，子类不实现就不能实例化。"""
    assert "vision_features" in ModelAdapter.__abstractmethods__

    class _NoVision(ModelAdapter):
        load = _Dummy.load
        build_inputs = _Dummy.build_inputs
        generate = _Dummy.generate
        generate_batch = _Dummy.generate_batch  # DEV-0011
        option_logits = _Dummy.option_logits  # DEV-0016
        hidden_states = _Dummy.hidden_states
        inject_query_tokens = _Dummy.inject_query_tokens
        add_grid_tokens = _Dummy.add_grid_tokens
        num_layers = _Dummy.num_layers
        hidden_size = _Dummy.hidden_size
        parse_boxes = _Dummy.parse_boxes  # DEV-0006：只缺 vision_features 一个

    with pytest.raises(TypeError, match="vision_features"):
        _NoVision()
    assert _Dummy().vision_features(_inputs()) is None


def test_model_inputs_validation():
    with pytest.raises(ValueError):
        ModelInputs({}, "image_last", (), Span(0, 1))
    with pytest.raises(ValueError):  # query_span 给了但没给 position_mode
        ModelInputs({}, "image_first", (), Span(0, 1), query_span=Span(1, 5))
    with pytest.raises(ValueError):
        ModelInputs({}, "image_first", (), Span(0, 1), query_span=Span(1, 5), query_position_mode="grid3d")
    mi = ModelInputs({}, "image_first", (), Span(0, 1), query_span=Span(1, 5), query_position_mode="grid2d")
    assert len(mi.query_span) == 4
    with pytest.raises(ValueError):
        Span(5, 3)


def test_generation_result_consistency():
    with pytest.raises(ValueError):
        GenerationResult("ab", (1, 2), 3, 10, 4, 0.1)
    with pytest.raises(ValueError):
        GenerationResult("ab", (1, 2), 2, -1, 4, 0.1)


def test_generation_result_new_fields_default_and_validation():
    """DEV-0011：stopped / n_image_pixels / batch_size / batch_wall_time_s 都有默认值，旧的构造方式照旧可用。"""
    r = GenerationResult("ab", (1, 2), 2, 10, 4, 0.1)
    assert (r.stopped, r.n_image_pixels, r.batch_size, r.batch_wall_time_s) == (None, None, 1, None)
    ok = GenerationResult("ab", (1, 2), 2, 10, 4, 0.25, stopped=True, n_image_pixels=4096, batch_size=4,
                          batch_wall_time_s=1.0)
    assert ok.wall_time_s * ok.batch_size == ok.batch_wall_time_s
    assert GenerationResult("", (), 0, 10, 0, 0.0, stopped=False, n_image_pixels=0).stopped is False
    with pytest.raises(TypeError):
        GenerationResult("ab", (1, 2), 2, 10, 4, 0.1, stopped=1)
    with pytest.raises(ValueError):  # 停下来了却没有 token
        GenerationResult("", (), 0, 10, 0, 0.0, stopped=True)
    with pytest.raises(ValueError):  # 有视觉 token 却 0 个像素
        GenerationResult("ab", (1, 2), 2, 10, 4, 0.1, n_image_pixels=0)
    with pytest.raises(ValueError):  # 没有视觉 token 却有像素
        GenerationResult("ab", (1, 2), 2, 10, 0, 0.1, n_image_pixels=16)
    with pytest.raises(ValueError):
        GenerationResult("ab", (1, 2), 2, 10, 4, 0.1, n_image_pixels=-1)
    with pytest.raises(ValueError):
        GenerationResult("ab", (1, 2), 2, 10, 4, 0.1, batch_size=0)
    with pytest.raises(ValueError):  # 均摊份额不是整批墙钟的 1/batch_size
        GenerationResult("ab", (1, 2), 2, 10, 4, 0.5, batch_size=4, batch_wall_time_s=1.0)


def test_image_token_info_n_pixels():
    assert ImageTokenInfo(Span(0, 4), (2, 2)).n_pixels is None  # 旧的构造方式
    assert ImageTokenInfo(Span(0, 4), (2, 2), n_pixels=4096).n_pixels == 4096
    for bad in (0, -5, 1.5, True):
        with pytest.raises(ValueError):
            ImageTokenInfo(Span(0, 4), (2, 2), n_pixels=bad)


def test_generate_batch_is_abstract_and_has_the_same_required_kwargs():
    assert "generate_batch" in ModelAdapter.__abstractmethods__
    assert base._REQUIRED_KWONLY["generate_batch"] == base._REQUIRED_KWONLY["generate"]
    with pytest.raises(TypeError, match="max_new_tokens"):

        class _Bad(_Dummy):
            def generate_batch(self, inputs_list, *, enable_thinking, max_new_tokens=64, allowed_token_ids=None, **kw):
                pass

    with pytest.raises(TypeError, match="enable_thinking"):

        class _Bad2(_Dummy):
            def generate_batch(self, inputs_list, enable_thinking, *, max_new_tokens, allowed_token_ids=None, **kw):
                pass

    m = _Dummy()
    with pytest.raises(TypeError):
        m.generate_batch([_inputs()], max_new_tokens=4)
    assert len(m.generate_batch([_inputs()] * 3, enable_thinking=False, max_new_tokens=4)) == 3


def test_warmup_runs_one_uncounted_greedy_generation():
    m = _Dummy()
    t = m.warmup()
    assert t >= 0 and m.calls == [("generate", 1, False, 2, {"do_sample": False})]
    m.warmup(batch_size=3, max_new_tokens=4)
    assert m.calls[-1] == ("generate_batch", 3, False, 4, {"do_sample": False})
    with pytest.raises(ValueError):
        m.warmup(batch_size=0)


def test_hidden_states_layers_match():
    with pytest.raises(ValueError):
        HiddenStates((1, 2), {1: None})
    with pytest.raises(ValueError):
        HiddenStates((1, 2), {1: None, 2: None}, query={2: None})
    hs = _Dummy().hidden_states(_inputs(), [4, 1])
    assert hs.layers == (4, 1)


def test_grid_token_budget():
    names = [f"<g{i}>" for i in range(MAX_NEW_GRID_TOKENS)]
    assert len(_Dummy().add_grid_tokens(names)) == MAX_NEW_GRID_TOKENS
    with pytest.raises(ValueError):
        check_grid_token_names(names + ["<extra>"])
    with pytest.raises(ValueError):
        check_grid_token_names(["<a>", "<a>"])


def test_base_source_has_no_layer_numbers_and_no_transformers():
    src = Path(base.__file__).read_text()
    assert "transformers" not in src.replace("不 import transformers", "")
    assert not re.search(r"layers?\s*[=:\[(]\s*[\[(]?\s*-?\d", src)
    assert not re.search(r"第\s*-?\d+\s*层|\d+\s*层", src)
