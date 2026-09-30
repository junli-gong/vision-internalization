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
        return GenerationResult("x", (7,), 1, 3, 0, 0.0)

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
