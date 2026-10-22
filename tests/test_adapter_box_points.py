"""DEV-0059（管理员审查的裁定 B）：parse_box_points——W2 的 box_points 提出（一个框加前景点、背景点）的解析与 0–1000 换算放在适配器里。
base 只有签名（其余家族 NotImplementedError）；Qwen3.5 实现：框走本家族的 parse_one_box，点是原生的 point_2d（同 bbox_2d 的 0–1000、
x 在前，按原图宽高换算）。不需要加载模型。"""

from __future__ import annotations

import inspect

import pytest

from src.adapters import qwen35
from src.adapters.base import BoxPointsParse, ModelAdapter
from src.adapters.gemma4 import Gemma4Adapter
from src.adapters.image_policy import PixelBudget
from src.adapters.internvl35 import InternVL35Adapter
from src.parsing import ParseFailure, parse_generation

KW = dict(width=200, height=100, max_points=2)
FULL = ('```json\n[{"bbox_2d": [100, 100, 500, 900], "label": "cat"}, {"point_2d": [300, 500], "label": "foreground"}, '
        '{"point_2d": [310, 510]}, {"point_2d": [320, 520], "label": "foreground"}, {"point_2d": [900, 900], "label": " Background "}, '
        '{"point_2d": [1200, 10], "label": "background"}, {"point_2d": [5, true]}, {"other": 1}]\n```')


def test_signature_and_other_families():
    sig = inspect.signature(ModelAdapter.parse_box_points)
    assert [(p.name, p.kind.name) for p in list(sig.parameters.values())[1:]] == \
        [("text", "POSITIONAL_OR_KEYWORD"), ("width", "KEYWORD_ONLY"), ("height", "KEYWORD_ONLY"), ("max_points", "KEYWORD_ONLY")]
    assert BoxPointsParse._fields == ("box", "fg_points", "bg_points", "n_points_dropped", "failure")
    for ad in (Gemma4Adapter(device="cpu", max_soft_tokens=70), InternVL35Adapter(device="cpu", crop_to_patches=False, max_patches=1)):
        with pytest.raises(NotImplementedError, match="parse_box_points"):
            ad.parse_box_points(FULL, **KW)


def test_qwen_parses_box_and_points():
    ad = qwen35.Qwen35Adapter(device="cpu", image_policy=PixelBudget(64))  # 不加载模型
    r = ad.parse_box_points(FULL, **KW)
    assert r == qwen35.parse_box_points(FULL, **KW)
    assert r.failure is None and r.box == (20.0, 10.0, 100.0, 90.0)  # 框与 parse_one_box 同一套换算
    assert r.box == tuple(qwen35.parse_one_box('[{"bbox_2d": [100, 100, 500, 900]}]', width=200, height=100).box)
    assert r.fg_points == ((60.0, 50.0), (62.0, 51.0))  # 没有 label 的算前景；第三个前景点超过上限
    assert r.bg_points == ((180.0, 90.0),)  # label 去空白、不分大小写
    assert r.n_points_dropped == 3  # 超上限 1 + 越界 1 + 不是两个数 1；{"other": 1} 忽略


@pytest.mark.parametrize("text, why", [
    ('[{"point_2d": [1, 2]}]', ParseFailure.NO_BOX),
    ('[{"bbox_2d": [1, 2, 3, 4]}, {"bbox_2d": [5, 6, 7, 8]}]', ParseFailure.MULTIPLE_BOXES),
    ('[{"bbox_2d": [1, 2, 3, 1200]}]', ParseFailure.OUT_OF_RANGE),
    ("no json here", ParseFailure.NO_BOX),
    ("", ParseFailure.NO_BOX),
    ("[1, 2]", ParseFailure.MALFORMED),
    ("```json\n[{\"bbox_2d\": [1, 2, 3\n```", ParseFailure.MALFORMED),
])
def test_qwen_failures(text, why):
    r = qwen35.parse_box_points(text, **KW)
    assert r.failure == why and r.box is None and r.fg_points is None and r.bg_points is None and r.n_points_dropped is None


def test_parse_generation_contract():
    """parse_generation 对思考段未闭合、截断、空回答先判，且要求对空串返回失败的 NamedTuple。"""
    ad = qwen35.Qwen35Adapter(device="cpu", image_policy=PixelBudget(64))
    view = lambda **k: type("R", (), {"thinking_unclosed": False, "stopped": True, "text": FULL, **k})()  # noqa: E731
    assert parse_generation(view(), ad.parse_box_points, **KW).failure is None
    assert parse_generation(view(stopped=False), ad.parse_box_points, **KW).failure == ParseFailure.TRUNCATED
    assert parse_generation(view(thinking_unclosed=True, text=""), ad.parse_box_points, **KW).failure == ParseFailure.THINKING_UNCLOSED
