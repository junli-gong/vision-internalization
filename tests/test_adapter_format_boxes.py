"""DEV-0027：format_boxes（parse_boxes 的逆；D56-5，C3 对照用本家族原生的框格式）与公开的停止 token 接口。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.adapters import gemma4, internvl35, qwen35, registry
from src.adapters.base import NATIVE_BOX_TOLERANCE, ModelAdapter, native_box_ints
from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy
from src.e3.tokens import family_stop_id

MODULES = {"qwen3.5": qwen35, "gemma4": gemma4, "internvl3.5": internvl35}
ADAPTERS = {
    "qwen3.5": lambda: qwen35.Qwen35Adapter(device="cpu", image_policy=PixelBudget(64)),
    "gemma4": lambda: gemma4.Gemma4Adapter(device="cpu", max_soft_tokens=70),
    "internvl3.5": lambda: internvl35.InternVL35Adapter(device="cpu", crop_to_patches=False, max_patches=1),
}
SIZES = [(640, 480), (333, 517), (1000, 1000), (37, 1201)]


def _boxes(w, h, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = np.sort(rng.uniform(0, w, 2))
        y = np.sort(rng.uniform(0, h, 2))
        out.append([float(x[0]), float(y[0]), float(x[1]), float(y[1])])
    return out


@pytest.mark.parametrize("family", list(MODULES))
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("n,with_labels", [(1, True), (1, False), (3, True), (4, False)])
def test_round_trip_within_native_precision(family, size, n, with_labels):
    w, h = size
    boxes = _boxes(w, h, n, seed=sum(map(ord, family)) * 7 + w * 3 + h + n)
    boxes[0] = [0.0, 0.0, float(w), float(h)]  # 贴边的框
    labels = [f"obj {k}" for k in range(n)] if with_labels else None
    if with_labels and n > 2:
        labels[1] = labels[0]  # 相邻同标签（InternVL 写进同一个 <box>）
        labels[2] = 'quote " and \\ backslash 中文'
    ad = ADAPTERS[family]()
    text = ad.format_boxes(boxes, width=w, height=h, labels=labels)
    assert text == MODULES[family].format_boxes(boxes, width=w, height=h, labels=labels)
    p = ad.parse_boxes(text, width=w, height=h)
    assert p.failure is None and len(p.boxes) == n
    assert list(p.labels) == (labels if labels is not None else [None] * n)
    for got, want in zip(p.boxes, boxes):  # 每个坐标差不超过半个刻度（W/2000、H/2000 像素）
        assert all(abs(g - v) <= (w if i % 2 == 0 else h) / 2000 + 1e-9 for i, (g, v) in enumerate(zip(got, want)))
    # 再格式化一次逐字相同（解析出的框已经在 0–1000 的整数格上）
    assert ad.format_boxes(list(p.boxes), width=w, height=h, labels=list(p.labels)) == text


def test_exact_native_strings_match_observed_outputs():
    """与 RUN-0005 / model_notes §7.5 的实测输出逐字相同（W = H = 1000 时像素坐标就是 0–1000 的刻度）。"""
    assert qwen35.format_boxes([[14, 114, 492, 983]], width=1000, height=1000, labels=["cat on the left"]) == \
        '```json\n[\n\t{"bbox_2d": [14, 114, 492, 983], "label": "cat on the left"}\n]\n```'
    assert gemma4.format_boxes([[13, 107, 496, 987]], width=1000, height=1000, labels=["the cat on the left"]) == \
        '```json\n[\n  {"box_2d": [107, 13, 987, 496], "label": "the cat on the left"}\n]\n```'
    assert internvl35.format_boxes([[24, 114, 492, 984]], width=1000, height=1000, labels=["the cat on the left"]) == \
        "<ref>the cat on the left</ref><box>[[24, 114, 492, 984]]</box>"
    two = qwen35.format_boxes([[1, 2, 3, 4], [5, 6, 7, 8]], width=1000, height=1000)
    assert two == '```json\n[\n\t{"bbox_2d": [1, 2, 3, 4]},\n\t{"bbox_2d": [5, 6, 7, 8]}\n]\n```'
    assert internvl35.format_boxes([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 1]], width=1000, height=1000,
                                   labels=["a", "a", None]) == "<ref>a</ref><box>[[1, 2, 3, 4], [5, 6, 7, 8]]</box>\n<box>[[0, 0, 1, 1]]</box>"


def test_zero_boxes():
    for mod in (qwen35, gemma4):
        text = mod.format_boxes([], width=640, height=480)
        assert text == "```json\n[]\n```"
        p = mod.parse_boxes(text, width=640, height=480)
        assert p.failure is None and p.boxes == ()
    with pytest.raises(ValueError, match="零个框"):
        internvl35.format_boxes([], width=640, height=480)


@pytest.mark.parametrize("family", list(MODULES))
def test_bad_boxes_rejected(family):
    f = MODULES[family].format_boxes
    with pytest.raises(ValueError, match="颠倒"):
        f([[10, 10, 5, 20]], width=100, height=100)
    with pytest.raises(ValueError, match="超出原图"):
        f([[0, 0, 101, 50]], width=100, height=100)
    with pytest.raises(ValueError, match="4 个有限数"):
        f([[0, 0, 1]], width=100, height=100)
    with pytest.raises(ValueError, match="labels 有"):
        f([[0, 0, 1, 1]], width=100, height=100, labels=["a", "b"])
    with pytest.raises(TypeError):
        f([[0, 0, 1, 1]], width=100, height=100, labels="a")
    with pytest.raises(TypeError):
        f([[0, 0, 1, 1]], 100, 100)  # 宽高只能按关键字给


def test_tolerance_clips_tiny_overshoot():
    tol_px = NATIVE_BOX_TOLERANCE / 1000 * 640
    assert native_box_ints([[0, 0, 640 + 0.9 * tol_px, 480]], width=640, height=480) == [(0, 0, 1000, 1000)]
    assert native_box_ints([[-0.9 * tol_px, 0, 10, 10]], width=640, height=480)[0][0] == 0
    with pytest.raises(ValueError):
        native_box_ints([[0, 0, 640 + 1.1 * tol_px, 480]], width=640, height=480)


def test_internvl_label_with_tags_rejected():
    with pytest.raises(ValueError, match="标签"):
        internvl35.format_boxes([[0, 0, 1, 1]], width=10, height=10, labels=["a</ref>"])


def test_format_boxes_width_height_must_be_keyword_only_without_default():
    with pytest.raises(TypeError, match="format_boxes"):
        class Bad(ModelAdapter):  # noqa: F841
            def format_boxes(self, boxes, *, width=1.0, height=1.0, labels=None):
                return ""


# ---------------------------------------------------------------------------
# 公开的停止 token 接口
# ---------------------------------------------------------------------------

STOPS = {
    "qwen3.5-0.8b": ("<|im_end|>", ("<|im_end|>", "<|endoftext|>")),
    "gemma-4-e2b-it": ("<turn|>", ("<turn|>", "<eos>")),
    "internvl3.5-2b-hf": ("<|im_end|>", ("<|im_end|>",)),
}
POLICIES = {"qwen3.5-0.8b": PixelBudget(64), "gemma-4-e2b-it": SoftTokenBudget(70), "internvl3.5-2b-hf": TilePolicy(False, 1)}


@pytest.mark.parametrize("key", list(STOPS))
def test_public_stop_token_interface(key):
    ad = make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True)
    eot, stops = STOPS[key]
    assert ad.end_of_turn_token_id() == ad.token_id(eot) == ad.stop_token_ids()[0]
    assert ad.stop_token_ids() == tuple(ad.token_id(s) for s in stops) == tuple(ad._stop_ids())
    assert family_stop_id(ad) == ad.end_of_turn_token_id()
    assert MODULES[registry.get_model(key).family].STOP_TOKENS == stops


def test_e3_reads_only_the_public_interface():
    class OnlyPublic:
        def end_of_turn_token_id(self):
            return 7

    assert family_stop_id(OnlyPublic()) == 7


def test_unloaded_adapter_stop_ids_raise():
    for make in ADAPTERS.values():
        with pytest.raises(RuntimeError):
            make().end_of_turn_token_id()
