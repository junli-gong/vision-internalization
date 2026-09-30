import numpy as np
import pytest

from src.constants import NORM1000_SCALE
from src.executors.geometry import (
    abs_to_norm1000,
    abs_to_unit,
    box_area,
    box_iou,
    norm1000_to_abs,
    polygon_iou,
    unit_to_abs,
)


def test_box_iou_hand_no_plus_one():
    a, b = [0, 0, 4, 2], [2, 1, 6, 5]
    # 交 = 2*1 = 2；并 = 8 + 16 - 2 = 22
    assert box_iou(a, b) == pytest.approx(2 / 22, abs=0)
    assert box_iou(b, a) == box_iou(a, b)
    assert box_area([1, 1, 3, 4]) == 6.0  # +1 约定会给 12
    assert box_iou([0, 0, 1, 1], [1, 0, 2, 1]) == 0.0  # 只接边不相交
    assert box_iou([1.5, 2.5, 3.5, 4.5], [1.5, 2.5, 3.5, 4.5]) == 1.0


def test_box_iou_rejects_degenerate():
    with pytest.raises(ValueError):
        box_iou([2, 0, 1, 1], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        box_iou([0, 0, 0, 0], [1, 1, 1, 1])


def test_polygon_iou_hand():
    sq = [[0, 0, 2, 0, 2, 2, 0, 2]]
    shifted = [np.array([[1, 0], [3, 0], [3, 2], [1, 2]])]
    assert polygon_iou(sq, shifted) == pytest.approx(2 / 6)
    tri = [[0, 0, 2, 0, 0, 2]]
    assert polygon_iou(sq, tri) == pytest.approx(2 / 4)
    # 两个多边形取并集
    assert polygon_iou([[0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 2, 0, 2, 1, 1, 1]], [[0, 0, 2, 0, 2, 1, 0, 1]]) == pytest.approx(1.0)


def test_polygon_iou_rejects_invalid():
    bowtie = [[0, 0, 2, 2, 2, 0, 0, 2]]
    with pytest.raises(ValueError):
        polygon_iou(bowtie, [[0, 0, 1, 0, 1, 1]])


def test_coordinate_conversions_hand_and_axes():
    W, H = 640, 480  # W ≠ H，抓轴序错误
    box = [64.0, 48.0, 320.0, 240.0]
    n = abs_to_norm1000(box, width=W, height=H)
    assert np.allclose(n, [100, 100, 500, 500])
    u = abs_to_unit(box, width=W, height=H)
    assert np.allclose(u, [0.1, 0.1, 0.5, 0.5])
    assert NORM1000_SCALE == 1000.0
    # x 只受 W 影响、y 只受 H 影响
    assert np.allclose(abs_to_unit([W, 0], width=W, height=H), [1, 0])
    assert np.allclose(abs_to_unit([0, H], width=W, height=H), [0, 1])


def test_coordinate_roundtrips():
    rng = np.random.default_rng(3)
    W, H = 1279.0, 721.0
    pts = rng.uniform(0, 1, size=(50, 8)) * np.array([W, H] * 4)  # 扁平多边形，x,y 交替
    for fwd, bwd in ((abs_to_norm1000, norm1000_to_abs), (abs_to_unit, unit_to_abs)):
        back = bwd(fwd(pts, width=W, height=H), width=W, height=H)
        np.testing.assert_allclose(back, pts, rtol=1e-12, atol=1e-9)
    # 1000 ↔ unit 之间也一致
    np.testing.assert_allclose(
        abs_to_norm1000(pts, width=W, height=H) / 1000.0, abs_to_unit(pts, width=W, height=H), rtol=1e-12
    )


def test_conversion_requires_positive_size_and_even_length():
    with pytest.raises(ValueError):
        abs_to_unit([1, 2], width=0, height=5)
    with pytest.raises(ValueError):
        abs_to_unit([1, 2, 3], width=4, height=5)
    with pytest.raises(TypeError):
        abs_to_unit([1, 2], 4, 5)  # 宽高必须显式按名字给
