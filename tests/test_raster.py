import numpy as np
import pytest
from pycocotools import mask as mask_utils

from src.executors.raster import box_to_mask, mask_to_rle, polygons_to_mask, rle_to_mask

H, W = 7, 11  # H ≠ W


def test_box_to_mask_integer_box_hand():
    m = box_to_mask([2, 1, 5, 3], H, W)
    assert m.shape == (H, W)
    expected = np.zeros((H, W), bool)
    expected[1:3, 2:5] = True  # 行 1..2、列 2..4
    assert np.array_equal(m, expected)


def test_box_to_mask_pixel_center_rule_and_ties():
    # 中心 2.5 恰在 x0 上 → (lo, hi] 不含；中心 4.5 恰在 x1 上 → 含
    m = box_to_mask([2.5, 0.2, 4.5, 0.8], H, W)
    assert np.flatnonzero(m[0]).tolist() == [3, 4]
    assert m.sum() == 2  # 只有第 0 行（中心 0.5 在 (0.2, 0.8] 内）
    # 框没覆盖任何像素中心
    assert box_to_mask([2.6, 1.6, 3.4, 2.4], H, W).sum() == 0


def test_box_to_mask_clips_outside_image():
    m = box_to_mask([-5, -5, 100, 2], H, W)
    assert m[:2].all() and not m[2:].any()


def test_box_to_mask_rejects_bad_boxes():
    with pytest.raises(ValueError):
        box_to_mask([5, 1, 2, 3], H, W)
    with pytest.raises(ValueError):
        box_to_mask([0, 0, np.nan, 3], H, W)


def _rect_poly(b):
    x0, y0, x1, y1 = b
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def test_box_rule_matches_pycocotools_rectangle_raster():
    """钉住 constants.BOX_RASTER_RULE：(lo, hi] 与 pycocotools 对矩形多边形的光栅化逐位一致。"""
    rng = np.random.default_rng(0)
    hh, ww = 37, 53
    for q in (None, 0.5, 0.1):
        for _ in range(300):
            x = np.sort(rng.uniform(-3, ww + 3, 2))
            y = np.sort(rng.uniform(-3, hh + 3, 2))
            b = [x[0], y[0], x[1], y[1]]
            if q:
                b = [round(v / q) * q for v in b]
            assert np.array_equal(box_to_mask(b, hh, ww), polygons_to_mask(_rect_poly(b), hh, ww)), b


def test_polygons_to_mask_triangle_and_union():
    # 直角三角形 (0,0)-(6,0)-(0,6)：像素中心 (j+.5, i+.5) 满足 j+i+1 < 6 的在内（边界上的归属由 pycocotools 定）
    tri = polygons_to_mask([[0, 0, 6, 0, 0, 6]], H, W)
    assert tri.shape == (H, W)
    ii, jj = np.mgrid[:H, :W]
    strictly_inside = (ii + jj + 1 < 6)
    strictly_outside = (ii + jj + 1 > 6) | (jj >= 6) | (ii >= 6)
    assert tri[strictly_inside].all() and not tri[strictly_outside].any()
    # 两个不相交矩形取并集，(k,2) 数组格式也可
    two = polygons_to_mask([np.array([[0, 0], [2, 0], [2, 2], [0, 2]]), [8, 4, 11, 4, 11, 7, 8, 7]], H, W)
    assert two.sum() == 4 + 9
    assert two[0:2, 0:2].all() and two[4:7, 8:11].all()


def test_polygons_to_mask_rejects_bad_input():
    with pytest.raises(ValueError):
        polygons_to_mask([[0, 0, 1]], H, W)
    with pytest.raises(ValueError):
        polygons_to_mask([[0, 0, 1, 1]], H, W)
    with pytest.raises(ValueError):
        polygons_to_mask([], H, W)


def test_rle_roundtrip_asymmetric():
    rng = np.random.default_rng(1)
    m = rng.random((H, W)) > 0.6
    m[0, W - 1] = True
    m[H - 1, 0] = False
    rle = mask_to_rle(m)
    assert rle["size"] == [H, W] and isinstance(rle["counts"], str)
    back = rle_to_mask(rle)
    assert back.dtype == bool and np.array_equal(back, m)


def test_rle_decode_uncompressed_matches_compressed():
    m = np.zeros((H, W), bool)
    m[2:5, 3:9] = True
    # 未压缩 RLE：列优先的游程
    flat = m.flatten(order="F").astype(int)
    counts, cur, run = [], 0, 0
    for v in flat:
        if v == cur:
            run += 1
        else:
            counts.append(run)
            cur, run = v, 1
    counts.append(run)
    assert np.array_equal(rle_to_mask({"size": [H, W], "counts": counts}), m)
    # 与 pycocotools 直接编码的一致
    ref = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    assert mask_to_rle(m)["counts"] == ref["counts"].decode("ascii")
