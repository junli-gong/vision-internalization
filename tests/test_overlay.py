"""叠加渲染：alpha 混合的整数化、轮廓、颜色顺序、逐位保留 mask 外的原图。"""

import inspect
import math
from fractions import Fraction

import numpy as np
import pytest

from src.constants import OVERLAY_ALPHA_DEN, OVERLAY_ALPHA_NUM, OVERLAY_CONTOUR_WIDTH, OVERLAY_DRAW_CONTOUR, OVERLAY_PALETTE
from src.executors import overlay as O
from src.executors.overlay import mask_contour, render_overlay

H, W = 37, 53  # H ≠ W


def _image(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def _masks():
    """三个互不相同的 mask：大小、位置、形状都不同，其中两个部分重叠，一个贴着图边。"""
    yy, xx = np.mgrid[:H, :W]
    a = ((yy - 15) / 9.0) ** 2 + ((xx - 18) / 13.0) ** 2 <= 1  # 椭圆
    b = np.zeros((H, W), bool)
    b[10:30, 25:45] = True  # 方块，与椭圆部分重叠
    c = np.zeros((H, W), bool)
    c[28:, 0:9] = True  # 贴左下角
    c[33:, 9:14] = True
    return [a, b, c]


def _blend_exact(base, color):
    """按有理数精确计算 (1-α)·I + α·C，再四舍五入（.5 进位）。"""
    alpha = Fraction(OVERLAY_ALPHA_NUM, OVERLAY_ALPHA_DEN)
    v = (1 - alpha) * int(base) + alpha * int(color)
    return math.floor(v + Fraction(1, 2))


# ---------------------------------------------------------------- 常量


def _srgb_to_lab(rgb):
    c = np.asarray(rgb, float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    t = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(t > d**3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def test_overlay_constants_pinned():
    assert len(OVERLAY_PALETTE) >= 8
    pal = np.asarray(OVERLAY_PALETTE)
    assert pal.shape == (len(OVERLAY_PALETTE), 3) and pal.min() >= 0 and pal.max() <= 255
    assert len({tuple(c) for c in OVERLAY_PALETTE}) == len(OVERLAY_PALETTE)
    assert OVERLAY_PALETTE[0] == (230, 25, 75)  # 单物体叠加用第一色（红）
    assert (OVERLAY_ALPHA_NUM, OVERLAY_ALPHA_DEN) == (1, 2) and 0 < OVERLAY_ALPHA_NUM < OVERLAY_ALPHA_DEN
    assert OVERLAY_DRAW_CONTOUR is True and OVERLAY_CONTOUR_WIDTH == 2


def test_palette_colors_are_easy_to_tell_apart():
    # 注释："两两 CIE76 色差 ≥ 25（最近的一对是紫与品红，约 29）"
    lab = _srgb_to_lab(OVERLAY_PALETTE)
    d = np.linalg.norm(lab[:, None] - lab[None], axis=-1)
    iu = np.triu_indices(len(lab), 1)
    assert d[iu].min() >= 25.0
    i, j = iu[0][np.argmin(d[iu])], iu[1][np.argmin(d[iu])]
    assert {i, j} == {5, 7} and 28.0 < d[i, j] < 30.0


def test_signature_takes_no_render_or_task_parameters():
    # 渲染参数只来自 constants（§12 训练/评测一致）；函数签名里也不出现任务身份
    assert list(inspect.signature(render_overlay).parameters) == ["image", "masks"]


# ---------------------------------------------------------------- 混合与整数化


def test_integer_rounding_rule_hand_examples():
    img = np.zeros((1, 4, 3), np.uint8)
    img[0, :, 0] = [0, 1, 254, 255]
    m = np.ones((1, 4), bool)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(O, "OVERLAY_DRAW_CONTOUR", False)
        mp.setattr(O, "OVERLAY_PALETTE", ((255, 0, 7),))
        out = render_overlay(img, [m])
    # α=1/2：(I + C + 1) // 2。R 通道 (0+255)/2=127.5→128，(1+255)/2=128，(254+255)/2=254.5→255，255
    assert out[0, :, 0].tolist() == [128, 128, 255, 255]
    # G 通道 (0+0)/2=0，(0+0)/2=0 ...；B 通道 (0+7)/2=3.5→4
    assert out[0, :, 1].tolist() == [0, 0, 0, 0] and out[0, :, 2].tolist() == [4, 4, 4, 4]


def test_fill_matches_exact_rational_blend_and_outside_is_untouched():
    img = _image(1)
    masks = _masks()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(O, "OVERLAY_DRAW_CONTOUR", False)
        out = render_overlay(img, masks)
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)
    covered = np.any(masks, axis=0)
    assert np.array_equal(out[~covered], img[~covered])  # mask 外逐位相同
    for i, j in zip(*np.nonzero(covered)):
        k = max(n for n, m in enumerate(masks) if m[i, j])  # 覆盖它的最后一个 mask
        for ch in range(3):
            assert out[i, j, ch] == _blend_exact(img[i, j, ch], OVERLAY_PALETTE[k][ch])


def test_other_alpha_values_follow_the_same_integer_rule():
    img = _image(2)
    m = _masks()[1]
    for num, den in [(2, 5), (3, 4), (1, 3)]:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(O, "OVERLAY_DRAW_CONTOUR", False)
            mp.setattr(O, "OVERLAY_ALPHA_NUM", num)
            mp.setattr(O, "OVERLAY_ALPHA_DEN", den)
            out = render_overlay(img, [m])
        a = Fraction(num, den)
        for i, j in list(zip(*np.nonzero(m)))[::17]:
            for ch in range(3):
                v = (1 - a) * int(img[i, j, ch]) + a * OVERLAY_PALETTE[0][ch]
                assert out[i, j, ch] == math.floor(v + Fraction(1, 2))


# ---------------------------------------------------------------- 轮廓


def _naive_contour(mask, width):
    """独立参照：像素在 mask 内、且 (2w+1)×(2w+1) 邻域里有背景（图外算背景）即为轮廓。"""
    padded = np.pad(mask, width, constant_values=False)
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] and not padded[i : i + 2 * width + 1, j : j + 2 * width + 1].all():
                out[i, j] = True
    return out


@pytest.mark.parametrize("width", [1, 2, 3])
def test_contour_definition(width):
    for m in _masks():
        assert np.array_equal(mask_contour(m, width), _naive_contour(m, width))
    # 宽度 1 即"mask 减 3×3 腐蚀"：手算一个 3×4 的实心块，只有正中两格不是轮廓
    block = np.zeros((5, 6), bool)
    block[1:4, 1:5] = True
    inner = np.zeros_like(block)
    inner[2, 2:4] = True
    assert np.array_equal(mask_contour(block, 1), block & ~inner)
    with pytest.raises(ValueError):
        mask_contour(block, 0)


def test_contour_is_solid_color_and_drawn_after_fill():
    img = _image(3)
    masks = _masks()
    out = render_overlay(img, masks)
    contours = [mask_contour(m, OVERLAY_CONTOUR_WIDTH) for m in masks]
    painted = np.zeros((H, W), bool)
    for k, c in enumerate(contours):
        # 后画的轮廓盖住先画的：只检查没被后面轮廓覆盖的部分
        later = np.any(contours[k + 1 :], axis=0) if k + 1 < len(contours) else np.zeros((H, W), bool)
        assert np.all(out[c & ~later] == np.array(OVERLAY_PALETTE[k], np.uint8))
        painted |= c
    covered = np.any(masks, axis=0)
    assert np.array_equal(out[~covered], img[~covered])
    # 贴图边的 mask（第 3 个）在图边上也有轮廓
    assert contours[2][H - 1, 0] and contours[2][H - 1, 13]
    # 轮廓之外的 mask 内部仍是混合色，不是实色
    inside = covered & ~painted
    assert inside.any()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(O, "OVERLAY_DRAW_CONTOUR", False)
        no_contour = render_overlay(img, masks)
    assert np.array_equal(out[inside], no_contour[inside])


# ---------------------------------------------------------------- 顺序、确定性、输入检查


def test_order_determines_colors_and_result_is_deterministic():
    img = _image(4)
    a, b, c = _masks()
    out1 = render_overlay(img, [a, b, c])
    assert np.array_equal(out1, render_overlay(img, [a, b, c]))
    assert np.array_equal(out1, render_overlay(img, np.stack([a, b, c])))  # (K, H, W) 数组同样处理
    swapped = render_overlay(img, [b, a, c])
    only_a = a & ~b & ~c & ~mask_contour(a) & ~mask_contour(b)
    assert only_a.any()
    i, j = np.argwhere(only_a)[0]
    assert out1[i, j].tolist() == [_blend_exact(img[i, j, ch], OVERLAY_PALETTE[0][ch]) for ch in range(3)]
    assert swapped[i, j].tolist() == [_blend_exact(img[i, j, ch], OVERLAY_PALETTE[1][ch]) for ch in range(3)]


def test_inputs_not_modified_and_empty_mask_list():
    img = _image(5)
    img0 = img.copy()
    masks = _masks()
    masks0 = [m.copy() for m in masks]
    render_overlay(img, masks)
    assert np.array_equal(img, img0) and all(np.array_equal(m, m0) for m, m0 in zip(masks, masks0))
    out = render_overlay(img, [])
    assert np.array_equal(out, img) and out is not img
    # 0/1 整数 mask 与 bool mask 等价
    assert np.array_equal(render_overlay(img, [masks[0].astype(np.uint8)]), render_overlay(img, [masks[0]]))


def test_result_does_not_depend_on_memory_layout():
    img = _image(7)
    masks = _masks()
    ref = render_overlay(img, masks)
    big = np.zeros((2 * H, 3 * W, 3), np.uint8)
    big[::2, ::3] = img
    for im in (np.asfortranarray(img), big[::2, ::3]):
        assert np.array_equal(render_overlay(im, [np.asfortranarray(m) for m in masks]), ref)


def test_input_checks():
    img = _image(6)
    m = _masks()[0]
    with pytest.raises(ValueError):
        render_overlay(img.astype(np.float64), [m])
    with pytest.raises(ValueError):
        render_overlay(img[..., 0], [m])
    with pytest.raises(ValueError):
        render_overlay(img, m)  # 单个二维数组：含义不明
    with pytest.raises(ValueError):
        render_overlay(img, [m.T])
    with pytest.raises(ValueError):
        render_overlay(img, [m.astype(np.float64)])
    with pytest.raises(ValueError):
        render_overlay(img, [m.astype(np.int64) * 2])
    with pytest.raises(ValueError):
        render_overlay(img, [m] * (len(OVERLAY_PALETTE) + 1))  # 颜色不够，不循环复用
