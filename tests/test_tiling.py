"""分块与重叠拼接。"""

import inspect

import numpy as np
import pytest

from src.constants import PIXEL_WINDOW_FORMAT, TILE_OVERLAP, TILE_SIZE
from src.executors.tiling import make_tiles, stitch


def test_tiling_constants_pinned():
    assert (TILE_SIZE, TILE_OVERLAP) == (256, 64)
    # 注释："256 是 32 的倍数……一块 8×8 个 token；重叠 64 px 即 2 个 token，步长 192"
    assert TILE_SIZE % 32 == 0 and TILE_SIZE // 32 == 8
    assert TILE_OVERLAP // 32 == 2 and TILE_SIZE - TILE_OVERLAP == 192
    assert PIXEL_WINDOW_FORMAT == "(r0, r1, c0, c1)_half_open"


def test_signatures_carry_no_task_identity():
    for fn in (make_tiles, stitch):
        assert not any("task" in name for name in inspect.signature(fn).parameters)


def _mmseg_slide_windows(h_img, w_img, crop, stride):
    """mmsegmentation EncoderDecoder.slide_inference 的窗口循环，逐行照抄（独立参照）。"""
    h_crop = w_crop = crop
    h_stride = w_stride = stride
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    out = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            out.append((y1, y2, x1, x2))
    return out


def _coverage(tiles, h, w):
    cnt = np.zeros((h, w), np.int64)
    for r0, r1, c0, c1 in tiles:
        cnt[r0:r1, c0:c1] += 1
    return cnt


def test_divisible_size_gives_exact_grid():
    tiles = make_tiles(512, 768, tile=256, overlap=0)
    assert tiles == [(r, r + 256, c, c + 256) for r in (0, 256) for c in (0, 256, 512)]
    assert np.all(_coverage(tiles, 512, 768) == 1)
    tiles = make_tiles(448, 640, tile=256, overlap=64)  # 448 = 256 + 192，640 = 256 + 2·192：步长整除
    assert tiles == [(r, r + 256, c, c + 256) for r in (0, 192) for c in (0, 192, 384)]


@pytest.mark.parametrize(
    "h,w,tile,overlap",
    [(300, 700, 256, 64), (481, 640, 256, 32), (480, 640, TILE_SIZE, TILE_OVERLAP), (97, 1000, 64, 16), (250, 90, 256, 64), (7, 11, 3, 2)],
)
def test_non_divisible_sizes_follow_mmseg_and_cover(h, w, tile, overlap):
    tiles = make_tiles(h, w, tile, overlap)
    assert tiles == _mmseg_slide_windows(h, w, tile, tile - overlap)
    cov = _coverage(tiles, h, w)
    assert cov.min() >= 1  # 每个像素至少被一块覆盖
    # 最后一块贴边对齐、不填充：每块边长 = min(tile, 图边长)，且都在图内
    for r0, r1, c0, c1 in tiles:
        assert 0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w
        assert (r1 - r0, c1 - c0) == (min(tile, h), min(tile, w))
    assert max(t[1] for t in tiles) == h and max(t[3] for t in tiles) == w
    # 相邻起点差 = 步长，只有最后一块可以更近（重叠更大）；没有重复的块
    for starts in (sorted({t[0] for t in tiles}), sorted({t[2] for t in tiles})):
        gaps = np.diff(starts)
        assert np.all(gaps[:-1] == tile - overlap) and np.all((gaps > 0) & (gaps <= tile - overlap))
    assert len(set(tiles)) == len(tiles)
    assert make_tiles(h, w, tile, overlap) == tiles  # 确定


def test_image_smaller_than_tile_is_one_tile():
    assert make_tiles(50, 70, tile=256, overlap=64) == [(0, 50, 0, 70)]
    assert make_tiles(50, 300, tile=256, overlap=64) == [(0, 50, 0, 256), (0, 50, 44, 300)]


def test_make_tiles_rejects_bad_args():
    for args in [(0, 10, 4, 1), (10, 10, 0, 0), (10, 10, 4, 4), (10, 10, 4, -1), (10, 10, 4.5, 1)]:
        with pytest.raises(ValueError):
            make_tiles(*args)


def test_stitch_restores_float64_image_bitwise():
    rng = np.random.default_rng(0)
    for h, w, tile, overlap, tail in [(300, 700, 256, 64, ()), (97, 130, 40, 13, (3,)), (481, 640, 256, 32, ())]:
        img = rng.standard_normal((h, w) + tail) * 10.0 ** rng.integers(-3, 4)
        tiles = make_tiles(h, w, tile, overlap)
        cov = _coverage(tiles, h, w)
        assert cov.max() >= 3  # 确实有 3 块及以上重叠的像素（求和再除不一定逐位还原）
        i, j = np.argwhere(cov == cov.max())[0]
        img[i, j] = -0.0  # 位模式与 +0.0 不同，也要原样还原
        out = stitch([img[r0:r1, c0:c1] for r0, r1, c0, c1 in tiles], tiles, h, w)
        assert out.dtype == np.float64
        assert np.array_equal(out.view(np.int64), img.view(np.int64))  # 逐位（比较位模式）


def test_stitch_constant_gives_constant():
    tiles = make_tiles(97, 130, 40, 13)
    for c in (0.1, -3.7, 1e300, 0.0):
        out = stitch([np.full((r1 - r0, c1 - c0), c) for r0, r1, c0, c1 in tiles], tiles, 97, 130)
        assert np.all(out == c)


def test_stitch_averages_with_weight_equal_to_cover_count():
    rng = np.random.default_rng(1)
    h, w = 97, 130
    tiles = make_tiles(h, w, 40, 13)
    vals = [rng.standard_normal((r1 - r0, c1 - c0, 2)) for r0, r1, c0, c1 in tiles]
    total = np.zeros((h, w, 2))
    for v, (r0, r1, c0, c1) in zip(vals, tiles):
        total[r0:r1, c0:c1] += v
    ref = total / _coverage(tiles, h, w)[..., None]
    np.testing.assert_allclose(stitch(vals, tiles, h, w), ref, rtol=0, atol=1e-13)
    # 手算一例：两块在一个像素上重叠，值 1 和 4 → 2.5
    out = stitch([np.ones((1, 2)), np.full((1, 2), 4.0)], [(0, 1, 0, 2), (0, 1, 1, 3)], 1, 3)
    assert out.tolist() == [[1.0, 2.5, 4.0]]


def test_stitch_does_not_depend_on_memory_layout():
    rng = np.random.default_rng(2)
    h, w = 97, 130
    tiles = make_tiles(h, w, 40, 13)
    field = rng.standard_normal((h, w, 3))
    big = np.zeros((2 * h, 3 * w, 3))
    big[::2, ::3] = field
    for f in (np.asfortranarray(field), big[::2, ::3]):
        vals = [f[r0:r1, c0:c1] for r0, r1, c0, c1 in tiles]
        assert np.array_equal(stitch(vals, tiles, h, w).view(np.int64), field.view(np.int64))


def test_stitch_rejects_bad_input():
    tiles = [(0, 2, 0, 2)]
    with pytest.raises(ValueError):
        stitch([np.zeros((2, 2))], tiles, 2, 3)  # 第 3 列没被覆盖
    with pytest.raises(ValueError):
        stitch([np.zeros((2, 3))], tiles, 2, 2)  # 形状与窗口不符
    with pytest.raises(ValueError):
        stitch([np.array([[0.0, np.nan], [0.0, 0.0]])], tiles, 2, 2)
    with pytest.raises(ValueError):
        stitch([np.zeros((2, 2))], [(0, 2, 0, 3)], 2, 2)  # 越界
    with pytest.raises(ValueError):
        stitch([np.zeros((2, 2)), np.zeros((2, 2))], tiles, 2, 2)  # 数量不一致
