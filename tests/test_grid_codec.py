import numpy as np
import pytest

from src.constants import D_MAX, D_MIN, DEPTH_NUM_BINS, GRID_SIZE_E1, GRID_SIZE_E3B
from src.grid.codec import (
    GridParseFailure,
    bins_to_depth,
    cell_edges,
    depth_bin_edges,
    depth_to_bins,
    grid_to_mask,
    grid_to_text,
    mask_to_grid,
    parse_grid_text,
)

# ---------------------------------------------------------------- mask ↔ 网格


def test_cell_edges_floor_rule():
    assert cell_edges(10, 4).tolist() == [0, 2, 5, 7, 10]  # floor(k*10/4)
    assert cell_edges(7, 3).tolist() == [0, 2, 4, 7]
    with pytest.raises(ValueError):
        cell_edges(3, 4)


def test_mask_to_grid_threshold_exactly_half():
    # 10×6 图，N=2：格为 5×3 = 15 像素；8/15 ≥ .5 → 1，7/15 < .5 → 0
    m = np.zeros((10, 6), bool)
    m[0:5, 0:3].flat[:8] = True
    m[0:5, 3:6].flat[:7] = True
    # 偶数格面积恰好一半：12×4，N=2 → 格 6×2 = 12 像素，6 个前景 → 1
    m2 = np.zeros((12, 4), bool)
    m2[0:6, 0:2].flat[:6] = True
    m2[6:12, 2:4].flat[:5] = True
    assert mask_to_grid(m, 2).tolist() == [[1, 0], [0, 0]]
    assert mask_to_grid(m2, 2).tolist() == [[1, 0], [0, 0]]


def test_grid_mask_roundtrip_asymmetric():
    rng = np.random.default_rng(0)
    for n, (h, w) in ((GRID_SIZE_E1, (37, 53)), (GRID_SIZE_E3B, (480, 640)), (GRID_SIZE_E3B, (333, 101))):
        g = (rng.random((n, n)) > 0.5).astype(np.uint8)
        assert not np.array_equal(g, g.T)
        m = grid_to_mask(g, h, w)
        assert m.shape == (h, w)
        assert np.array_equal(mask_to_grid(m, n), g)


def test_grid_to_mask_exact_backfill():
    g = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], np.uint8)
    m = grid_to_mask(g, 7, 10)  # 行边界 0,2,4,7；列边界 0,3,6,10
    assert m[0:2, 0:3].all() and not m[0:2, 3:].any()
    assert m[2:4, 6:10].all() and m[2:4, :6].sum() == 0
    assert m[4:7, 3:6].all() and m[4:7].sum() == 9


def test_mask_to_grid_axes_not_swapped():
    m = np.zeros((20, 40), bool)
    m[:, 30:] = True  # 右侧竖条
    g = mask_to_grid(m, 4)
    assert g[:, 3].all() and g[:, :3].sum() == 0


# ---------------------------------------------------------------- 深度分档


def test_depth_bin_edges():
    e = depth_bin_edges()
    assert len(e) == DEPTH_NUM_BINS + 1
    assert e[0] == pytest.approx(D_MIN, rel=1e-12) and e[-1] == pytest.approx(D_MAX, rel=1e-12)
    r = e[1:] / e[:-1]
    assert np.allclose(r, r[0])  # 对数均匀


def test_depth_bins_roundtrip_centers_and_clipping():
    k = np.arange(DEPTH_NUM_BINS)
    centers = bins_to_depth(k)
    e = depth_bin_edges()
    assert np.allclose(centers, np.sqrt(e[:-1] * e[1:]))  # 几何中心
    assert np.array_equal(depth_to_bins(centers), k)
    assert depth_to_bins(np.array([0.01, D_MIN, D_MAX, 50.0])).tolist() == [0, 0, DEPTH_NUM_BINS - 1, DEPTH_NUM_BINS - 1]
    # 档内任意值解码后与原值的对数误差不超过半档
    d = np.exp(np.random.default_rng(0).uniform(np.log(D_MIN), np.log(D_MAX), 1000))
    half = np.log(e[1] / e[0]) / 2
    assert np.max(np.abs(np.log(bins_to_depth(depth_to_bins(d)) / d))) <= half + 1e-12


def test_depth_bins_reject_invalid():
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            depth_to_bins(np.array([1.0, bad]))
    with pytest.raises(ValueError):
        bins_to_depth(np.array([DEPTH_NUM_BINS]))
    with pytest.raises(ValueError):
        bins_to_depth(np.array([1.0]))


# ---------------------------------------------------------------- 数字字符网格


def _g(n=4, seed=0):
    g = (np.random.default_rng(seed).random((n, n)) > 0.5).astype(np.uint8)
    g[0, n - 1], g[n - 1, 0] = 1, 0  # 非对称
    return g


def test_text_roundtrip():
    g = _g(GRID_SIZE_E1)
    t = grid_to_text(g)
    lines = t.split("\n")
    assert len(lines) == GRID_SIZE_E1 and all(len(ln) == GRID_SIZE_E1 and set(ln) <= {"0", "1"} for ln in lines)
    p = parse_grid_text(t, GRID_SIZE_E1)
    assert p.failure is None and np.array_equal(p.grid, g)


def test_parse_accepts_separators_and_surrounding_prose():
    g = _g(4)
    rows = ["0 1 1 0", "1,0,0,1", "1, 1, 0, 0,", "0\t0 ,1 1"]
    expected = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
    text = "Here is the 4x4 mask:\n```\n" + "\n".join(rows) + "\n```\nDone."
    p = parse_grid_text(text, 4)
    assert p.failure is None and np.array_equal(p.grid, expected)
    # 前后的说明文字里有数字（带字母）不影响
    p2 = parse_grid_text("Step 1: think.\n" + grid_to_text(g) + "\nThat is 16 cells.", 4)
    assert p2.failure is None and np.array_equal(p2.grid, g)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("I cannot see the object.", GridParseFailure.NO_GRID),
        ("", GridParseFailure.NO_GRID),
        ("0110\n1001\n1100\n\n0011", GridParseFailure.MULTIPLE_BLOCKS),
        ("0110\n1001\n1100\nabc\n0011", GridParseFailure.MULTIPLE_BLOCKS),
        ("0110\n1021\n1100\n0011", GridParseFailure.ILLEGAL_CHAR),
        ("[0,1,1,0]\n1001\n1100\n0011", GridParseFailure.ILLEGAL_CHAR),
        ("0110\n1001\n1100", GridParseFailure.WRONG_ROW_COUNT),
        ("0110\n1001\n1100\n0011\n0000", GridParseFailure.WRONG_ROW_COUNT),
        ("0110\n1001\n110\n0011", GridParseFailure.WRONG_COL_COUNT),
        ("0110\n10011\n1100\n0011", GridParseFailure.WRONG_COL_COUNT),
    ],
)
def test_parse_failures(text, reason):
    p = parse_grid_text(text, 4)
    assert p.grid is None and p.failure is reason


def test_parse_never_pads_or_truncates():
    # 16×16 期望下给 15 行：不补齐
    g = _g(16)
    t = "\n".join(grid_to_text(g).split("\n")[:15])
    assert parse_grid_text(t, 16).failure is GridParseFailure.WRONG_ROW_COUNT
    # 每行 17 个数字：不截断
    t2 = "\n".join(ln + "0" for ln in grid_to_text(g).split("\n"))
    assert parse_grid_text(t2, 16).failure is GridParseFailure.WRONG_COL_COUNT


def test_failure_enum_is_finite_and_stringy():
    assert {f.value for f in GridParseFailure} == {
        "no_grid",
        "multiple_blocks",
        "illegal_char",
        "wrong_row_count",
        "wrong_col_count",
    }
