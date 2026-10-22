"""DEV-0058：W2 的挑选图渲染（src/w2/render.py；接口见 docs/w2_plan.md §2）。

钉住：两种布局的尺寸（拼图的长边、单图不超过 640×480、不放大）；编号顺序（第 k 个候选在第 k 格 / 第 k 张，参照在最前、不写字）；
叠加的规则（mask 外的像素与缩放后的原图逐位相同，mask 内是整数混合，轮廓与描边的颜色）；逐位确定；样式的补全与校验；
在 Qwen3.5 的 PixelBudget(300) 下用适配器的图像处理现算视觉 token 数；docs/w2_tools.md 列出全部样式键与默认值。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.w2 import render as RD
from src.w2.candidates import Candidate
from src.w2.render import DEFAULT_STYLE, render_selection, resolve_style

DOCS = Path(__file__).resolve().parents[1] / "docs" / "w2_tools.md"


def _image(w=640, h=480, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _quadrant_cands(w, h, k=8):
    """第 i 个候选是第 i 条竖带（宽 w/k）：编号顺序看得出来。"""
    out = []
    for i in range(k):
        m = np.zeros((h, w), bool)
        m[h // 4 : 3 * h // 4, i * w // k : (i + 1) * w // k] = True
        out.append(Candidate(f"c{i + 1}", m, True, {}))
    return out


def _flat_cands(w, h, k):
    return [Candidate(f"c{i + 1}", np.zeros((h, w), bool), True, {}) for i in range(k)]


# ---------------------------------------------------------------------------
# 尺寸
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wh,k,ref,want", [
    ((640, 480), 8, True, (638, 482)),  # 3×3，格子 210×158
    ((640, 480), 8, False, (638, 482)),  # 8 格也排成 3×3（最后一格空着）
    ((480, 640), 8, True, (482, 638)),
    ((640, 427), 8, True, (638, 428)),
    ((640, 480), 6, False, (638, 320)),  # 3 列 × 2 行：2 × 158 + 4
    ((320, 240), 8, True, (638, 482)),  # 小图放大到拼图的长边
])
def test_grid_size(wh, k, ref, want):
    w, h = wh
    out = render_selection(_image(w, h), _flat_cands(w, h, k), layout="grid", style={**DEFAULT_STYLE, "reference": ref})
    assert len(out) == 1 and out[0].size == want and out[0].mode == "RGB"
    assert max(out[0].size) <= DEFAULT_STYLE["grid_long_side"]


@pytest.mark.parametrize("wh,want", [((640, 480), (640, 480)), ((480, 640), (480, 640)), ((500, 375), (500, 375)),
                                     ((1280, 960), (640, 480)), ((640, 640), (480, 480)), ((960, 1280), (480, 640)),
                                     ((100, 80), (100, 80))])
def test_separate_size_never_exceeds_640x480_and_never_upscales(wh, want):
    w, h = wh
    out = render_selection(_image(w, h), _flat_cands(w, h, 3), layout="separate", style=DEFAULT_STYLE)
    assert len(out) == 4 and all(im.size == want for im in out)
    out2 = render_selection(_image(w, h), _flat_cands(w, h, 3), layout="separate", style={"reference": False})
    assert len(out2) == 3


def test_grid_helpers():
    assert RD.grid_shape(9) == (3, 3) and RD.grid_shape(8) == (3, 3) and RD.grid_shape(6) == (2, 3) and RD.grid_shape(1) == (1, 1)
    assert RD.grid_shape(8, 4) == (2, 4)
    assert RD.grid_tile_size(640, 480, 3, 3, 640, 4) == (210, 158)
    with pytest.raises(ValueError):
        RD.grid_shape(0)


# ---------------------------------------------------------------------------
# 编号顺序与叠加
# ---------------------------------------------------------------------------


def _cell(arr, k, st, tw, th, cols):
    g = st["gutter_px"]
    r, c = divmod(k, cols)
    return arr[r * (th + g) : r * (th + g) + th, c * (tw + g) : c * (tw + g) + tw]


def test_grid_order_reference_first_and_overlay_rule():
    w, h, k = 640, 480, 8
    im = _image(w, h)
    cands = _quadrant_cands(w, h, k)
    st = resolve_style(DEFAULT_STYLE)
    arr = np.asarray(render_selection(im, cands, layout="grid", style=DEFAULT_STYLE)[0])
    tw, th = RD.grid_tile_size(w, h, 3, 3, st["grid_long_side"], st["gutter_px"])
    small = RD._resize_rgb(np.asarray(im), tw, th)
    # 参照格：与缩放后的原图逐位相同（没有叠加、没有字）
    assert np.array_equal(_cell(arr, 0, st, tw, th, 3), small)
    fill = np.asarray(st["fill_rgb"])
    for i in range(1, k + 1):
        cell = _cell(arr, i, st, tw, th, 3).astype(int)
        m = RD._resize_mask(cands[i - 1].mask, tw, th)
        # 填充之后的像素朝填充色偏：第 i 格在第 i 条竖带里
        inner = m.copy()  # 离边界 ≥ 3 px 的 mask 内部
        for _ in range(3):
            inner &= np.roll(inner, 1, 0) & np.roll(inner, -1, 0) & np.roll(inner, 1, 1) & np.roll(inner, -1, 1)
        num = int(round(st["fill_alpha"] * RD.ALPHA_DEN))
        want = (small[inner].astype(np.int64) * (RD.ALPHA_DEN - num) + fill * num + RD.ALPHA_DEN // 2) // RD.ALPHA_DEN
        assert inner.any() and np.array_equal(cell[inner], want), i
        # mask 外、离 mask 与编号都远的像素与原图相同
        far = ~m
        for _ in range(st["halo_px"] + 2):
            far &= np.roll(far, 1, 0) & np.roll(far, -1, 0) & np.roll(far, 1, 1) & np.roll(far, -1, 1)
        far[:40, :40] = False
        assert np.array_equal(cell[far], small[far].astype(int))
        # 轮廓与描边
        assert (cell == np.asarray(st["contour_rgb"])).all(axis=-1).sum() > 0
        # 左上角是黑底的编号
        assert (cell[:6, :6] == np.asarray(st["label_bg_rgb"])).all()
        assert (cell[:30, :30] == np.asarray(st["label_text_rgb"])).all(axis=-1).any()


def test_separate_order_matches_candidates():
    w, h = 320, 240
    im = _image(w, h, seed=2)
    cands = _quadrant_cands(w, h, 4)
    out = render_selection(im, cands, layout="separate", style=DEFAULT_STYLE)
    assert len(out) == 5
    assert np.array_equal(np.asarray(out[0]), np.asarray(im))  # 参照：原图原样（不放大、没有字）
    for i, img in enumerate(out[1:], start=1):
        a = np.asarray(img).astype(int)
        diff = np.abs(a - np.asarray(im).astype(int)).sum(axis=-1) > 0
        diff[:50, :50] = False  # 编号
        cols = np.flatnonzero(diff.any(axis=0))
        band = (i - 1) * w // 4, i * w // 4
        assert cols.min() >= band[0] - 2 and cols.max() <= band[1] + 1, (i, cols.min(), cols.max())
        assert diff[h // 2, (band[0] + band[1]) // 2]


def test_reference_label_and_empty_mask():
    w, h = 160, 120
    im = _image(w, h, seed=4)
    cands = [Candidate("empty", np.zeros((h, w), bool), True, {})]
    out = render_selection(im, cands, layout="separate", style={"reference_label": "ref"})
    a0 = np.asarray(out[0])
    assert (a0[:4, :4] == 0).all()  # 参照写了字：左上角黑底
    a1 = np.asarray(out[1])
    rest = np.ones((h, w), bool)
    rest[:48, :48] = False
    assert np.array_equal(a1[rest], np.asarray(im)[rest])  # 空 mask：只有编号


def test_render_is_bitwise_deterministic():
    w, h = 640, 427
    im = _image(w, h, seed=7)
    cands = _quadrant_cands(w, h, 8)
    for layout in RD.LAYOUTS:
        a = [np.asarray(x).tobytes() for x in render_selection(im, cands, layout=layout, style=DEFAULT_STYLE)]
        b = [np.asarray(x).tobytes() for x in render_selection(im.copy(), cands, layout=layout, style=dict(DEFAULT_STYLE))]
        assert a == b


# ---------------------------------------------------------------------------
# 样式与参数
# ---------------------------------------------------------------------------


def test_style_resolution_and_validation():
    full = resolve_style(None)
    assert set(full) == set(DEFAULT_STYLE) and full["fill_rgb"] == list(DEFAULT_STYLE["fill_rgb"])
    assert resolve_style({"fill_alpha": 0.25})["fill_alpha"] == 0.25 and resolve_style({"fill_alpha": 0.25})["gutter_px"] == 4
    for bad in ({"nope": 1}, {"fill_rgb": (1, 2)}, {"fill_rgb": (1, 2, 300)}, {"fill_alpha": 1.5}, {"contour_px": -1},
                {"grid_cols": 2.5}, {"reference": 1}, {"reference_label": "x" * 40}, {"separate_max_wh": (640,)},
                {"grid_long_side": True}):
        with pytest.raises(ValueError):
            resolve_style(bad)
    im = _image(64, 48)
    cands = _flat_cands(64, 48, 2)
    with pytest.raises(ValueError):
        render_selection(im, cands, layout="mosaic", style=DEFAULT_STYLE)
    with pytest.raises(ValueError):
        render_selection(im, [], layout="grid", style=DEFAULT_STYLE)
    with pytest.raises(ValueError):
        render_selection(im, [Candidate("x", np.zeros((10, 10), bool), True, {})], layout="grid", style=DEFAULT_STYLE)
    with pytest.raises(TypeError):
        render_selection(np.asarray(im), cands, layout="grid", style=DEFAULT_STYLE)
    # 关掉轮廓与描边、alpha = 0：mask 内外都与原图相同（只剩编号）
    out = render_selection(im, [Candidate("x", np.ones((48, 64), bool), True, {})], layout="separate",
                           style={"fill_alpha": 0.0, "contour_px": 0, "halo_px": 0, "reference": False})
    a = np.asarray(out[0])
    assert np.array_equal(a[20:, 20:], np.asarray(im)[20:, 20:])


def test_grid_cols_override():
    im = _image(640, 480)
    out = render_selection(im, _flat_cands(640, 480, 8), layout="grid", style={"grid_cols": 3, "reference": False})
    assert out[0].size == (638, 482)
    out = render_selection(im, _flat_cands(640, 480, 8), layout="grid", style={"grid_cols": 4, "reference": False})
    assert out[0].size == (4 * 157 + 3 * 4, 2 * 118 + 4)


# ---------------------------------------------------------------------------
# 视觉 token 数（适配器的图像处理现算）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny300():
    from src.adapters import qwen35
    from src.adapters.image_policy import E1_PIXEL_BUDGET_TOKENS, PixelBudget

    try:
        comps = qwen35.load_pinned_components()
    except Exception as e:  # noqa: BLE001 — 取不到小文件时给出原因后失败，不跳过（同 tests/test_adapter_qwen35_pixel_budget.py）
        pytest.fail(f"取不到 Qwen3.5 的小文件：{e!r}（需要联网下载一次）", pytrace=False)
    assert E1_PIXEL_BUDGET_TOKENS == 300
    return qwen35.make_tiny_adapter(comps, image_policy=PixelBudget(E1_PIXEL_BUDGET_TOKENS))


def test_visual_tokens_under_pixel_budget_300(tiny300):
    w, h = 640, 480
    im = _image(w, h)
    cands = _quadrant_cands(w, h, 8)
    grid = render_selection(im, cands, layout="grid", style=DEFAULT_STYLE)
    sep = render_selection(im, cands, layout="separate", style=DEFAULT_STYLE)
    assert RD.visual_tokens(tiny300, grid) == [300]
    assert RD.visual_tokens(tiny300, sep) == [300] * 9
    im2 = _image(640, 427)
    grid2 = render_selection(im2, _quadrant_cands(640, 427, 8), layout="grid", style=DEFAULT_STYLE)
    assert RD.visual_tokens(tiny300, grid2) == [330] and RD.visual_tokens(tiny300, [im2]) == [330]


# ---------------------------------------------------------------------------
# 文档
# ---------------------------------------------------------------------------


def test_docs_list_every_style_key_and_default():
    text = DOCS.read_text(encoding="utf-8")
    for k, v in resolve_style(None).items():
        shown = f"`{k}`"
        assert shown in text, k
        val = f"`{v!r}`" if isinstance(v, str) else f"`{str(v).lower() if isinstance(v, bool) else v}`"
        assert val in text, (k, val)
    assert f"`DEFAULT_LAYOUT` = `{RD.DEFAULT_LAYOUT}`" in text
