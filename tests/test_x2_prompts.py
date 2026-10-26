"""DEV-0069：X-2 的提示与解析（src/explore/x2_prompts.py）与修正叠加图（src/explore/x2_render.py）。

- 提示原文里有指代句、提出的框的网格整数、n / m；目标文本能被严格解析回同样的值；合法与各类非法输出的分类；parse_generation 的截断、空回答；
- 坐标换算与适配器一致：E1 的原生框（网格整数）经 Qwen 适配器解析成原图像素后，box_q 还原出同样的四个整数；网格 → 像素与适配器同一个函数；
- 生成上限够写下最坏情况的目标（tiny 适配器的 tokenizer 实数）；
- 光栅化与 X-1 的 B 样条通道（oracle_sim + guidance.rasterize_loops，框约束）逐位相同；
- 叠加图：尺寸同 W2 separate 的单图，叠加部分与 W2 的挑选渲染逐位相同，框画在约定的像素上，确定；
- 这三个模块与 x2_data 不 import 学习型模块（干净的子进程）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.adapters import qwen35
from src.adapters.base import native_box_ints
from src.executors.geometry import norm1000_to_abs
from src.executors.raster import box_to_mask
from src.explore import guidance as G
from src.explore import oracle_sim as OS
from src.explore import x2_prompts as XP
from src.explore import x2_render as XR
from src.parsing import parse_generation
from src.w2 import render as WR
from src.w2.config import load_w2_config

REPO = Path(__file__).resolve().parents[1]
F = XP.X2Failure


# ---------------------------------------------------------------------------
# 提示与目标文本
# ---------------------------------------------------------------------------


def test_prompts_carry_box_and_counts():
    sp = XP.spline_prompt("  the red cup ", [12, 30, 400, 999], 16)
    assert "the red cup" in sp and "[12, 30, 400, 999]" in sp and "exactly 16 control points" in sp and "exactly 16 points" in sp
    assert sp.startswith("The object the red cup lies inside the box") and "0-1000 grid" in sp and "x first" in sp
    co = XP.correct_prompt("dog on left", [0, 0, 1000, 1000], 2)
    assert "dog on left" in co and "[0, 0, 1000, 1000]" in co and "at most 2 local regions" in co
    assert '[{"sign": "+", "box": [x1, y1, x2, y2]}, ...]' in co and "output []" in co and '"-" for an extra part' in co
    assert XP.spline_prompt("a", [1, 2, 3, 4], 12) != XP.spline_prompt("a", [1, 2, 3, 4], 16)
    for bad in ([1, 2, 3], [5, 0, 4, 9], [0, 0, 1001, 5], [0.5, 0, 4, 5], [True, 0, 4, 5]):
        with pytest.raises(ValueError):
            XP.spline_prompt("a", bad)
    with pytest.raises(ValueError):
        XP.spline_prompt("   ", [0, 0, 1, 1])
    with pytest.raises(ValueError):
        XP.spline_prompt("a", [0, 0, 1, 1], 3)
    with pytest.raises(ValueError):
        XP.correct_prompt("a", [0, 0, 1, 1], 0)


def test_target_texts_round_trip():
    pts = [[0, 1000], [17, 3], [999, 500], [250, 250]]
    t = XP.spline_text(pts)
    assert t == "[[0, 1000], [17, 3], [999, 500], [250, 250]]"
    assert XP.parse_spline(t, n=4) == (pts, None)
    items = [{"sign": "+", "box": [1, 2, 3, 4]}, {"sign": "-", "box": [10, 10, 10, 20]}]
    t = XP.corrections_text(items)
    assert t == '[{"sign": "+", "box": [1, 2, 3, 4]}, {"sign": "-", "box": [10, 10, 10, 20]}]'
    assert XP.parse_corrections(t, m=2) == (items, None)
    assert XP.corrections_text([]) == "[]" and XP.parse_corrections("[]", m=2) == ([], None)
    assert XP.parse_corrections(" [ ]\n", m=1) == ([], None)


@pytest.mark.parametrize("text,fail", [
    ("", F.EMPTY), ("   ", F.EMPTY),
    ("```json\n[[1, 2], [3, 4], [5, 6], [7, 8]]\n```", F.MALFORMED_JSON),
    ("[[1, 2], [3, 4], [5, 6], [7, 8]] done", F.MALFORMED_JSON),
    ("[[1, 2], [3, 4]] [[5, 6]]", F.MALFORMED_JSON),
    ("[[1, 2], [3, 4], [5, 6], [7, 8]", F.MALFORMED_JSON),
    ('{"points": [[1, 2]]}', F.NOT_LIST), ("7", F.NOT_LIST),
    ("[[1, 2], [3, 4, 5], [5, 6], [7, 8]]", F.BAD_POINT), ("[[1, 2], 3, [5, 6], [7, 8]]", F.BAD_POINT),
    ("[[1, 2], [3.0, 4], [5, 6], [7, 8]]", F.NON_INTEGER), ('[[1, 2], ["3", 4], [5, 6], [7, 8]]', F.NON_INTEGER),
    ("[[1, 2], [true, 4], [5, 6], [7, 8]]", F.NON_INTEGER),
    ("[[1, 2], [-1, 4], [5, 6], [7, 8]]", F.OUT_OF_RANGE), ("[[1, 2], [3, 1001], [5, 6], [7, 8]]", F.OUT_OF_RANGE),
    ("[[1, 2], [3, 4], [5, 6]]", F.WRONG_COUNT), ("[[1, 2], [3, 4], [5, 6], [7, 8], [9, 9]]", F.WRONG_COUNT), ("[]", F.WRONG_COUNT),
])
def test_parse_spline_failures(text, fail):
    p = XP.parse_spline(text, n=4)
    assert p.value is None and p.failure is fail and XP.failure_name(p.failure) == fail.value


@pytest.mark.parametrize("text,fail", [
    ("", F.EMPTY), ("none", F.MALFORMED_JSON), ('```json\n[]\n```', F.MALFORMED_JSON), ('[] ok', F.MALFORMED_JSON),
    ('{"sign": "+", "box": [1, 2, 3, 4]}', F.NOT_LIST),
    ('[{"sign": "+"}]', F.BAD_ITEM), ('[{"sign": "+", "box": [1, 2, 3, 4], "why": "x"}]', F.BAD_ITEM), ('[[1, 2, 3, 4]]', F.BAD_ITEM),
    ('[{"sign": "+", "box": [1, 2, 3]}]', F.BAD_ITEM), ('[{"sign": "+", "box": "1,2,3,4"}]', F.BAD_ITEM),
    ('[{"sign": "add", "box": [1, 2, 3, 4]}]', F.BAD_SIGN), ('[{"sign": "\\u2212", "box": [1, 2, 3, 4]}]', F.BAD_SIGN),
    ('[{"sign": 1, "box": [1, 2, 3, 4]}]', F.BAD_SIGN),
    ('[{"sign": "+", "box": [1, 2, 3.5, 4]}]', F.NON_INTEGER), ('[{"sign": "-", "box": [1, 2, 3, 1001]}]', F.OUT_OF_RANGE),
    ('[{"sign": "-", "box": [5, 2, 3, 4]}]', F.INVERTED), ('[{"sign": "-", "box": [1, 9, 3, 4]}]', F.INVERTED),
    ('[{"sign": "+", "box": [1, 2, 3, 4]}, {"sign": "-", "box": [1, 2, 3, 4]}, {"sign": "+", "box": [5, 6, 7, 8]}]', F.TOO_MANY),
])
def test_parse_corrections_failures(text, fail):
    p = XP.parse_corrections(text, m=2)
    assert p.value is None and p.failure is fail


def test_parse_generation_wraps_truncation_and_empty():
    ok = SimpleNamespace(thinking_unclosed=False, stopped=True, text="[[1, 2], [3, 4], [5, 6], [7, 8]]")
    assert parse_generation(ok, XP.parse_spline, n=4).value == [[1, 2], [3, 4], [5, 6], [7, 8]]
    tr = SimpleNamespace(thinking_unclosed=False, stopped=False, text="[[1, 2], [3, 4], [5, 6], [7, 8]]")
    p = parse_generation(tr, XP.parse_spline, n=4)
    assert p.value is None and XP.failure_name(p.failure) == "truncated"  # 截断一律失败（E1 的协议）
    em = SimpleNamespace(thinking_unclosed=False, stopped=True, text="  ")
    assert XP.failure_name(parse_generation(em, XP.parse_corrections, m=2).failure) == "empty"


# ---------------------------------------------------------------------------
# 坐标与适配器一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wh", [(640, 480), (480, 640), (427, 640), (128, 96), (1000, 1000), (333, 517)])
def test_box_grid_round_trip_matches_adapter(wh):
    w, h = wh
    rng = np.random.default_rng(w * 7 + h)
    for _ in range(200):
        a, c = sorted(int(v) for v in rng.integers(0, 1001, 2))
        b, d = sorted(int(v) for v in rng.integers(0, 1001, 2))
        q = [a, b, c, d]
        text = json.dumps([{"bbox_2d": q, "label": "x"}])
        parsed = qwen35.parse_one_box(text, width=w, height=h)  # E1 refexp_box 的提出框 = 适配器的解析
        box = [float(v) for v in parsed.box]
        assert XP.box_q(box, w, h) == q  # 提示里写的就是模型写的那四个整数
        assert native_box_ints([box], width=w, height=h) == [tuple(q)]  # 与适配器的取整一致（网格点上没有 .5 的并列）
        assert XP.box_abs(q, w, h) == pytest.approx(box, abs=0)  # 网格 → 像素 = 适配器的同一个换算
        np.testing.assert_array_equal(XP.points_abs([[a, b], [c, d]], w, h), norm1000_to_abs(np.array([[a, b], [c, d]], float), width=w, height=h))


def test_proposals_abs_and_spline_mask_match_x1_channel():
    yy, xx = np.mgrid[0:96, 0:128]
    gt = ((xx - 60.5) ** 2 / 30 ** 2 + (yy - 50.5) ** 2 / 20 ** 2) <= 1
    box = [28.0, 29.0, 93.0, 72.0]
    sp = OS.oracle_spline(gt, 16)
    q = sp["loops_q"][0]["ctrl"]
    ours = XP.spline_mask(q, box, gt.shape)
    ref = G.rasterize_loops(sp["loops"], gt.shape, box=box)  # X-1 的 B 样条直接光栅化（框约束）
    np.testing.assert_array_equal(ours, ref)
    assert (ours & gt).sum() / (ours | gt).sum() > 0.95
    assert not (ours & ~box_to_mask(box, 96, 128)).any()  # 夹在框内
    props = XP.proposals_abs([{"sign": "-", "box": [0, 0, 500, 500]}], 128, 96)
    assert props == [([0.0, 0.0, 64.0, 48.0], "-")]


def test_generation_budget_fits_worst_case_targets():
    from src.adapters.factory import make_adapter
    import torch

    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, tiny=True)
    tok = lambda s: len(ad.tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    for n in (12, 16, 24):
        worst = XP.spline_text([[1000, 1000]] * n)
        typical = XP.spline_text([[123 + i, 987 - i] for i in range(n)])
        assert tok(worst) < XP.max_new_tokens_spline(n) and tok(typical) < XP.max_new_tokens_spline(n)
        assert tok(typical) < 0.6 * XP.max_new_tokens_spline(n)
    for m in (1, 2, 3):
        worst = XP.corrections_text([{"sign": "-", "box": [1000, 1000, 1000, 1000]}] * m)
        assert tok(worst) < XP.max_new_tokens_correct(m)


# ---------------------------------------------------------------------------
# 叠加图
# ---------------------------------------------------------------------------


def _img(w: int, h: int) -> Image.Image:
    rng = np.random.default_rng(w + h)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("wh", [(1280, 720), (480, 640), (128, 96)])
def test_overlay_matches_w2_rendering_and_draws_box(wh):
    w, h = wh
    style = load_w2_config().style
    img = _img(w, h)
    mask = np.zeros((h, w), bool)
    mask[h // 4 : h // 2, w // 3 : w // 2] = True
    box = [w * 0.2, h * 0.15, w * 0.7, h * 0.8]
    ov = XR.render_overlay(img, mask, box, style=style)
    tw, th = WR.separate_size(w, h, style["separate_max_wh"])
    assert ov.size == (tw, th) == XR.overlay_size(w, h, style)
    st = {**style, "reference": False}
    tile = np.asarray(WR.render_selection(img, [SimpleNamespace(mask=mask)], layout="separate", style=st)[0])
    arr = np.asarray(ov)
    nobox = np.asarray(XR.render_overlay(img, mask, None, style=style))
    # W2 的单图 = 同样的叠加 + 左上角的编号：编号块以外逐位相同
    lab = np.any(tile != nobox, axis=2)
    rows, cols = np.nonzero(lab)
    assert rows.max() < 60 and cols.max() < 60
    keep = np.ones((th, tw), bool)
    keep[:60, :60] = False
    np.testing.assert_array_equal(tile[keep], nobox[keep])
    # 框：只改框线上的像素，颜色 BOX_RGB
    diff = np.any(arr != nobox, axis=2)
    c0, r0 = int(np.floor(box[0] * tw / w)), int(np.floor(box[1] * th / h))
    c1, r1 = int(np.ceil(box[2] * tw / w)) - 1, int(np.ceil(box[3] * th / h)) - 1
    ring = np.zeros((th, tw), bool)
    ring[r0 : r1 + 1, c0 : c1 + 1] = True
    ring[r0 + XR.BOX_PX : r1 + 1 - XR.BOX_PX, c0 + XR.BOX_PX : c1 + 1 - XR.BOX_PX] = False
    assert not (diff & ~ring).any() and (arr[ring] == np.array(XR.BOX_RGB, np.uint8)).all()
    assert np.array_equal(arr, np.asarray(XR.render_overlay(img, mask, box, style=style)))  # 确定
    with pytest.raises(ValueError):
        XR.render_overlay(img, mask[:, :-1], box, style=style)


def test_overlay_empty_mask_and_tiny_box():
    style = load_w2_config().style
    img = _img(64, 48)
    m = np.zeros((48, 64), bool)
    ov = np.asarray(XR.render_overlay(img, m, [10.0, 10.0, 11.0, 11.0], style=style))
    base = np.asarray(img)
    diff = np.any(ov != base, axis=2)
    assert diff.sum() == 1 and diff[10, 10]  # 1 px 的框整块涂满，其余是原图


# ---------------------------------------------------------------------------
# 不 import 学习型模块
# ---------------------------------------------------------------------------


def test_x2_tool_modules_import_no_learned_modules():
    code = ("import json, sys; import src.explore.x2_prompts, src.explore.x2_render, src.explore.x2_data; "
            "import src.explore.guidance as G; "
            "print(json.dumps({'loaded': G.loaded_learned_modules(), 'roots': sorted({m.split('.')[0] for m in sys.modules})}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True)
    doc = json.loads(r.stdout.strip().splitlines()[-1])
    assert doc["loaded"] == [] and not set(doc["roots"]) & set(G.BANNED_MODULE_ROOTS)
    assert "cv2" in doc["roots"] and "PIL" in doc["roots"]
