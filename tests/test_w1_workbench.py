"""DEV-0050：W1 工作台（src/w1/workbench.py）。

钉住：框、多边形、样条的光栅化与 src/executors/raster.py 的规则一致；贴边工具在明确的合成情形下给出预期的 mask；
zoom 的覆盖范围与坐标换算；check 的叠加图与原图同尺寸；specs() 的 JSON Schema 合法、关掉的组不出现且调用报错；
出错不抛异常；日志字段；同样的调用序列结果逐位相同；docs/w1_tools.md 列出每个工具。

DEV-0054（视图坐标，view_frames）：关时 specs() 的 tools_digest 与改动前相同、行为不变；开时视图编号递增、按视图给的坐标与
直接给换算后的原图坐标逐位同结果、整图视图的换算是恒等、未知编号与超过 MAX_VIEWS 报错、结果确定。
"""

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.executors.geometry import norm1000_to_abs
from src.executors.overlay import render_overlay
from src.executors.raster import box_to_mask, polygons_to_mask
from src.w1 import workbench as wbm
from src.w1.workbench import GROUPS, Submission, ToolResult, Workbench, grabcut_mask, spline_polygon

H, W = 96, 128  # H ≠ W，坐标换算的 x、y 不对称
CY, CX, R = 48, 60, 22  # 圆心（行、列）与半径（px）
DOCS = Path(__file__).resolve().parents[1] / "docs" / "w1_tools.md"


def _scene(seed=0):
    """噪声灰背景上一个纯色红圆，外加一个纯色蓝方块。"""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(110, 12, (H, W, 3)), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    disk = (xx + 0.5 - CX) ** 2 + (yy + 0.5 - CY) ** 2 <= R**2
    img[disk] = (220, 50, 40)
    img[70:90, 100:120] = (40, 80, 210)
    return img, disk


@pytest.fixture()
def scene():
    img, disk = _scene()
    return img, disk, Workbench(Image.fromarray(img))


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def _ok(r: ToolResult) -> ToolResult:
    assert r.error is None, r.text
    return r


def _abs(*coords):
    return norm1000_to_abs(np.asarray(coords, dtype=np.float64), width=W, height=H)


# ---------------------------------------------------------------------------
# hand：光栅化与 raster.py 一致
# ---------------------------------------------------------------------------


def test_draw_box_matches_raster(scene):
    _, _, wb = scene
    _ok(wb.call("draw_box", {"name": "b", "x1": "123.4", "y1": "201", "x2": "654", "y2": "777.7"}))
    assert np.array_equal(wb.masks["b"], box_to_mask(_abs(123.4, 201, 654, 777.7), H, W))


def test_draw_polygon_matches_raster(scene):
    _, _, wb = scene
    pts = [[100, 120], [800, 90], [650, 700], [300, 930]]
    _ok(wb.call("draw_polygon", {"name": "p", "points": json.dumps(pts)}))
    assert np.array_equal(wb.masks["p"], polygons_to_mask([_abs(*np.ravel(pts))], H, W))


def test_draw_spline_matches_raster_and_is_smooth(scene):
    _, disk, wb = scene
    # 圆周上 8 个控制点：样条应当接近这个圆
    ang = np.arange(8) * 2 * np.pi / 8
    pts_abs = np.stack([CX + R * np.cos(ang), CY + R * np.sin(ang)], axis=1)
    pts = np.stack([pts_abs[:, 0] / W * 1000, pts_abs[:, 1] / H * 1000], axis=1).round(3)
    _ok(wb.call("draw_spline", {"name": "s", "points": json.dumps(pts.tolist())}))
    expected = polygons_to_mask([spline_polygon(_abs(*pts.ravel()).reshape(-1, 2)).ravel()], H, W)
    assert np.array_equal(wb.masks["s"], expected)
    assert _iou(wb.masks["s"], disk) > 0.95
    # 样条经过控制点，比同样 8 个点连成的多边形更接近圆
    poly = polygons_to_mask([_abs(*pts.ravel())], H, W)
    assert _iou(wb.masks["s"], disk) > _iou(poly, disk)


def test_spline_polygon_passes_through_control_points():
    p = np.array([[10.0, 10.0], [50.0, 12.0], [40.0, 60.0], [5.0, 40.0]])
    s = spline_polygon(p, samples_per_segment=8)
    assert s.shape == (32, 2)
    assert np.allclose(s[::8], p)
    with pytest.raises(ValueError):
        spline_polygon(np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]))


def test_draw_points_positive_and_negative(scene):
    _, _, wb = scene
    _ok(wb.call("draw_points", {"name": "q", "points": "[[500, 500]]", "radius": "4"}))
    x, y = _abs(500, 500)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= 16
    assert np.array_equal(wb.masks["q"], disc)
    # 负点擦掉圆盘（mode=union 时作用于已有 mask）
    _ok(wb.call("draw_points", {"name": "q", "points": "[[500, 500]]", "labels": "[0]", "radius": "2", "mode": "union"}))
    assert wb.masks["q"].sum() == disc.sum() - ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= 4).sum()
    # 半径 0：只有点所在的那个像素
    _ok(wb.call("draw_points", {"name": "one", "points": "[[0, 0], [1000, 1000]]", "radius": "0"}))
    assert wb.masks["one"].sum() == 2 and wb.masks["one"][0, 0] and wb.masks["one"][H - 1, W - 1]


def test_mask_modes(scene):
    _, _, wb = scene
    a = box_to_mask(_abs(0, 0, 500, 500), H, W)
    b = box_to_mask(_abs(250, 250, 750, 750), H, W)
    _ok(wb.call("draw_box", {"name": "m", "x1": 0, "y1": 0, "x2": 500, "y2": 500}))
    _ok(wb.call("draw_box", {"name": "m", "x1": 250, "y1": 250, "x2": 750, "y2": 750, "mode": "union"}))
    assert np.array_equal(wb.masks["m"], a | b)
    _ok(wb.call("draw_box", {"name": "m", "x1": 250, "y1": 250, "x2": 750, "y2": 750, "mode": "subtract"}))
    assert np.array_equal(wb.masks["m"], a & ~b)
    _ok(wb.call("draw_box", {"name": "m", "x1": 0, "y1": 0, "x2": 500, "y2": 500, "mode": "new"}))
    _ok(wb.call("draw_box", {"name": "m", "x1": 250, "y1": 250, "x2": 750, "y2": 750, "mode": "intersect"}))
    assert np.array_equal(wb.masks["m"], a & b)
    # combine_masks 与 delete_mask
    _ok(wb.call("draw_box", {"name": "n", "x1": 250, "y1": 250, "x2": 750, "y2": 750}))
    _ok(wb.call("combine_masks", {"name": "m", "other": "n"}))
    assert np.array_equal(wb.masks["m"], b)
    _ok(wb.call("delete_mask", {"name": "n"}))
    assert "n" not in wb.masks
    # subtract / intersect 需要已有的 mask
    r = wb.call("draw_box", {"name": "zz", "x1": 0, "y1": 0, "x2": 10, "y2": 10, "mode": "subtract"})
    assert r.error and "does not exist" in r.error


# ---------------------------------------------------------------------------
# snap：明确的合成情形
# ---------------------------------------------------------------------------


def test_grabcut_from_box_and_from_mask(scene):
    img, disk, wb = scene
    box = [(CX - R - 6) / W * 1000, (CY - R - 6) / H * 1000, (CX + R + 6) / W * 1000, (CY + R + 6) / H * 1000]
    _ok(wb.call("grabcut", {"out": "g", "box": json.dumps(box)}))
    assert _iou(wb.masks["g"], disk) > 0.95
    # W1-F 用的是同一个函数：同一个框给出逐位相同的结果
    assert np.array_equal(wb.masks["g"], grabcut_mask(img, box_abs=_abs(*box)))
    # 从 mask 初始化（膨胀过的圆）
    big = cv2.dilate(disk.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    wb2 = Workbench(Image.fromarray(img))
    wb2._set_mask("init", big)  # 测试里直接放一个初始 mask
    _ok(wb2.call("grabcut", {"out": "g", "mask": "init", "iterations": "5"}))
    assert _iou(wb2.masks["g"], disk) > 0.9
    # 两种初始化都给 / 都不给 → 报错
    assert wb.call("grabcut", {"out": "x"}).error
    assert wb.call("grabcut", {"out": "x", "box": "[0, 0, 500, 500]", "mask": "g"}).error


def test_grabcut_mask_init_is_bounded_by_margin():
    """mask 初始化：结果不会越出 mask 外接框每边外扩 GRABCUT_MASK_MARGIN 的范围（更远处是确定背景）。"""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)  # 纯噪声：没有可贴的边，GrabCut 不受约束时会四处扩散
    init = np.zeros((H, W), bool)
    init[40:60, 50:80] = True
    out = grabcut_mask(img, init_mask=init)
    mr, mc = math.ceil(wbm.GRABCUT_MASK_MARGIN * 20), math.ceil(wbm.GRABCUT_MASK_MARGIN * 30)
    allowed = np.zeros((H, W), bool)
    allowed[40 - mr : 60 + mr, 50 - mc : 80 + mc] = True
    assert not (out & ~allowed).any()
    with pytest.raises(ValueError):
        grabcut_mask(img, init_mask=np.zeros((H, W), bool))  # 没有前景
    with pytest.raises(ValueError):
        grabcut_mask(img, box_abs=[0, 0, W, H])  # 框盖住整张图：没有背景


def test_toolspec_is_the_adapter_definition_when_available():
    """DEV-0049 合入后 ToolSpec 必须就是 src.adapters.chat.ToolSpec（约定只定义一处）；合入前用同字段的占位。"""
    import dataclasses

    try:
        from src.adapters.chat import ToolSpec as AdapterToolSpec
    except ModuleNotFoundError:
        AdapterToolSpec = None
    if AdapterToolSpec is not None:
        assert wbm.ToolSpec is AdapterToolSpec
    assert [f.name for f in dataclasses.fields(wbm.ToolSpec)] == ["name", "description", "parameters"]


def test_watershed_grows_seed_to_disk(scene):
    _, disk, wb = scene
    _ok(wb.call("draw_points", {"name": "seed", "points": json.dumps([[CX / W * 1000, CY / H * 1000]]), "radius": "2"}))
    _ok(wb.call("watershed", {"out": "w", "fg": "seed"}))
    assert _iou(wb.masks["w"], disk) > 0.95
    # 显式背景种子：与前景重叠时报错
    _ok(wb.call("draw_box", {"name": "bg", "x1": 0, "y1": 0, "x2": 1000, "y2": 30}))
    _ok(wb.call("watershed", {"out": "w2", "fg": "seed", "bg": "bg"}))
    assert _iou(wb.masks["w2"], disk) > 0.95
    _ok(wb.call("draw_box", {"name": "bad", "x1": 0, "y1": 0, "x2": 1000, "y2": 1000}))
    assert "overlap" in wb.call("watershed", {"out": "w3", "fg": "seed", "bg": "bad"}).error


def test_superpixels_and_selection(scene):
    _, disk, wb = scene
    assert "superpixels first" in wb.call("select_superpixels", {"name": "s", "ids": "[1]"}).error
    r = _ok(wb.call("superpixels", {"n_segments": "40"}))
    assert len(r.images) == 1 and r.images[0].size == (W, H)
    lab = wb._superpixels
    k = int(lab.max())
    assert set(np.unique(lab)) == set(range(1, k + 1))
    m = re.search(r"(\d+) superpixels with ids 1\.\.(\d+)", r.text)
    assert m and int(m.group(1)) == k
    # 圆内的点选中的超像素都基本落在圆内
    _ok(wb.call("select_superpixels", {"name": "s", "points": json.dumps([[CX / W * 1000, CY / H * 1000]])}))
    sel = wb.masks["s"]
    assert sel.sum() > 0 and (sel & disk).sum() / sel.sum() > 0.9
    # 按编号选：与标号图一致
    _ok(wb.call("select_superpixels", {"name": "t", "ids": "[1, 2]"}))
    assert np.array_equal(wb.masks["t"], np.isin(lab, [1, 2]))
    assert "out of range" in wb.call("select_superpixels", {"name": "t", "ids": json.dumps([k + 1])}).error


def test_refine_snaps_shifted_mask_to_edges(scene):
    img, disk, wb = scene
    shifted = np.roll(disk, 3, axis=1)  # 向右偏 3 px 的圆
    wb._set_mask("m", shifted)
    _ok(wb.call("refine", {"name": "m", "out": "r"}))
    assert _iou(wb.masks["r"], disk) > _iou(shifted, disk)
    assert _iou(wb.masks["r"], disk) > 0.9
    assert np.array_equal(wb.masks["m"], shifted)  # 写到 out，不改输入


def test_morph_ops(scene):
    _, disk, wb = scene
    wb._set_mask("d", disk)
    _ok(wb.call("morph", {"name": "d", "op": "dilate", "radius": "2", "out": "dd"}))
    _ok(wb.call("morph", {"name": "d", "op": "erode", "radius": "2", "out": "de"}))
    assert wb.masks["de"].sum() < disk.sum() < wb.masks["dd"].sum()
    assert np.array_equal(wb.masks["dd"], cv2.dilate(disk.astype(np.uint8), wbm.disk_footprint(2)) > 0)
    ring = disk & ~(cv2.erode(disk.astype(np.uint8), wbm.disk_footprint(5)) > 0)
    wb._set_mask("ring", ring)
    _ok(wb.call("morph", {"name": "ring", "op": "fill_holes"}))
    assert np.array_equal(wb.masks["ring"], disk)
    speck = disk.copy()
    speck[5, 5] = True
    wb._set_mask("sp", speck)
    _ok(wb.call("morph", {"name": "sp", "op": "open", "radius": "1"}))
    assert not wb.masks["sp"][5, 5] and _iou(wb.masks["sp"], disk) > 0.98


def test_components_listing_and_keep(scene):
    _, disk, wb = scene
    small = np.zeros((H, W), bool)
    small[2:6, 2:6] = True
    wb._set_mask("two", disk | small)
    r = _ok(wb.call("components", {"name": "two"}))
    assert "2 8-connected components" in r.text
    assert r.text.index(f"#1: {disk.sum()} px") < r.text.index("#2: 16 px")
    _ok(wb.call("components", {"name": "two", "keep": "1", "out": "big"}))
    assert np.array_equal(wb.masks["big"], disk)
    _ok(wb.call("components", {"name": "two", "keep": "[2]", "out": "lil"}))
    assert np.array_equal(wb.masks["lil"], small)
    assert "out of range" in wb.call("components", {"name": "two", "keep": "[3]"}).error


def test_edges_follow_the_disk_boundary(scene):
    _, disk, wb = scene
    r = _ok(wb.call("edges", {"out": "e"}))
    assert len(r.images) == 1 and r.images[0].size == (W, H)
    e = wb.masks["e"]
    band = (cv2.dilate(disk.astype(np.uint8), wbm.disk_footprint(2)) > 0) & ~(cv2.erode(disk.astype(np.uint8), wbm.disk_footprint(2)) > 0)
    ring_hit = (e & band).sum() / max(e[:, :90].sum(), 1)  # 只看圆所在的左边部分（右边还有蓝方块）
    assert ring_hit > 0.9
    assert wb.call("edges", {"low": "200", "high": "100"}).error


# ---------------------------------------------------------------------------
# eye 与 check
# ---------------------------------------------------------------------------


def test_zoom_coverage_and_conversion(scene):
    img, _, wb = scene
    r = _ok(wb.call("zoom", {"x1": "250", "y1": "250", "x2": "750", "y2": "750"}))
    view = np.asarray(r.images[0])
    assert max(view.shape[:2]) == wbm.ZOOM_LONG_SIDE
    # W=128、H=96：x 250..750 → 列 32..95，y 250..750 → 行 24..71
    assert "x = 250..750, y = 250..750" in r.text and "columns 32..95, rows 24..71" in r.text
    exp = cv2.resize(img[24:72, 32:96], (640, 480), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(view, exp)
    # 不在像素边界上的区域向外取整，文字报告实际覆盖的范围
    r = _ok(wb.call("zoom", {"x1": "251", "y1": "0", "x2": "500", "y2": "1000"}))
    assert "x = 250..500, y = 0..1000" in r.text
    # 视图里 (u, v) 的换算公式与覆盖范围一致
    assert "x = 250 + u*250" in r.text


def test_read_pixels_values(scene):
    img, _, wb = scene
    r = _ok(wb.call("read_pixels", {"points": json.dumps([[CX / W * 1000, CY / H * 1000], [0, 0]])}))
    lines = r.text.splitlines()
    assert "RGB (220, 50, 40)" in lines[0]
    v = img[0, 0].astype(float)
    assert f"RGB ({int(v[0])}, {int(v[1])}, {int(v[2])})" in lines[1]
    assert f"luminance {float(v @ np.asarray(wbm.LUMA_WEIGHTS)):.1f}" in lines[1]
    r = _ok(wb.call("read_pixels", {"points": "[[0, 0]]", "radius": "1"}))
    assert "mean over 4 px" in r.text  # 角上的 3×3 窗口截在图内，只剩 2×2


def test_tile_view(scene):
    _, _, wb = scene
    r = _ok(wb.call("tile_view", {"rows": "2", "cols": "4"}))
    assert r.images[0].size == (W, H)
    assert len([ln for ln in r.text.splitlines() if ln.startswith("cell ")]) == 8
    assert "cell 8 (row 2, col 4): x = 750..1000, y = 500..1000" in r.text


def test_show_overlays_the_original_image(scene):
    img, disk, wb = scene
    wb._set_mask("d", disk)
    r = _ok(wb.call("show", {"name": "d"}))
    ov = np.asarray(r.images[0])
    assert ov.shape == img.shape
    assert np.array_equal(ov, render_overlay(img, [disk]))
    assert np.array_equal(ov[~disk], img[~disk])  # mask 外与原图逐位相同
    # 带 region：与 zoom 同一套裁剪
    r = _ok(wb.call("show", {"name": "d", "region": "[250, 250, 750, 750]"}))
    assert max(np.asarray(r.images[0]).shape[:2]) == wbm.ZOOM_LONG_SIDE
    assert f"{disk.sum()} px" in r.text


# 每个工具一组合法参数；新增工具必须加进来（下面的测试先核对它与 specs() 的名字集合相同）。
_ALL_TOOL_ARGS = {
    "zoom": {"x1": "200", "y1": "200", "x2": "800", "y2": "800"},
    "read_pixels": {"points": "[[470, 500]]"},
    "tile_view": {"rows": "2", "cols": "2"},
    "draw_box": {"name": "n", "x1": "100", "y1": "100", "x2": "600", "y2": "600"},
    "draw_points": {"name": "n", "points": "[[470, 500]]"},
    "draw_polygon": {"name": "n", "points": "[[300, 250], [700, 250], [500, 750]]"},
    "draw_spline": {"name": "n", "points": "[[470, 270], [640, 500], [470, 730], [300, 500]]"},
    "combine_masks": {"name": "n", "other": "m"},
    "delete_mask": {"name": "m"},
    "edges": {},
    "grabcut": {"out": "g", "mask": "m"},
    "watershed": {"out": "w", "fg": "seed"},
    "superpixels": {"n_segments": "30"},
    "select_superpixels": {"name": "s", "points": "[[470, 500]]"},
    "refine": {"name": "m", "out": "r"},
    "morph": {"name": "m", "op": "dilate"},
    "components": {"name": "m"},
    "show": {"name": "m"},
    "run_python": {"code": "result_mask = masks['m']", "out": "c"},
    "submit_mask": {"name": "m"},
    "submit_box": {"x1": "100", "y1": "100", "x2": "600", "y2": "600"},
    "submit_count": {"n": "3"},
}
# 允许返回图的工具：check 组（叠加 mask 的唯一出口），eye 组与 snap 组里两个"看图本身"的视图（与 mask 无关）
_MASK_OVERLAY_TOOLS = {"show"}
_IMAGE_VIEW_TOOLS = {"zoom", "tile_view", "edges", "superpixels"}


def test_only_check_overlays_a_mask_on_the_original():
    """管理员裁定（DEV-0050 审查）：把 mask 叠回原图只有 check 组的 show 做；hand、snap（除 edges、superpixels 两个视图）、
    code、submit 只返回文字（含 mask 的面积、占比、0–1000 外接框）。eye 与两个视图的图与 mask 无关（换一个 mask 逐位不变），
    show 的图随 mask 变。这样 A2（去掉 check）之后，模型没有任何办法把结果叠回原图再看。"""
    img, disk = _scene()
    box = box_to_mask(_abs(150, 150, 650, 650), H, W)
    seed = np.zeros((H, W), bool)
    seed[CY - 1 : CY + 2, CX - 1 : CX + 2] = True
    wb0 = Workbench(Image.fromarray(img))
    assert set(_ALL_TOOL_ARGS) == {s.name for s in wb0.specs()}
    assert "render" not in _ALL_TOOL_ARGS and "unknown tool" in wb0.call("render", {"name": "m"}).error
    assert {n for n, t in wb0._tools.items() if t.group == "check"} == _MASK_OVERLAY_TOOLS

    def run(tool, m):
        wb = Workbench(Image.fromarray(img))
        wb._set_mask("m", m)
        wb._set_mask("seed", seed)
        if tool == "select_superpixels":
            _ok(wb.call("superpixels", {"n_segments": "30"}))
        return _ok(wb.call(tool, _ALL_TOOL_ARGS[tool]))

    for tool in _ALL_TOOL_ARGS:
        a, b = run(tool, disk), run(tool, box)
        if tool not in _MASK_OVERLAY_TOOLS | _IMAGE_VIEW_TOOLS:
            assert a.images == [] and b.images == [], tool
        elif tool in _IMAGE_VIEW_TOOLS:
            assert len(a.images) == 1 and np.array_equal(np.asarray(a.images[0]), np.asarray(b.images[0])), tool
        else:
            assert not np.array_equal(np.asarray(a.images[0]), np.asarray(b.images[0])), tool
            assert np.array_equal(np.asarray(a.images[0]), render_overlay(img, [disk]))


def test_hand_tools_report_area_and_box(scene):
    """hand 组只回文字：结果 mask 的面积（像素与占比）和 0–1000 外接框，模型照样有反馈。"""
    _, disk, wb = scene
    r = _ok(wb.call("draw_box", {"name": "b", "x1": "250", "y1": "250", "x2": "750", "y2": "750"}))
    m = wb.masks["b"]
    assert r.images == []
    assert f"Mask 'b': {int(m.sum())} px ({100 * m.sum() / (H * W):.2f}% of the {W}x{H} image)" in r.text
    assert "box [250, 250, 750, 750] (0-1000)" in r.text
    wb._set_mask("d", disk)
    for tool, args in [("combine_masks", {"name": "b", "other": "d", "mode": "intersect"}),
                       ("draw_points", {"name": "p", "points": "[[100, 100]]"}),
                       ("draw_polygon", {"name": "q", "points": "[[300, 250], [700, 250], [500, 750]]"}),
                       ("draw_spline", {"name": "s", "points": "[[470, 270], [640, 500], [470, 730], [300, 500]]"})]:
        r = _ok(wb.call(tool, args))
        assert r.images == [] and re.search(r"\d+ px \(\d+\.\d\d% of the 128x96 image\), box \[\d+, \d+, \d+, \d+\] \(0-1000\)", r.text), tool


# ---------------------------------------------------------------------------
# code 与 submit
# ---------------------------------------------------------------------------


def test_run_python_through_workbench(scene):
    _, disk, wb = scene
    wb._set_mask("d", disk)
    r = _ok(wb.call("run_python", {"code": "import numpy as np\nprint(int(masks['d'].sum()), H, W)\nresult_mask = ~masks['d']", "out": "inv"}))
    assert f"{disk.sum()} {H} {W}" in r.text
    assert np.array_equal(wb.masks["inv"], ~disk)
    r = wb.call("run_python", {"code": "import torch"})
    assert r.error and r.error.startswith("rejected")
    r = wb.call("run_python", {"code": "import numpy as np\nnp.load('x.npy')"})
    assert r.error and r.error.startswith("violation")
    entry = wb.log()[-1]
    assert entry["code"] == "import numpy as np\nnp.load('x.npy')" and entry["sandbox"]["status"] == "violation"


def test_submissions(scene):
    _, disk, wb = scene
    assert wb.submission is None
    wb._set_mask("d", disk)
    _ok(wb.call("submit_mask", {"name": "d"}))
    s = wb.submission
    assert isinstance(s, Submission) and s.kind == "mask" and s.source == "d" and np.array_equal(s.mask, disk)
    _ok(wb.call("submit_box", {"x1": "100", "y1": "200", "x2": "300", "y2": "400"}))
    assert wb.submission.kind == "box" and wb.submission.box == tuple(float(v) for v in _abs(100, 200, 300, 400))
    _ok(wb.call("submit_count", {"n": "7"}))
    assert wb.submission.kind == "count" and wb.submission.count == 7
    assert wb.call("submit_count", {"n": "-1"}).error
    assert wb.call("submit_count", {"n": "2.5"}).error


# ---------------------------------------------------------------------------
# specs、消融、错误、日志、确定性
# ---------------------------------------------------------------------------

_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean"}


def _check_schema(s: dict, path: str) -> None:
    """JSON Schema（draft 2020-12 中本工作台用到的子集）的结构检查。"""
    assert isinstance(s, dict), path
    assert s.get("type") in _JSON_TYPES, path
    assert not any(k.startswith("x-") for k in s), path
    allowed = {"type", "description", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
               "minimum", "maximum", "enum", "default"}
    assert set(s) <= allowed, (path, set(s) - allowed)
    if s["type"] == "object":
        assert isinstance(s["properties"], dict) and s["properties"]
        assert set(s.get("required", [])) <= set(s["properties"]), path
        for k, v in s["properties"].items():
            assert re.fullmatch(r"[a-z][a-z0-9_]*", k), k
            assert isinstance(v.get("description"), str) and v["description"], f"{path}.{k}"
            _check_schema(v, f"{path}.{k}")
    if s["type"] == "array":
        _check_schema(s["items"], path + "[]")
        assert s.get("minItems", 0) <= s.get("maxItems", math.inf)
    if s["type"] in ("number", "integer") and "minimum" in s and "maximum" in s:
        assert s["minimum"] <= s["maximum"], path
    if "enum" in s:
        assert len(s["enum"]) == len(set(s["enum"])) and s["enum"]


@pytest.mark.parametrize("view_frames", [False, True])
def test_specs_are_valid_json_schema(view_frames):
    wb = Workbench(Image.fromarray(_scene()[0]), view_frames=view_frames)
    specs = wb.specs()
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    assert {wb._tools[n].group for n in names} == set(GROUPS)
    for s in specs:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", s.name)
        assert isinstance(s.description, str) and s.description.endswith(".")
        p = s.parameters
        json.dumps(p)  # 可序列化
        assert p["type"] == "object" and p["additionalProperties"] is False
        if not p["properties"]:
            continue
        _check_schema(p, s.name)
        # 默认值本身满足约束：按默认值调用解析器不报错
        for k, v in p["properties"].items():
            if "default" in v:
                wbm._parse_value(json.dumps(v["default"]) if v["type"] != "string" else v["default"], wb._tools[s.name].properties[k], k)


@pytest.mark.parametrize("view_frames", [False, True])
def test_tool_texts_are_task_agnostic(view_frames):
    wb = Workbench(Image.fromarray(_scene()[0]), view_frames=view_frames)
    blob = json.dumps([[s.name, s.description, s.parameters] for s in wb.specs()]).lower()
    for word in ("refer", "referring", "segment the", "count the", "object described", "expression", "refcoco", "countbench"):
        assert word not in blob, word


@pytest.mark.parametrize("group", GROUPS)
def test_disabled_group_is_hidden_and_refused(group):
    img, _ = _scene()
    full = Workbench(Image.fromarray(img))
    wb = Workbench(Image.fromarray(img), disabled_groups=frozenset({group}))
    in_group = [n for n, t in full._tools.items() if t.group == group]
    assert in_group
    assert not set(in_group) & {s.name for s in wb.specs()}
    assert len(wb.specs()) == len(full.specs()) - len(in_group)
    for n in in_group:
        r = wb.call(n, {})
        assert r.error == wb.call("no_such_tool", {}).error.replace("no_such_tool", n)


def test_ablation_presets():
    img, _ = _scene()
    a3 = Workbench(Image.fromarray(img), disabled_groups=frozenset({"snap", "code"}))
    names = {s.name for s in a3.specs()}
    assert "grabcut" not in names and "run_python" not in names and "draw_polygon" in names and "show" in names
    with pytest.raises(ValueError):
        Workbench(Image.fromarray(img), disabled_groups=frozenset({"nope"}))


@pytest.mark.parametrize(
    "tool,args,needle",
    [
        ("zoom", {"x1": "0", "y1": "0", "x2": "1200", "y2": "500"}, "out of range"),
        ("zoom", {"x1": "600", "y1": "0", "x2": "500", "y2": "500"}, "x1 < x2"),
        ("zoom", {"x1": "0", "y1": "0", "x2": "500"}, "missing"),
        ("zoom", {"x1": "0", "y1": "0", "x2": "500", "y2": "500", "zoom": "2"}, "unexpected"),
        ("read_pixels", {"points": "[[1, 2], [3]]"}, "items"),
        ("read_pixels", {"points": "not json"}, "JSON array"),
        ("draw_box", {"name": "9bad", "x1": "0", "y1": "0", "x2": "5", "y2": "5"}, "not a valid name"),
        ("draw_box", {"name": "b", "x1": "0", "y1": "0", "x2": "5", "y2": "5", "mode": "xor"}, "one of"),
        ("draw_polygon", {"name": "p", "points": "[[1, 2], [3, 4]]"}, "items"),
        ("show", {"name": "ghost"}, "unknown mask"),
        ("morph", {"name": "ghost", "op": "dilate"}, "unknown mask"),
        ("tile_view", {"rows": "0", "cols": "2"}, "out of range"),
        ("draw_points", {"name": "p", "points": "[[1, 2]]", "labels": "[1, 1]"}, "one entry per point"),
        ("draw_points", {"name": "p", "points": "[[1, 2]]", "labels": "[2]"}, "one of"),
        ("no_such_tool", {}, "unknown tool"),
    ],
)
def test_errors_are_returned_not_raised(scene, tool, args, needle):
    _, _, wb = scene
    r = wb.call(tool, args)
    assert r.error is not None and needle in r.error, r.error
    assert r.text.startswith("Error: ") and r.images == []
    assert wb.log()[-1]["ok"] is False and wb.log()[-1]["error"] == r.error


def test_log_records_calls(scene):
    _, disk, wb = scene
    _ok(wb.call("draw_box", {"name": "b", "x1": "0", "y1": "0", "x2": "500", "y2": "500"}))
    _ok(wb.call("zoom", {"x1": "0", "y1": "0", "x2": "500", "y2": "500"}))
    wb.call("show", {"name": "ghost"})
    log = wb.log()
    assert [e["tool"] for e in log] == ["draw_box", "zoom", "show"]
    assert log[0]["arguments"] == {"name": "b", "x1": "0", "y1": "0", "x2": "500", "y2": "500"}
    assert log[0]["masks_written"] == {"b": int(wb.masks["b"].sum())} and log[0]["group"] == "hand"
    assert log[1]["images"][0]["width"] == 640 and len(log[1]["images"][0]["sha256"]) == 64
    assert log[1]["images"][0]["id"] in wb.images
    assert log[2]["ok"] is False and log[2]["masks_written"] == {}
    assert all(e["wall_time_s"] >= 0 for e in log)
    json.dumps(log)


_SEQUENCE = [
    ("tile_view", {"rows": "3", "cols": "3"}),
    ("zoom", {"x1": "200", "y1": "100", "x2": "800", "y2": "900"}),
    ("read_pixels", {"points": "[[470, 500], [10, 10]]", "radius": "2"}),
    ("draw_box", {"name": "box", "x1": "250", "y1": "200", "x2": "700", "y2": "800"}),
    ("grabcut", {"out": "g", "box": "[250, 200, 700, 800]"}),
    ("draw_points", {"name": "seed", "points": "[[470, 500]]", "radius": "2"}),
    ("watershed", {"out": "w", "fg": "seed"}),
    ("superpixels", {"n_segments": "30"}),
    ("select_superpixels", {"name": "s", "points": "[[470, 500]]"}),
    ("edges", {"out": "e"}),
    ("refine", {"name": "box", "out": "r"}),
    ("morph", {"name": "g", "op": "close", "radius": "2"}),
    ("components", {"name": "e", "keep": "[1]", "out": "e1"}),
    ("draw_spline", {"name": "sp", "points": "[[470, 270], [640, 500], [470, 730], [300, 500]]"}),
    ("run_python", {"code": "import numpy as np\nprint(int(masks['g'].sum()))\nresult_mask = masks['g'] & masks['sp']", "out": "c"}),
    ("show", {"name": "c", "region": "[200, 200, 800, 800]"}),
    ("show", {"name": "w"}),
    ("submit_mask", {"name": "c"}),
]


def _run_sequence(perturb_rng: bool):
    img, _ = _scene(seed=3)
    wb = Workbench(Image.fromarray(img))
    out = []
    for name, args in _SEQUENCE:
        if perturb_rng:  # 打乱 OpenCV 的全局随机数状态：GrabCut 每次调用前重设种子，结果不受影响
            cv2.setRNGSeed(12345)
            cv2.randu(np.zeros((10, 10), np.float32), 0, 1)
        r = wb.call(name, args)
        assert r.error is None, (name, r.text)
        out.append((r.text, [np.asarray(im).tobytes() for im in r.images]))
    log = wb.log()
    for e in log:
        e.pop("wall_time_s")
        if e["sandbox"]:
            e["sandbox"].pop("wall_time_s")
    return out, {k: v.tobytes() for k, v in wb.masks.items()}, wb.submission.mask.tobytes(), log


def test_workbench_is_deterministic():
    a = _run_sequence(perturb_rng=False)
    b = _run_sequence(perturb_rng=True)
    assert a[0] == b[0]  # 文字与图逐位相同
    assert a[1] == b[1] and a[2] == b[2] and a[3] == b[3]


def test_docs_list_every_tool_and_group():
    text = DOCS.read_text(encoding="utf-8")
    img, _ = _scene()
    wb = Workbench(Image.fromarray(img))
    for s in wb.specs():
        assert f"`{s.name}`" in text, s.name
        for p in s.parameters["properties"]:
            assert f"`{p}`" in text, (s.name, p)
    for g in GROUPS:
        assert f"`{g}`" in text, g
    # 文档里写成 "`常量名` = 数值" 的固定参数逐个与代码核对
    from src.w1 import sandbox as sbm

    found = {}
    for name, num, unit in re.findall(r"`([A-Z][A-Z0-9_]+)` = ([0-9.]+)( GB)?", text):
        found[name] = float(num) * (1024**3 if unit else 1)
        code_value = getattr(wbm, name, getattr(sbm, name, None))
        assert code_value is not None, name
        assert float(code_value) == found[name], (name, code_value, found[name])
    for const in ("ZOOM_LONG_SIDE", "GRABCUT_ITERATIONS", "SLIC_N_SEGMENTS", "SLIC_COMPACTNESS", "CANNY_LOW", "CANNY_HIGH",
                  "MORPH_RADIUS_DEFAULT", "POINT_RADIUS_DEFAULT", "SPLINE_SAMPLES_PER_SEGMENT", "SPLINE_ALPHA",
                  "GRABCUT_RNG_SEED", "SLIC_SIGMA", "REFINE_THRESHOLD", "CANNY_BLUR_KSIZE", "CANNY_BLUR_SIGMA",
                  "STDOUT_LIMIT_CHARS", "DEFAULT_TIMEOUT_S", "DEFAULT_MEMORY_BYTES", "MAX_MASKS", "COMPONENTS_LIST_MAX", "MAX_VIEWS"):
        assert const in found, const
    # view_frames 开时多出来的 view 参数也要写进文档
    for s in Workbench(Image.fromarray(img), view_frames=True).specs():
        for p in s.parameters["properties"]:
            assert f"`{p}`" in text, (s.name, p)


# ---------------------------------------------------------------------------
# 视图坐标（view_frames；DEV-0054）
# ---------------------------------------------------------------------------

# specs() 的 tools_digest（src/w1/loop.tools_digest）。关：与 DEV-0054 之前逐字节相同（改动前在 c697ad4e 上算得）；
# 开：带坐标的 10 个工具各多一个 view 参数，坐标参数的说明涵盖视图。改了工具说明就要改这里，并同步 docs/w1_tools.md。
DIGEST_VIEW_FRAMES_OFF = "dcc91402712dd16de304c385f64197d73a6a119b1dcaeb252fa3368b041f8a11"
DIGEST_VIEW_FRAMES_ON = "0bdcaf73354329f7dad411d774d4eec2c32295fd2db7a81968ab3a9ddcf6e038"
# 开时坐标参数的说明（给模型的原文，docs/w1_tools.md 第 9 节逐字照录）
_VIEW_COORD_DESCRIPTIONS = {
    "Left edge, 0-1000 (full image, or the view given by 'view').",
    "Top edge, 0-1000 (full image, or the view given by 'view').",
    "Right edge, 0-1000 (full image, or the view given by 'view').",
    "Bottom edge, 0-1000 (full image, or the view given by 'view').",
    "Points [[x, y], ...] in 0-1000 coordinates of the full image or of the view given by 'view'.",
    "Vertices [[x, y], ...] in 0-1000 coordinates of the full image or of the view given by 'view', at least 3.",
    "Points on the curve [[x, y], ...] in order, 0-1000 coordinates of the full image or of the view given by 'view', at least 3.",
    "Initial box [x1, y1, x2, y2] in 0-1000 coordinates of the full image or of the view given by 'view'.",
    "Points [[x, y], ...] in 0-1000 coordinates of the full image or of the view given by 'view'; the superpixel containing each point "
    "is selected.",
    "Optional region [x1, y1, x2, y2] in 0-1000 coordinates of the full image or of the view given by 'view'.",
}
_VIEW_TOOLS = {"zoom", "read_pixels", "draw_box", "draw_points", "draw_polygon", "draw_spline", "grabcut", "select_superpixels",
               "show", "submit_box"}
_IN_VIEW_NOTE = "Coordinates given in view {k}, converted to full-image 0-1000 coordinates. "
_VIEW_PREFIX = re.compile(r"^(This is view \d+\. )+")


def _wb(img=None, **kw) -> Workbench:
    return Workbench(Image.fromarray(_scene()[0] if img is None else img), **kw)


def _strip_view_ids(text: str) -> str:
    return _VIEW_PREFIX.sub("", text)


def test_view_frames_off_keeps_specs_digest():
    from src.w1.loop import tools_digest

    img, _ = _scene()
    assert tools_digest(Workbench(Image.fromarray(img)).specs()) == DIGEST_VIEW_FRAMES_OFF
    assert tools_digest(Workbench(Image.fromarray(img), view_frames=False).specs()) == DIGEST_VIEW_FRAMES_OFF
    on = Workbench(Image.fromarray(img), view_frames=True)
    assert tools_digest(on.specs()) == DIGEST_VIEW_FRAMES_ON
    # 开时与关时的差别只有两处：带坐标的工具在参数表最后多一个 view；这些工具的坐标参数的说明涵盖视图。其余逐字节相同。
    off_specs = {s.name: s for s in Workbench(Image.fromarray(img)).specs()}
    assert [s.name for s in on.specs()] == list(off_specs)
    coord_descs = set()
    for s in on.specs():
        props = dict(s.parameters["properties"])
        off_props = off_specs[s.name].parameters["properties"]
        if s.name in _VIEW_TOOLS:
            assert list(props)[-1] == "view", s.name
            assert props.pop("view") == {"type": "integer", "minimum": 0, "maximum": wbm.MAX_VIEWS - 1, "default": 0,
                                         "description": wbm.VIEW_PARAM_DESCRIPTION}
            for p, _ in on._tools[s.name].coords:
                d_on, d_off = props[p]["description"], off_props[p]["description"]
                assert d_on != d_off and "view given by 'view'" in d_on and "full image" in d_on, (s.name, p)
                assert "full-image" not in d_on and "(full image)" not in d_on, (s.name, p)
                coord_descs.add(d_on)
                props[p] = {**props[p], "description": d_off}
        assert s.description == off_specs[s.name].description
        assert props == off_props, s.name
        assert s.parameters["required"] == off_specs[s.name].parameters["required"]
    assert coord_descs == _VIEW_COORD_DESCRIPTIONS
    assert {s.name for s in on.specs() if "view" in s.parameters["properties"]} == _VIEW_TOOLS
    # 共享的 schema 字典没有被开时的改写改动：之后新建的关时工作台说明不变
    assert tools_digest(Workbench(Image.fromarray(img)).specs()) == DIGEST_VIEW_FRAMES_OFF


def test_view_relative_rewrite_refuses_silent_no_op():
    """开时改写坐标参数的说明：说明里没有"原图坐标"的提法就报错（以后改了说明，不会静默漏改）。"""
    props = {"x": {"type": "number", "minimum": 0, "maximum": 1000, "description": "Some coordinate."}}
    with pytest.raises(AssertionError):
        wbm._view_relative(props, (("x", "x"),))
    shared = {"type": "number", "description": "Left edge, 0-1000 (full image)."}
    out = wbm._view_relative({"x1": shared}, (("x1", "x"),))
    assert out["x1"]["description"] == "Left edge, 0-1000 (full image, or the view given by 'view')."
    assert shared["description"] == "Left edge, 0-1000 (full image)."  # 共享的 schema 字典不被改动


def test_view_tools_are_exactly_the_tools_with_coordinates():
    """每个取值为 0–1000 坐标的参数都声明在 coords 里（view_frames 开时随 view 换算），没有漏的；返回图的工具都标了 emits_image。"""
    wb = _wb(view_frames=True)
    for name, t in wb._tools.items():
        declared = {p for p, _ in t.coords}
        for p, schema in t.properties.items():
            inner = schema
            while inner["type"] == "array":
                inner = inner["items"]
            is_coord = inner["type"] == "number" and inner.get("minimum") == 0 and inner.get("maximum") == 1000
            assert is_coord == (p in declared), (name, p)
        assert bool(t.coords) == (name in _VIEW_TOOLS), name
        assert t.emits_image == (name in _IMAGE_VIEW_TOOLS | _MASK_OVERLAY_TOOLS), name


def test_view_frames_constructor():
    img, _ = _scene()
    wb = Workbench(Image.fromarray(img), disabled_groups=frozenset(), view_frames=True)  # DEV-0053 的透传方式
    assert wb.view_frames is True and wb.views == (wbm.FULL_VIEW,)
    assert Workbench(Image.fromarray(img)).view_frames is False
    with pytest.raises(ValueError):
        Workbench(Image.fromarray(img), view_frames="yes")
    # 关时 view 是多余的参数，与改动前的报错相同
    r = Workbench(Image.fromarray(img)).call("draw_box", {"name": "b", "x1": "0", "y1": "0", "x2": "5", "y2": "5", "view": "0"})
    assert r.error and "unexpected parameter(s) ['view']" in r.error


def _run_seq(seq, *, view_frames: bool, perturb_rng: bool = False, img=None):
    wb = _wb(img, view_frames=view_frames)
    outs = []
    for name, args in seq:
        if perturb_rng:
            cv2.setRNGSeed(12345)
            cv2.randu(np.zeros((10, 10), np.float32), 0, 1)
        r = wb.call(name, args)
        outs.append((r.text, r.error, [np.asarray(im).tobytes() for im in r.images]))
    log = wb.log()
    for e in log:
        e.pop("wall_time_s")
        if e["sandbox"]:
            e["sandbox"].pop("wall_time_s")
    s = wb.submission
    sub = (s.kind, s.mask.tobytes() if s.mask is not None else None, s.box, s.count) if s else None
    return wb, outs, {k: v.tobytes() for k, v in wb.masks.items()}, sub, log


def test_view_frames_on_without_view_only_adds_view_ids():
    """开着、但不用 view 参数：图、mask、提交与关时逐位相同；文字只在最前面多了 "This is view k."；日志只多 view / views 两个键。"""
    _, a_out, a_masks, a_sub, a_log = _run_seq(_SEQUENCE, view_frames=False)
    wb, b_out, b_masks, b_sub, b_log = _run_seq(_SEQUENCE, view_frames=True)
    assert a_masks == b_masks and a_sub == b_sub
    k = 0
    for (name, _), (ta, ea, ia), (tb, eb, ib) in zip(_SEQUENCE, a_out, b_out):
        assert ea is None and eb is None and ia == ib, name
        assert not ta.startswith("This is view")
        if ib:
            k += 1
            assert tb == f"This is view {k}. " + ta, name
        else:
            assert tb == ta, name
    assert len(wb.views) == k + 1 and k == 6
    for ea, eb in zip(a_log, b_log):
        assert set(eb) - set(ea) == {"view", "views"} and eb["view"] is None
        assert {x: y for x, y in eb.items() if x not in ("view", "views")} == ea


def test_view_numbering_and_regions():
    wb = _wb(view_frames=True)
    r = _ok(wb.call("tile_view", {"rows": "2", "cols": "2"}))
    assert r.text.startswith("This is view 1. The 128x96 image")
    r = _ok(wb.call("zoom", {"x1": "250", "y1": "250", "x2": "750", "y2": "750"}))
    assert r.text.startswith("This is view 2. Zoomed view") and "x = 250 + u*500" in r.text  # 原来的换算式保留
    _ok(wb.call("draw_box", {"name": "m", "x1": "100", "y1": "100", "x2": "600", "y2": "600"}))  # 不返回图：不登记
    assert len(wb.views) == 3
    assert wb.call("zoom", {"x1": "600", "y1": "0", "x2": "500", "y2": "500"}).error  # 出错：不登记、不占编号
    assert wb.call("show", {"name": "ghost"}).error
    r = _ok(wb.call("edges", {}))
    assert r.text.startswith("This is view 3. Canny")
    r = _ok(wb.call("superpixels", {"n_segments": "20"}))
    assert r.text.startswith("This is view 4. SLIC")
    r = _ok(wb.call("show", {"name": "m"}))
    assert r.text.startswith("This is view 5. Mask 'm' overlaid")
    r = _ok(wb.call("show", {"name": "m", "region": "[251, 0, 500, 1000]"}))  # 向外取整：实际覆盖 x 250..500
    assert r.text.startswith("This is view 6. Mask 'm' overlaid") and "x = 250..500, y = 0..1000" in r.text
    full = wbm.FULL_VIEW
    assert wb.views == (full, full, (250.0, 250.0, 750.0, 750.0), full, full, full, (250.0, 0.0, 500.0, 1000.0))
    # 日志：每次调用登记的视图与用了哪个 view
    log = wb.log()
    assert [e["views"] for e in log] == [
        [{"view": 1, "region": list(full)}], [{"view": 2, "region": [250.0, 250.0, 750.0, 750.0]}], [], [], [],
        [{"view": 3, "region": list(full)}], [{"view": 4, "region": list(full)}], [{"view": 5, "region": list(full)}],
        [{"view": 6, "region": [250.0, 0.0, 500.0, 1000.0]}]]
    assert all(e["view"] is None for e in log)
    # 视图上的嵌套 zoom：登记的是换回原图之后实际覆盖的区域
    r = _ok(wb.call("zoom", {"x1": "0", "y1": "0", "x2": "500", "y2": "500", "view": "2"}))
    assert r.text.startswith("This is view 7. " + _IN_VIEW_NOTE.format(k=2) + "Zoomed view")
    assert "x = 250..500, y = 250..500" in r.text and wb.views[7] == (250.0, 250.0, 500.0, 500.0)
    assert wb.log()[-1]["view"] == 2 and wb.log()[-1]["views"] == [{"view": 7, "region": [250.0, 250.0, 500.0, 500.0]}]


# 每个带坐标的工具：按视图 1 给的参数，与换算成原图 0–1000 之后的同一组参数。视图 1 = zoom [250, 250, 750, 750]（128x96 上恰好覆盖
# 列 32..95、行 24..71，区域无舍入），比例 0.5：x_full = 250 + x/2，换算没有浮点误差，两边的结果必须逐位相同。
_VIEW_CALLS = {
    "zoom": ({"x1": "200", "y1": "100", "x2": "600", "y2": "900"}, {"x1": "350", "y1": "300", "x2": "550", "y2": "700"}),
    "read_pixels": ({"points": "[[100, 200], [900, 800], [437.5, 500]]", "radius": "1"},
                    {"points": "[[300, 350], [700, 650], [468.75, 500]]", "radius": "1"}),
    "draw_box": ({"name": "n", "x1": "123", "y1": "201", "x2": "655", "y2": "777"},
                 {"name": "n", "x1": "311.5", "y1": "350.5", "x2": "577.5", "y2": "638.5"}),
    "draw_points": ({"name": "n", "points": "[[100, 200], [900, 800], [440, 520]]", "labels": "[1, 1, 0]", "radius": "4"},
                    {"name": "n", "points": "[[300, 350], [700, 650], [470, 510]]", "labels": "[1, 1, 0]", "radius": "4"}),
    "draw_polygon": ({"name": "n", "points": "[[100, 100], [900, 200], [500, 900]]"},
                     {"name": "n", "points": "[[300, 300], [700, 350], [500, 700]]"}),
    "draw_spline": ({"name": "n", "points": "[[440, 40], [780, 500], [440, 960], [100, 500]]"},
                    {"name": "n", "points": "[[470, 270], [640, 500], [470, 730], [300, 500]]"}),
    "grabcut": ({"out": "g", "box": "[100, 100, 900, 900]"}, {"out": "g", "box": "[300, 300, 700, 700]"}),
    "select_superpixels": ({"name": "s", "points": "[[100, 200], [437.5, 500]]"}, {"name": "s", "points": "[[300, 350], [468.75, 500]]"}),
    "show": ({"name": "m", "region": "[0, 0, 1000, 1000]"}, {"name": "m", "region": "[250, 250, 750, 750]"}),
    "submit_box": ({"x1": "200", "y1": "100", "x2": "600", "y2": "900"}, {"x1": "350", "y1": "300", "x2": "550", "y2": "700"}),
}


@pytest.mark.parametrize("tool", sorted(_VIEW_CALLS))
def test_view_coordinates_match_full_image_coordinates(tool):
    """zoom 一块区域，再用 view = 1 调用：与直接用换算后的原图坐标调用（开、关两种工作台）逐位同结果；文字只多一句说明。"""
    assert set(_VIEW_CALLS) == _VIEW_TOOLS
    img, disk = _scene()
    in_view, in_full = _VIEW_CALLS[tool]

    def run(view_frames, args):
        wb = Workbench(Image.fromarray(img), view_frames=view_frames)
        wb._set_mask("m", disk)
        _ok(wb.call("zoom", {"x1": "250", "y1": "250", "x2": "750", "y2": "750"}))
        if tool == "select_superpixels":
            _ok(wb.call("superpixels", {"n_segments": "30"}))
        r = _ok(wb.call(tool, args))
        s = wb.submission
        return (r, {k: v.tobytes() for k, v in wb.masks.items()}, [np.asarray(im).tobytes() for im in r.images],
                (s.kind, s.box) if s else None, wb)

    a, a_masks, a_imgs, a_sub, wb_a = run(True, {**in_view, "view": "1"})
    b, b_masks, b_imgs, b_sub, wb_b = run(True, in_full)
    c, c_masks, c_imgs, c_sub, _ = run(False, in_full)
    assert wb_a.views[1] == (250.0, 250.0, 750.0, 750.0)
    assert a_masks == b_masks == c_masks and a_imgs == b_imgs == c_imgs and a_sub == b_sub == c_sub
    note = _IN_VIEW_NOTE.format(k=1)
    assert note in a.text and a.text.replace(note, "", 1) == b.text  # 回显的坐标是原图 0–1000，与直接给原图坐标时相同
    assert _strip_view_ids(b.text) == c.text
    assert wb_a.log()[-1]["view"] == 1 and wb_b.log()[-1]["view"] is None
    assert wb_a.views == wb_b.views
    if tool == "read_pixels":
        assert "RGB (220.0, 50.0, 40.0)" in a.text.splitlines()[-1]  # 视图 1 里的 (437.5, 500) 是红圆的圆心


def test_view_frame_matches_what_the_zoomed_image_shows():
    """语义核对（区域不是二进小数，换算有浮点舍入）：视图 [0, 0, 1000, 1000] 的框正好是这次 zoom 显示的那些原图像素；
    按 view 给的比例位置 (u, v) 读到的原图像素，正是视图里同一比例位置所对应的那个原图像素。"""
    img, _ = _scene()
    wb = _wb(img, view_frames=True)
    r = _ok(wb.call("zoom", {"x1": "100", "y1": "100", "x2": "333", "y2": "666"}))
    assert "columns 12..42, rows 9..63" in r.text
    _ok(wb.call("draw_box", {"name": "all", "x1": "0", "y1": "0", "x2": "1000", "y2": "1000", "view": "1"}))
    exp = np.zeros((H, W), bool)
    exp[9:64, 12:43] = True
    assert np.array_equal(wb.masks["all"], exp)
    for u, v in [(0.5, 0.5), (0.1, 0.9), (0.95, 0.05), (0.0, 0.0), (0.999, 0.999)]:
        rp = _ok(wb.call("read_pixels", {"points": json.dumps([[u * 1000, v * 1000]]), "view": "1"}))
        m = re.search(r"pixel \(col (\d+), row (\d+)\)", rp.text)
        c, rr = int(m.group(1)), int(m.group(2))
        # 视图覆盖 31 列 × 55 行：比例位置 u 落在窗口的第 floor(u·31) 列，行同理
        assert c == 12 + int(u * 31) and rr == 9 + int(v * 55), (u, v, c, rr)


def test_full_image_views_are_identity():
    """覆盖整图的视图（原图 = 视图 0、tile_view、edges、superpixels、不带 region 的 show）：换算逐位是恒等。"""
    rng = np.random.default_rng(7)
    xs = np.concatenate([[0.0, 1000.0, 0.1, 999.9, 123.456789, 1 / 3, 2 / 3 * 1000], rng.uniform(0, 1000, 200)])
    for x, y in zip(xs, xs[::-1]):
        assert wbm.view_point_to_full(wbm.FULL_VIEW, x, y) == (float(x), float(y))
    img, disk = _scene()
    wb = _wb(img, view_frames=True)
    wb._set_mask("m", disk)
    for name, args in [("tile_view", {"rows": "2", "cols": "3"}), ("edges", {}), ("superpixels", {"n_segments": "20"}),
                       ("show", {"name": "m"})]:
        _ok(wb.call(name, args))
    assert wb.views[1:] == (wbm.FULL_VIEW,) * 4
    calls = [("draw_box", {"name": "b", "x1": "123.4", "y1": "201.7", "x2": "654.321", "y2": "777.7"}),
             ("draw_points", {"name": "p", "points": "[[333.3, 666.7], [10.01, 990.99]]", "radius": "3.5"}),
             ("draw_polygon", {"name": "q", "points": "[[100.1, 120.2], [800.8, 90.9], [650.65, 700.7], [300.3, 930.93]]"}),
             ("read_pixels", {"points": "[[468.7, 499.9], [0, 0], [1000, 1000]]"}),
             ("zoom", {"x1": "251.1", "y1": "3.3", "x2": "500.5", "y2": "999.9"})]
    for k in range(5):
        for name, args in calls:
            if name.startswith("draw_"):
                args = {**args, "name": f"{args['name']}{k}"}  # 每次都是新 mask（文字里是 "Created"）
            ref = _wb(img, view_frames=True)
            want = _ok(ref.call(name, args))
            got = _ok(wb.call(name, {**args, "view": str(k)}))
            assert [np.asarray(i).tobytes() for i in got.images] == [np.asarray(i).tobytes() for i in want.images], (k, name)
            if name.startswith("draw_"):
                assert np.array_equal(wb.masks[args["name"]], ref.masks[args["name"]]), (k, name)
            note = _IN_VIEW_NOTE.format(k=k) if k else ""
            assert _strip_view_ids(got.text) == note + _strip_view_ids(want.text), (k, name)


def test_view_point_to_full_stays_inside_the_view():
    rng = np.random.default_rng(11)
    for _ in range(500):
        a, b = np.sort(rng.uniform(0, 1000, 2))
        c, d = np.sort(rng.uniform(0, 1000, 2))
        region = (float(a), float(c), float(b), float(d))
        for x, y in [(0.0, 0.0), (1000.0, 1000.0), tuple(rng.uniform(0, 1000, 2))]:
            fx, fy = wbm.view_point_to_full(region, x, y)
            assert a <= fx <= b and c <= fy <= d
            assert abs(fx - (a + x / 1000 * (b - a))) < 1e-9 and abs(fy - (c + y / 1000 * (d - c))) < 1e-9  # 任务书的式子


def test_unknown_view_is_an_error_not_an_exception():
    wb = _wb(view_frames=True)
    r = wb.call("draw_box", {"name": "b", "x1": "0", "y1": "0", "x2": "5", "y2": "5", "view": "1"})
    assert r.error == "unknown view 1 (existing views: 0; 0 = the original image)" and r.text == "Error: " + r.error
    assert "b" not in wb.masks
    _ok(wb.call("zoom", {"x1": "0", "y1": "0", "x2": "500", "y2": "500"}))
    for v, needle in [("2", "unknown view 2 (existing views: 0..1;"), (str(wbm.MAX_VIEWS), "out of range [0, 63]"),
                      ("-1", "out of range"), ("1.5", "must be an integer"), ("one", "must be an integer")]:
        r = wb.call("read_pixels", {"points": "[[1, 2]]", "view": v})
        assert r.error is not None and needle in r.error, (v, r.error)
        assert r.text.startswith("Error: ") and r.images == []
        assert wb.log()[-1]["ok"] is False and wb.log()[-1]["views"] == []
    # 视图坐标仍按 0–1000 校验（在视图坐标里校验）
    r = wb.call("draw_box", {"name": "b", "x1": "0", "y1": "0", "x2": "1001", "y2": "5", "view": "1"})
    assert r.error and "out of range" in r.error
    # 换算之后的报错注明坐标已换算（报错里的坐标是原图 0–1000）
    r = wb.call("draw_box", {"name": "b", "x1": "600", "y1": "0", "x2": "500", "y2": "5", "view": "1"})
    assert r.error.startswith("box must satisfy x1 < x2 and y1 < y2; got (300, 0, 250, 2.5)"), r.error
    assert r.error.endswith("(coordinates were converted from view 1 to full-image 0-1000 coordinates)")
    assert len(wb.views) == 2


def test_max_views():
    img, disk = _scene()
    wb = _wb(img, view_frames=True)
    wb._set_mask("m", disk)
    for k in range(1, wbm.MAX_VIEWS):
        r = _ok(wb.call("tile_view", {"rows": "1", "cols": "1"}))
        assert r.text.startswith(f"This is view {k}. ")
    assert len(wb.views) == wbm.MAX_VIEWS
    for name, args in [("zoom", {"x1": "0", "y1": "0", "x2": "500", "y2": "500"}), ("tile_view", {"rows": "1", "cols": "1"}),
                       ("edges", {"out": "e"}), ("superpixels", {}), ("show", {"name": "m"})]:
        r = wb.call(name, args)
        assert r.error and r.error.startswith(f"too many views (limit {wbm.MAX_VIEWS}"), name
    assert "e" not in wb.masks and wb._superpixels is None  # 在执行之前拦住：没有写了一半的状态
    assert len(wb.views) == wbm.MAX_VIEWS
    # 不返回图的工具照常可用，也能按最后一个视图给坐标
    _ok(wb.call("draw_box", {"name": "b", "x1": "0", "y1": "0", "x2": "500", "y2": "500", "view": str(wbm.MAX_VIEWS - 1)}))


_VIEW_SEQUENCE = [
    ("zoom", {"x1": "200", "y1": "100", "x2": "800", "y2": "900"}),
    ("zoom", {"x1": "300", "y1": "250", "x2": "700", "y2": "750", "view": "1"}),
    ("read_pixels", {"points": "[[470, 500], [10, 10]]", "radius": "2", "view": "2"}),
    ("draw_box", {"name": "box", "x1": "250", "y1": "200", "x2": "700", "y2": "800", "view": "1"}),
    ("grabcut", {"out": "g", "box": "[100, 50, 900, 950]", "view": "2"}),
    ("draw_points", {"name": "seed", "points": "[[500, 500]]", "radius": "2", "view": "2"}),
    ("watershed", {"out": "w", "fg": "seed"}),
    ("superpixels", {"n_segments": "30"}),
    ("select_superpixels", {"name": "s", "points": "[[470, 500]]", "view": "3"}),
    ("draw_polygon", {"name": "p", "points": "[[100, 100], [900, 150], [500, 950]]", "view": "2"}),
    ("draw_spline", {"name": "sp", "points": "[[470, 270], [640, 500], [470, 730], [300, 500]]", "view": "1"}),
    ("zoom", {"x1": "0", "y1": "0", "x2": "500", "y2": "500", "view": "9"}),
    ("show", {"name": "g", "region": "[0, 0, 1000, 1000]", "view": "2"}),
    ("submit_box", {"x1": "100", "y1": "50", "x2": "900", "y2": "950", "view": "2"}),
    ("submit_mask", {"name": "g"}),
]


def test_view_frames_are_deterministic():
    a = _run_seq(_VIEW_SEQUENCE, view_frames=True)
    b = _run_seq(_VIEW_SEQUENCE, view_frames=True, perturb_rng=True)
    assert a[0].views == b[0].views and len(a[0].views) == 5
    assert a[1:] == b[1:]  # 文字、报错、图、mask、提交、日志逐位相同
    errs = [o[1] for o in a[1]]
    assert "unknown view 9" in errs[11] and all(e is None for i, e in enumerate(errs) if i != 11)
    assert [e["view"] for e in a[4]] == [None, 1, 2, 1, 2, 2, None, None, 3, 2, 1, None, 2, 2, None]


def test_docs_describe_view_frames():
    text = DOCS.read_text(encoding="utf-8")
    assert "视图坐标（`view_frames`，默认关" in text
    assert "`MAX_VIEWS` = 64" in text  # 数值由 test_docs_list_every_tool_and_group 与代码核对
    assert wbm.VIEW_PARAM_DESCRIPTION in text  # 给模型的原文逐字照录
    for d in _VIEW_COORD_DESCRIPTIONS:
        assert d in text, d
    assert DIGEST_VIEW_FRAMES_OFF in text and DIGEST_VIEW_FRAMES_ON in text
    assert "This is view k." in text and _IN_VIEW_NOTE.format(k="k").strip() in text
    assert "(coordinates were converted from view k to full-image 0-1000 coordinates)" in text
    for name in _VIEW_TOOLS:
        assert f"`{name}`" in text
