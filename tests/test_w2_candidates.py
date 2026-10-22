"""DEV-0058：W2 的候选生成器（src/w2/candidates.py；接口见 docs/w2_plan.md §1）。

钉住：1 号候选 grabcut_default 与 W1-F（src/w1/baselines.w1f_episode + workbench_grabcut）逐位相同；清单的校验与缺省值；
DEFAULT_SPEC 是 8 项且参数写全；每种生成器都能跑通，各种边界情况（框贴边、框极小、框退化、框盖满全图、没有点、点在框外、
点在图外）下 K 不变、失败时用框代替且 ok = False；box 候选与"自己的框"的光栅化相同；后处理；同样输入逐位相同（中途打乱
OpenCV 的全局随机数也一样）；docs/w2_tools.md 列出全部类型、参数与默认清单。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.executors.raster import box_to_mask, polygons_to_mask, rle_to_mask
from src.w2 import candidates as C
from src.w2.candidates import DEFAULT_SPEC, GENERATOR_TYPES, Candidate, CandidateSpec, generate_candidates

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "w2_tools.md"
NYU = REPO / "tests" / "fixtures" / "e2" / "root" / "nyu_depth_train" / "sync" / "living_room_0004" / "rgb_00001.jpg"
H, W = 96, 128
CY, CX, R = 48, 60, 22

# 每种类型至少一项、外加后处理与 soft 初始化（测试用；第一项按约定是 grabcut_default）
ALL_SPEC = CandidateSpec(generators=(
    {"type": "grabcut_default"},
    {"type": "grabcut", "box_pct": 10},
    {"type": "grabcut", "init": "mask", "box_pct": -10},
    {"type": "grabcut", "iterations": 2},
    {"type": "grabcut_points"},
    {"type": "grabcut_points", "init": "mask", "seed_radius_pct": 8},
    {"type": "grabcut_default", "post": ["largest_cc", "fill_holes"]},
    {"type": "slic", "n_segments": 20},
    {"type": "slic", "n_segments": 80, "threshold_pct": 70},
    {"type": "watershed"},
    {"type": "watershed", "use_points": False, "bg_margin_pct": 10},
    {"type": "box"},
))


def _scene(seed=0):
    """噪声灰背景上一个纯色红圆，外加一个纯色蓝方块（同 tests/test_w1_workbench.py）。"""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(110, 12, (H, W, 3)), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    disk = (xx + 0.5 - CX) ** 2 + (yy + 0.5 - CY) ** 2 <= R**2
    img[disk] = (220, 50, 40)
    img[70:90, 100:120] = (40, 80, 210)
    return img, disk


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


DISK_BOX = [CX - R - 4.0, CY - R - 4.0, CX + R + 4.0, CY + R + 4.0]


def _check_invariants(cands, spec, img, box):
    h, w = img.shape[:2]
    bm = box_to_mask(box, h, w)
    assert [c.name for c in cands] == list(spec.names) and len(cands) == spec.k
    for i, c in enumerate(cands, start=1):
        assert isinstance(c, Candidate)
        assert c.mask.dtype == bool and c.mask.shape == (h, w)
        assert c.info["index"] == i and c.info["ok"] == c.ok and c.info["area"] == int(c.mask.sum())
        if c.ok:
            assert c.info["error"] is None
        else:
            assert c.info["error"] and np.array_equal(c.mask, bm), c.name
        json.dumps(c.info)  # info 可写 JSON


# ---------------------------------------------------------------------------
# 1 号候选 = W1-F
# ---------------------------------------------------------------------------


def _w1f(im, box):
    from src.data.records import Sample
    from src.w1 import baselines as B

    w, h = im.size
    s = Sample(id="syn/x/1", task="refexp", images=("/nonexistent.png",), image_keys=("k",), fields={"expression": "x"},
               gt={"box": [0.0, 0.0, 10.0, 10.0], "mask": {"polygons": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]], "height": h, "width": w}},
               source="syn", split="x")
    rec, _ = B.w1f_episode(s, {"id": s.id, "failure": None, "parsed": {"box": box}}, im, B.workbench_grabcut())
    return rec


ONLY_W1F = CandidateSpec(generators=({"type": "grabcut_default"},))


@pytest.mark.parametrize("box", [[200.0, 150.0, 420.0, 380.0], [12.5, 7.25, 300.75, 470.0], [455.3, 20.9, 640.0, 233.1]])
def test_first_candidate_is_w1f_bitwise_on_a_real_size_image(box):
    im = Image.open(NYU).convert("RGB")
    rec = _w1f(im, box)
    assert rec["failure"] is None
    c1 = generate_candidates(np.asarray(im.convert("RGB")), box, spec=ONLY_W1F)[0]
    assert c1.name == "gc_default" and c1.ok and c1.info["type"] == "grabcut_default"
    assert np.array_equal(c1.mask, rle_to_mask(rec["answer"]["mask"]["rle"]))


def test_first_candidate_is_w1f_inside_the_default_spec():
    """整份 DEFAULT_SPEC 里的 1 号（后面还有共用基础计算的 gc_default+lcc+fill）与 W1-F 逐位相同。"""
    img, _ = _scene()
    im = Image.fromarray(img)
    for box in (DISK_BOX, [3.5, 2.25, 77.75, 60.0]):
        rec = _w1f(im, box)
        cands = generate_candidates(img, box, spec=DEFAULT_SPEC)
        assert cands[0].name == "gc_default" and cands[0].ok
        assert np.array_equal(cands[0].mask, rle_to_mask(rec["answer"]["mask"]["rle"]))
        # 共用基础计算的后处理变体：基础结果相同、后处理另算，1 号不被改动
        lcc = cands[DEFAULT_SPEC.names.index("gc_default+lcc+fill")]
        assert lcc.info["reused"] and np.array_equal(lcc.mask, C.apply_post(cands[0].mask, ["largest_cc", "fill_holes"]))


def test_first_candidate_when_w1f_fails_is_the_box():
    """W1-F 的 GrabCut 报错（框盖满全图：没有背景像素）时，W1-F 记 tool_error、按空 mask 计分；1 号候选按 §1 的失败规则用框本身、
    ok = False。这是两者唯一的不同（DEV-0059 的 w1f 设置若要与 W1-F 逐位相同，失败时应取空 mask）。"""
    img, _ = _scene()
    im = Image.fromarray(img)
    box = [0.0, 0.0, float(W), float(H)]
    rec = _w1f(im, box)
    assert rec["failure"] == "tool_error" and rec["answer"] is None
    c1 = generate_candidates(img, box, spec=ONLY_W1F)[0]
    assert not c1.ok and "没有背景像素" in c1.info["error"] and c1.mask.all()


# ---------------------------------------------------------------------------
# 清单
# ---------------------------------------------------------------------------


def test_default_spec_has_8_explicit_generators():
    assert DEFAULT_SPEC.k == 8 and DEFAULT_SPEC.generators[0] == {"type": "grabcut_default"}
    assert DEFAULT_SPEC.names == ("gc_default", "gc_box_plus10", "gc_soft_box_minus10", "gc_default+lcc+fill",
                                  "slic_s30_t50", "slic_s100_t50", "ws_pts", "box")
    # 每项的参数都写全（DEV-0059 原样抄进 configs/w2.json，不依赖代码里的缺省值）
    for g, r in zip(DEFAULT_SPEC.generators, DEFAULT_SPEC.resolved):
        assert set(GENERATOR_TYPES[g["type"]]["params"]) <= set(g)
        assert {k: v for k, v in r.items() if k not in ("name", "post")} == {k: v for k, v in g.items() if k != "post"}
    assert CandidateSpec.from_dict(json.loads(json.dumps(DEFAULT_SPEC.as_dict()))) == DEFAULT_SPEC
    assert not DEFAULT_SPEC.uses_points or any(GENERATOR_TYPES[g["type"]]["uses_points"] for g in DEFAULT_SPEC.generators)


def test_spec_validation():
    ok = ({"type": "grabcut_default"},)
    bad = [
        (),  # 空
        ({"type": "grabcut"},),  # 第一项不是 grabcut_default
        ({"type": "grabcut_default", "post": ["largest_cc"]},),  # 第一项必须恰为 {"type": "grabcut_default"}
        ok + ({"type": "nope"},),
        ok + ({"type": "grabcut", "iters": 5},),  # 未知参数
        ok + ({"type": "grabcut", "iterations": 0},),
        ok + ({"type": "grabcut", "iterations": 2.5},),
        ok + ({"type": "grabcut", "iterations": True},),
        ok + ({"type": "grabcut", "box_pct": -50},),
        ok + ({"type": "grabcut", "init": "lasso"},),
        ok + ({"type": "slic", "threshold_pct": 0},),
        ok + ({"type": "watershed", "use_points": "yes"},),
        ok + ({"type": "box", "post": ["open"]},),  # open 要半径
        ok + ({"type": "box", "post": ["open:0"]},),
        ok + ({"type": "box", "post": ["blur:3"]},),
        ok + ({"type": "box", "post": "largest_cc"},),
        ok + ({"type": "box"}, {"type": "box"}),  # 候选名重复
        ok + ({"type": "box", "name": ""},),
        ok + ({"type": "grabcut", "box_pct": float("nan")},),
    ]
    for gens in bad:
        with pytest.raises(ValueError):
            CandidateSpec(generators=gens)
    with pytest.raises(ValueError):
        CandidateSpec(generators={"type": "grabcut_default"})  # 不是列表
    # 名字：自动生成；显式 name 覆盖；同一类型不同参数不同名
    s = CandidateSpec(generators=ok + ({"type": "box", "name": "the_box"}, {"type": "grabcut", "iterations": 10},
                                       {"type": "grabcut"}, {"type": "grabcut", "box_pct": -7.5, "init": "mask"},
                                       {"type": "slic", "compactness": 20, "context_pct": 50}, {"type": "watershed", "use_points": False},
                                       {"type": "grabcut_points", "seed_radius_pct": 2}, {"type": "box", "post": ["open:3", "largest_cc"]}))
    assert s.names == ("gc_default", "the_box", "gc_iter10", "gc_rect", "gc_soft_box_minus7.5", "slic_s50_t50_c20_ctx50",
                       "ws_center", "gc_pts_r2", "box+open3+lcc")
    assert s.k == 9 and s.uses_points
    # 缺省值写不写，digest 相同；参数不同，digest 不同
    a = CandidateSpec(generators=ok + ({"type": "grabcut"},))
    b = CandidateSpec(generators=ok + ({"type": "grabcut", "iterations": 5, "box_pct": 0.0, "init": "rect"},))
    c = CandidateSpec(generators=ok + ({"type": "grabcut", "iterations": 6},))
    assert a.digest() == b.digest() != c.digest() and hash(a) == hash(b)
    # 原样保存（深拷贝，外部改动不影响）
    gens = [{"type": "grabcut_default"}, {"type": "box", "post": ["largest_cc"]}]
    s2 = CandidateSpec(generators=tuple(gens))
    gens[1]["post"].append("fill_holes")
    assert s2.generators[1] == {"type": "box", "post": ["largest_cc"]}
    with pytest.raises(TypeError):
        generate_candidates(np.zeros((4, 4, 3), np.uint8), [0, 0, 2, 2], spec=ok)


# ---------------------------------------------------------------------------
# 每种生成器与边界情况
# ---------------------------------------------------------------------------


def test_every_generator_runs_on_a_clear_scene():
    img, disk = _scene()
    fg = [[CX + 0.5, CY + 0.5], [CX - 8.0, CY + 6.0]]
    bg = [[5.0, 5.0], [110.0, 80.0]]  # 第二个点在蓝方块上
    cands = generate_candidates(img, DISK_BOX, fg_points_abs=fg, bg_points_abs=bg, spec=ALL_SPEC)
    _check_invariants(cands, ALL_SPEC, img, DISK_BOX)
    by = {c.name: c for c in cands}
    assert all(c.ok for c in cands), [(c.name, c.info["error"]) for c in cands]
    for name in ("gc_default", "gc_box_plus10", "gc_soft_box_minus10", "gc_iter2", "gc_pts", "gc_pts_soft_r8", "gc_default+lcc+fill",
                 "ws_pts", "ws_center_margin10"):
        assert _iou(by[name].mask, disk) > 0.85, (name, _iou(by[name].mask, disk))
    bm = box_to_mask(DISK_BOX, H, W)
    for name in ("slic_s20_t50", "slic_s80_t70"):
        m = by[name].mask
        c0, r0, c1, r1 = by[name].info["crop"]
        outside = np.ones_like(m)
        outside[r0:r1, c0:c1] = False
        assert m.any() and not (m & outside).any() and by[name].info["n_superpixels"] >= 2
        assert max(_iou(m, bm), _iou(m, disk)) > 0.7, name  # 超像素贴着框或物体
    assert np.array_equal(by["box"].mask, bm)
    assert by["ws_pts"].info["fg_marker"] == "points" and by["ws_center_margin10"].info["fg_marker"] == "center"
    assert by["gc_pts"].info["n_fg_points"] == 2 and by["gc_pts"].info["n_bg_points"] == 2


EDGE_BOXES = {
    "edge_top_left": [0.0, 0.0, 40.0, 30.0],
    "edge_bottom_right": [W - 40.0, H - 30.0, float(W), float(H)],
    "partly_outside": [-10.0, -10.0, 30.0, 30.0],
    "tiny": [60.2, 48.1, 61.9, 49.8],
    "zero_width": [60.0, 20.0, 60.0, 70.0],
    "full_image": [0.0, 0.0, float(W), float(H)],
}
POINTS = {
    "none": (None, None),
    "inside": ([[CX, CY]], [[2.0, 2.0]]),
    "outside_box": ([[120.0, 90.0]], [[60.0, 48.0]]),
    "outside_image": ([[-5.0, 10.0], [W + 3.0, 10.0]], [[10.0, H + 1.0]]),
}


@pytest.mark.parametrize("pts", list(POINTS), ids=list(POINTS))
@pytest.mark.parametrize("box", list(EDGE_BOXES), ids=list(EDGE_BOXES))
def test_edge_cases_keep_k_and_fall_back_to_the_box(box, pts):
    img, _ = _scene()
    b = EDGE_BOXES[box]
    fg, bg = POINTS[pts]
    cands = generate_candidates(img, b, fg_points_abs=fg, bg_points_abs=bg, spec=ALL_SPEC)
    _check_invariants(cands, ALL_SPEC, img, b)
    by = {c.name: c for c in cands}
    assert by["box"].ok and np.array_equal(by["box"].mask, box_to_mask(b, H, W))
    if pts in ("none", "outside_image"):
        assert not by["gc_pts"].ok and "没有落在图内的点" in by["gc_pts"].info["error"]
        if by["ws_pts"].ok:  # 没有点：分水岭退回框中心的前景标记
            assert by["ws_pts"].info["fg_marker"] == "center"
    if box == "tiny":  # 约 2 px 的框：光栅化 4 px；内缩 30% 之后中心标记为空
        assert by["box"].mask.sum() == 4 and not by["ws_center_margin10"].ok and "前景标记为空" in by["ws_center_margin10"].info["error"]
    if pts == "outside_image":
        assert by["gc_pts"].info["dropped_points"] == {"fg": 2, "bg": 1}
    if box == "full_image":  # GrabCut 的框初始化没有背景像素；分水岭的背景标记为空
        for name in ("gc_default", "gc_box_plus10", "gc_iter2", "gc_default+lcc+fill"):
            assert not by[name].ok and "没有背景像素" in by[name].info["error"]
        assert not by["ws_center_margin10"].ok and "背景标记为空" in by["ws_center_margin10"].info["error"]
    if box == "zero_width":  # 框里没有像素：GrabCut、SLIC、分水岭（中心标记）都失败，框本身是空 mask
        for name in ("gc_default", "gc_box_plus10", "slic_s20_t50", "ws_center_margin10"):
            assert not by[name].ok
        assert not by["box"].mask.any()
    if box in ("edge_top_left", "edge_bottom_right", "partly_outside") and pts == "inside":
        assert all(by[n].ok for n in ("gc_default", "gc_soft_box_minus10", "slic_s20_t50", "slic_s80_t70", "ws_pts", "gc_pts"))


def test_points_outside_the_box_are_used():
    img, disk = _scene()
    box = [CX - 10.0, CY - 10.0, CX + 10.0, CY + 10.0]  # 框比圆小
    fg = [[CX - 18.0, CY], [CX + 18.0, CY]]  # 前景点在框外、圆内
    cands = generate_candidates(img, box, fg_points_abs=fg, spec=ALL_SPEC)
    by = {c.name: c for c in cands}
    assert by["gc_pts"].ok and by["gc_pts"].mask[CY, CX - 18] and by["gc_pts"].mask[CY, CX + 18]
    assert by["ws_pts"].ok and by["ws_pts"].info["fg_marker"] == "points" and by["ws_pts"].mask[CY, CX - 18]


def test_bad_inputs_raise():
    img, _ = _scene()
    with pytest.raises(ValueError):
        generate_candidates(img.astype(np.float32), DISK_BOX, spec=ALL_SPEC)
    with pytest.raises(ValueError):
        generate_candidates(img[..., 0], DISK_BOX, spec=ALL_SPEC)
    with pytest.raises(ValueError):
        generate_candidates(img, [50.0, 10.0, 40.0, 30.0], spec=ALL_SPEC)  # x1 < x0：调用方的错
    with pytest.raises(ValueError):
        generate_candidates(img, [0.0, 0.0, float("inf"), 30.0], spec=ALL_SPEC)
    with pytest.raises(ValueError):
        generate_candidates(img, DISK_BOX, fg_points_abs=[[1.0, float("nan")]], spec=ALL_SPEC)
    with pytest.raises(ValueError):
        generate_candidates(img, DISK_BOX, fg_points_abs=[1.0, 2.0, 3.0], spec=ALL_SPEC)
    # 空的点列表等于没有点
    a = generate_candidates(img, DISK_BOX, fg_points_abs=[], spec=ALL_SPEC)
    b = generate_candidates(img, DISK_BOX, spec=ALL_SPEC)
    assert all(np.array_equal(x.mask, y.mask) and x.ok == y.ok for x, y in zip(a, b))


def test_box_candidate_equals_the_own_box_raster():
    """box 候选与"自己的框"（scripts/summarize_e1_posthoc.own_box_record：四角多边形经 pycocotools 光栅化）逐位相同。"""
    from src.w1 import baselines as B

    box_polygon = B._summary().box_polygon
    img, _ = _scene()
    rng = np.random.default_rng(3)
    spec = CandidateSpec(generators=({"type": "grabcut_default"}, {"type": "box"}))
    for _ in range(40):
        x0, x1 = sorted(rng.uniform(-5, W + 5, 2))
        y0, y1 = sorted(rng.uniform(-5, H + 5, 2))
        box = [float(x0), float(y0), float(x1), float(y1)]
        got = generate_candidates(img, box, spec=spec)[1].mask
        assert np.array_equal(got, polygons_to_mask(box_polygon(box), H, W)), box


def test_post_ops():
    m = np.zeros((40, 50), bool)
    m[2:12, 2:12] = True  # 100 px
    m[20:35, 20:35] = True  # 225 px
    m[25:28, 25:28] = False  # 洞
    lcc = C.apply_post(m, ["largest_cc"])
    assert lcc.sum() == 225 - 9 and not lcc[5, 5]
    filled = C.apply_post(m, ["largest_cc", "fill_holes"])
    assert filled.sum() == 225 and filled[26, 26]
    assert not C.apply_post(np.zeros((5, 5), bool), ["largest_cc", "fill_holes"]).any()
    # 同面积取扫描顺序在前的
    t = np.zeros((10, 10), bool)
    t[0:2, 6:8] = True
    t[5:7, 0:2] = True
    assert C.apply_post(t, ["largest_cc"])[0, 6] and not C.apply_post(t, ["largest_cc"])[5, 0]
    # open / close / dilate / erode 与工作台的 morph 同一个实现
    from src.w1 import workbench as WB

    for op in ("open", "close", "dilate", "erode"):
        assert np.array_equal(C.apply_post(m, [f"{op}:2"]), WB.morph_mask(m, op, 2))
    out = C.apply_post(m, [])
    assert np.array_equal(out, m) and out is not m


def test_failed_base_with_post_still_falls_back_to_the_box():
    img, _ = _scene()
    box = [0.0, 0.0, float(W), float(H)]
    cands = generate_candidates(img, box, spec=DEFAULT_SPEC)
    by = {c.name: c for c in cands}
    assert not by["gc_default"].ok and not by["gc_default+lcc+fill"].ok and by["gc_default+lcc+fill"].mask.all()


def test_duplicates_are_kept_and_counted():
    img, _ = _scene()
    spec = CandidateSpec(generators=({"type": "grabcut_default"}, {"type": "grabcut", "name": "gc_same"}, {"type": "box"}))
    cands = generate_candidates(img, DISK_BOX, spec=spec)
    assert np.array_equal(cands[0].mask, cands[1].mask) and cands[1].info["reused"] is False  # 名字不同、参数不同 → 各算一次
    assert C.duplicate_rate(cands) == pytest.approx(1 / 3)
    assert cands[0].mask is not cands[1].mask
    assert C.duplicate_rate([]) == 0.0


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------


def test_deterministic_even_if_opencv_rng_is_disturbed():
    img, _ = _scene(seed=5)
    fg, bg = [[CX, CY]], [[4.0, 90.0]]
    first = generate_candidates(img, DISK_BOX, fg_points_abs=fg, bg_points_abs=bg, spec=ALL_SPEC)
    cv2.setRNGSeed(987654)
    cv2.randu(np.empty((64, 64), np.float32), 0.0, 1.0)
    np.random.seed(11)
    second = generate_candidates(img, DISK_BOX, fg_points_abs=fg, bg_points_abs=bg, spec=ALL_SPEC)
    cv2.setRNGSeed(13)
    third = generate_candidates(img, DISK_BOX, fg_points_abs=fg, bg_points_abs=bg, spec=ALL_SPEC)
    for a, b, c in zip(first, second, third):
        assert a.name == b.name == c.name and a.ok == b.ok == c.ok
        assert np.array_equal(a.mask, b.mask) and np.array_equal(a.mask, c.mask), a.name
        strip = lambda i: {k: v for k, v in i.items() if k != "time_s"}  # noqa: E731
        assert strip(a.info) == strip(b.info)


# ---------------------------------------------------------------------------
# 文档
# ---------------------------------------------------------------------------


def test_docs_list_every_generator_param_and_the_default_spec():
    text = DOCS.read_text(encoding="utf-8")
    for t, d in GENERATOR_TYPES.items():
        assert f"`{t}`" in text, t
        for p, schema in d["params"].items():
            assert f"`{p}`" in text, (t, p)
    for name in DEFAULT_SPEC.names:
        assert f"`{name}`" in text, name
    for op in C.POST_OPS:
        assert f"`{op}" in text, op
    for const in ("SEED_RADIUS_MIN_PX", "BOX_PCT_MIN", "BOX_PCT_MAX"):
        assert f"`{const}` = {getattr(C, const):g}" in text, const
    # DEFAULT_SPEC 的 JSON 原样出现在文档里（DEV-0059 照抄）
    assert json.dumps(DEFAULT_SPEC.as_dict()["generators"][1]) in text
