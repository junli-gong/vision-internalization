"""DEV-0068：引导工具库 src/explore/guidance.py。

钉住：不 import 学习型模块（干净的子进程里核对，也不用 OpenCV 的 dnn 等）；镜像的常量与函数与 src/w1/workbench.py、
src/w2/candidates.py 逐位相同；生成器库 v2 的前 4 个 = W2 冻结的 K = 4；偶奇填充与 box_to_mask 一致；B 样条拟合的 IoU 随 n 上升、
与 LLMSeg 的 scipy 构造是同一条曲线、框约束、洞；软锁定只消耗承诺、不翻转；effect gate；先删后补；局部 GrabCut 的补与删。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.executors.raster import box_to_mask
from src.explore import guidance as G

REPO = Path(__file__).resolve().parents[1]
NYU = REPO / "tests" / "fixtures" / "e2" / "root" / "nyu_depth_train" / "sync" / "living_room_0004" / "rgb_00001.jpg"
H, W = 96, 128
CY, CX, R = 48, 60, 22


def _scene(seed=0):
    """噪声灰背景上一个纯色红圆，外加一个纯色蓝方块（同 tests/test_w2_candidates.py）。"""
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


# ---------------------------------------------------------------------------
# 不 import 学习型模块
# ---------------------------------------------------------------------------


def test_no_learned_modules_imported_in_clean_process():
    code = ("import json, sys; import src.explore.guidance as G, src.explore.oracle_sim, src.explore.ceiling; "
            "print(json.dumps({'loaded': G.loaded_learned_modules(), "
            "'roots': sorted({m.split('.')[0] for m in sys.modules})}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True)
    doc = json.loads(r.stdout.strip().splitlines()[-1])
    assert doc["loaded"] == []
    assert not set(doc["roots"]) & set(G.BANNED_MODULE_ROOTS)
    assert "skimage" in doc["roots"] and "cv2" in doc["roots"]  # 审计本身是活的：确实 import 了这些


def test_explore_sources_do_not_use_learned_opencv_modules_or_import_banned():
    bad = re.compile(r"cv2\s*\.\s*(dnn|dnn_superres|ximgproc|saliency|face|text)\b")
    imp = re.compile(r"^\s*(?:from|import)\s+(" + "|".join(G.BANNED_MODULE_ROOTS) + r")\b", re.M)
    for p in sorted((REPO / "src" / "explore").glob("*.py")):
        src = p.read_text(encoding="utf-8")
        assert not bad.search(src), p
        top = [m for m in imp.finditer(src) if not src[: m.start()].rstrip().endswith(":")]
        if p.name != "oracle_sim.py":
            assert not top, (p, [m.group(0) for m in top])
    # oracle_sim 只在 load_tokenizer 里（函数体内）延迟 import transformers
    src = (REPO / "src" / "explore" / "oracle_sim.py").read_text(encoding="utf-8")
    assert re.findall(r"^from transformers", src, re.M) == []


def test_loaded_learned_modules_detects():
    import importlib.util

    assert "json" not in G.BANNED_MODULE_ROOTS
    if importlib.util.find_spec("torch") is None:
        pytest.skip("没有 torch")
    r = subprocess.run([sys.executable, "-c", "import torch, src.explore.guidance as G; print(G.loaded_learned_modules())"],
                       capture_output=True, text=True, cwd=REPO, check=True)
    assert "torch" in r.stdout


# ---------------------------------------------------------------------------
# 镜像的常量与函数
# ---------------------------------------------------------------------------


def test_mirrored_constants_equal_originals():
    from src.w1 import workbench as WB
    from src.w2 import candidates as C
    from src.w2 import runner as W2R
    from src.explore import ceiling as X

    for k in ("GRABCUT_ITERATIONS", "GRABCUT_MAX_ITERATIONS", "GRABCUT_RNG_SEED", "GRABCUT_MASK_MARGIN", "SLIC_COMPACTNESS",
              "SLIC_SIGMA", "MORPH_RADIUS_DEFAULT", "MORPH_RADIUS_MAX"):
        assert getattr(G, k) == getattr(WB, k), k
    assert G.SEED_RADIUS_MIN_PX == C.SEED_RADIUS_MIN_PX
    assert G.SEED_RADIUS_PCT_DEFAULT == C.GENERATOR_TYPES["grabcut_points"]["params"]["seed_radius_pct"]["default"]
    assert G.POST_OPS == C.POST_OPS
    assert X.THREAD_ENV_VARS == W2R.THREAD_ENV_VARS


def test_mirrored_functions_bitwise_equal():
    from src.w1 import workbench as WB
    from src.w2 import candidates as C

    img, disk = _scene()
    rng = np.random.default_rng(3)
    for box in ([30.0, 20.5, 95.2, 80.0], [0.0, 0.0, 128.0, 80.0], [5.0, 5.0, 40.0, 30.0]):
        assert np.array_equal(G.grabcut_mask(img, box_abs=box), WB.grabcut_mask(img, box_abs=box))
    init = box_to_mask([40, 30, 90, 70], H, W)
    fg = WB.point_disks((H, W), np.array([[60.0, 48.0]]), 4.0)
    bg = WB.point_disks((H, W), np.array([[10.0, 10.0], [110.0, 80.0]]), 4.0)
    assert np.array_equal(G.grabcut_mask(img, init_mask=init, fg=fg, bg=bg, iterations=3),
                          WB.grabcut_mask(img, init_mask=init, fg=fg, bg=bg, iterations=3))
    pts = rng.uniform(0, [W, H], size=(5, 2))
    assert np.array_equal(G.point_disks((H, W), pts, 6.5), WB.point_disks((H, W), pts, 6.5))
    assert np.array_equal(G.color_gradient(img), WB.color_gradient(img))
    m = rng.random((H, W)) > 0.6
    for op in ("dilate", "erode", "open", "close", "fill_holes"):
        assert np.array_equal(G.morph_mask(m, op, 3), WB.morph_mask(m, op, 3)), op
    a, b = G.components_by_area(m), WB.components_by_area(m)
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]
    post = ["open:2", "largest_cc", "fill_holes", "close:4", "dilate:1", "erode:2"]
    assert np.array_equal(G.apply_post(m, post), C.apply_post(m, post))
    for bad in (["open"], ["largest_cc:3"], ["blur:2"], "open:2"):
        with pytest.raises(ValueError):
            G.parse_post(bad)


def _w2_frozen_spec():
    from src.w2.candidates import CandidateSpec

    doc = json.loads((REPO / "configs" / "w2.json").read_text(encoding="utf-8"))
    return CandidateSpec(generators=tuple(doc["candidates"]["generators"]))


@pytest.mark.parametrize("which", ["scene", "nyu"])
def test_library_first_four_equal_w2_frozen_k4(which):
    from PIL import Image

    from src.w2.candidates import generate_candidates

    spec = _w2_frozen_spec()
    assert spec.names == G.W2_FROZEN_NAMES
    if which == "scene":
        img, _ = _scene()
        boxes = (DISK_BOX, [0.0, 0.0, 128.0, 96.0], [3.0, 3.0, 4.0, 4.0])  # 普通、盖满全图（失败）、极小
    else:
        img = np.asarray(Image.open(NYU).convert("RGB"))
        h, w = img.shape[:2]
        boxes = ([0.2 * w, 0.25 * h, 0.7 * w, 0.8 * h],)
    for box in boxes:
        ref = generate_candidates(img, box, spec=spec)
        mine = G.generate_v2(img, box, G.W2_FROZEN_NAMES)
        for a, b in zip(ref, mine):
            assert a.name == b.name and a.ok == b.ok, (box, a.name)
            assert np.array_equal(a.mask, b.mask), (box, a.name)


# ---------------------------------------------------------------------------
# 生成器库 v2
# ---------------------------------------------------------------------------


def test_library_order_matches_config_and_resolves():
    cfg = json.loads((REPO / "configs" / "explore_ceiling.json").read_text(encoding="utf-8"))
    assert tuple(cfg["generators"]["order"]) == G.LIBRARY_V2_ORDER
    assert len(G.LIBRARY_V2_ORDER) == 16 and len(set(G.LIBRARY_V2_ORDER)) == 16
    for nm in G.LIBRARY_V2_ORDER:
        r = G.resolve_v2(nm)
        assert r["type"] in G.GENERATOR_TYPES_V2 and r["name"] == nm
    with pytest.raises(ValueError):
        G.resolve_v2("x", {"type": "grabcut", "iterations": 0})
    with pytest.raises(ValueError):
        G.resolve_v2("x", {"type": "grabcut", "nope": 1})
    with pytest.raises(ValueError):
        G.resolve_v2("x", {"type": "superpixel", "select": "magic"})


def test_generate_v2_all_run_deterministic_and_fail_to_box():
    img, disk = _scene()
    a = G.generate_v2(img, DISK_BOX, G.LIBRARY_V2_ORDER)
    b = G.generate_v2(img, DISK_BOX, G.LIBRARY_V2_ORDER)
    assert [r.name for r in a] == list(G.LIBRARY_V2_ORDER)
    for x, y in zip(a, b):
        assert x.mask.shape == (H, W) and x.mask.dtype == bool
        assert np.array_equal(x.mask, y.mask), x.name
        assert x.ok, (x.name, x.info.get("error"))
    best = max(_iou(r.mask, disk) for r in a)
    assert best > 0.9
    # 框盖满全图：分水岭、GMM、超像素（flood/color）都没有背景，失败时用框本身
    full = [0.0, 0.0, float(W), float(H)]
    res = {r.name: r for r in G.generate_v2(img, full, G.LIBRARY_V2_ORDER)}
    for nm in ("ws_pts", "gmm_k3_lab+lcc+fill", "slic_s50_flood"):
        assert not res[nm].ok and np.array_equal(res[nm].mask, box_to_mask(full, H, W)), nm
        assert res[nm].info["error"]


# ---------------------------------------------------------------------------
# B 样条
# ---------------------------------------------------------------------------


def test_fill_polygon_matches_box_to_mask_for_rectangles():
    for bx in ([3.2, 4.7, 30.5, 40.0], [0, 0, 70, 50], [10.5, 10.5, 20.5, 20.5], [-5, -3, 12.3, 60], [7.0, 7.0, 7.0, 9.0]):
        poly = np.array([[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]], dtype=float)
        assert np.array_equal(G.fill_polygon_evenodd(poly, 50, 70), box_to_mask(bx, 50, 70)), bx
        assert np.array_equal(G.fill_polygon_evenodd(poly[::-1], 50, 70), box_to_mask(bx, 50, 70)), bx


def test_fill_polygon_even_odd_star_has_hole():
    ang = np.pi / 2 + np.arange(5) * 4 * np.pi / 5  # 五角星（自交）
    star = np.stack([50 + 40 * np.cos(ang), 50 - 40 * np.sin(ang)], axis=1)
    m = G.fill_polygon_evenodd(star, 100, 100)
    assert not m[50, 50]  # 中心五边形按偶奇规则是洞
    assert m[20, 50]  # 上面的角是实的


def _circle_rect():
    yy, xx = np.mgrid[0:200, 0:300]
    circle = (xx + 0.5 - 150) ** 2 + (yy + 0.5 - 100) ** 2 <= 70**2
    rect = np.zeros((200, 300), bool)
    rect[40:160, 60:250] = True
    return circle, rect


def test_bspline_fit_iou_rises_with_n():
    circle, rect = _circle_rect()
    ious = {}
    for nm, m in (("circle", circle), ("rect", rect)):
        ious[nm] = [_iou(G.rasterize_loops(G.fit_mask_bspline(m, n), m.shape), m) for n in (4, 8, 12, 16, 24)]
    r = ious["rect"]
    assert all(b > a for a, b in zip(r, r[1:])), r
    c = ious["circle"]
    assert c[-1] > c[0] and min(c[1:]) > 0.99, c


def test_eval_matches_llmseg_scipy_construction():
    from scipy.interpolate import BSpline

    ctrl = np.random.default_rng(0).normal(size=(9, 2)) * 50 + 100
    count, degree = len(ctrl), 3
    kv = np.arange(-degree, count + degree + 1, dtype=float)
    factor, fraction = divmod(count + degree + 1, count)
    wrapped = np.roll(np.concatenate((ctrl,) * factor + (ctrl[:fraction],)), -1, axis=0)
    theirs = BSpline(kv, wrapped, degree, extrapolate=False)(np.linspace(0, count, 20001))
    theirs = theirs[np.isfinite(theirs).all(axis=1)]
    mine = G.eval_closed_bspline(ctrl, 4000)
    d1 = np.sqrt(((mine[::10, None, :] - theirs[None, :, :]) ** 2).sum(-1)).min(1)
    d2 = np.sqrt(((theirs[::50, None, :] - mine[None, :, :]) ** 2).sum(-1)).min(1)
    assert d1.max() < 0.05 and d2.max() < 0.2  # 同一条闭合曲线（只差参数的起点）


def test_bspline_samples_rule():
    assert G.bspline_samples([0, 0, 5, 5]) == 64
    assert G.bspline_samples([0, 0, 2000, 2000]) == 2000
    assert G.bspline_samples([10, 10, 110.3, 60.2]) == round(2 * (100.3 + 50.2))


def test_box_constraint_and_holes():
    circle, _ = _circle_rect()
    loops = G.fit_mask_bspline(circle, 16)
    box = [120.0, 60.0, 260.0, 140.0]
    m = G.rasterize_loops(loops, circle.shape, box=box)
    bm = box_to_mask(box, *circle.shape)
    assert not (m & ~bm).any()
    assert _iou(m, circle & bm) > 0.97
    yy, xx = np.mgrid[0:200, 0:300]
    d2 = (xx + 0.5 - 150) ** 2 + (yy + 0.5 - 100) ** 2
    ring = (d2 <= 80**2) & (d2 >= 40**2)
    with_h = G.rasterize_loops(G.fit_mask_bspline(ring, 16, holes=True), ring.shape)
    no_h = G.rasterize_loops(G.fit_mask_bspline(ring, 16, holes=False), ring.shape)
    assert not with_h[100, 150] and no_h[100, 150]
    assert _iou(with_h, ring) > 0.97
    assert [lp["role"] for lp in G.fit_mask_bspline(ring, 8, holes=True)] == ["outer", "hole"]
    assert G.fit_mask_bspline(np.zeros((10, 10), bool), 8) == []


def test_spline_grabcut_modes_and_failure():
    img, disk = _scene()
    loops = G.fit_mask_bspline(np.roll(disk, 2, axis=1), 12)  # 稍有偏移的样条
    for mode in G.SPLINE_GC_MODES:
        r = G.spline_grabcut(img, DISK_BOX, loops, mode=mode)
        assert r.ok and _iou(r.mask, disk) > 0.9, mode
        assert not (r.mask & ~box_to_mask(DISK_BOX, H, W)).any()
    r = G.spline_grabcut(img, DISK_BOX, [], mode="soft")
    assert not r.ok and not r.mask.any() and "error" in r.info
    with pytest.raises(ValueError):
        G.spline_grabcut(img, DISK_BOX, loops, mode="hard")


def test_contour_band():
    reg = np.zeros((20, 20), bool)
    reg[5:15, 5:15] = True
    band = G.contour_band(reg, 1.0)
    assert band[5, 10] and band[4, 10] and not band[10, 10] and not band[1, 10]


# ---------------------------------------------------------------------------
# 点 → GrabCut
# ---------------------------------------------------------------------------


def test_points_grabcut():
    img, disk = _scene()
    r = G.points_grabcut(img, DISK_BOX, [[CX, CY], [CX - 10, CY + 5]], [[DISK_BOX[0] + 2, DISK_BOX[1] + 2]])
    assert r.ok and _iou(r.mask, disk) > 0.9
    assert r.info["n_fg"] == 2 and r.info["n_bg"] == 1
    assert not (r.mask & ~box_to_mask(DISK_BOX, H, W)).any()
    # 没有前景点、框内又是可能背景：GrabCut 没有前景，失败时用框本身
    f = G.points_grabcut(img, DISK_BOX, [], [[CX, CY]])
    assert not f.ok and np.array_equal(f.mask, box_to_mask(DISK_BOX, H, W))
    g = G.points_grabcut(img, DISK_BOX, [], None, box_init="pr_fgd", context_pct=None)
    assert g.ok and np.array_equal(g.mask, G.grabcut_mask(img, box_abs=DISK_BOX))  # 没有点时 = W1-F
    a = G.points_grabcut(img, DISK_BOX, [[CX, CY]])
    b = G.points_grabcut(img, DISK_BOX, [[CX, CY]])
    assert np.array_equal(a.mask, b.mask)


# ---------------------------------------------------------------------------
# 带符号的局部修正
# ---------------------------------------------------------------------------


def test_apply_signed_formula4():
    mask = np.array([[False, True, True, False]])
    lock = np.array([[0.0, 2.0, 1.0, -1.0]])
    reg = np.ones((1, 4), bool)
    m, lk = mask.copy(), lock.copy()
    G.apply_signed(m, lk, reg, "-", alpha=1.0)
    # 未承诺：直接写、Λ = −1；Λ = 2：只消耗到 1，不翻转；Λ = 1：消耗到 0 即翻转、Λ = −1；同号：加强到 −2
    assert m.tolist() == [[False, True, False, False]]
    assert lk.tolist() == [[-1.0, 1.0, -1.0, -2.0]]
    m, lk = mask.copy(), lock.copy()
    G.apply_signed(m, lk, reg, "-", alpha=0.5)  # LLMSeg 代码里的 α
    assert m.tolist() == [[False, True, True, False]] and lk.tolist() == [[-1.0, 1.5, 0.5, -2.0]]


def test_soft_lock_reverse_correction_only_consumes_commitment():
    img, disk = _scene()
    box = [CX - 8.0, CY - 8.0, CX + 8.0, CY + 8.0]
    empty = np.zeros((H, W), bool)
    m, lk, _ = G.correction_round(img, empty, None, [(box, "+")], rule="direct", soft_lock=True)
    m, lk, _ = G.correction_round(img, m, lk, [(box, "+")], rule="direct", soft_lock=True, gate=False)  # 同向：承诺加强到 2
    roi = box_to_mask(box, H, W)
    assert m[roi].all() and (lk[roi] == 2.0).all()
    m2, lk2, info = G.correction_round(img, m, lk, [(box, "-")], rule="direct", soft_lock=True)
    assert np.array_equal(m2, m)  # 反向的修正只消耗承诺，不翻转
    assert (lk2[roi] == 1.0).all() and info["n_applied"] == 1
    m3, lk3, _ = G.correction_round(img, m2, lk2, [(box, "-")], rule="direct", soft_lock=True)
    assert not m3[roi].any() and (lk3[roi] == -1.0).all()  # 承诺耗尽才翻转
    m4, _, _ = G.correction_round(img, m, lk, [(box, "-")], rule="direct", soft_lock=False)
    assert not m4[roi].any()  # 不用软锁定：直接删


def test_effect_gate_thresholds():
    mask = np.zeros((40, 40), bool)
    lock = np.zeros((40, 40))
    box = [0.0, 0.0, 10.0, 10.0]  # 100 px：门槛 max(16, ceil(2)) = 16
    mask[:10, :10] = True
    mask[0, :10] = False
    mask[1, :5] = False  # 15 个背景像素
    assert G.achievable_px("+", box, mask, lock) == (15, 100)
    assert not G.effect_gate("+", box, mask, lock)
    mask[1, 5] = False
    assert G.effect_gate("+", box, mask, lock)
    big = [0.0, 0.0, 40.0, 40.0]  # 1600 px：门槛 ceil(0.02 × 1600) = 32
    m = np.ones((40, 40), bool)
    m[0, :31] = False
    assert not G.effect_gate("+", big, m, lock)
    m[0, 31] = False
    assert G.effect_gate("+", big, m, lock)
    # 承诺挡住：背景像素都已承诺为背景（Λ < 0）时，"+" 照样可以改动；Λ > 0 的前景像素对 "−" 也可以
    assert G.achievable_px("-", box, mask, np.full((40, 40), 3.0))[0] == int(mask[:10, :10].sum())
    # 已经正确且承诺的区域：同向的提议被门挡掉
    img, _ = _scene()
    full = np.zeros((H, W), bool)
    full[10:30, 10:30] = True
    _, info = G.correction_round(img, full, None, [([10.0, 10.0, 30.0, 30.0], "+")], rule="direct")[1:]
    assert info["n_gated"] == 1 and info["n_applied"] == 0


def test_negatives_before_positives_and_confine():
    img, _ = _scene()
    empty = np.zeros((H, W), bool)
    a, b = [10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 40.0, 40.0]
    m, lk, _ = G.correction_round(img, empty, None, [(a, "+"), (b, "-")], rule="direct", gate=False)
    ov = box_to_mask(a, H, W) & box_to_mask(b, H, W)
    assert m[ov].all()  # 同一轮先删后补：重叠处最后是补
    m, _, info = G.correction_round(img, empty, None, [([90.0, 80.0, 120.0, 95.0], "+")], rule="direct",
                                    confine_box=[0.0, 0.0, 50.0, 50.0])
    assert not m.any() and info["n_outside_confine"] == 1
    m, _, _ = G.correction_round(img, empty, None, [([40.0, 40.0, 70.0, 70.0], "+")], rule="direct", confine_box=[0.0, 0.0, 50.0, 50.0])
    assert m.any() and not (m & ~box_to_mask([0.0, 0.0, 50.0, 50.0], H, W)).any()


def test_local_grabcut_rule_adds_and_removes_by_color():
    img, disk = _scene()
    yy, xx = np.mgrid[0:H, 0:W]
    cur = disk & (xx < CX + 5)  # 少了右边一块
    add_box = [CX + 4.0, CY - R - 2.0, CX + R + 3.0, CY + R + 2.0]
    reg, inf = G.correction_region(img, cur, add_box, "+", rule="grabcut")
    assert inf["rule"] == "grabcut"
    missing = disk & ~cur
    assert (reg & missing).sum() >= 0.9 * missing.sum()
    roi = box_to_mask(add_box, H, W)
    assert (reg & roi & ~disk).sum() <= 0.25 * (roi & ~disk).sum()  # 框里的背景大多不补
    cur2 = disk.copy()
    cur2[70:90, 100:120] = True  # 多出蓝方块
    rm_box = [96.0, 66.0, 124.0, 94.0]
    reg2, _ = G.correction_region(img, cur2, rm_box, "-", rule="grabcut")
    blue = np.zeros((H, W), bool)
    blue[70:90, 100:120] = True
    assert (reg2 & blue).sum() >= 0.9 * blue.sum()
    # 孤立的多余块：裁剪区里没有前景，GrabCut 跑不起来，退回 direct（整框删掉，正是要的）
    m, _, info = G.correction_round(img, cur2, None, [(rm_box, "-")], rule="grabcut")
    assert _iou(m, disk) == 1.0 and info["n_fallback_direct"] == 1
    # 连着物体的多余块（圆右边一条灰背景）：局部 GrabCut 删掉灰的、留下红的
    cur3 = disk.copy()
    cur3[38:58, CX + R - 2 : CX + R + 8] = True
    rm3 = [CX + R - 10.0, 34.0, CX + R + 10.0, 62.0]
    m3, _, info3 = G.correction_round(img, cur3, None, [(rm3, "-")], rule="grabcut")
    assert info3["n_fallback_direct"] == 0
    assert _iou(m3, disk) > 0.97 and _iou(m3, disk) > _iou(cur3, disk) + 0.03
    md, _, _ = G.correction_round(img, cur3, None, [(rm3, "-")], rule="direct")
    assert _iou(m3, disk) > _iou(md, disk)  # direct 连圆的一块一起删了
    # 跑不起来时退回 direct：整张图都是 mask、修正框盖满裁剪区域 → 没有背景
    allm = np.ones((H, W), bool)
    reg3, inf3 = G.correction_region(img, allm, [0.0, 0.0, float(W), float(H)], "+", rule="grabcut")
    assert inf3["rule"] == "direct" and "fallback" in inf3 and reg3.all()


def test_region_cache_is_bitwise_transparent():
    img, disk = _scene()
    yy, xx = np.mgrid[0:H, 0:W]
    start = disk & (xx < CX + 6)
    start[70:90, 100:110] = True
    props = [([CX + 2.0, CY - R - 2.0, CX + R + 3.0, CY + R + 2.0], "+"), ([96.0, 66.0, 112.0, 94.0], "-")]
    cache: dict = {}
    hits = {}
    for lk in (True, False):
        la, cur_a, lb, cur_b = None, start, None, start
        hits[lk] = []
        for _ in range(3):
            cur_a, la, ia = G.correction_round(img, cur_a, la, props, rule="grabcut", soft_lock=lk)
            cur_b, lb, ib = G.correction_round(img, cur_b, lb, props, rule="grabcut", soft_lock=lk, region_cache=cache)
            assert np.array_equal(cur_a, cur_b) and np.array_equal(la, lb)
            hits[lk].append((ib["n_region_cached"], ib["n_applied"]))
    assert hits[True][0][0] == 0
    assert hits[False][0][0] == hits[False][0][1] > 0  # 第二遍（不用软锁定）第 1 轮的区域全部取自缓存


def test_correction_round_does_not_mutate_inputs():
    img, disk = _scene()
    cur = disk.copy()
    lock = np.zeros((H, W))
    G.correction_round(img, cur, lock, [([90.0, 60.0, 128.0, 96.0], "+")], rule="grabcut")
    assert np.array_equal(cur, disk) and not lock.any()


def test_grabcut_crop_and_color_space():
    img, disk = _scene()
    a = G.grabcut_mask(img, box_abs=DISK_BOX, context_pct=50.0)
    assert _iou(a, disk) > 0.9
    r0, r1, c0, c1 = G.context_crop(DISK_BOX, 50.0, H, W)
    outside = np.ones((H, W), bool)
    outside[r0:r1, c0:c1] = False
    assert not (a & outside).any()
    for cs in G.COLOR_SPACES:
        assert _iou(G.grabcut_mask(img, box_abs=DISK_BOX, color_space=cs), disk) > 0.85, cs
    with pytest.raises(ValueError):
        G.grabcut_mask(img, box_abs=DISK_BOX, color_space="xyz")
    with pytest.raises(ValueError):
        G.run_grabcut(img, np.full((H, W), cv2.GC_BGD, np.uint8))
