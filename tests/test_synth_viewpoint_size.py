"""DEV-0018：b 视角变化放大物体（D28-13：被点名的物体在 32×32 网格上至少约占 100 格；任务版本 3 → 4）。

- 下限的定义：min_target_px = max(VIEW_MIN_VISIBLE_PX, ceil(100 · H · W / 32²))；两个视角里都满足；网格上实际约 100 格。
- 凸包面积在整数单位里精确；像素中心数 ≤ 凸包面积 + 半周长 + 1（Pick 定理），所以 0.9 的预筛余量不会去掉合格候选。
- 预筛与分步渲染只省时间：关掉预筛（恒为真），样本逐位不变。
- 报告：放大后目标的格数分布、两种平凡基线的 IoU 分布（按移动类型）。放大前（任务版本 3）的数字见 DEV-0018 完成总结。
"""

import math

import numpy as np

import src.synth.viewpoint as V
from src.synth import generate
from src.synth.constants import FLOOR_ID, VIEW_FLOOR_RADIUS_M, VIEW_MIN_TARGET_CELLS, VIEW_PLACE_RADIUS_M, VIEW_RADIUS_M
from src.synth.raster import UNITS
from src.synth.render3d import Scene3D, posed_vertices, render_posed, project_units


def test_min_target_px_definition():
    assert V.min_target_px(512, 512) == 25600 == VIEW_MIN_TARGET_CELLS * 512 * 512 // 1024
    assert V.min_target_px(384, 512) == 19200
    assert V.min_target_px(100, 100) == math.ceil(100 * 100 * 100 / 1024)
    assert V.min_target_px(64, 64) == 400
    assert VIEW_PLACE_RADIUS_M + VIEW_RADIUS_M[1] <= VIEW_FLOOR_RADIUS_M + 1e-12  # 物体不伸出地板


def test_hull_area_is_exact():
    f = 512.0  # 512×512 画布：f = 512，Z = f 时 u = X + 256
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0), (2.0, 7.0)]
    p = np.array([[x, y, f] for x, y in pts])
    assert V.hull_area2_units(p, 512, 512) == 2 * 100 * UNITS * UNITS
    tri = np.array([[0.0, 0.0, f], [8.0, 0.0, f], [0.0, 6.0, f], [4.0, 0.0, f]])  # 共线点不改变面积
    assert V.hull_area2_units(tri, 512, 512) == 2 * 24 * UNITS * UNITS
    assert V.hull_area2_units(np.array([[0.0, 0.0, f], [1.0, 1.0, f]]), 512, 512) == 0


def _hull_perimeter_px(p, h, w) -> float:
    uu, vv = project_units(p, h, w)
    pts = sorted(set(zip((uu / UNITS).tolist(), (vv / UNITS).tolist())))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for q in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    hull = lo[:-1] + up[:-1]
    return sum(math.dist(hull[k], hull[(k + 1) % len(hull)]) for k in range(len(hull)))


def test_pixel_count_never_exceeds_the_hull_bound():
    """预筛的前提：每个物体完整轮廓的像素数 ≤ 凸包面积 + 半周长 + 1；对足够大的物体，这个余量 < 10%。"""
    for i in range(10):
        s = generate("viewpoint", "val", 600 + i)
        scene = Scene3D.from_dict(s.meta["scene"])
        fr = render_posed(scene, scene.camera, posed_vertices(scene), want_rgb=False, want_amodal=True)
        for k, p in enumerate(posed_vertices(scene)):
            area = V.hull_area2_units(p, scene.height, scene.width) / (2 * UNITS * UNITS)
            per = _hull_perimeter_px(p, scene.height, scene.width)
            n = int(fr.amodal[k].sum())
            assert n <= area + per / 2 + 1, (i, k)
            if n >= V.min_target_px(scene.height, scene.width):
                assert area >= 0.9 * n


def test_targets_are_big_in_both_views():
    cells = []
    for split in ("val", "test"):
        for i in range(24):
            s = generate("viewpoint", split, i)
            min_px = s.meta["min_visible_px"]
            assert min_px == V.min_target_px(*s.target_mask.shape)
            assert min(s.meta["visible_px"]) >= min_px
            assert int(s.target_mask.sum()) == s.meta["visible_px"][1]
            assert int(s.baselines["copy"].mask.sum()) == s.meta["visible_px"][0]
            assert 2 <= len(s.meta["scene"]["objects"]) <= 3
            assert np.all(s.extras["id_map"][s.extras["id_map"] > len(s.meta["scene"]["objects"])] == FLOOR_ID)
            cells += [int(s.target_grid.sum()), int(s.baselines["copy"].grid.sum())]
    assert min(cells) >= 90 and float(np.median(cells)) >= 110  # "约 100 格"：网格的占比阈值让少数样本落在 90 多


def test_precheck_does_not_change_samples(monkeypatch):
    want = [generate("viewpoint", "train", 700 + i).digest() for i in range(16)]
    monkeypatch.setattr(V, "_hull_big_enough", lambda *a, **k: True)
    assert [generate("viewpoint", "train", 700 + i).digest() for i in range(16)] == want


def _summary(v) -> str:
    v = np.asarray(v, dtype=float)
    q = np.quantile(v, [0.1, 0.5, 0.9])
    return f"mean {v.mean():.3f} [p10 {q[0]:.3f} p50 {q[1]:.3f} p90 {q[2]:.3f}]"


def test_report_cells_and_baseline_iou_after_enlarging():
    n = 25
    lines = ["", "b viewpoint（任务版本 5）：目标格数（32×32，新视角）与两种平凡基线的网格 IoU，val 划分每档 25 条"]
    allc = []
    for b in range(4):
        per = {"orbit": {"cells": [], "copy": [], "translate": []}, "shift": {"cells": [], "copy": [], "translate": []}}
        for i in range(n):
            s = generate("viewpoint", "val", 3000 + 4 * i + b)
            kind = "orbit" if s.meta["move"]["type"] == "orbit" else "shift"
            per[kind]["cells"].append(int(s.target_grid.sum()))
            for k in ("copy", "translate"):
                per[kind][k].append(s.meta["baseline_iou"][k]["grid"])
        for kind, v in per.items():
            if v["cells"]:
                allc += v["cells"]
                lines.append(f"  bin {b} {kind} n={len(v['cells'])}: cells p50 {np.median(v['cells']):.0f}; "
                             f"copy {_summary(v['copy'])}; translate {_summary(v['translate'])}")
    lines.append(f"  all cells: {_summary(allc)}")
    print("\n".join(lines))
    assert np.median(allc) >= 110
