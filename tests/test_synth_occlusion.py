"""c 遮挡补全：难度档、目标唯一性与 z 序、平凡基线（构造例子）、非平凡配额、速度。"""

import time

import numpy as np
import shapely

from src.synth import generate
from src.synth.constants import (
    CANVAS_MARGIN_PX,
    OCC_NONTRIVIAL_HULL_IOU,
    OCC_NONTRIVIAL_QUOTA,
    OCCLUSION_BINS,
)
from src.synth.occlusion import convex_hull_mask
from src.synth.sample import iou
from src.synth.scene import SceneSpec, object_masks, visible_masks
from src.synth.shapes import polygon_inside

H, W = 96, 160


def test_difficulty_bins_hold_and_match_the_masks():
    for b, (lo, hi) in enumerate(OCCLUSION_BINS):
        for i in range(8):
            s = generate("occlusion", "train", 100 + i, bin=b)
            d = s.meta["difficulty"]
            assert d["bin"] == b and d["bin_range"] == [lo, hi] and lo <= d["value"] < hi
            scene = SceneSpec.from_dict(s.meta["scene"])
            amodal = object_masks(scene)
            vis = visible_masks(scene, amodal)
            t = s.meta["target_obj_id"]
            a, v = np.count_nonzero(amodal[t]), np.count_nonzero(vis[t])
            assert d["value"] == (a - v) / a
            assert np.array_equal(s.target_mask, amodal[t])
            assert np.array_equal(s.baselines["visible"].mask, vis[t])


def test_target_is_unique_and_occluders_are_in_front():
    for i in range(12):
        s = generate("occlusion", "val", 200 + i)
        scene = SceneSpec.from_dict(s.meta["scene"])
        t = s.meta["target_obj_id"]
        desc = scene.objects[t].description
        assert desc == s.meta["target_description"] and desc in s.instruction
        assert sum(o.description == desc for o in scene.objects) == 1
        roles = s.meta["object_roles"]
        assert roles[t] == "target" and 1 <= roles.count("occluder") <= 3 and roles.count("distractor") <= 2
        zt = scene.objects[t].z
        tgeom = shapely.polygons(scene.objects[t].polygon())
        for o, r in zip(scene.objects, roles):
            assert polygon_inside(o.polygon(), scene.height, scene.width, CANVAS_MARGIN_PX)
            if r == "occluder":
                assert o.z > zt
            elif r == "distractor":
                assert o.z < zt and not shapely.intersects(tgeom, shapely.polygons(o.polygon()))


def _center_in_polygon(poly, h, w):
    """像素中心严格在多边形内的精确 mask，以及中心离边界的距离。"""
    g = shapely.polygons(poly)
    yy, xx = np.mgrid[:h, :w]
    pts = shapely.points(xx.ravel() + 0.5, yy.ravel() + 0.5)
    inside = shapely.contains(g, pts).reshape(h, w)
    dist = shapely.distance(g.exterior, pts).reshape(h, w)
    return inside, dist


def test_convex_hull_baseline_on_constructed_examples():
    # 凸的可见区域（矩形）：凸包就是它自己
    rect = np.zeros((H, W), bool)
    rect[10:20, 10:40] = True
    assert np.array_equal(convex_hull_mask(rect), rect)
    assert not convex_hull_mask(np.zeros((H, W), bool)).any()
    # L 形可见区域：凸包 = 像素方块之并的凸包，顶点 (10,10) (40,10) (40,20) (20,50) (10,50)
    ell = np.zeros((H, W), bool)
    ell[10:20, 10:40] = True
    ell[20:50, 10:20] = True
    hull = convex_hull_mask(ell)
    exact, dist = _center_in_polygon(np.array([[10, 10], [40, 10], [40, 20], [20, 50], [10, 50]], float), H, W)
    far = dist > 0.3  # 离边界 0.3 px 以内的像素中心，光栅化规则可以不同
    assert np.array_equal(hull[far], exact[far])
    assert np.all(hull[ell]) and abs(np.count_nonzero(hull) - (600 + 300)) <= 20  # 加上的三角形面积 = 20·30/2
    # 再取一次凸包：只可能在斜边上多出不到半个像素宽的一条（像素方块的凸包比中心光栅化略大），不会缩小
    h2 = convex_hull_mask(hull)
    assert np.all(h2[hull]) and np.count_nonzero(h2 & ~hull) <= 40


def test_nontrivial_quota_and_threshold():
    per_bin = 24
    for b in range(len(OCCLUSION_BINS)):
        flags = []
        for i in range(per_bin):
            s = generate("occlusion", "train", 300 + i, bin=b)
            hull_iou = s.meta["baseline_iou"]["convex_hull"]["full"]
            assert s.meta["nontrivial"] == (hull_iou < OCC_NONTRIVIAL_HULL_IOU)
            if s.meta["nontrivial_required"]:
                assert hull_iou < OCC_NONTRIVIAL_HULL_IOU
            assert hull_iou == iou(s.baselines["convex_hull"].mask, s.target_mask)
            assert np.all(s.baselines["convex_hull"].mask[s.baselines["visible"].mask])
            flags.append(s.meta["nontrivial"])
        assert np.mean(flags) >= OCC_NONTRIVIAL_QUOTA, (b, np.mean(flags))


def test_speed_at_least_50_per_second_single_core():
    generate("occlusion", "train", 0)  # 预热（import、单位形状缓存）
    t = time.process_time()
    n = 60
    for i in range(n):
        generate("occlusion", "train", 1000 + i)
    rate = n / (time.process_time() - t)
    print(f"occlusion: {rate:.1f} samples/s (CPU time, single process)")
    assert rate >= 50
