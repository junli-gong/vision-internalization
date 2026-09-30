"""d 短时物理：步数分档、pymunk 确定性、合理性（静止保持静止、放下的物体落地、无明显穿透）、
碰撞形状与渲染一致、匀速外推基线、速度。"""

import math
import time

import numpy as np
import pymunk

from src.synth import generate
from src.synth import shapes as S
from src.synth.constants import (
    PHYSICS_K_BINS,
    PHYSICS_OVERLAP_TOL,
    PHYSICS_PENETRATION_TOL_PX,
    PHYSICS_REST_TOL_DEG,
    PHYSICS_REST_TOL_PX,
    PHYSICS_STEP_SECONDS,
    PHYSICS_SUPPORT_TOL_PX,
)
from src.synth.physics import Simulation, ground_y, max_overlap_fraction
from src.synth.scene import Background, Ground, ObjectSpec, SceneSpec

H, W = 480, 600
BG = Background("solid", (236, 236, 236), (236, 236, 236))
RECT = [v.index for v in S.VARIANTS["rectangle"]]


def _find(action, n=4, start=0, **cfg):
    out, i = [], start
    while len(out) < n:
        s = generate("physics", "train", i, **cfg)
        if s.meta["action"] == action:
            out.append(s)
        i += 1
    return out


def _resting(oid, fam, var, cx, r, yg):
    down = float((S.unit_polygon(fam, var)[:, 1] * r).max())
    return ObjectSpec(oid, fam, var, 0, S.qcoord(cx), S.qcoord(yg - down), S.qcoord(r), 0.0, 0)


def test_step_bins_and_instruction():
    for b, k in enumerate(PHYSICS_K_BINS):
        for i in range(4):
            s = generate("physics", "val", 20 + i, bin=b)
            d = s.meta["difficulty"]
            assert d["closed"] and d["value"] == k == s.meta["k"] and d["bin_range"] == [k, k]
            assert f"{k} " in s.instruction and s.meta["target_description"] in s.instruction
            assert abs(s.meta["seconds"] - k / 30) < 1e-12


def test_simulation_is_bitwise_deterministic():
    for s in _find("collide", 2) + _find("drop", 2):
        scene = SceneSpec.from_dict(s.meta["scene"])
        vel = {int(k): tuple(v) for k, v in s.meta["initial_velocity_px_per_s"].items()}
        runs = []
        for _ in range(2):
            sim = Simulation(scene, vel)
            traj = []
            for _ in range(40):
                sim.step()
                traj.append([(b.position.x, b.position.y, b.angle, b.velocity.x, b.velocity.y) for b in sim.bodies])
            runs.append(np.array(traj))
        assert runs[0].tobytes() == runs[1].tobytes()


def test_resting_objects_stay_at_rest():
    yg = ground_y(H)
    sq = next(v.index for v in S.VARIANTS["rectangle"] if v.noun == "square")
    objs = (
        _resting(0, "rectangle", RECT[0], 100.0, 50.0, yg),
        _resting(1, "circle", 0, 250.0, 40.0, yg),
        _resting(2, "rectangle", sq, 420.0, 60.0, yg),
    )
    scene = SceneSpec(H, W, BG, objs, Ground(yg, (70, 70, 80)))
    sim = Simulation(scene)
    sim.step(40)
    for o, (cx, cy, a) in zip(objs, sim.poses()):
        assert math.hypot(cx - o.cx, cy - o.cy) <= PHYSICS_REST_TOL_PX
        assert abs(a - o.angle) <= PHYSICS_REST_TOL_DEG
    assert max(sim.speeds()) < 1.0
    # 生成的样本里，未被碰到的静止物体同样不动
    for i in range(24):
        s = generate("physics", "test", 60 + i)
        assert s.meta["untouched_max_drift_px"] <= PHYSICS_REST_TOL_PX


def _supported(sim, k, yg):
    """物体 k 静止并且落在地面（最低点贴地面）或另一个物体上（与之有接触）。"""
    b = sim.bodies[k]
    others = []
    b.each_arbiter(lambda arb: others.extend(s for s in arb.shapes if s.body is not b and s.body.body_type == pymunk.Body.DYNAMIC))
    o = sim.scene_now().objects[k]
    lowest = float(o.polygon()[:, 1].max())
    return abs(lowest - yg) <= PHYSICS_SUPPORT_TOL_PX or bool(others)


def test_dropped_objects_land_on_ground_or_another_object():
    stacked = 0
    for s in _find("drop", 8):
        scene = SceneSpec.from_dict(s.meta["scene"])
        k = s.meta["object_roles"].index("dropped")
        sim = Simulation(scene)
        sim.step(120)  # 4 秒
        assert sim.speeds()[k] < 2.0, sim.speeds()[k]
        assert _supported(sim, k, scene.ground.y)
        stacked += "support" in s.meta["object_roles"]
    assert stacked >= 1  # 覆盖到落在另一个物体上的情形


def test_no_interpenetration_in_hard_cases_without_rejection():
    """不经生成器的拒绝：最高速相撞、从最高处砸到盒子上，每步检查穿透与重叠。"""
    yg = ground_y(H)
    sq = next(v.index for v in S.VARIANTS["rectangle"] if v.noun == "square")
    a = _resting(0, "rectangle", RECT[2], 200.0, 0.12 * W, yg)
    b = _resting(1, "circle", 0, 330.0, 0.06 * W, yg)
    c = _resting(2, "rectangle", RECT[0], 480.0, 0.12 * W, yg)
    drop = ObjectSpec(3, "rectangle", sq, 0, 480.0, S.qcoord(yg - 0.45 * W - 150), 0.12 * W, 30.0, 0)
    scene = SceneSpec(H, W, BG, (a, b, c, drop), Ground(yg, (70, 70, 80)))
    sim = Simulation(scene, {0: (0.75 * W, 0.0), 1: (-0.75 * W, 0.0)})
    for _ in range(60):
        sim.step()
        assert max_overlap_fraction(sim.scene_now()) <= PHYSICS_OVERLAP_TOL
    assert sim.max_penetration <= PHYSICS_PENETRATION_TOL_PX
    assert sim.touched[0] and sim.touched[1] and sim.touched[2] and sim.touched[3]
    for i in range(24):
        s = generate("physics", "train", 200 + i)
        assert s.meta["max_penetration_px"] <= PHYSICS_PENETRATION_TOL_PX
        assert s.meta["max_overlap_fraction"] <= PHYSICS_OVERLAP_TOL


def test_collision_shapes_match_rendered_polygons():
    for s in _find("drop", 3) + _find("push", 2):
        scene = SceneSpec.from_dict(s.meta["scene"])
        sim = Simulation(scene)
        for o, body, shp in zip(scene.objects, sim.bodies, [next(iter(b.shapes)) for b in sim.bodies]):
            if o.family == "circle":
                assert abs(shp.radius - o.radius) < 1e-9
                continue
            world = np.array([tuple(body.local_to_world(v)) for v in shp.get_vertices()])
            poly = o.polygon()
            # pymunk 可能重排顶点：逐点找最近的渲染顶点
            d = np.sqrt(((world[:, None, :] - poly[None, :, :]) ** 2).sum(-1)).min(axis=1)
            assert d.max() < 0.01, d.max()


def test_const_velocity_baseline_is_an_exact_integer_shift():
    """W = 600：速度 n/100·W = 6n px/s，k 步位移 = 6n·k/30 = n·k/5 px，恰是整数像素。"""
    for s in _find("push", 3, canvas_hw=(H, W)) + _find("collide", 3, canvas_hw=(H, W)):
        t = s.meta["target_obj_id"]
        vx = s.meta["initial_velocity_px_per_s"][str(t)][0]
        k = s.meta["k"]
        shift = round(vx * k * PHYSICS_STEP_SECONDS)
        assert abs(vx * k * PHYSICS_STEP_SECONDS - shift) < 1e-6
        copy = s.baselines["copy"].mask
        expect = np.zeros_like(copy)
        if shift >= 0:
            expect[:, shift:] = copy[:, : W - shift]
        else:
            expect[:, :shift] = copy[:, -shift:]
        assert np.array_equal(s.baselines["const_velocity"].mask, expect)
        speed_text = f"{abs(vx) / W:.2f} image widths per second"
        assert speed_text in s.instruction
    for s in _find("drop", 2):
        assert np.array_equal(s.baselines["const_velocity"].mask, s.baselines["copy"].mask)


def test_speed_report():
    generate("physics", "train", 0)
    t = time.process_time()
    n = 40
    for i in range(n):
        generate("physics", "train", 1000 + i)
    rate = n / (time.process_time() - t)
    print(f"physics: {rate:.1f} samples/s (CPU time, single process)")
    assert rate >= 10  # 任务书只要求报告实测速度；这里只挡明显的退化
