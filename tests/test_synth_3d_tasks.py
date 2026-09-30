"""E5 第二部分的两类任务（DEV-0010）：a 出平面三维旋转、b 视角变化。

- 文字决定答案：只从指令文字反解出旋转 / 相机移动，再渲染，结果与目标逐位相同（场景取自元数据，即图像里的内容）。
- a 排除平面内旋转：所有样本的旋转轴都是相机的水平轴或竖直轴，与指向实体中心的视线垂直。
- a 的对称等价：报告的角不超过该轴上旋转周期的一半；周期 < 360° 时，差一个周期的旋转给出几乎相同的 mask。
- 难度档；平凡基线在构造例子上正确；报告各档两种非真值基线与真值仿射上界的 IoU 分布（均值与分位数）。
- 划分：各划分的实体身份与场景种子两两不相交；速度：单核 ≥ 20 样本/秒。
- 可复现：逐字节重现、与全局随机状态无关、换 PYTHONHASHSEED 的新进程结果相同（本文件）。
  钉住的摘要与桩测试（平台超越函数、np.mean、np.linalg / BLAS 换成抛异常的桩后摘要不变）在 tests/test_synth_repro.py，
  a、b 都在 ANY_PLATFORM_TASKS 里（任何平台逐位相同）。
"""

import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.constants import D_MAX, D_MIN
from src.synth import SynthDataset, generate
from src.synth import solids as SO
from src.synth.constants import (
    CANVAS_MARGIN_PX,
    FLOOR_ID,
    ROT3D_AXES,
    ROT3D_BINS,
    SPLITS,
    VIEW_MIN_VISIBLE_FRAC,
    VIEW_ORBIT_BINS,
    VIEW_SHIFT_BINS_CM,
)
from src.synth.render3d import (
    CameraSpec,
    Scene3D,
    SolidSpec,
    intrinsics,
    mat_mul,
    mat_vec,
    render_scene,
    rot3,
)
from src.synth.rotation3d import AXIS_PHRASES, DIRECTION_PHRASES, instruction_affine
from src.synth.sample import iou, value_in_bin
from src.synth.shapes import max_equivalent_rotation
from src.synth.viewpoint import centroid_shift_px, move_camera
from src.synth.warp2d import affine_warp, mask_moments, moment_matched_affine, shift_mask

H, W = 96, 160


# ---------------------------------------------------------------------------
# 只从文字反解
# ---------------------------------------------------------------------------

_AXIS_BY_PHRASE = {v: k for k, v in AXIS_PHRASES.items()}
_SIGN_BY_PHRASE = {v: k for k, v in DIRECTION_PHRASES.items()}


def parse_rotation(instruction: str):
    m = re.search(r"(\d+) degrees about (" + "|".join(map(re.escape, _AXIS_BY_PHRASE)) + ")", instruction)
    axis = _AXIS_BY_PHRASE[m[2]]
    phrases = [p for p in _SIGN_BY_PHRASE if p in instruction]
    assert len(phrases) == 1, instruction
    ax2, sign = _SIGN_BY_PHRASE[phrases[0]]
    assert ax2 == axis
    return axis, sign * int(m[1])


def parse_move(instruction: str):
    if m := re.search(r"circles (\d+) degrees to the (left|right)", instruction):
        return "orbit", int(m[1]) * (1 if m[2] == "right" else -1)
    if m := re.search(r"slides (\d\.\d\d) m to its (right|left)", instruction):
        return m[2], round(float(m[1]) * 100)
    if m := re.search(r"(rises|drops) (\d\.\d\d) m straight (up|down)", instruction):
        return m[3], round(float(m[2]) * 100)
    if m := re.search(r"moves (\d\.\d\d) m (forward|backward) along its line of sight", instruction):
        return m[2], round(float(m[1]) * 100)
    raise AssertionError(instruction)


@pytest.mark.parametrize("canvas", [(512, 512), (384, 512)])
def test_text_alone_determines_the_rotation_target(canvas):
    for i in range(12):
        s = generate("rotation3d", "train", 300 + i, canvas_hw=canvas)
        scene = Scene3D.from_dict(s.meta["scene"])
        assert s.meta["target_description"] in s.instruction and scene.objects[0].description == s.meta["target_description"]
        axis, signed = parse_rotation(s.instruction)
        assert (axis, signed) == (s.meta["rotation"]["axis"], s.meta["rotation"]["signed_angle_deg"])
        before = render_scene(scene)
        after = render_scene(scene, extra={0: rot3({"horizontal": "x", "vertical": "y"}[axis], signed)}, want_rgb=False)
        assert np.array_equal(after.mask(0), s.target_mask), (canvas, i, s.instruction)
        assert np.array_equal(before.mask(0), s.baselines["copy"].mask)
        assert np.array_equal(before.rgb, s.image)
        assert np.array_equal(after.depth_m(), s.extras["depth"]) and np.array_equal(before.depth_m(), s.extras["input_depth"])


@pytest.mark.parametrize("canvas", [(512, 512), (384, 512)])
def test_text_alone_determines_the_viewpoint_target(canvas):
    for i in range(12):
        s = generate("viewpoint", "train", 300 + i, canvas_hw=canvas)
        scene = Scene3D.from_dict(s.meta["scene"])
        named = [o.obj_id for o in scene.objects if f"the {o.description}" in s.instruction]
        assert named == [s.meta["target_obj_id"]], s.instruction  # 颜色 + 名词在场景里唯一
        move, amount = parse_move(s.instruction)
        assert (move, amount) == (s.meta["move"]["type"], s.meta["move"]["amount"])
        cam1 = move_camera(scene.camera, move, amount)
        assert cam1.to_dict() == s.meta["camera_after"]
        f0 = render_scene(scene)
        f1 = render_scene(scene, cam1, want_rgb=False)
        t = named[0]
        assert np.array_equal(f1.mask(t), s.target_mask), (canvas, i, s.instruction)
        assert np.array_equal(f1.id_map, s.extras["id_map"]) and np.array_equal(f0.id_map, s.extras["input_id_map"])
        assert np.array_equal(f0.rgb, s.image) and np.array_equal(f0.mask(t), s.baselines["copy"].mask)
        assert np.array_equal(s.object_masks, np.stack([f1.mask(k) for k in range(len(scene.objects))]))


# ---------------------------------------------------------------------------
# a：排除平面内旋转、对称等价、难度
# ---------------------------------------------------------------------------


def test_rotation_axes_are_perpendicular_to_the_line_of_sight():
    for i in range(80):
        s = generate("rotation3d", "val", 500 + i)
        rot = s.meta["rotation"]
        assert rot["axis"] in ROT3D_AXES and rot["axis_to_line_of_sight_deg"] == 90
        scene = Scene3D.from_dict(s.meta["scene"])
        cam = scene.camera
        center = mat_vec(cam.basis(), tuple(scene.objects[0].center[k] - cam.position()[k] for k in range(3)))
        # 实体中心在光轴上（投影在画布中心），视线 = 相机 z 轴；旋转轴是相机 x 或 y 轴
        assert abs(center[0]) < 1e-9 and abs(center[1]) < 1e-9 and center[2] > 1.0
        axis_vec = {"horizontal": (1.0, 0.0, 0.0), "vertical": (0.0, 1.0, 0.0)}[rot["axis"]]
        r = rot3({"horizontal": "x", "vertical": "y"}[rot["axis"]], rot["signed_angle_deg"])
        assert mat_vec(r, axis_vec) == axis_vec  # 这条轴在旋转下不动
        cos_los = sum(a * c for a, c in zip(axis_vec, center)) / math.sqrt(sum(c * c for c in center))
        assert abs(cos_los) < 1e-9  # 夹角 90°（≥ 60° 的要求自然满足）
        # 外接球的投影在画布里：旋转前后的轮廓都不碰画布边缘
        for m in (s.target_mask, s.baselines["copy"].mask):
            assert not (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


def test_direction_words_match_the_geometry():
    """方向说明的含义与几何一致（不经过 DIRECTION_PHRASES 表）：水平轴正转，最上面的顶点离相机更近；
    竖直轴正转，最靠近相机的顶点向左移。负转相反。"""
    from src.synth.render3d import camera_vertices

    obj = SolidSpec(0, "cone", 1, 0, "upright", 20, 0.5, (0.0, 0.0, 0.0))
    cam = CameraSpec(0, 20, 2.5)
    p0 = camera_vertices(obj, cam)
    top = int(np.argmin(p0[:, 1]))  # 相机 y 向下：y 最小即最上面（圆锥的尖）
    front = int(np.argmin(p0[:, 2]))
    for sign in (1, -1):
        ph = camera_vertices(obj, cam, rot3("x", sign * 40))
        assert (ph[top, 2] < p0[top, 2]) == (sign == 1)
        assert DIRECTION_PHRASES[("horizontal", sign)] == ("its top moves toward the camera" if sign == 1 else "its top moves away from the camera")
        pv = camera_vertices(obj, cam, rot3("y", sign * 40))
        assert (pv[front, 0] < p0[front, 0]) == (sign == 1)
        assert DIRECTION_PHRASES[("vertical", sign)].endswith("to the left" if sign == 1 else "to the right")
    # b：环绕"向右"= 相机向它自己的右边移动
    cam1 = move_camera(cam, "orbit", 30)
    right = cam.basis()[0]
    d = [cam1.position()[k] - cam.position()[k] for k in range(3)]
    assert sum(a * b for a, b in zip(d, right)) > 0.5


def _period_of(sample) -> float:
    scene = Scene3D.from_dict(sample.meta["scene"])
    o = scene.objects[0]
    m = mat_mul(scene.camera.basis(), o.rotation())
    local = m[0] if sample.meta["rotation"]["axis"] == "horizontal" else m[1]
    return SO.rotation_period_about(o.family, o.variant, local)


def test_reported_angles_are_minimal_under_the_solid_symmetry():
    seen_symmetric = 0
    for i in range(120):
        s = generate("rotation3d", "test", 700 + i)
        rot = s.meta["rotation"]
        period = _period_of(s)
        assert period == rot["symmetry_period_deg"] and period != 0.0
        assert 1 <= rot["angle_deg"] <= max_equivalent_rotation(period) == rot["max_equivalent_deg"]
        assert not SO.is_isotropic(s.meta["scene"]["objects"][0]["family"], s.meta["scene"]["objects"][0]["variant"])
        if period < 360.0 and seen_symmetric < 6:
            # 差一个周期的旋转是等价的：mask 只差在折线化的棱上
            seen_symmetric += 1
            scene = Scene3D.from_dict(s.meta["scene"])
            letter = {"horizontal": "x", "vertical": "y"}[rot["axis"]]
            other = render_scene(scene, extra={0: rot3(letter, rot["signed_angle_deg"] + period)}, want_rgb=False).mask(0)
            assert iou(other, s.target_mask) > 0.97, (i, period)
    assert seen_symmetric >= 3  # 圆柱、圆环、旋转椭球绕与轴垂直的水平轴时周期为 180°，这类样本常见


def test_constructed_symmetric_poses_limit_the_angle():
    cube = next(v.index for v in SO.SOLID_VARIANTS["box"] if v.noun == "cube")
    obj = SolidSpec(0, "box", cube, 0, "upright", 0, 0.5, (0.0, 0.0, 0.0))
    cam = CameraSpec(0, 30, 2.5)
    m = mat_mul(cam.basis(), obj.rotation())
    assert SO.rotation_period_about("box", cube, m[0]) == 90.0  # 水平轴正好是立方体的 4 重轴
    assert max_equivalent_rotation(90.0) == 45
    lying = SolidSpec(0, "cylinder", 1, 0, "side_z", 0, 0.5, (0.0, 0.0, 0.0))  # 轴沿世界 x，正是相机的水平轴
    assert SO.rotation_period_about("cylinder", 1, mat_mul(cam.basis(), lying.rotation())[0]) == 0.0
    assert max_equivalent_rotation(0.0) == 0


def test_rotation_difficulty_bins():
    for b, (lo, hi) in enumerate(ROT3D_BINS):
        for i in range(8):
            s = generate("rotation3d", "val", 900 + i, bin=b)
            d = s.meta["difficulty"]
            assert d["bin"] == b and d["bin_range"] == [lo, hi] and value_in_bin(d)
            assert d["value"] == abs(s.meta["rotation"]["signed_angle_deg"])
            assert d["upper_closed"] == (b == len(ROT3D_BINS) - 1)
            assert lo <= d["value"] <= hi if d["upper_closed"] else lo <= d["value"] < hi


# ---------------------------------------------------------------------------
# b：合格条件与难度
# ---------------------------------------------------------------------------


def test_viewpoint_constraints_and_bins():
    for b in range(4):
        for i in range(8):
            s = generate("viewpoint", "val", 900 + i, bin=b)
            d, mv = s.meta["difficulty"], s.meta["move"]
            assert d["bin"] == b and value_in_bin(d) and d["upper_closed"] == (b == 3)
            if mv["type"] == "orbit":
                assert d["name"] == "orbit_deg" and d["bin_range"] == list(VIEW_ORBIT_BINS[b]) and d["value"] == abs(mv["amount"])
            else:
                assert d["name"] == "shift_cm" and d["bin_range"] == list(VIEW_SHIFT_BINS_CM[b]) and d["value"] == mv["amount"]
            assert min(s.meta["visible_fraction"]) >= VIEW_MIN_VISIBLE_FRAC
            t = s.meta["target_obj_id"]
            for key in ("id_map", "input_id_map"):
                rows, cols = np.nonzero(s.extras[key] == t + 1)
                assert rows.min() >= CANVAS_MARGIN_PX - 1 and cols.min() >= CANVAS_MARGIN_PX - 1
                assert rows.max() <= s.target_mask.shape[0] - CANVAS_MARGIN_PX and cols.max() <= s.target_mask.shape[1] - CANVAS_MARGIN_PX
            assert s.baselines["copy"].mask.any() and not np.array_equal(s.baselines["copy"].mask, s.target_mask)
            assert np.array_equal(s.target_mask, s.extras["id_map"] == t + 1)
            for key in ("depth", "input_depth"):
                dm = s.extras[key]
                assert dm.dtype == np.float32 and dm[dm > 0].min() >= D_MIN and dm.max() <= D_MAX
            assert np.array_equal(s.extras["depth"] > 0, s.extras["id_map"] > 0)
            assert np.all(s.extras["id_map"][s.extras["id_map"] > len(s.meta["scene"]["objects"])] == FLOOR_ID)


def test_depth_grids_and_ranges():
    for task in ("rotation3d", "viewpoint"):
        for i in range(6):
            s = generate(task, "train", 40 + i, grid_size=16)
            g = s.extras["depth_grid"]
            assert g.shape == (16, 16) and g.dtype == np.float32
            assert np.array_equal(g > 0, s.target_grid == 1)
            assert g[g > 0].min() >= D_MIN and g.max() <= D_MAX
            lo, hi = s.meta["depth_range_m"]
            dm = s.extras["depth"]
            assert lo == float(dm[dm > 0].min()) and hi == float(dm.max())


# ---------------------------------------------------------------------------
# 平凡基线：构造例子
# ---------------------------------------------------------------------------


def _l_mask(h, w):
    m = np.zeros((h, w), bool)
    m[20:70, 30:45] = True
    m[55:70, 30:90] = True
    return m


def test_instruction_affine_on_constructed_examples():
    m = _l_mask(H, W)
    a, t = instruction_affine("vertical", 180, H, W)  # 正交近似下绕竖直轴转 180° = 左右镜像
    assert a == ((-1.0, 0.0), (0.0, 1.0)) and t == (float(W), 0.0)
    assert np.array_equal(affine_warp(m, a, t), m[:, ::-1])
    a, t = instruction_affine("horizontal", 180, H, W)
    assert np.array_equal(affine_warp(m, a, t), m[::-1, :])
    a, t = instruction_affine("horizontal", 60, H, W)  # cos 60° = 0.5：以画布中心为轴、上下压扁一半
    assert a == ((1.0, 0.0), (0.0, 0.5))
    rect = np.zeros((H, W), bool)
    rect[10:40, 50:70] = True  # 行 [10, 40)：像是 [48 + (10 − 48)/2, 48 + (40 − 48)/2) = [29, 44)
    want = np.zeros_like(rect)
    want[29:44, 50:70] = True
    assert np.array_equal(affine_warp(rect, a, t), want)
    mom0, mom1 = mask_moments(rect), mask_moments(want)
    assert abs(mom1.my - (48 + 0.5 * (mom0.my - 48))) < 0.5 and abs(mom1.syy - 0.25 * mom0.syy) < 0.2 * mom0.syy
    a, t = instruction_affine("vertical", -90, H, W)  # 恰好 90°：M 奇异，像是零面积的线段，结果为空
    assert a[0][0] == 0.0 and not affine_warp(m, a, t).any()


def test_translate_baseline_on_constructed_example():
    """相机向右平移 d：场景质心（深度 Z）在图像上向左移 f·d/Z 像素，竖直方向不动。"""
    objs = (
        SolidSpec(0, "box", 1, 0, "upright", 0, 0.4, (0.5, 0.3, 0.0)),
        SolidSpec(1, "cone", 0, 1, "upright", 0, 0.4, (-0.5, 0.3, 0.0)),
    )
    cam0 = CameraSpec(0, 0, 4.0, target=(0.0, 0.3, 0.0))
    scene = Scene3D(256, 256, (255, 255, 255), cam0, objs)
    cam1 = move_camera(cam0, "right", 37)
    dx, dy = centroid_shift_px(scene, cam0, cam1)
    f, _, _ = intrinsics(256, 256)
    assert (dx, dy) == (round(-f * 0.37 / 4.0), 0)
    m = np.zeros((256, 256), bool)
    m[100:120, 10:30] = True
    shifted = shift_mask(m, dx, dy)
    assert shifted.sum() == m.sum() - 20 * max(0, -dx - 10) and np.array_equal(shifted[100:120, : 30 + dx], m[100:120, -dx:30])
    assert np.array_equal(shift_mask(shift_mask(m, 5, -7), -5, 7)[10:-10, 10:-10], m[10:-10, 10:-10])


def test_gt_affine_upper_bound_recovers_a_true_affine():
    m = _l_mask(H, W)
    c, s = math.cos(math.radians(35)), math.sin(math.radians(35))
    a = ((1.2 * c, -0.7 * s), (1.2 * s, 0.7 * c))
    t = (80.0 - (a[0][0] * 60 + a[0][1] * 50), 48.0 - (a[1][0] * 60 + a[1][1] * 50))
    dst = affine_warp(m, a, t)
    fit = moment_matched_affine(m, dst)
    assert fit.iou > 0.93, fit
    mirrored = m[:, ::-1]
    assert moment_matched_affine(m, mirrored).iou > 0.97  # 需要镜像的也找得到


# ---------------------------------------------------------------------------
# IoU 分布（报告）
# ---------------------------------------------------------------------------


def _summary(values) -> str:
    v = np.asarray(values)
    q = np.quantile(v, [0.1, 0.5, 0.9])
    return f"mean {v.mean():.3f} [p10 {q[0]:.3f} p50 {q[1]:.3f} p90 {q[2]:.3f}]"


def test_report_baseline_iou_distributions():
    """各档两种非真值基线与真值仿射上界（a）、两种非真值基线（b，按移动类型分）的 IoU 分布。完整报告见 DEV-0010 总结。"""
    n = 16
    lines = ["", "a rotation3d (full-res IoU vs target):"]
    for b, (lo, hi) in enumerate(ROT3D_BINS):
        rows = [generate("rotation3d", "val", 2000 + 100 * b + i, bin=b).meta["baseline_iou"] for i in range(n)]
        vals = {k: [r[k]["full"] for r in rows] for k in ("copy", "instr_affine", "gt_affine_upper")}
        lines.append(f"  bin {b} [{lo},{hi}{']' if b == 3 else ')'}: " + "; ".join(f"{k} {_summary(v)}" for k, v in vals.items()))
        assert np.mean(vals["gt_affine_upper"]) >= np.mean(vals["copy"]) - 0.02
    lines.append("b viewpoint (full-res IoU vs target):")
    for b in range(4):
        per = {"orbit": {"copy": [], "translate": []}, "shift": {"copy": [], "translate": []}}
        for i in range(n):
            s = generate("viewpoint", "val", 2000 + 100 * b + i, bin=b)
            kind = "orbit" if s.meta["move"]["type"] == "orbit" else "shift"
            for k in ("copy", "translate"):
                per[kind][k].append(s.meta["baseline_iou"][k]["full"])
        for kind, vals in per.items():
            if vals["copy"]:
                lines.append(f"  bin {b} {kind} (n={len(vals['copy'])}): " + "; ".join(f"{k} {_summary(v)}" for k, v in vals.items()))
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# 划分、留出族、速度
# ---------------------------------------------------------------------------


def test_identities_and_scene_seeds_disjoint_across_splits():
    table = SO.solid_identity_split_table()
    for task, n in (("rotation3d", 40), ("viewpoint", 40)):
        idents = {s: set() for s in SPLITS}
        seeds = {s: set() for s in SPLITS}
        for split in SPLITS:
            for i in range(n):
                smp = generate(task, split, i)
                seeds[split].add(smp.meta["scene_seed"])
                for o in smp.meta["scene"]["objects"]:
                    ident = (o["family"], o["variant"], o["appearance"])
                    assert table[ident] == split
                    idents[split].add(ident)
        for a in range(3):
            for b in range(a + 1, 3):
                assert not idents[SPLITS[a]] & idents[SPLITS[b]], (task, a, b)
                assert not seeds[SPLITS[a]] & seeds[SPLITS[b]], (task, a, b)
        assert all(idents[s] for s in SPLITS)


def test_holdout_family_for_3d_tasks():
    for task in ("rotation3d", "viewpoint"):
        for split in ("train", "val"):
            for i in range(6):
                s = generate(task, split, i, holdout_family="cone")
                assert all(o["family"] != "cone" for o in s.meta["scene"]["objects"])
        with pytest.raises(ValueError):
            generate(task, "train", 0, holdout_family="crescent")  # 2D 形状族不是实体族
    with pytest.raises(ValueError):
        generate("occlusion", "train", 0, holdout_family="torus")


@pytest.mark.parametrize("task", ["rotation3d", "viewpoint"])
def test_speed_at_least_20_per_second_single_core(task):
    generate(task, "train", 0)
    t = time.process_time()
    n = 40
    for i in range(n):
        generate(task, "train", 3000 + i)
    rate = n / (time.process_time() - t)
    print(f"{task}: {rate:.1f} samples/s (CPU time, single process)")
    assert rate >= 20


# ---------------------------------------------------------------------------
# 可复现、数据集接口
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
CASES_3D = [(t, sp, i) for t in ("rotation3d", "viewpoint") for sp in SPLITS for i in (0, 7)]


def test_reproducible_bytes_global_state_and_hash_seed():
    ref = [generate(t, sp, i).digest() for t, sp, i in CASES_3D]
    np.random.seed(4321)
    random.seed(99)
    np.random.random(11)
    again = [generate(t, sp, i) for t, sp, i in CASES_3D]
    assert [s.digest() for s in again] == ref
    s0, s1 = generate("viewpoint", "test", 7), again[-1]
    assert s0.image.tobytes() == s1.image.tobytes() and all(s0.extras[k].tobytes() == s1.extras[k].tobytes() for k in s0.extras)
    code = "from src.synth import generate\n" f"for t, s, i in {CASES_3D!r}:\n" "    print(generate(t, s, i).digest())\n"
    for seed in ("1", "31337"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(REPO))
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr
        assert r.stdout.split() == ref


def test_dataset_class_for_3d_tasks():
    for task in ("a", "b"):
        ds = SynthDataset(task, "val", 3, start=20, grid_size=16)
        assert [s.digest() for s in ds] == [generate(task, "val", 20 + i, grid_size=16).digest() for i in range(3)]
        assert ds[0].task == {"a": "rotation3d", "b": "viewpoint"}[task] and ds[0].target_grid.shape == (16, 16)
