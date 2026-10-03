"""a 出平面三维旋转：一个实体的图像 + 文字给出的旋转（轴与角度）→ 旋转后该实体的 mask（另有深度图与深度网格）。

场景
- 一个实体，中心（体积质心）在世界原点；摆放姿态 SOLID_REST_POSES 之一、偏航角（整数度）均匀、外接半径 ROT3D_RADIUS_M。
- 相机方位角 0、俯角 ROT3D_ELEVATION_DEG（整数度）、距离 ROT3D_DISTANCE_M，看向原点：实体中心投影在画布中心。
  深度都在 [D − r, D + r] ⊂ [1.15, 3.45] m 里，在 D_MIN–D_MAX（0.5–10 m）之内；外接球的投影半径 ≤ f·r/√(D² − r²) < 0.4·min(H, W)，
  旋转后仍在外接球里，所以始终完整在画布内。
- 纯色背景（BACKGROUND_COLORS 之一），没有地板（旋转会让实体穿过地板）。

旋转
- 排除平面内旋转（任务书二选一的第一种）：轴只取相机坐标系的水平轴（x，图像的左右方向）或竖直轴（y，图像的上下方向），
  过实体中心。实体中心在光轴上，所以轴与视线严格垂直（夹角 90°）；绕视线（相机 z）的旋转从不出现。
- 角度 θ 为整数度，方向由文字给出（相机坐标系里的右手旋转为正）：
  * 水平轴，正：实体的上部转向相机；负：上部转离相机；
  * 竖直轴，正：朝向相机的一面转向左边；负：转向右边。
- 对称实体的旋转等价（与 e 同一做法）：按理想实体在这条轴上的旋转周期 P（solids.rotation_period_about；P 由姿态决定）
  把 |θ| 限制在 shapes.max_equivalent_rotation(P) = min(180, P/2) 以内，于是报告的总是最小的等价角。
  P = 0（绕连续对称轴，例如直立的圆柱绕竖直轴——俯角不为 0 时不会出现，但侧放时可能）时这条轴不能用，换姿态重来。
  所以任意旋转不变的球不当 a 的实体。档位的角度范围与 [15, 上限] 没有交集时同样重来。
- 难度 = |θ|，档 ROT3D_BINS：[15, 30)、[30, 60)、[60, 90)、[90, 180]。

目标、基线与附加输出
- target_mask：旋转后实体的 mask（= 旋转后物体 ID 图里的实体）；object_masks：同一张，K = 1。
- 平凡基线（不看真值）：
  * "copy"：当前（旋转前）的 mask；
  * "instr_affine"：由指令推得的二维仿射。正交近似下把 3D 旋转作用在当前 mask 的质心与二阶矩上：把 mask 看成位于旋转中心
    深度、没有厚度的平面图形（深度方差为 0），旋转后正交投影，质心 μ → c₀ + M(μ − c₀)，协方差 Σ → M Σ Mᵀ，
    其中 M 是旋转矩阵左上角的 2×2 块（水平轴：diag(1, cos θ)；竖直轴：diag(cos θ, 1)），c₀ = 画布中心（旋转中心的投影）。
    把这组矩变过去的仿射就是 x' = c₀ + M(x − c₀)，作用在当前 mask 上。θ = 90° 时 M 奇异，结果为空。
  * "copy_flat"（DEV-0018）：mask 同 copy，起伏取常数档（relief.FLAT_LEVEL）——"预测一个平面"，只在起伏上与 copy 不同。
- 参考上界（用了真值，不参与"在想象"的判定）："gt_affine_upper"：warp2d.moment_matched_affine 从真值按矩匹配拟合的
  二维仿射作用在当前 mask 上，作为"任何二维方法的上界"参考线。meta["baseline_roles"] 标明各基线的角色。
- extras："depth"（旋转后的深度图，float32 米，0 = 背景）、"depth_grid"（旋转后实体的 N×N 深度网格，见
  render3d.depth_grid）、"input_depth"（旋转前的深度图）。
- 起伏（DEV-0018，D28-13，任务版本 4；定义见 src/synth/relief.py）：extras["relief_grid"] 是目标的 N×N 起伏代码
  （uint8，0 背景、k + 1 = 档 k，33 个取值；半径取实体的外接半径），extras["baseline_relief_<基线名>"] 是各基线的起伏代码：
  copy = 输入视角的起伏；instr_affine、gt_affine_upper = 输入深度图连同 mask 一起做同一个仿射（最近邻）后的起伏；
  copy_flat = 常数档。基线起伏代码 > 0 的格子与该基线的网格逐位相同。meta["relief"] 记半径、档宽、目标的格数 / 平均深度 /
  标准差 / 截断格数，以及各基线的起伏误差（relief.relief_score）。
- 凸起先验基线 copy_convex（DEV-0028，D59-4）：不存进样本，由 relief.derived_baselines 从 copy 的网格现算（于是 a 的摘要与
  任务版本 4 都不变）。

跨机器逐位复现（D24 的标准：a 在任何平台上逐位相同）：见 render3d.py、warp2d.py 的说明；元数据只写整数、
2 的幂网格上的量化值与整数计数之比。
"""

from __future__ import annotations

import numpy as np

from src.synth.constants import (
    BACKGROUND_COLORS,
    MAX_ATTEMPTS,
    ROT3D_AXES,
    ROT3D_BINS,
    ROT3D_DISTANCE_M,
    ROT3D_ELEVATION_DEG,
    ROT3D_RADIUS_M,
    SCENE_QUANTUM_M,
    SOLID_REST_POSES,
)
from src.synth.dmath import rot_cs
from src.synth.render3d import (
    CameraSpec,
    Mat3,
    Scene3D,
    SolidSpec,
    depth_grid,
    depth_range_m,
    mat_mul,
    posed_vertices,
    qscene,
    render_posed,
    rot3,
    vertices_ok,
)
from src.synth.sample import (
    GenContext,
    Sample,
    SynthGenerationError,
    baseline_iou_meta,
    difficulty_meta,
    make_prediction,
)
from src.synth.seeding import coin, pick, randint, uniform
from src.synth.shapes import max_equivalent_rotation
from src.synth.solids import rotation_period_about, sample_solid_identity
from src.synth.relief import (
    RELIEF_TARGET_KEY,
    baseline_relief_key,
    flat_codes,
    relief_grid,
    relief_meta,
)
from src.synth.warp2d import Moments, affine_warp_values, mask_moments, moment_matched_affine

AXIS_PHRASES = {
    "horizontal": "the horizontal axis (the image's left-right direction)",
    "vertical": "the vertical axis (the image's up-down direction)",
}
# (轴, 符号) → 方向说明；符号 +1 = 相机坐标系里的右手旋转（x 右、y 下、z 前）
DIRECTION_PHRASES = {
    ("horizontal", 1): "its top moves toward the camera",
    ("horizontal", -1): "its top moves away from the camera",
    ("vertical", 1): "the side facing the camera moves to the left",
    ("vertical", -1): "the side facing the camera moves to the right",
}
_CAMERA_AXIS = {"horizontal": "x", "vertical": "y"}
TEMPLATES = (
    "Imagine the {obj} rotating {angle} degrees about {axis} through its center, so that {direction}. "
    "Which pixels does the {obj} cover after the rotation?",
    "The {obj} turns {angle} degrees about {axis} through its center ({direction}); the camera stays where it is. "
    "Draw the {obj} after the turn.",
    "Rotate the {obj} in your mind by {angle} degrees about {axis} passing through its center, so that {direction}. "
    "Where does the {obj} appear now?",
    "Suppose the {obj} is rotated by {angle} degrees about {axis} through its center, so that {direction}. "
    "Give the mask of the {obj} after the rotation.",
)
BASELINE_ROLES = {"copy": "trivial", "copy_flat": "trivial", "instr_affine": "trivial", "gt_affine_upper": "reference_upper_bound"}
RELIEF_DEFINITION = "src/synth/relief.py（D28-13）"  # meta["relief"]["definition"]（进摘要，不改）


def rotation_matrix(axis: str, signed_deg: int) -> Mat3:
    """相机坐标系里绕水平轴（x）或竖直轴（y）的右手旋转。"""
    return rot3(_CAMERA_AXIS[axis], signed_deg)


def instruction_affine(axis: str, signed_deg: int, height: int, width: int) -> tuple[tuple[tuple[float, float], tuple[float, float]], tuple[float, float]]:
    """由指令推得的二维仿射 x' = A·x + t（见模块说明）：A = M，t = c₀ − M·c₀，c₀ = 画布中心。"""
    c, _s = rot_cs(signed_deg)
    m = ((1.0, 0.0), (0.0, c)) if axis == "horizontal" else ((c, 0.0), (0.0, 1.0))
    cx, cy = width / 2.0, height / 2.0
    return m, (cx - m[0][0] * cx, cy - m[1][1] * cy)


def _nondegenerate(m: Moments) -> bool:
    return m.sxx > 0.0 and m.syy > 0.0 and m.sxx * m.syy - m.sxy * m.sxy > 0.0


def _attempt(ctx: GenContext, ident, lo: int, hi_incl: int):
    rng = ctx.rng
    h, w = ctx.height, ctx.width
    rest = pick(rng, SOLID_REST_POSES)
    yaw = randint(rng, 0, 359)
    elev = randint(rng, *ROT3D_ELEVATION_DEG)
    radius = qscene(uniform(rng, *ROT3D_RADIUS_M), SCENE_QUANTUM_M)
    dist = qscene(uniform(rng, *ROT3D_DISTANCE_M), SCENE_QUANTUM_M)
    axis = pick(rng, ROT3D_AXES)
    bg = tuple(pick(rng, BACKGROUND_COLORS))
    camera = CameraSpec(0, elev, dist)
    obj = SolidSpec(0, ident[0], int(ident[1]), int(ident[2]), rest, yaw, radius, (0.0, 0.0, 0.0))
    # 相机轴在实体局部坐标里的方向 = (B · R_obj)ᵀ 的对应列 = B · R_obj 的对应行
    m = mat_mul(camera.basis(), obj.rotation())
    local_axis = m[0] if axis == "horizontal" else m[1]
    period = rotation_period_about(obj.family, obj.variant, local_axis)
    top = min(hi_incl, max_equivalent_rotation(period))
    if top < lo:
        return None
    theta = randint(rng, lo, top)
    sign = 1 if coin(rng, 0.5) else -1
    scene = Scene3D(h, w, bg, camera, (obj,))
    v0 = posed_vertices(scene)
    v1 = posed_vertices(scene, extra={0: rotation_matrix(axis, sign * theta)})
    if not (vertices_ok(v0[0], h, w) and vertices_ok(v1[0], h, w)):
        return None
    before = render_posed(scene, camera, v0, want_rgb=True)
    after = render_posed(scene, camera, v1, want_rgb=False)
    m0, m1 = before.mask(0), after.mask(0)
    if not m0.any() or not m1.any() or np.array_equal(m0, m1):
        return None
    mom0, mom1 = mask_moments(m0), mask_moments(m1)
    if not (_nondegenerate(mom0) and _nondegenerate(mom1)):  # 矩匹配要求协方差正定（实体的轮廓实际上总满足）
        return None
    return {
        "scene": scene,
        "axis": axis,
        "theta": theta,
        "sign": sign,
        "period": period,
        "before": before,
        "after": after,
        "m0": m0,
        "m1": m1,
        "moments": (mom0, mom1),
    }


def generate(ctx: GenContext) -> Sample:
    rng = ctx.rng
    lo, hi = ROT3D_BINS[ctx.bin]
    hi_incl = hi if ctx.bin == len(ROT3D_BINS) - 1 else hi - 1
    # 实体身份每个样本只抽一次，重试只改姿态、轴与角度（拒绝不会改变实体族的分布）
    ident = sample_solid_identity(rng, ctx.split, ctx.holdout_family, rotatable_only=True)
    res = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = _attempt(ctx, ident, lo, hi_incl)
        if res is not None:
            break
    if res is None:
        raise SynthGenerationError(f"rotation3d {ctx.split}#{ctx.index} 档 {ctx.bin}：{MAX_ATTEMPTS} 次尝试都不合格")
    h, w = ctx.height, ctx.width
    scene: Scene3D = res["scene"]
    obj = scene.objects[0]
    axis, theta, sign = res["axis"], res["theta"], res["sign"]
    desc = obj.description
    direction = DIRECTION_PHRASES[(axis, sign)]
    template_id = int(rng.integers(len(TEMPLATES)))
    instruction = TEMPLATES[template_id].format(obj=desc, angle=theta, axis=AXIS_PHRASES[axis], direction=direction)

    m0, m1 = res["m0"], res["m1"]
    n = ctx.grid_size
    before, after = res["before"], res["after"]
    a_instr, t_instr = instruction_affine(axis, sign * theta, h, w)
    # 二维仿射基线的起伏：把输入视角的深度图连同 mask 一起做同一个仿射（最近邻取样），再按目标的规则算起伏
    fit = moment_matched_affine(m0, m1, moments=res["moments"], values=before.depth_units)
    m_instr, d_instr = affine_warp_values(m0, before.depth_units, a_instr, t_instr)
    m_fit, d_fit = fit.mask, fit.values
    target = make_prediction(m1, n)
    copy = make_prediction(m0, n)
    p_instr, p_fit = make_prediction(m_instr, n), make_prediction(m_fit, n)
    baselines = {
        "copy": copy,
        "copy_flat": copy,
        "instr_affine": p_instr,
        "gt_affine_upper": p_fit,
    }
    # 网格已由 make_prediction 算好，传进去省掉重算（DEV-0028 提速，结果逐位不变）
    rel_target = relief_grid(after.depth_units, m1, n, obj.radius, target.grid)
    rel_copy = relief_grid(before.depth_units, m0, n, obj.radius, copy.grid)
    rel_base = {
        "copy": rel_copy,
        "copy_flat": None,
        "instr_affine": relief_grid(d_instr, m_instr, n, obj.radius, p_instr.grid),
        "gt_affine_upper": relief_grid(d_fit, m_fit, n, obj.radius, p_fit.grid),
    }
    base_codes = {k: (flat_codes(copy.grid) if v is None else v.codes) for k, v in rel_base.items()}
    extras = {
        "depth": after.depth_m(),
        "depth_grid": depth_grid(after.depth_units, m1, n, target.grid),
        "input_depth": before.depth_m(),
        RELIEF_TARGET_KEY: rel_target.codes,
        **{baseline_relief_key(k): v for k, v in base_codes.items()},
    }
    meta_relief = relief_meta(RELIEF_DEFINITION, obj.radius, rel_target, rel_copy, base_codes)
    meta = ctx.base_meta()
    meta.update(
        {
            "difficulty": difficulty_meta("rotation_deg", theta, ctx.bin, ROT3D_BINS, upper_closed_last=True),
            "template_id": template_id,
            "scene": scene.to_dict(),
            "rotation": {
                "axis": axis,
                "camera_axis": _CAMERA_AXIS[axis],
                "angle_deg": theta,
                "sign": sign,
                "signed_angle_deg": sign * theta,
                "direction": direction,
                "axis_to_line_of_sight_deg": 90,
                "symmetry_period_deg": res["period"],
                "max_equivalent_deg": max_equivalent_rotation(res["period"]),
            },
            "target_obj_id": 0,
            "target_description": desc,
            "depth_range_m": depth_range_m(after.depth_units),
            "input_depth_range_m": depth_range_m(before.depth_units),
            "instr_affine": {"matrix": [list(a_instr[0]), list(a_instr[1])], "offset_px": list(t_instr)},
            "gt_affine_upper": {"rotation_deg": fit.rotation_deg, "reflect": fit.reflect},
            "baseline_roles": dict(BASELINE_ROLES),
            "baseline_iou": baseline_iou_meta(baselines, target.mask, target.grid),
            "relief": meta_relief,
            "attempts": attempt,
        }
    )
    return Sample(
        task=ctx.task,
        split=ctx.split,
        index=ctx.index,
        image=before.rgb,
        instruction=instruction,
        target_mask=target.mask,
        target_grid=target.grid,
        object_masks=m1[None].copy(),
        baselines=baselines,
        meta=meta,
        extras=extras,
    )
