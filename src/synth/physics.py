"""d 短时物理（pymunk 2D 刚体，侧视）：初始帧 + 动作文字 + 步数 k → k 步之后各物体的 mask。

世界
- 画布即世界，y 向下；地面是静态长方体，上表面在 y_g = round(PHYSICS_GROUND_FRAC · H)，左右远超画布。
- 长度单位 = 画布宽 W（像素）：重力 PHYSICS_GRAVITY·W px/s² 向下；文字里的速度以"image widths per second"给出。
- 刚体只有盒子（rectangle 族）与圆（circle 族）；质量 = 面积 × 密度，质心在单位形状的原点，
  碰撞形状与渲染用的多边形相同（圆的碰撞形状是真圆，渲染是 128 边内接多边形，差 ≤ 0.03 px）。
- 摩擦、弹性、密度都是全局常数（图像里看不出的量不随样本变化），见 constants。

时间
- 一步 = PHYSICS_STEP_SECONDS = 1/30 秒，内部细分为 PHYSICS_SUBSTEPS = 10 个子步（pymunk.step(1/300)）。
- 难度 = k ∈ PHYSICS_K_BINS = {5, 10, 20, 40}，每个 k 一档。

动作（初速度由文字给出，数值是整数/100 W/s，与仿真用的值完全相同）
- drop：一个物体从静止释放，最低点离支撑面 PHYSICS_DROP_GAP·W；以 PHYSICS_STACK_PROB 的概率正下方有一个静止物体。
- push：地面上一个静止物体获得水平冲量 J = m·v（等价于水平初速 v）；以 PHYSICS_OBSTACLE_PROB 的概率前方有障碍物。
- collide：地面上两个物体相向运动。
- 另有 0–2 个与动作无关的静止物体。所有物体在第 0 步和第 k 步都完整在画布内，不相互重叠（否则重来）。

目标与基线
- target_mask：指令点名的物体（drop/push 的施动物体，collide 里两者之一）在第 k 步的 mask；
  object_masks：全部物体在第 k 步的 mask。
- 平凡基线："copy"（目标的初始 mask）；"const_velocity"（目标按文字给出的初速度匀速平移 k 步；drop 的初速为 0，等于 copy）。

合理性（生成时作为拒绝条件，并记进元数据；tests/test_synth_physics.py 另在不设拒绝的场景上检查）
- 任一接触点的穿透深度 ≤ PHYSICS_PENETRATION_TOL_PX（每步检查 pymunk 的接触距离）；
- 第 k 步两两多边形的重叠面积 ≤ PHYSICS_OVERLAP_TOL × 较小者面积（shapely，独立于 pymunk）。

确定性：同一进程、同一 pymunk 构建下逐位相同（按同样顺序加入物体、同样的步长序列）。
第 k 步的位姿量化后再渲染（坐标 1/256 px、角度 1/256 度）。跨机器（arm64 开发机 vs x86 服务器）是否逐位相同
取决于 Chipmunk 的编译浮点行为，由 tests/test_synth_repro.py 的钉住摘要在服务器上核对。
"""

from __future__ import annotations

import math

import numpy as np
import pymunk
import shapely

from src.synth.constants import (
    ANGLE_QUANTUM_DEG,
    BACKGROUND_COLORS,
    CANVAS_MARGIN_PX,
    MAX_ATTEMPTS,
    PHYSICS_ACTIONS,
    PHYSICS_COLLIDE_GAP,
    PHYSICS_DENSITY,
    PHYSICS_DROP_GAP,
    PHYSICS_ELASTICITY_GROUND,
    PHYSICS_ELASTICITY_OBJECT,
    PHYSICS_FAMILIES,
    PHYSICS_FRICTION_GROUND,
    PHYSICS_FRICTION_OBJECT,
    PHYSICS_GRAVITY,
    PHYSICS_GROUND_COLOR,
    PHYSICS_GROUND_FRAC,
    PHYSICS_ITERATIONS,
    PHYSICS_K_BINS,
    PHYSICS_MIN_GAP,
    PHYSICS_NUM_BYSTANDERS,
    PHYSICS_OBSTACLE_GAP,
    PHYSICS_OBSTACLE_PROB,
    PHYSICS_OVERLAP_TOL,
    PHYSICS_PENETRATION_TOL_PX,
    PHYSICS_RADIUS,
    PHYSICS_SPEED_HUNDREDTHS,
    PHYSICS_STACK_PROB,
    PHYSICS_STEP_SECONDS,
    PHYSICS_SUBSTEPS,
)
from src.synth.sample import (
    GenContext,
    Prediction,
    Sample,
    SynthGenerationError,
    baseline_iou_meta,
    difficulty_meta,
    make_prediction,
)
from src.synth.scene import Background, Ground, ObjectSpec, SceneSpec, object_masks, render_image
from src.synth.seeding import coin, pick, randint, uniform
from src.synth.shapes import describe, polygon_inside, qcoord, quantize, rasterize, sample_identity, unit_polygon

TEMPLATES = (
    "{action} Where will the {target} be after {k} time steps? One time step is 1/30 of a second.",
    "{action} Predict the mask of the {target} {k} steps later (each step lasts 1/30 s).",
    "{action} Show where the {target} will be after {k} steps of 1/30 second each.",
    "{action} After {k} steps (1/30 s per step), which pixels will the {target} occupy?",
)
_DIR = {1: "to the right", -1: "to the left"}


def ground_y(height: int) -> float:
    return float(round(PHYSICS_GROUND_FRAC * height))


# ---------------------------------------------------------------------------
# 仿真
# ---------------------------------------------------------------------------


class Simulation:
    """一个场景的 pymunk 世界。物体按 scene.objects 的顺序加入；velocities: obj_id -> (vx, vy) px/s。"""

    def __init__(self, scene: SceneSpec, velocities: dict[int, tuple[float, float]] | None = None) -> None:
        if scene.ground is None:
            raise ValueError("物理场景必须有地面")
        self.scene = scene
        w, h = scene.width, scene.height
        sp = pymunk.Space()
        sp.gravity = (0.0, PHYSICS_GRAVITY * w)
        sp.iterations = PHYSICS_ITERATIONS
        yg = scene.ground.y
        ground = pymunk.Poly(sp.static_body, [(-2.0 * w, yg), (3.0 * w, yg), (3.0 * w, yg + h), (-2.0 * w, yg + h)])
        ground.friction = PHYSICS_FRICTION_GROUND
        ground.elasticity = PHYSICS_ELASTICITY_GROUND
        sp.add(ground)
        self.bodies: list[pymunk.Body] = []
        shape_index: dict[pymunk.Shape, int] = {ground: -1}
        for k, o in enumerate(scene.objects):
            if o.family not in PHYSICS_FAMILIES:
                raise ValueError(f"物理场景只支持 {PHYSICS_FAMILIES}，收到 {o.family}")
            if o.family == "circle":
                mass = PHYSICS_DENSITY * math.pi * o.radius * o.radius
                body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0.0, o.radius))
                shp = pymunk.Circle(body, o.radius)
            else:
                verts = [(float(x), float(y)) for x, y in unit_polygon(o.family, o.variant) * o.radius]
                area = abs(shapely.polygons(np.asarray(verts)).area)
                mass = PHYSICS_DENSITY * area
                body = pymunk.Body(mass, pymunk.moment_for_poly(mass, verts))
                shp = pymunk.Poly(body, verts)
            body.position = (o.cx, o.cy)
            body.angle = math.radians(o.angle)
            shp.friction = PHYSICS_FRICTION_OBJECT
            shp.elasticity = PHYSICS_ELASTICITY_OBJECT
            sp.add(body, shp)
            shape_index[shp] = k
            self.bodies.append(body)
        for oid, (vx, vy) in (velocities or {}).items():
            k = next(i for i, o in enumerate(scene.objects) if o.obj_id == oid)
            b = self.bodies[k]
            b.apply_impulse_at_local_point((b.mass * vx, b.mass * vy), (0.0, 0.0))  # 冲量 J = m·v，作用在质心
        self.touched = [False] * len(self.bodies)  # 曾与另一个物体（非地面）接触

        def _begin(arb, _space, _data):
            i, j = (shape_index.get(s, -1) for s in arb.shapes)
            if i >= 0 and j >= 0:
                self.touched[i] = self.touched[j] = True

        sp.on_collision(None, None, begin=_begin)
        self.space = sp
        self.steps_done = 0
        self.max_penetration = 0.0  # 各步末接触点穿透深度的最大值（px，正数）

    def step(self, n: int = 1) -> None:
        dt = PHYSICS_STEP_SECONDS / PHYSICS_SUBSTEPS
        pens = []

        def _collect(arb):
            for p in arb.contact_point_set.points:
                pens.append(-p.distance)

        for _ in range(n):
            for _ in range(PHYSICS_SUBSTEPS):
                self.space.step(dt)
            self.steps_done += 1
            pens.clear()
            for b in self.bodies:
                b.each_arbiter(_collect)
            if pens:
                self.max_penetration = max(self.max_penetration, max(pens))

    def poses(self) -> list[tuple[float, float, float]]:
        """每个物体当前的 (cx, cy, angle 度)，量化后（与渲染一致）。"""
        return [
            (qcoord(b.position.x), qcoord(b.position.y), quantize(math.degrees(b.angle), ANGLE_QUANTUM_DEG)) for b in self.bodies
        ]

    def scene_now(self) -> SceneSpec:
        objs = tuple(o.moved(cx=cx, cy=cy, angle=a) for o, (cx, cy, a) in zip(self.scene.objects, self.poses()))
        return SceneSpec(self.scene.height, self.scene.width, self.scene.background, objs, self.scene.ground)

    def speeds(self) -> list[float]:
        return [math.sqrt(b.velocity.x ** 2 + b.velocity.y ** 2) for b in self.bodies]


def max_overlap_fraction(scene: SceneSpec) -> float:
    """两两多边形重叠面积 / 较小者面积 的最大值（shapely，独立于 pymunk）。"""
    polys = [shapely.polygons(o.polygon()) for o in scene.objects]
    worst = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if shapely.intersects(polys[i], polys[j]):
                a = shapely.intersection(polys[i], polys[j]).area
                worst = max(worst, a / min(polys[i].area, polys[j].area))
    return worst


# ---------------------------------------------------------------------------
# 场景构造
# ---------------------------------------------------------------------------


def _half_extents(family: str, variant: int, radius: float, angle: float) -> tuple[float, float, float]:
    """旋转后形状相对质心的 (左伸, 右伸, 下伸)，像素。"""
    u = unit_polygon(family, variant) * radius
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    x = u[:, 0] * c - u[:, 1] * s
    y = u[:, 0] * s + u[:, 1] * c
    return float(-x.min()), float(x.max()), float(y.max())


class _Layout:
    """地面上已占用的水平区间，保证新放的物体之间有最小间隙。"""

    def __init__(self, width: int) -> None:
        self.w = width
        self.gap = PHYSICS_MIN_GAP * width
        self.spans: list[tuple[float, float]] = []

    def free(self, lo: float, hi: float) -> bool:
        if lo < CANVAS_MARGIN_PX or hi > self.w - CANVAS_MARGIN_PX:
            return False
        return all(hi + self.gap <= a or lo - self.gap >= b for a, b in self.spans)

    def add(self, lo: float, hi: float) -> None:
        self.spans.append((lo, hi))


def _new_identity(rng, ctx: GenContext, taken: set[str]):
    for _ in range(100):
        ident = sample_identity(rng, ctx.split, PHYSICS_FAMILIES, ctx.holdout_family)
        if describe(*ident) not in taken:
            return ident
    raise SynthGenerationError("物理场景：找不到简称不重复的身份")  # pragma: no cover


def _resting(ident, radius: float, cx: float, yg: float) -> tuple:
    """地面上静止的物体（角度 0，最低点贴地面上表面）。"""
    _, _, down = _half_extents(ident[0], ident[1], radius, 0.0)
    return (ident, qcoord(cx), qcoord(yg - down), radius, 0.0)


def _place_resting(rng, layout: _Layout, ident, radius: float, yg: float, tries: int = 20):
    left, right, _ = _half_extents(ident[0], ident[1], radius, 0.0)
    for _ in range(tries):
        cx = uniform(rng, left + CANVAS_MARGIN_PX, layout.w - right - CANVAS_MARGIN_PX)
        if layout.free(cx - left, cx + right):
            layout.add(cx - left, cx + right)
            return _resting(ident, radius, cx, yg)
    return None


def _speed(rng) -> tuple[int, float]:
    n = randint(rng, *PHYSICS_SPEED_HUNDREDTHS)
    return n, n / 100.0


def _attempt(ctx: GenContext, action: str, k: int):
    rng = ctx.rng
    h, w = ctx.height, ctx.width
    yg = ground_y(h)
    layout = _Layout(w)
    taken: set[str] = set()
    placed: list[tuple] = []  # (ident, cx, cy, radius, angle)
    roles: list[str] = []
    vel: dict[int, tuple[int, int]] = {}  # placed 下标 -> (方向, 速度百分数)

    def radius() -> float:
        return qcoord(uniform(rng, *PHYSICS_RADIUS) * w)

    def ident_new():
        i = _new_identity(rng, ctx, taken)
        taken.add(describe(*i))
        return i

    if action == "drop":
        d_ident, d_r = ident_new(), radius()
        d_angle = quantize(uniform(rng, 0.0, 360.0), ANGLE_QUANTUM_DEG)
        dl, dr, dd = _half_extents(d_ident[0], d_ident[1], d_r, d_angle)
        support = yg
        if coin(rng, PHYSICS_STACK_PROB):
            s_ident = ident_new()
            s_obj = _place_resting(rng, layout, s_ident, radius(), yg)
            if s_obj is None:
                return None
            placed.append(s_obj)
            roles.append("support")
            s_left, s_right, s_down = _half_extents(s_ident[0], s_ident[1], s_obj[3], 0.0)
            s_up = _half_extents(s_ident[0], s_ident[1], s_obj[3], 180.0)[2]  # 上伸 = 旋转 180° 后的下伸
            dcx = s_obj[1] + uniform(rng, -0.5, 0.5) * min(s_left, s_right)
            support = s_obj[2] - s_up
        else:
            dcx = uniform(rng, dl + CANVAS_MARGIN_PX, w - dr - CANVAS_MARGIN_PX)
            if not layout.free(dcx - dl, dcx + dr):
                return None
            layout.add(dcx - dl, dcx + dr)
        gap = uniform(rng, *PHYSICS_DROP_GAP) * w
        placed.append((d_ident, qcoord(dcx), qcoord(support - gap - dd), d_r, d_angle))
        roles.append("dropped")
        target_k = len(placed) - 1
    elif action == "push":
        p_ident, p_r = ident_new(), radius()
        p_obj = _place_resting(rng, layout, p_ident, p_r, yg)
        if p_obj is None:
            return None
        direction = 1 if coin(rng, 0.5) else -1
        n, _ = _speed(rng)
        placed.append(p_obj)
        roles.append("pushed")
        vel[0] = (direction, n)
        target_k = 0
        if coin(rng, PHYSICS_OBSTACLE_PROB):
            o_ident, o_r = ident_new(), radius()
            pl, pr, _ = _half_extents(p_ident[0], p_ident[1], p_r, 0.0)
            ol, orr, _ = _half_extents(o_ident[0], o_ident[1], o_r, 0.0)
            g = uniform(rng, *PHYSICS_OBSTACLE_GAP) * w
            ocx = p_obj[1] + (pr + g + ol if direction > 0 else -(pl + g + orr))
            if not layout.free(ocx - ol, ocx + orr):
                return None
            layout.add(ocx - ol, ocx + orr)
            placed.append(_resting(o_ident, o_r, ocx, yg))
            roles.append("obstacle")
    elif action == "collide":
        a_ident, a_r = ident_new(), radius()
        b_ident, b_r = ident_new(), radius()
        al, ar, _ = _half_extents(a_ident[0], a_ident[1], a_r, 0.0)
        bl, br, _ = _half_extents(b_ident[0], b_ident[1], b_r, 0.0)
        g = uniform(rng, *PHYSICS_COLLIDE_GAP) * w
        acx = uniform(rng, al + CANVAS_MARGIN_PX, w - (ar + g + bl + br) - CANVAS_MARGIN_PX)
        bcx = acx + ar + g + bl
        if not (layout.free(acx - al, acx + ar) and layout.free(bcx - bl, bcx + br)):
            return None
        layout.add(acx - al, acx + ar)
        layout.add(bcx - bl, bcx + br)
        placed.append(_resting(a_ident, a_r, acx, yg))
        placed.append(_resting(b_ident, b_r, bcx, yg))
        roles += ["collider", "collider"]
        vel[0] = (1, _speed(rng)[0])
        vel[1] = (-1, _speed(rng)[0])
        target_k = 0 if coin(rng, 0.5) else 1
    else:  # pragma: no cover
        raise ValueError(action)

    for _ in range(randint(rng, *PHYSICS_NUM_BYSTANDERS)):
        ob = _place_resting(rng, layout, ident_new(), radius(), yg)
        if ob is not None:
            placed.append(ob)
            roles.append("bystander")

    # obj_id 打乱；z 一律 0（物体互不重叠）
    perm = list(range(len(placed)))
    for i in range(len(perm) - 1, 0, -1):
        j = int(rng.integers(i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    objs = [None] * len(placed)
    for idx, (ident, cx, cy, r, a) in enumerate(placed):
        objs[perm[idx]] = ObjectSpec(perm[idx], ident[0], int(ident[1]), int(ident[2]), cx, cy, r, a, 0)
    bg_color = tuple(pick(rng, BACKGROUND_COLORS))
    scene = SceneSpec(h, w, Background("solid", bg_color, bg_color), tuple(objs), Ground(yg, tuple(PHYSICS_GROUND_COLOR)))
    velocities = {perm[idx]: (dirn * n * w / 100.0, 0.0) for idx, (dirn, n) in vel.items()}

    # 第 0 步：完整在画布内、互不重叠
    if not all(polygon_inside(o.polygon(), h, w, CANVAS_MARGIN_PX) for o in scene.objects):
        return None
    if max_overlap_fraction(scene) > 0.0:
        return None

    sim = Simulation(scene, velocities)
    sim.step(k)
    final = sim.scene_now()
    if not all(polygon_inside(o.polygon(), h, w, CANVAS_MARGIN_PX) for o in final.objects):
        return None
    if sim.max_penetration > PHYSICS_PENETRATION_TOL_PX:
        return None
    overlap = max_overlap_fraction(final)
    if overlap > PHYSICS_OVERLAP_TOL:
        return None
    return {
        "scene": scene,
        "final": final,
        "sim": sim,
        "target": perm[target_k],
        "roles": {perm[i]: roles[i] for i in range(len(placed))},
        "vel_text": {perm[idx]: (dirn, n) for idx, (dirn, n) in vel.items()},
        "velocities": velocities,
        "overlap": overlap,
    }


def _action_text(action: str, scene: SceneSpec, target: int, roles: dict, vel_text: dict) -> str:
    def name(oid):
        return f"the {scene.objects[oid].description}"

    def cap(text: str) -> str:
        return text[:1].upper() + text[1:]

    if action == "drop":
        return f"{cap(name(target))} is released from rest."
    if action == "push":
        (oid, (dirn, n)), = vel_text.items()
        return f"{cap(name(oid))} is pushed {_DIR[dirn]} and starts moving at {n / 100:.2f} image widths per second."
    (a, (da, na)), (b, (db, nb)) = sorted(vel_text.items(), key=lambda kv: -kv[1][0])  # 先说向右的那个
    return (
        f"{cap(name(a))} moves {_DIR[da]} at {na / 100:.2f} image widths per second and {name(b)} moves "
        f"{_DIR[db]} at {nb / 100:.2f} image widths per second, toward each other."
    )


def generate(ctx: GenContext) -> Sample:
    rng = ctx.rng
    k = PHYSICS_K_BINS[ctx.bin]
    action = pick(rng, PHYSICS_ACTIONS)
    res = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = _attempt(ctx, action, k)
        if res is not None:
            break
    if res is None:
        raise SynthGenerationError(f"physics {ctx.split}#{ctx.index} k={k} {action}：{MAX_ATTEMPTS} 次尝试都不合格")
    scene: SceneSpec = res["scene"]
    final: SceneSpec = res["final"]
    sim: Simulation = res["sim"]
    t = res["target"]
    h, w = ctx.height, ctx.width

    init_masks = object_masks(scene)
    final_masks = object_masks(final)
    image = render_image(scene, init_masks)

    template_id = int(rng.integers(len(TEMPLATES)))
    action_text = _action_text(action, scene, t, res["roles"], res["vel_text"])
    instruction = TEMPLATES[template_id].format(action=action_text, target=scene.objects[t].description, k=k)

    vx, vy = res["velocities"].get(t, (0.0, 0.0))
    t_obj = scene.objects[t]
    moved = t_obj.moved(cx=t_obj.cx + vx * k * PHYSICS_STEP_SECONDS, cy=t_obj.cy + vy * k * PHYSICS_STEP_SECONDS)
    target = make_prediction(final_masks[t], ctx.grid_size)
    baselines: dict[str, Prediction] = {
        "copy": make_prediction(init_masks[t], ctx.grid_size),
        "const_velocity": make_prediction(rasterize(moved.polygon(), h, w), ctx.grid_size),
    }

    untouched_drift = [
        math.sqrt((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2)
        for i, (a, b) in enumerate(zip(scene.objects, final.objects))
        if i not in res["velocities"] and res["roles"][i] != "dropped" and not sim.touched[i]
    ]
    meta = ctx.base_meta()
    meta.update(
        {
            "difficulty": difficulty_meta("steps", k, ctx.bin, PHYSICS_K_BINS, closed=True),
            "template_id": template_id,
            "scene": scene.to_dict(),
            "action": action,
            "action_text": action_text,
            "target_obj_id": t,
            "target_description": scene.objects[t].description,
            "object_roles": [res["roles"][i] for i in range(len(scene.objects))],
            "k": k,
            "seconds": k * PHYSICS_STEP_SECONDS,
            "step_seconds": PHYSICS_STEP_SECONDS,
            "substeps": PHYSICS_SUBSTEPS,
            "initial_velocity_px_per_s": {str(i): list(v) for i, v in sorted(res["velocities"].items())},
            "final_poses": [{"obj_id": o.obj_id, "cx": o.cx, "cy": o.cy, "angle": o.angle} for o in final.objects],
            "target_displacement_px": math.sqrt((final.objects[t].cx - t_obj.cx) ** 2 + (final.objects[t].cy - t_obj.cy) ** 2),
            "touched": [bool(x) for x in sim.touched],
            "max_penetration_px": sim.max_penetration,
            "max_overlap_fraction": res["overlap"],
            "untouched_max_drift_px": max(untouched_drift, default=0.0),
            "baseline_iou": baseline_iou_meta(baselines, target.mask, target.grid),
            "attempts": attempt,
            "provenance": {"pymunk": pymunk.version, "chipmunk": pymunk.chipmunk_version},
        }
    )
    return Sample(
        task=ctx.task,
        split=ctx.split,
        index=ctx.index,
        image=image,
        instruction=instruction,
        target_mask=target.mask,
        target_grid=target.grid,
        object_masks=final_masks,
        baselines=baselines,
        meta=meta,
    )
