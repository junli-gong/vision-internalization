"""b 视角变化：多物体场景的图像 + 文字给出的相机移动 → 新视角下被点名物体的 mask（另存两个视角的物体 ID 图与深度图）。

场景
- 圆形地板：世界原点为中心、y = 0、半径 VIEW_FLOOR_RADIUS_M（折线化为 VIEW_FLOOR_SEGMENTS 边形），纯色、平面着色；纯色背景。
- 3–4 个实体（VIEW_NUM_OBJECTS），身份从该划分的实体身份池抽（球也可以），"颜色 + 名词"的简称在场景里互不相同；
  摆放姿态、偏航角（整数度）均匀，外接半径 VIEW_RADIUS_M；最低点贴在地板上（中心高度向上取到 SCENE_QUANTUM_M），
  中心的水平位置均匀落在半径 VIEW_PLACE_RADIUS_M 的圆内；两两外接半径之和加 VIEW_MIN_GAP_M 以内不放，所以互不穿插。
- 相机看向地板中心：方位角（整数度）均匀、俯角 VIEW_ELEVATION_DEG（整数度）、距离 VIEW_DISTANCE_M。

相机移动（文字里的数就是生成用的精确值）
- orbit：绕地板中心水平环绕 θ 度（整数），向右（相机向它自己的右边移动，方位角增大）或向左；高度、到中心的距离不变，仍看向中心。
- 平移（不转向）：向右 / 左 d 米（相机的水平轴）、向上 / 下 d 米（竖直）、向前 / 后 d 米（沿视线）；d = 整数厘米 / 100。
  平移向量 = d × 相机轴（dmath 的三角函数），分量量化到 VERTEX_QUANTUM_M。
- 每个样本先定移动类型（VIEW_ORBIT_PROB 的概率环绕，否则 6 种平移均分），重试不改类型。
- 难度：第 k 档 = 环绕角 VIEW_ORBIT_BINS[k] 或平移量 VIEW_SHIFT_BINS_CM[k]（前三档 [lo, hi)，最后一档两端都含）。

合格条件（不满足就整个场景重来）
- 两个视角里：相机高度 ≥ VIEW_MIN_CAMERA_HEIGHT_M；全部物体与地板顶点的深度在 D_MIN–D_MAX 内（留 1 cm 余量）。
- 被点名的物体：两个视角里都完整在画布内（顶点投影离边缘 ≥ CANVAS_MARGIN_PX），可见像素 ≥ VIEW_MIN_VISIBLE_PX，
  且可见像素 / 自身完整轮廓的像素 ≥ VIEW_MIN_VISIBLE_FRAC；新视角的 mask 与旧视角不同。
  满足条件的物体里按随机顺序取第一个当目标。

目标、基线与附加输出
- target_mask：新视角下目标的可见 mask（= 新视角物体 ID 图里的目标）；object_masks：新视角下全部物体的可见 mask。
- 平凡基线："copy"（旧视角里目标的可见 mask）；"translate"（把它按场景质心——全部物体中心的平均——投影的位移平移，
  位移四舍五入到整像素）。环绕时场景质心接近环绕中心，这个位移很小，translate 接近 copy。
- extras："id_map"（新视角的物体 ID 图，uint8：0 没有表面、k + 1 物体 k、FLOOR_ID 地板）、"input_id_map"（旧视角）、
  "depth"、"input_depth"（float32 米，0 = 没有表面）、"depth_grid"（新视角里目标的 N×N 深度网格）。

跨机器逐位复现（D24 的标准：b 在任何平台上逐位相同）：见 render3d.py、warp2d.py 的说明。
"""

from __future__ import annotations

import numpy as np

from src.synth.constants import (
    BACKGROUND_COLORS,
    CANVAS_MARGIN_PX,
    MAX_ATTEMPTS,
    SCENE_QUANTUM_M,
    SOLID_REST_POSES,
    VERTEX_QUANTUM_M,
    VIEW_DISTANCE_M,
    VIEW_ELEVATION_DEG,
    VIEW_FLOOR_COLOR,
    VIEW_FLOOR_RADIUS_M,
    VIEW_FLOOR_SEGMENTS,
    VIEW_MIN_CAMERA_HEIGHT_M,
    VIEW_MIN_GAP_M,
    VIEW_MIN_VISIBLE_FRAC,
    VIEW_MIN_VISIBLE_PX,
    VIEW_MOVES,
    VIEW_NUM_OBJECTS,
    VIEW_ORBIT_BINS,
    VIEW_ORBIT_PROB,
    VIEW_PLACE_RADIUS_M,
    VIEW_RADIUS_M,
    VIEW_SHIFT_BINS_CM,
)
from src.synth.render3d import (
    CameraSpec,
    FloorSpec,
    Scene3D,
    SolidSpec,
    ceil_to,
    depth_grid,
    depth_range_m,
    floor_vertices,
    inside_canvas,
    posed_vertices,
    project_float,
    qscene,
    render_posed,
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
from src.synth.solids import REST_MATRICES, describe_solid, sample_solid_identity, unit_mesh
from src.synth.warp2d import shift_mask

TEMPLATES = (
    "{move} Where is the {target} in the new view?",
    "{move} Draw the {target} as the camera now sees it.",
    "{move} Which pixels does the {target} cover in the new view?",
    "{move} Give the mask of the {target} as seen from the new camera position.",
)
BASELINE_ROLES = {"copy": "trivial", "translate": "trivial"}


def move_sentence(move: str, amount) -> str:
    """移动的文字。orbit：amount = 带符号的整数度（正 = 向右）；平移：amount = 整数厘米。"""
    if move == "orbit":
        side = "right" if amount > 0 else "left"
        return (
            f"The camera circles {abs(amount)} degrees to the {side} around the center of the round floor, "
            "staying at the same height and the same distance from that center and still looking at it."
        )
    d = f"{amount / 100:.2f}"
    if move in ("right", "left"):
        return f"The camera slides {d} m to its {move}, without turning."
    if move == "up":
        return f"The camera rises {d} m straight up, without turning."
    if move == "down":
        return f"The camera drops {d} m straight down, without turning."
    if move in ("forward", "backward"):
        return f"The camera moves {d} m {move} along its line of sight, without turning."
    raise ValueError(f"未知的相机移动 {move!r}")


def move_camera(camera: CameraSpec, move: str, amount) -> CameraSpec:
    """移动后的相机。orbit：amount = 带符号的整数度；平移：amount = 整数厘米（d = amount / 100 米）。"""
    if move == "orbit":
        return CameraSpec((camera.azimuth_deg + int(amount)) % 360, camera.elevation_deg, camera.distance, camera.target, camera.shift)
    d = amount / 100.0
    right, _down, fwd = camera.basis()
    vec = {
        "right": right,
        "left": tuple(-x for x in right),
        "up": (0.0, 1.0, 0.0),
        "down": (0.0, -1.0, 0.0),
        "forward": fwd,
        "backward": tuple(-x for x in fwd),
    }[move]
    shift = tuple(qscene(camera.shift[i] + d * vec[i], VERTEX_QUANTUM_M) for i in range(3))
    return CameraSpec(camera.azimuth_deg, camera.elevation_deg, camera.distance, camera.target, shift)


def _resting_height(family: str, variant: int, rest: str, radius: float) -> float:
    """最低点贴地时中心的高度（米），向上取到 SCENE_QUANTUM_M：姿态旋转的 y 行是精确的 0/±1，偏航不改变高度。"""
    v, _ = unit_mesh(family, variant)
    up = REST_MATRICES[rest][1]
    lowest = float(((v[:, 0] * up[0] + v[:, 1] * up[1]) + v[:, 2] * up[2]).min())
    return ceil_to(-lowest * radius, SCENE_QUANTUM_M)


def _place_objects(rng, ctx: GenContext, n: int) -> list[SolidSpec]:
    taken: set[str] = set()
    placed: list[SolidSpec] = []
    for _ in range(n):
        for _try in range(30):
            ident = sample_solid_identity(rng, ctx.split, ctx.holdout_family)
            desc = describe_solid(*ident)
            if desc in taken:
                continue
            rest = pick(rng, SOLID_REST_POSES)
            yaw = randint(rng, 0, 359)
            radius = qscene(uniform(rng, *VIEW_RADIUS_M), SCENE_QUANTUM_M)
            x = qscene(uniform(rng, -VIEW_PLACE_RADIUS_M, VIEW_PLACE_RADIUS_M), SCENE_QUANTUM_M)
            z = qscene(uniform(rng, -VIEW_PLACE_RADIUS_M, VIEW_PLACE_RADIUS_M), SCENE_QUANTUM_M)
            if x * x + z * z > VIEW_PLACE_RADIUS_M * VIEW_PLACE_RADIUS_M:
                continue
            ok = True
            for o in placed:
                gap = o.radius + radius + VIEW_MIN_GAP_M
                dx, dz = x - o.center[0], z - o.center[2]
                if dx * dx + dz * dz < gap * gap:
                    ok = False
                    break
            if not ok:
                continue
            y = _resting_height(ident[0], ident[1], rest, radius)
            placed.append(SolidSpec(len(placed), ident[0], int(ident[1]), int(ident[2]), rest, yaw, radius, (x, y, z)))
            taken.add(desc)
            break
    return placed


def _shuffled(rng, n: int) -> list[int]:
    """Fisher–Yates，只用 integers()。"""
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(i + 1))
        p[i], p[j] = p[j], p[i]
    return p


def scene_centroid(scene: Scene3D) -> tuple[float, float, float]:
    """全部物体中心的平均（按 obj_id 顺序逐个相加，再除以个数）。"""
    n = len(scene.objects)
    sx = sy = sz = 0.0
    for o in scene.objects:
        sx, sy, sz = sx + o.center[0], sy + o.center[1], sz + o.center[2]
    return (sx / n, sy / n, sz / n)


def centroid_shift_px(scene: Scene3D, cam0: CameraSpec, cam1: CameraSpec) -> tuple[int, int]:
    """场景质心在两个视角里投影位置之差，四舍五入（取偶）到整像素：(dx, dy)。"""
    c = scene_centroid(scene)
    uv = []
    for cam in (cam0, cam1):
        b, pos = cam.basis(), cam.position()
        rel = (c[0] - pos[0], c[1] - pos[1], c[2] - pos[2])
        p = np.array([[(b[i][0] * rel[0] + b[i][1] * rel[1]) + b[i][2] * rel[2] for i in range(3)]], dtype=np.float64)
        u, v = project_float(p, scene.height, scene.width)
        uv.append((float(u[0]), float(v[0])))
    return int(round(uv[1][0] - uv[0][0])), int(round(uv[1][1] - uv[0][1]))


def _attempt(ctx: GenContext, move: str):
    rng = ctx.rng
    h, w = ctx.height, ctx.width
    last = ctx.bin == len(VIEW_ORBIT_BINS) - 1
    bg = tuple(pick(rng, BACKGROUND_COLORS))
    cam0 = CameraSpec(
        randint(rng, 0, 359),
        randint(rng, *VIEW_ELEVATION_DEG),
        qscene(uniform(rng, *VIEW_DISTANCE_M), SCENE_QUANTUM_M),
    )
    objs = _place_objects(rng, ctx, randint(rng, *VIEW_NUM_OBJECTS))
    if len(objs) < VIEW_NUM_OBJECTS[0]:
        return None
    if move == "orbit":
        lo, hi = VIEW_ORBIT_BINS[ctx.bin]
        value = randint(rng, lo, hi if last else hi - 1)
        amount = value if coin(rng, 0.5) else -value
    else:
        lo, hi = VIEW_SHIFT_BINS_CM[ctx.bin]
        value = amount = randint(rng, lo, hi if last else hi - 1)
    cam1 = move_camera(cam0, move, amount)
    floor = FloorSpec(VIEW_FLOOR_RADIUS_M, VIEW_FLOOR_SEGMENTS, tuple(VIEW_FLOOR_COLOR))
    scene = Scene3D(h, w, bg, cam0, tuple(objs), floor)
    if min(cam0.position()[1], cam1.position()[1]) < VIEW_MIN_CAMERA_HEIGHT_M:
        return None
    verts0, verts1 = posed_vertices(scene, cam0), posed_vertices(scene, cam1)
    for cam, vs in ((cam0, verts0), (cam1, verts1)):
        if not vertices_ok(floor_vertices(floor, cam), h, w) or not all(vertices_ok(p, h, w) for p in vs):
            return None
    cands = [k for k in range(len(objs)) if inside_canvas(verts0[k], h, w, CANVAS_MARGIN_PX) and inside_canvas(verts1[k], h, w, CANVAS_MARGIN_PX)]
    if not cands:
        return None
    f0 = render_posed(scene, cam0, verts0, want_rgb=True, want_amodal=True)
    f1 = render_posed(scene, cam1, verts1, want_rgb=False, want_amodal=True)
    stats = None
    for pos in _shuffled(rng, len(cands)):
        k = cands[pos]
        m0, m1 = f0.mask(k), f1.mask(k)
        n0, n1 = int(np.count_nonzero(m0)), int(np.count_nonzero(m1))
        a0, a1 = int(np.count_nonzero(f0.amodal[k])), int(np.count_nonzero(f1.amodal[k]))
        if min(n0, n1) < VIEW_MIN_VISIBLE_PX:
            continue
        if n0 < VIEW_MIN_VISIBLE_FRAC * a0 or n1 < VIEW_MIN_VISIBLE_FRAC * a1:
            continue
        if np.array_equal(m0, m1):
            continue
        stats = (k, n0, n1, a0, a1)
        break
    if stats is None:
        return None
    return {"scene": scene, "cam1": cam1, "f0": f0, "f1": f1, "stats": stats, "amount": amount, "value": value}


def generate(ctx: GenContext) -> Sample:
    rng = ctx.rng
    move = "orbit" if coin(rng, VIEW_ORBIT_PROB) else pick(rng, VIEW_MOVES[1:])
    res = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = _attempt(ctx, move)
        if res is not None:
            break
    if res is None:
        raise SynthGenerationError(f"viewpoint {ctx.split}#{ctx.index} 档 {ctx.bin} {move}：{MAX_ATTEMPTS} 次尝试都不合格")
    scene: Scene3D = res["scene"]
    cam1: CameraSpec = res["cam1"]
    f0, f1 = res["f0"], res["f1"]
    t, n0, n1, a0, a1 = res["stats"]
    desc = scene.objects[t].description
    move_text = move_sentence(move, res["amount"])
    template_id = int(rng.integers(len(TEMPLATES)))
    instruction = TEMPLATES[template_id].format(move=move_text, target=desc)

    m0, m1 = f0.mask(t), f1.mask(t)
    dx, dy = centroid_shift_px(scene, scene.camera, cam1)
    target = make_prediction(m1, ctx.grid_size)
    baselines = {
        "copy": make_prediction(m0, ctx.grid_size),
        "translate": make_prediction(shift_mask(m0, dx, dy), ctx.grid_size),
    }
    object_masks = np.stack([f1.mask(k) for k in range(len(scene.objects))])
    extras = {
        "id_map": f1.id_map,
        "input_id_map": f0.id_map,
        "depth": f1.depth_m(),
        "input_depth": f0.depth_m(),
        "depth_grid": depth_grid(f1.depth_units, m1, ctx.grid_size),
    }
    if move == "orbit":
        difficulty = difficulty_meta("orbit_deg", res["value"], ctx.bin, VIEW_ORBIT_BINS, upper_closed_last=True)
    else:
        difficulty = difficulty_meta("shift_cm", res["value"], ctx.bin, VIEW_SHIFT_BINS_CM, upper_closed_last=True)
    meta = ctx.base_meta()
    meta.update(
        {
            "difficulty": difficulty,
            "template_id": template_id,
            "scene": scene.to_dict(),
            "camera_after": cam1.to_dict(),
            "move": {
                "type": move,
                "amount": res["amount"],
                "unit": "deg" if move == "orbit" else "cm",
                "text": move_text,
            },
            "target_obj_id": t,
            "target_description": desc,
            "visible_px": [n0, n1],
            "amodal_px": [a0, a1],
            "visible_fraction": [n0 / a0, n1 / a1],
            "translate_shift_px": [dx, dy],
            "depth_range_m": depth_range_m(f1.depth_units),
            "input_depth_range_m": depth_range_m(f0.depth_units),
            "baseline_roles": dict(BASELINE_ROLES),
            "baseline_iou": baseline_iou_meta(baselines, target.mask, target.grid),
            "attempts": attempt,
        }
    )
    return Sample(
        task=ctx.task,
        split=ctx.split,
        index=ctx.index,
        image=f0.rgb,
        instruction=instruction,
        target_mask=target.mask,
        target_grid=target.grid,
        object_masks=object_masks,
        baselines=baselines,
        meta=meta,
        extras=extras,
    )
