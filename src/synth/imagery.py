"""e 文本到意象：只有文字（没有图像）。文字描述一个 2D 场景和对其中一个物体的变换，要预测变换后该物体的网格。

坐标表述（写进每条指令的约定段）
- 归一化坐标，两位小数：(0, 0) 是画布左上角，(1, 1) 是右下角，x 向右、y 向下；x 以画布宽 W、y 以画布高 H 为单位。
- 物体 = 外观 + 形状（含离散化的形状参数）+ 质心 (x, y) + 外接半径 r（以 min(H, W) 为单位）+ 顺时针旋转角（整数度）。
  角度 0 时的朝向与 shapes.py 的单位形状一致。文字里的每个数都是生成真值时用的精确值（没有隐藏精度）：
  质心 = 整数/100，半径 = 整数/100，角度 = 整数度；像素坐标 = 整数 × W / 100。
- 变换只作用于目标物体，按固定顺序：先绕质心缩放 s（两位小数），再绕质心旋转 θ（整数度，顺/逆时针），
  最后平移 (tx, ty)（两位小数，右/左、下/上）。组合类型只出现其中的几项。
- 对 n 重旋转对称的形状，|θ| 不超过半个周期（180/n 度）；圆不旋转。所以旋转从不是恒等变换。

难度
- 目标各像素中心在变换下的平均位移 / min(H, W)（旋转已限制在半周期内，即对称意义下的最小位移）。
  档为半开区间 IMAGERY_BINS。各类型的分量另记在 meta["transform"]。

目标与基线
- target_mask / target_grid：变换后的目标；object_masks：变换后的整幅布局（其余物体不动）。
- 平凡基线 "no_transform"：按文字直接放置、不做变换的目标。
- 图像为 None。
"""

from __future__ import annotations

import math

import numpy as np
import shapely

from src.synth.constants import (
    BACKGROUND_COLORS,
    CANVAS_MARGIN_PX,
    IMAGERY_BIN_MARGIN,
    IMAGERY_BINS,
    IMAGERY_BISECTION_STEPS,
    IMAGERY_MAX_RADIUS_HUNDREDTHS,
    IMAGERY_MIN_RADIUS_HUNDREDTHS,
    IMAGERY_NUM_OBJECTS,
    IMAGERY_RADIUS_HUNDREDTHS,
    IMAGERY_SCALE_RANGE,
    IMAGERY_TRANSFORMS,
    MAX_ATTEMPTS,
    SHAPE_FAMILIES,
)
from src.synth.sample import (
    GenContext,
    Sample,
    SynthGenerationError,
    baseline_iou_meta,
    difficulty_meta,
    make_prediction,
)
from src.synth.scene import Background, ObjectSpec, SceneSpec, object_masks
from src.synth.seeding import coin, pick, randint, uniform
from src.synth.shapes import appearance, describe, polygon_inside, qcoord, rasterize, rot_cs, sample_identity, variant

CONVENTION = (
    "Coordinates are fractions of the canvas width and height, from (0, 0) at the top-left corner to (1, 1) at the "
    "bottom-right; x grows to the right and y grows downward. Each object is given by its center (centroid) and its "
    "radius, the distance from the center to its farthest point, as a fraction of the canvas's shorter side. Rotations "
    "are in degrees; when not rotated, ellipses and rectangles are wider than tall, triangles, polygons and stars point "
    "up, crosses are upright plus signs, L-shapes have their corner at the bottom left, and crescents open to the right."
)
TEMPLATES = (
    "{conv} The canvas contains {objects}. {transform} Draw the {target} after this change.",
    "{conv} Scene: {objects}. {transform} Where is the {target} now? Give its mask.",
    "Picture a blank canvas. {conv} On it are {objects}. {transform} Show the {target} in its new position.",
    "{conv} Imagine {objects}. {transform} Which pixels does the {target} cover afterwards?",
)


def _max_rotation(period: float) -> int:
    """|θ| 的上限（整数度）：无对称 180；n 重对称取半周期；圆 0。"""
    if period == 0.0:
        return 0
    return int(math.floor(min(180.0, period / 2.0)))


def _object_phrase(o: ObjectSpec, x_h: int, y_h: int, r_h: int, a_deg: int) -> str:
    v = variant(o.family, o.variant)
    s = f"a {appearance(o.appearance).phrase} {v.phrase} centered at ({x_h / 100:.2f}, {y_h / 100:.2f}) with radius {r_h / 100:.2f}"
    if v.period_deg != 0.0 and a_deg != 0:
        s += f", rotated {a_deg} degrees clockwise"
    return s


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return "; ".join(items[:-1]) + f"; and {items[-1]}"


def _transform_sentence(target: str, s_h: int | None, theta: int | None, t_h: tuple[int, int] | None) -> str:
    parts = []
    if s_h is not None:
        parts.append(f"scaled by {s_h / 100:.2f} about its center")
    if theta is not None:
        parts.append(f"turned {abs(theta)} degrees {'clockwise' if theta > 0 else 'counterclockwise'} about its center")
    if t_h is not None:
        mv = []
        if t_h[0]:
            mv.append(f"{abs(t_h[0]) / 100:.2f} to the {'right' if t_h[0] > 0 else 'left'}")
        if t_h[1]:
            mv.append(f"{abs(t_h[1]) / 100:.2f} {'down' if t_h[1] > 0 else 'up'}")
        parts.append("moved " + " and ".join(mv))
    if len(parts) == 1:
        return f"The {target} is {parts[0]}."
    if len(parts) == 2:
        return f"The {target} is first {parts[0]}, and then {parts[1]}."
    return f"The {target} is first {parts[0]}, then {parts[1]}, and finally {parts[2]}."


class _Target:
    """目标物体的像素中心与变换后的平均位移。"""

    def __init__(self, mask: np.ndarray, cx: float, cy: float, short: int) -> None:
        rows, cols = np.nonzero(mask)
        self.px = cols.astype(np.float64) + 0.5
        self.py = rows.astype(np.float64) + 0.5
        self.dx = self.px - cx
        self.dy = self.py - cy
        self.short = float(short)
        self.rho = float(np.mean(np.sqrt(self.dx * self.dx + self.dy * self.dy))) / self.short

    def mpd(self, s: float, theta_deg: float, tx_px: float, ty_px: float) -> float:
        c, sn = rot_cs(theta_deg)
        ex = s * (self.dx * c - self.dy * sn) - self.dx + tx_px
        ey = s * (self.dx * sn + self.dy * c) - self.dy + ty_px
        return float(np.mean(np.sqrt(ex * ex + ey * ey))) / self.short


def _place_objects(rng, ctx: GenContext):
    h, w, short = ctx.height, ctx.width, ctx.short_side
    n = randint(rng, *IMAGERY_NUM_OBJECTS)
    taken: set[str] = set()
    out = []  # (ident, x_h, y_h, r_h, a_deg)
    geoms = []
    for _ in range(n):
        for _try in range(20):
            ident = sample_identity(rng, ctx.split, SHAPE_FAMILIES, ctx.holdout_family)
            if describe(*ident) in taken:
                continue
            r_h = randint(rng, *IMAGERY_RADIUS_HUNDREDTHS)
            r_px = r_h * short / 100.0
            period = variant(ident[0], ident[1]).period_deg
            a_deg = 0 if period == 0.0 else int(rng.integers(int(round(period))))
            x_lo = math.ceil((r_px + CANVAS_MARGIN_PX + 1) * 100 / w)
            x_hi = math.floor((w - r_px - CANVAS_MARGIN_PX - 1) * 100 / w)
            y_lo = math.ceil((r_px + CANVAS_MARGIN_PX + 1) * 100 / h)
            y_hi = math.floor((h - r_px - CANVAS_MARGIN_PX - 1) * 100 / h)
            if x_hi < x_lo or y_hi < y_lo:
                continue
            x_h, y_h = randint(rng, x_lo, x_hi), randint(rng, y_lo, y_hi)
            o = _spec(0, ident, x_h, y_h, r_h, a_deg, w, h, short)
            g = shapely.polygons(o.polygon())
            if any(shapely.intersects(g.buffer(1.0), g2) for g2 in geoms):
                continue
            geoms.append(g)
            taken.add(describe(*ident))
            out.append((ident, x_h, y_h, r_h, a_deg))
            break
    return out


def _spec(oid: int, ident, x_h: int, y_h: int, r_h: int, a_deg: int, w: int, h: int, short: int, s_h: int = 100) -> ObjectSpec:
    """文字里的整数量 → ObjectSpec（像素坐标 = 整数 × 尺寸 / 100；缩放后半径 = r_h·s_h·short / 10000）。"""
    return ObjectSpec(
        oid,
        ident[0],
        int(ident[1]),
        int(ident[2]),
        qcoord(x_h * w / 100.0),
        qcoord(y_h * h / 100.0),
        qcoord(r_h * s_h * short / 10000.0),
        float(a_deg),
        0,
    )


def _solve(rng, kind: str, tgt: _Target, period: float, m_star: float, ctx: GenContext, r_h: int):
    """为给定类型抽分量，使平均位移 ≈ m_star。返回 (s_h, θ, (tx_h, ty_h))，不用的分量为 None；不可行返回 None。"""
    w, h, short = ctx.width, ctx.height, ctx.short_side
    th_max = _max_rotation(period)
    use_s, use_r, use_t = "scale" in kind, "rotate" in kind, "translate" in kind
    if use_r and th_max < 1:
        return None
    s_h = theta = None
    if kind == "rotate":
        x = m_star / (2.0 * tgt.rho)
        if x >= math.sin(math.radians(th_max) / 2.0):
            return None
        theta = int(round(math.degrees(2.0 * math.asin(x))))
        if theta < 1:
            return None
        return None, theta if coin(rng, 0.5) else -theta, None
    if kind == "scale":
        sign = 1 if coin(rng, 0.5) else -1
        for sg in (sign, -sign):
            s = 1.0 + sg * m_star / tgt.rho
            if IMAGERY_SCALE_RANGE[0] <= s <= IMAGERY_SCALE_RANGE[1]:
                s_h = int(round(100 * s))
                if s_h != 100 and IMAGERY_MIN_RADIUS_HUNDREDTHS * 100 <= r_h * s_h <= IMAGERY_MAX_RADIUS_HUNDREDTHS * 100:
                    return s_h, None, None
        return None
    # 平移（可带缩放、旋转）：先抽非平移分量，再沿随机方向二分平移长度
    for _ in range(10):
        s_h = randint(rng, int(IMAGERY_SCALE_RANGE[0] * 100), int(IMAGERY_SCALE_RANGE[1] * 100)) if use_s else None
        if s_h is not None and (abs(s_h - 100) < 5 or not IMAGERY_MIN_RADIUS_HUNDREDTHS * 100 <= r_h * s_h <= IMAGERY_MAX_RADIUS_HUNDREDTHS * 100):
            continue
        theta = (randint(rng, 1, th_max) * (1 if coin(rng, 0.5) else -1)) if use_r else None
        s = (s_h or 100) / 100.0
        if tgt.mpd(s, theta or 0, 0.0, 0.0) < m_star:
            break
    else:
        return None
    phi = uniform(rng, 0.0, 2.0 * math.pi)
    ux, uy = math.cos(phi), math.sin(phi)
    lo, hi = 0.0, 1.5 * short
    for _ in range(IMAGERY_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if tgt.mpd(s, theta or 0, mid * ux, mid * uy) < m_star:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    t_h = (int(round(lam * ux * 100 / w)), int(round(lam * uy * 100 / h)))
    if t_h == (0, 0):
        return None
    return s_h, theta, t_h


def _attempt(ctx: GenContext, lo: float, hi: float):
    rng = ctx.rng
    h, w, short = ctx.height, ctx.width, ctx.short_side
    placed = _place_objects(rng, ctx)
    if not placed:
        return None
    t_k = int(rng.integers(len(placed)))
    ident, x_h, y_h, r_h, a_deg = placed[t_k]
    period = variant(ident[0], ident[1]).period_deg
    kind = pick(rng, IMAGERY_TRANSFORMS)
    if "rotate" in kind and _max_rotation(period) < 1:
        return None
    t0 = _spec(0, ident, x_h, y_h, r_h, a_deg, w, h, short)
    tgt = _Target(rasterize(t0.polygon(), h, w), t0.cx, t0.cy, short)
    m_star = uniform(rng, lo + IMAGERY_BIN_MARGIN, hi - IMAGERY_BIN_MARGIN)
    sol = _solve(rng, kind, tgt, period, m_star, ctx, r_h)
    if sol is None:
        return None
    s_h, theta, t_h = sol
    tx_h, ty_h = t_h if t_h is not None else (0, 0)
    t1 = _spec(0, ident, x_h + tx_h, y_h + ty_h, r_h, a_deg + (theta or 0), w, h, short, s_h or 100)
    if not polygon_inside(t1.polygon(), h, w, CANVAS_MARGIN_PX):
        return None
    tx_px = t1.cx - t0.cx
    ty_px = t1.cy - t0.cy
    m = tgt.mpd((s_h or 100) / 100.0, theta or 0, tx_px, ty_px)
    if not lo <= m < hi:
        return None
    return placed, t_k, kind, s_h, theta, t_h, m


def generate(ctx: GenContext) -> Sample:
    lo, hi = IMAGERY_BINS[ctx.bin]
    res = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = _attempt(ctx, lo, hi)
        if res is not None:
            break
    if res is None:
        raise SynthGenerationError(f"imagery {ctx.split}#{ctx.index} 档 {ctx.bin}：{MAX_ATTEMPTS} 次尝试都不合格")
    rng = ctx.rng
    placed, t_k, kind, s_h, theta, t_h, m = res
    h, w, short = ctx.height, ctx.width, ctx.short_side

    # obj_id = 在文字里出现的顺序（目标的位置随机，由 t_k 决定）
    before = tuple(_spec(i, p[0], *p[1:], w, h, short) for i, p in enumerate(placed))
    ident, x_h, y_h, r_h, a_deg = placed[t_k]
    tx_h, ty_h = t_h if t_h is not None else (0, 0)
    t_after = _spec(t_k, ident, x_h + tx_h, y_h + ty_h, r_h, a_deg + (theta or 0), w, h, short, s_h or 100)
    after = tuple(t_after if i == t_k else o for i, o in enumerate(before))
    bg = Background("solid", tuple(BACKGROUND_COLORS[0]), tuple(BACKGROUND_COLORS[0]))
    scene = SceneSpec(h, w, bg, before)
    scene_after = SceneSpec(h, w, bg, after)
    masks_after = object_masks(scene_after)
    no_transform_mask = rasterize(before[t_k].polygon(), h, w)

    target_ref = before[t_k].description
    objects_text = _join([_object_phrase(o, *p[1:]) for o, p in zip(before, placed)])
    transform_text = _transform_sentence(target_ref, s_h, theta, t_h)
    template_id = int(rng.integers(len(TEMPLATES)))
    instruction = TEMPLATES[template_id].format(conv=CONVENTION, objects=objects_text, transform=transform_text, target=target_ref)

    target = make_prediction(masks_after[t_k], ctx.grid_size)
    baselines = {"no_transform": make_prediction(no_transform_mask, ctx.grid_size)}
    meta = ctx.base_meta()
    meta.update(
        {
            "difficulty": difficulty_meta("mean_displacement", m, ctx.bin, IMAGERY_BINS),
            "template_id": template_id,
            "scene": scene.to_dict(),
            "scene_after": scene_after.to_dict(),
            "target_obj_id": t_k,
            "target_description": target_ref,
            "text_objects": [
                {"obj_id": i, "x": p[1] / 100, "y": p[2] / 100, "radius": p[3] / 100, "angle_deg": p[4]} for i, p in enumerate(placed)
            ],
            "transform": {
                "type": kind,
                "scale": None if s_h is None else s_h / 100,
                "rotate_deg": theta,
                "translate": None if t_h is None else [t_h[0] / 100, t_h[1] / 100],
            },
            "baseline_iou": baseline_iou_meta(baselines, target.mask, target.grid),
            "attempts": attempt,
        }
    )
    return Sample(
        task=ctx.task,
        split=ctx.split,
        index=ctx.index,
        image=None,
        instruction=instruction,
        target_mask=target.mask,
        target_grid=target.grid,
        object_masks=masks_after,
        baselines=baselines,
        meta=meta,
    )
