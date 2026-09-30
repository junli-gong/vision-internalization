"""c 遮挡补全：若干 2D 物体按 z 序叠画，目标被一到三个遮挡物部分遮住，要预测目标的完整（非模态）mask。

生成（每次尝试，全部随机量来自样本的 rng）
1. 目标：身份从该划分的身份池里抽（先族、再变体、再外观），每个样本只抽一次、不随重试改变（否则拒绝采样会让
   目标形状族偏向容易满足约束的非凸形状）；外接半径 OCC_TARGET_RADIUS × min(H, W)；任意角度；
   质心均匀落在使目标完整留在画布内的范围里。以下各步每次尝试重抽。
2. 遮挡物 1–3 个（z 在目标之上）：身份的简称不得与目标相同；每个有随机方向 φ（度，均匀）与基础偏移
   δ ∈ OCC_OFFSET_RANGE，圆心 = 目标质心 + λ·δ·(r_目标 + r_遮挡物)·rot_cs(φ)。
   在 [OCC_BIN_MARGIN 收窄后的档内] 均匀抽目标遮挡比例 f*，用 shapely 的面积对公共缩放 λ 二分，使几何遮挡比例 ≈ f*。
3. 干扰物 0–2 个（z 在目标之下，不与目标相交），简称不得与目标相同。
4. 光栅化后按像素重算遮挡比例 f = 1 - |可见| / |非模态|，不在档内就重来；全部物体必须完整留在画布内。
5. 非平凡约束：每个样本先以 OCC_NONTRIVIAL_QUOTA 的概率抽一个"必须非平凡"标记；带标记的样本拒绝
   "可见部分凸包与真值 IoU ≥ OCC_NONTRIVIAL_HULL_IOU"的场景。于是每档非平凡样本占比 ≥ 配额（大样本意义上）。

目标与基线
- target_mask：目标的非模态 mask；object_masks：全部物体的非模态 mask（可见 mask 用 scene.visible_masks 重算）。
- 平凡基线："visible"（目标的可见部分）、"convex_hull"（可见部分的凸包，见 convex_hull_mask）。
- 目标在指令里用"外观 + 名词"指明，场景里没有别的物体有同样的简称。

跨机器逐位复现（D24：c 在任何平台上逐位相同）
- 位姿、方向都只经过量化与 dmath 的确定三角函数；mask 用 raster.py 的精确整数光栅化；遮挡比例、凸包 IoU 是整数计数之比。
  写进样本与元数据的每个量都由这些确定的计算得出。
- shapely（GEOS，编译过的 C++）还用在三处，都不会让结果随平台变化：
  * λ 的二分、"遮挡比例 ≥ f*"的初判、非平凡预检，用的是 GEOS 算的面积之比，只拿来与随机抽的 f* 或固定阈值比大小，
    本身不进样本。编译器是否融合 FMA 会让这个比值在两个平台上相差 ~1e-15，比较结果只有在比值与 f*（或阈值）
    相差不到这个量级时才会翻转：每次比较的概率 ~1e-14，也没有量化或取整去放大它。
    改成像素级的精确判定要在每次二分时光栅化遮挡物（平均约 5 次尝试 × 14 步），c 的速度会降到要求以下。
  * 干扰物与目标不相交（intersects），是 GEOS 的鲁棒谓词，对网格上的精确坐标给出精确结果。
  * 凸包：整数点集的凸包由精确的方向判定得出，顶点是输入点本身；光栅化只取决于凸包围成的区域。
"""

from __future__ import annotations

import numpy as np
import shapely

from src.synth.constants import (
    ANGLE_QUANTUM_DEG,
    BACKGROUND_COLORS,
    BACKGROUND_KINDS,
    CANVAS_MARGIN_PX,
    MAX_ATTEMPTS,
    OCC_BIN_MARGIN,
    OCC_BISECTION_STEPS,
    OCC_DISTRACTOR_RADIUS,
    OCC_GEOM_PRECHECK_SLACK,
    OCC_NONTRIVIAL_HULL_IOU,
    OCC_NONTRIVIAL_QUOTA,
    OCC_NUM_DISTRACTORS,
    OCC_NUM_OCCLUDERS,
    OCC_OCCLUDER_RADIUS,
    OCC_OFFSET_RANGE,
    OCC_TARGET_RADIUS,
    OCCLUSION_BINS,
    SHAPE_FAMILIES,
)
from src.synth.sample import (
    GenContext,
    Sample,
    SynthGenerationError,
    baseline_iou_meta,
    difficulty_meta,
    iou,
    make_prediction,
)
from src.synth.scene import Background, ObjectSpec, SceneSpec, object_masks, render_image, visible_masks
from src.synth.seeding import coin, pick, randint, uniform
from src.synth.shapes import describe, polygon_inside, pose_polygon, qcoord, quantize, rasterize, rot_cs, sample_identity

TEMPLATES = (
    "The {obj} is partly hidden behind other objects. Draw its complete shape, including the hidden parts.",
    "Segment the entire {obj}, including the parts covered by other objects.",
    "Imagine that the objects in front of the {obj} were removed. Which pixels would the {obj} cover?",
    "Give the full (amodal) mask of the {obj}, as if nothing were in front of it.",
)


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """可见部分的凸包：各可见像素方块（闭集）之并的凸包，用同一光栅化规则画出，再并上 mask 本身。

    只需每行最左、最右可见像素的四个角点；凸包用 shapely（GEOS 的方向判定是精确的，整数输入得到精确凸包）。
    空 mask 返回空 mask。
    """
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    if not m.any():
        return np.zeros_like(m)
    rows = np.nonzero(m.any(axis=1))[0]
    sub = m[rows]
    first = sub.argmax(axis=1)
    last = sub.shape[1] - 1 - sub[:, ::-1].argmax(axis=1)
    xs = np.concatenate([first, first, last + 1, last + 1]).astype(np.float64)
    ys = np.concatenate([rows, rows + 1, rows, rows + 1]).astype(np.float64)
    hull = shapely.convex_hull(shapely.multipoints(np.stack([xs, ys], axis=1)))
    coords = np.asarray(hull.exterior.coords, dtype=np.float64)[:-1]
    return rasterize(coords, h, w) | m


def _background(rng) -> Background:
    kind = pick(rng, BACKGROUND_KINDS)
    top = pick(rng, BACKGROUND_COLORS)
    bottom = pick(rng, BACKGROUND_COLORS) if kind == "vgradient" else top
    return Background(kind, tuple(top), tuple(bottom))


def _other_identity(rng, ctx: GenContext, avoid_desc: str):
    for _ in range(100):
        ident = sample_identity(rng, ctx.split, SHAPE_FAMILIES, ctx.holdout_family)
        if describe(*ident) != avoid_desc:
            return ident
    raise SynthGenerationError("找不到与目标简称不同的身份")  # pragma: no cover


def _angle(rng) -> float:
    return quantize(uniform(rng, 0.0, 360.0), ANGLE_QUANTUM_DEG)


def _shuffled(rng, n: int) -> list[int]:
    """Fisher–Yates，只用 integers()。"""
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(i + 1))
        p[i], p[j] = p[j], p[i]
    return p


def _attempt(ctx: GenContext, lo: float, hi: float, need_nontrivial: bool, t_ident):
    rng = ctx.rng
    h, w, s = ctx.height, ctx.width, ctx.short_side
    m = CANVAS_MARGIN_PX

    # 1. 目标（身份在 generate 里每个样本只抽一次，重试只改几何，拒绝不会改变目标形状族的分布）
    t_desc = describe(*t_ident)
    r_t = qcoord(uniform(rng, *OCC_TARGET_RADIUS) * s)
    a_t = _angle(rng)
    cx_t = qcoord(uniform(rng, r_t + m + 1, w - r_t - m - 1))
    cy_t = qcoord(uniform(rng, r_t + m + 1, h - r_t - m - 1))
    t_poly = pose_polygon(*t_ident[:2], cx_t, cy_t, r_t, a_t)
    t_geom = shapely.polygons(t_poly)
    t_area = t_geom.area

    # 2. 遮挡物：先在原点摆好形状，再按 λ 平移
    n_occ = randint(rng, *OCC_NUM_OCCLUDERS)
    occ = []
    for _ in range(n_occ):
        ident = _other_identity(rng, ctx, t_desc)
        r_o = qcoord(uniform(rng, *OCC_OCCLUDER_RADIUS) * s)
        a_o = _angle(rng)
        cos_phi, sin_phi = rot_cs(uniform(rng, 0.0, 360.0))  # 随机方向；确定的三角函数
        delta = uniform(rng, *OCC_OFFSET_RANGE)
        base = pose_polygon(*ident[:2], 0.0, 0.0, r_o, a_o)
        step = delta * (r_t + r_o)
        occ.append((ident, r_o, a_o, base, step * cos_phi, step * sin_phi))
    # 被要求非平凡时 f ≥ 1 - 阈值（凸目标的凸包 IoU ≥ 可见 IoU = 1 - f，f 更小必然不合格）
    f_lo = max(lo, 1.0 - OCC_NONTRIVIAL_HULL_IOU) if need_nontrivial else lo
    f_star = uniform(rng, f_lo + OCC_BIN_MARGIN, hi - OCC_BIN_MARGIN)

    def frac(lam: float) -> float:
        polys = [shapely.polygons(o[3] + np.array([cx_t + lam * o[4], cy_t + lam * o[5]])) for o in occ]
        return shapely.intersection(t_geom, shapely.union_all(polys)).area / t_area

    if frac(0.0) < f_star:
        return None
    lam_lo, lam_hi = 0.0, 1.0 / OCC_OFFSET_RANGE[0]
    for _ in range(OCC_BISECTION_STEPS):
        mid = 0.5 * (lam_lo + lam_hi)
        if frac(mid) >= f_star:
            lam_lo = mid
        else:
            lam_hi = mid
    lam = 0.5 * (lam_lo + lam_hi)
    occ_objs = []
    occ_geoms = []
    for ident, r_o, a_o, _base, ox, oy in occ:
        cx, cy = qcoord(cx_t + lam * ox), qcoord(cy_t + lam * oy)
        poly = pose_polygon(*ident[:2], cx, cy, r_o, a_o)
        if not polygon_inside(poly, h, w, m):
            return None
        occ_objs.append((ident, cx, cy, r_o, a_o))
        occ_geoms.append(shapely.polygons(poly))
    if need_nontrivial:  # 先用连续几何粗判凸包 IoU，明显不合格的不必光栅化
        vis_geom = shapely.difference(t_geom, shapely.union_all(occ_geoms))
        hull_geom = shapely.convex_hull(vis_geom)
        g_iou = shapely.intersection(hull_geom, t_geom).area / shapely.union(hull_geom, t_geom).area
        if g_iou >= OCC_NONTRIVIAL_HULL_IOU + OCC_GEOM_PRECHECK_SLACK:
            return None

    # 3. 干扰物（在目标之下，不与目标相交）
    dis_objs = []
    for _ in range(randint(rng, *OCC_NUM_DISTRACTORS)):
        ident = _other_identity(rng, ctx, t_desc)
        r_d = qcoord(uniform(rng, *OCC_DISTRACTOR_RADIUS) * s)
        a_d = _angle(rng)
        for _try in range(10):
            cx = qcoord(uniform(rng, r_d + m + 1, w - r_d - m - 1))
            cy = qcoord(uniform(rng, r_d + m + 1, h - r_d - m - 1))
            if not shapely.intersects(t_geom, shapely.polygons(pose_polygon(*ident[:2], cx, cy, r_d, a_d))):
                dis_objs.append((ident, cx, cy, r_d, a_d))
                break

    # 4. z 序与 obj_id（打乱，目标的位置不固定）
    roles = [("distractor", o) for o in dis_objs] + [("target", (t_ident, cx_t, cy_t, r_t, a_t))]
    roles += [("occluder", o) for o in occ_objs]
    z_dis = _shuffled(rng, len(dis_objs))
    z_occ = _shuffled(rng, len(occ_objs))
    zs = z_dis + [len(dis_objs)] + [len(dis_objs) + 1 + z for z in z_occ]
    perm = _shuffled(rng, len(roles))  # perm[k] = 第 k 个角色的 obj_id
    objs = [None] * len(roles)
    for k, (role, (ident, cx, cy, r, a)) in enumerate(roles):
        oid = perm[k]
        objs[oid] = ObjectSpec(oid, ident[0], int(ident[1]), int(ident[2]), cx, cy, r, a, int(zs[k]))
    scene = SceneSpec(h, w, _background(rng), tuple(objs))
    t_id = perm[len(dis_objs)]

    # 5. 按像素核对难度与非平凡约束
    amodal = object_masks(scene)
    vis = visible_masks(scene, amodal)
    a_cnt = int(np.count_nonzero(amodal[t_id]))
    v_cnt = int(np.count_nonzero(vis[t_id]))
    f_pix = (a_cnt - v_cnt) / a_cnt
    if not lo <= f_pix < hi:
        return None
    hull = convex_hull_mask(vis[t_id])
    hull_iou = iou(hull, amodal[t_id])
    if need_nontrivial and hull_iou >= OCC_NONTRIVIAL_HULL_IOU:
        return None
    return {
        "scene": scene,
        "target": t_id,
        "roles": {perm[k]: role for k, (role, _) in enumerate(roles)},
        "amodal": amodal,
        "visible": vis,
        "hull": hull,
        "hull_iou": hull_iou,
        "f_pix": f_pix,
    }


def generate(ctx: GenContext) -> Sample:
    lo, hi = OCCLUSION_BINS[ctx.bin]
    need_nontrivial = coin(ctx.rng, OCC_NONTRIVIAL_QUOTA)
    t_ident = sample_identity(ctx.rng, ctx.split, SHAPE_FAMILIES, ctx.holdout_family)
    res = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = _attempt(ctx, lo, hi, need_nontrivial, t_ident)
        if res is not None:
            break
    if res is None:
        raise SynthGenerationError(f"occlusion {ctx.split}#{ctx.index} 档 {ctx.bin}：{MAX_ATTEMPTS} 次尝试都不合格")
    rng = ctx.rng
    scene: SceneSpec = res["scene"]
    t_id = res["target"]
    t_desc = scene.objects[t_id].description
    template_id = int(rng.integers(len(TEMPLATES)))
    instruction = TEMPLATES[template_id].format(obj=t_desc)

    target = make_prediction(res["amodal"][t_id], ctx.grid_size)
    baselines = {
        "visible": make_prediction(res["visible"][t_id], ctx.grid_size),
        "convex_hull": make_prediction(res["hull"], ctx.grid_size),
    }
    meta = ctx.base_meta()
    meta.update(
        {
            "difficulty": difficulty_meta("occluded_fraction", res["f_pix"], ctx.bin, OCCLUSION_BINS),
            "template_id": template_id,
            "scene": scene.to_dict(),
            "target_obj_id": t_id,
            "target_description": t_desc,
            "object_roles": [res["roles"][k] for k in range(len(scene.objects))],
            "nontrivial_required": need_nontrivial,
            "nontrivial": bool(res["hull_iou"] < OCC_NONTRIVIAL_HULL_IOU),
            "baseline_iou": baseline_iou_meta(baselines, target.mask, target.grid),
            "attempts": attempt,
        }
    )
    return Sample(
        task=ctx.task,
        split=ctx.split,
        index=ctx.index,
        image=render_image(scene, res["amodal"]),
        instruction=instruction,
        target_mask=target.mask,
        target_grid=target.grid,
        object_masks=res["amodal"],
        baselines=baselines,
        meta=meta,
    )
