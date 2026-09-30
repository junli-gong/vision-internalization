"""3D 实体（DEV-0010）：程序化低面数网格、对称性、外观、身份与身份划分。无学习组件，不读任何外部资产，不用 trimesh。

网格（局部坐标：右手系，y 朝上；原点 = 网格的体积质心；外接半径 1）
- 顶点在 decimal 里构造（三角函数走 dmath 的精确参数约简），移到体积质心、除以最远顶点的距离，再量化到
  UNIT_SHAPE_QUANTUM（2^-24）。三角面的顶点顺序使 (v1 − v0) × (v2 − v0) 朝外。
- 构造时检查：网格闭合且定向一致（每条有向边恰好出现一次、它的反向边也恰好出现一次），体积为正；不满足就抛异常。
- 族与变体（参数都在 constants）：
  * box：长方体，(x, y, z) 相对尺寸，12 个三角面；(1, 1, 1) 是立方体；
  * sphere：二十面体细分 SPHERE_SUBDIVISIONS 次（320 面），y 方向缩放（1.0 = 球，其余为旋转椭球）；
  * cylinder：ROUND_SEGMENTS 棱的棱柱近似，上下底面以中心为扇心；参数 = 高 / 直径；
  * cone：底面 ROUND_SEGMENTS 边、以中心为扇心；参数 = 高 / 直径；
  * torus：ROUND_SEGMENTS × TORUS_TUBE_SEGMENTS 个四边形（各分成两个三角面），轴为 y；参数 = 管半径 / 中心圆半径；
  * prism：正 n 边形底面（外接圆半径 1，顶点在 xz 平面里 90° + 360°k/n 的方向，(cos, 0, sin)），轴为 y；
  * l_prism：xy 平面里高 1、宽 L_PRISM_WIDTH、臂宽 t 的 L 形（竖臂在左、横臂在下），沿 z 拉伸厚度 d。
- "中心" = 体积质心：实体的每条旋转对称轴都过它，所以"绕中心旋转"与下面的对称性一致。

对称性（按理想实体，不按折线化的网格）
- rotation_period_about(族, 变体, 轴)：绕过中心、方向为"轴"（局部坐标）的直线，实体的旋转周期（度）：
  0 = 任意角度都不变（连续对称）；360 = 没有非平凡的旋转对称；其余为 360/k（k 重对称）。与 2D 的 period_deg 同一约定，
  shapes.max_equivalent_rotation 把它换成可报告的最大角（半个周期）。
  * 球：任意轴连续；
  * 旋转椭球、圆柱、圆环：y 轴连续，与 y 垂直的任意轴 2 重（上下对称）；
  * 圆锥：y 轴连续；
  * 长方体：三条坐标轴 2 重（另两边相等时 4 重）；两边相等时这两边所在平面的两条对角线 2 重；立方体另有 4 条体对角线 3 重；
  * n 棱柱：y 轴 n 重；与 y 垂直、方向角为 90° + j·180°/n（j = 0..n-1）的 n 条轴 2 重；
  * L 形块：没有（两臂不等长）。
  折线化的网格只在其中一部分下严格不变（例如 16 棱的圆柱只有 16 重），a 的旋转角按理想实体的周期约简（见 rotation3d）。
- 轴方向是浮点（由确定的三角函数与逐元素运算得出），平行、垂直的判定用 SYMMETRY_TOL = 1e-9 的相对容差：
  整数度的姿态下，真的对齐时偏差 ~1e-12，不对齐时至少是千分之一的量级，判定没有歧义（tests/test_synth_3d_render.py 钉住）。

外观与身份
- 外观 = COLORS 里 10 种纯色之一（shapes.APPEARANCES 的前 10 个，图案都是 solid），平面着色。
- 身份 = (族, 变体序号, 外观序号)；身份键 "族:变体:外观"（3D 族名与 2D 族名不重复）。
- 划分规则与第一部分相同（shapes.assign_splits）：每个 (族, 变体) 的 10 个身份按 sha256(盐 | 身份键) 排序，
  8/1/1 分到 train/val/test；可选"留出整族"（该族的全部身份只进 test）。
"""

from __future__ import annotations

import decimal
import functools
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

import numpy as np

from src.synth.constants import (
    BOX_DIMS,
    COLORS,
    CONE_HEIGHT_RATIOS,
    CYLINDER_HEIGHT_RATIOS,
    L_PRISM_VARIANTS,
    L_PRISM_WIDTH,
    PRISM_VARIANTS,
    ROUND_SEGMENTS,
    SOLID_FAMILIES,
    SOLID_REST_POSES,
    SPHERE_SUBDIVISIONS,
    SPHERE_Y_RATIOS,
    TORUS_TUBE_RATIOS,
    TORUS_TUBE_SEGMENTS,
    UNIT_SHAPE_QUANTUM,
)
from src.synth.dmath import CTX, cos_sin_deg, quantize_dec, rot_cs
from src.synth.shapes import APPEARANCES, Identity, _dec, appearance, assign_splits

SYMMETRY_TOL = 1e-9

# 实体用的外观：纯色的 10 种（APPEARANCES 按 图案 × 颜色 排列，前 10 个是 solid）
SOLID_APPEARANCES = tuple(APPEARANCES[: len(COLORS)])
if any(a.pattern != "solid" for a in SOLID_APPEARANCES):  # pragma: no cover - 常量表顺序被改
    raise ImportError("APPEARANCES 的前 len(COLORS) 个必须是纯色")


# ---------------------------------------------------------------------------
# 族与变体
# ---------------------------------------------------------------------------

_PRISM_NOUNS = {3: "triangular prism", 6: "hexagonal prism"}


@dataclass(frozen=True)
class SolidVariant:
    family: str
    index: int
    params: tuple
    noun: str  # 指令里的名词："cube"、"box"、"sphere"、"ellipsoid"、"cylinder"……


def _build_variants() -> dict[str, tuple[SolidVariant, ...]]:
    out: dict[str, list[SolidVariant]] = {f: [] for f in SOLID_FAMILIES}

    def add(family, params, noun):
        out[family].append(SolidVariant(family, len(out[family]), tuple(params), noun))

    for d in BOX_DIMS:
        add("box", d, "cube" if d[0] == d[1] == d[2] else "box")
    for q in SPHERE_Y_RATIOS:
        add("sphere", (q,), "sphere" if q == 1.0 else "ellipsoid")
    for r in CYLINDER_HEIGHT_RATIOS:
        add("cylinder", (r,), "cylinder")
    for r in CONE_HEIGHT_RATIOS:
        add("cone", (r,), "cone")
    for r in TORUS_TUBE_RATIOS:
        add("torus", (r,), "torus")
    for n, r in PRISM_VARIANTS:
        add("prism", (n, r), _PRISM_NOUNS[n])
    for t, d in L_PRISM_VARIANTS:
        add("l_prism", (t, d), "L-shaped block")
    return {f: tuple(v) for f, v in out.items()}


SOLID_VARIANTS: dict[str, tuple[SolidVariant, ...]] = _build_variants()


def solid_variant(family: str, index: int) -> SolidVariant:
    if family not in SOLID_VARIANTS:
        raise ValueError(f"未知实体族 {family!r}")
    vs = SOLID_VARIANTS[family]
    if not 0 <= int(index) < len(vs):
        raise ValueError(f"{family} 没有变体 {index}（共 {len(vs)} 个）")
    return vs[int(index)]


def describe_solid(family: str, variant_index: int, appearance_index: int) -> str:
    """指令里的简称：颜色 + 名词，如 "red cylinder"。"""
    return f"{appearance(appearance_index).phrase} {solid_variant(family, variant_index).noun}"


def is_isotropic(family: str, variant_index: int) -> bool:
    """任意旋转都不变（球）：a 不能用它做旋转目标。"""
    v = solid_variant(family, variant_index)
    return v.family == "sphere" and v.params[0] == 1.0


# ---------------------------------------------------------------------------
# 网格构造（decimal）
# ---------------------------------------------------------------------------

P3 = tuple[Decimal, Decimal, Decimal]


def _det(a: P3, b: P3, c: P3) -> Decimal:
    """a · (b × c)：四面体 (0, a, b, c) 有向体积的 6 倍。调用方已进入 CTX。"""
    return a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def _cross(a: P3, b: P3) -> P3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a: P3, b: P3) -> P3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _orient_convex(verts: list[P3], faces: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """凸网格、原点严格在内：det(v0, v1, v2) > 0 即朝外；为负就交换后两个顶点。"""
    out = []
    with decimal.localcontext(CTX):
        for a, b, c in faces:
            d = _det(verts[a], verts[b], verts[c])
            if d == 0:  # pragma: no cover - 构造写错
                raise ValueError("退化的三角面（或原点在面所在平面上）")
            out.append((a, b, c) if d > 0 else (a, c, b))
    return out


def _ring(n: int, start_deg, radius: Decimal, y: Decimal) -> list[P3]:
    """xz 平面里半径 radius、高度 y 的正 n 边形顶点，方向角 start + 360k/n（精确有理数），点 = (r cos, y, r sin)。"""
    pts = []
    with decimal.localcontext(CTX):
        for k in range(n):
            c, s = cos_sin_deg(Fraction(start_deg) + Fraction(360 * k, n))
            pts.append((radius * c, y, radius * s))
    return pts


def _box(dims) -> tuple[list[P3], list[tuple[int, int, int]]]:
    with decimal.localcontext(CTX):
        half = [_dec(d) / 2 for d in dims]
        verts = [(sx * half[0], sy * half[1], sz * half[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = []
    for axis in range(3):
        b, c = [a for a in range(3) if a != axis]
        for side in (-1, 1):
            quad = []
            for sb, sc in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
                s = [0, 0, 0]
                s[axis], s[b], s[c] = side, sb, sc
                quad.append(4 * (s[0] > 0) + 2 * (s[1] > 0) + (s[2] > 0))
            faces += [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
    return verts, _orient_convex(verts, faces)


# 正二十面体：12 个顶点 (0, ±1, ±φ) 的循环排列，20 个面（朝向由 _orient_convex 统一）
_ICOSA_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)  # fmt: skip


def _unit(p: P3) -> P3:
    n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    return (p[0] / n, p[1] / n, p[2] / n)


def _ellipsoid(y_ratio, subdivisions: int) -> tuple[list[P3], list[tuple[int, int, int]]]:
    with decimal.localcontext(CTX):
        phi = (1 + Decimal(5).sqrt()) / 2
        one = Decimal(1)
        raw = [(-one, phi, 0), (one, phi, 0), (-one, -phi, 0), (one, -phi, 0), (0, -one, phi), (0, one, phi), (0, -one, -phi),
               (0, one, -phi), (phi, 0, -one), (phi, 0, one), (-phi, 0, -one), (-phi, 0, one)]  # fmt: skip
        verts = [_unit((Decimal(x), Decimal(y), Decimal(z))) for x, y, z in raw]
        faces = list(_ICOSA_FACES)
        for _ in range(subdivisions):
            mids: dict[tuple[int, int], int] = {}

            def mid(i: int, j: int) -> int:
                key = (min(i, j), max(i, j))
                if key not in mids:
                    a, b = verts[i], verts[j]
                    verts.append(_unit(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)))
                    mids[key] = len(verts) - 1
                return mids[key]

            new_faces = []
            for a, b, c in faces:
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            faces = new_faces
        q = _dec(y_ratio)
        verts = [(x, y * q, z) for x, y, z in verts]
    return verts, _orient_convex(verts, faces)


def _ngon_prism(n: int, start_deg, half_height: Decimal) -> tuple[list[P3], list[tuple[int, int, int]]]:
    """正 n 棱柱（圆柱也用它）：底环 0..n-1、顶环 n..2n-1、底心 2n、顶心 2n+1；侧面每段两个三角面，底面以中心为扇心。"""
    one = Decimal(1)
    verts = _ring(n, start_deg, one, -half_height) + _ring(n, start_deg, one, half_height)
    verts += [(Decimal(0), -half_height, Decimal(0)), (Decimal(0), half_height, Decimal(0))]
    bc, tc = 2 * n, 2 * n + 1
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        faces += [(k, k1, n + k1), (k, n + k1, n + k), (tc, n + k, n + k1), (bc, k1, k)]
    return verts, _orient_convex(verts, faces)


def _cone(n: int, half_height: Decimal) -> tuple[list[P3], list[tuple[int, int, int]]]:
    """底环 0..n-1（y = -h）、顶点 n（y = +h）、底心 n+1。原点（半高处）严格在锥内，可以用 _orient_convex。"""
    verts = _ring(n, 0, Decimal(1), -half_height) + [(Decimal(0), half_height, Decimal(0)), (Decimal(0), -half_height, Decimal(0))]
    apex, bc = n, n + 1
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        faces += [(apex, k1, k), (bc, k, k1)]
    return verts, _orient_convex(verts, faces)


def _torus(n_major: int, n_minor: int, tube) -> tuple[list[P3], list[tuple[int, int, int]]]:
    """中心圆半径 1、管半径 tube，轴为 y。不是凸的：每个面的朝向按"法向指向离开管中心线的一侧"逐个定。"""
    with decimal.localcontext(CTX):
        rho = _dec(tube)
        verts = []
        for i in range(n_major):
            cu, su = cos_sin_deg(Fraction(360 * i, n_major))
            for j in range(n_minor):
                cv, sv = cos_sin_deg(Fraction(360 * j, n_minor))
                w = 1 + rho * cv
                verts.append((w * cu, rho * sv, w * su))

        def idx(i: int, j: int) -> int:
            return (i % n_major) * n_minor + (j % n_minor)

        faces = []
        for i in range(n_major):
            for j in range(n_minor):
                a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
                for tri in ((a, b, c), (a, c, d)):
                    p0, p1, p2 = (verts[k] for k in tri)
                    nrm = _cross(_sub(p1, p0), _sub(p2, p0))
                    g = ((p0[0] + p1[0] + p2[0]) / 3, (p0[1] + p1[1] + p2[1]) / 3, (p0[2] + p1[2] + p2[2]) / 3)
                    r = (g[0] * g[0] + g[2] * g[2]).sqrt()
                    out = (g[0] - g[0] / r, g[1], g[2] - g[2] / r)  # 质心减去它在中心圆上的最近点
                    dot = nrm[0] * out[0] + nrm[1] * out[1] + nrm[2] * out[2]
                    if dot == 0:  # pragma: no cover
                        raise ValueError("圆环面的朝向无法判定")
                    faces.append(tri if dot > 0 else (tri[0], tri[2], tri[1]))
    return verts, faces


def _l_prism(arm, depth) -> tuple[list[P3], list[tuple[int, int, int]]]:
    """侧面 L 形在 xy 平面里逆时针（从 +z 看）：p0 外角（左下）、p1、p2、p3 内角、p4、p5；前面 z = +d/2（0..5），后面 z = -d/2（6..11）。

    L 形关于外角 p0 是星形的，端面用以 p0 为扇心的 4 个三角面；侧面每条边两个三角面。朝向在构造里写定，由 _check_mesh 核对。
    """
    with decimal.localcontext(CTX):
        t, d, w = _dec(arm), _dec(depth), _dec(L_PRISM_WIDTH)
        half = Decimal("0.5")
        x0, x1, y0, y1 = -w / 2, w / 2, -half, half
        prof = [(x0, y0), (x1, y0), (x1, y0 + t), (x0 + t, y0 + t), (x0 + t, y1), (x0, y1)]
        verts = [(x, y, d / 2) for x, y in prof] + [(x, y, -d / 2) for x, y in prof]
    faces = []
    for k in range(1, 5):
        faces += [(0, k, k + 1), (6, 6 + k + 1, 6 + k)]
    for k in range(6):
        k1 = (k + 1) % 6
        faces += [(k, 6 + k, 6 + k1), (k, 6 + k1, k1)]
    return verts, faces


def _raw_mesh(v: SolidVariant) -> tuple[list[P3], list[tuple[int, int, int]]]:
    f, p = v.family, v.params
    with decimal.localcontext(CTX):
        if f == "box":
            return _box(p)
        if f == "sphere":
            return _ellipsoid(p[0], SPHERE_SUBDIVISIONS)
        # 圆柱、圆锥、棱柱：底面（外接圆）半径 1，高 = 高径比 × 直径 2，所以半高 = 高径比
        if f == "cylinder":
            return _ngon_prism(ROUND_SEGMENTS, 0, _dec(p[0]))
        if f == "cone":
            return _cone(ROUND_SEGMENTS, _dec(p[0]))
        if f == "torus":
            return _torus(ROUND_SEGMENTS, TORUS_TUBE_SEGMENTS, p[0])
        if f == "prism":
            return _ngon_prism(int(p[0]), 90, _dec(p[1]))
        if f == "l_prism":
            return _l_prism(p[0], p[1])
    raise ValueError(f"未知实体族 {f!r}")  # pragma: no cover


def _check_mesh(verts: list[P3], faces: list[tuple[int, int, int]]) -> None:
    """闭合、定向一致、体积为正；否则抛 ValueError。"""
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        if len({a, b, c}) != 3:
            raise ValueError("三角面的顶点重复")
        for e in ((a, b), (b, c), (c, a)):
            edges[e] = edges.get(e, 0) + 1
    for (a, b), n in edges.items():
        if n != 1 or edges.get((b, a), 0) != 1:
            raise ValueError(f"网格不闭合或定向不一致：有向边 {(a, b)} 出现 {n} 次，反向边 {edges.get((b, a), 0)} 次")
    with decimal.localcontext(CTX):
        vol6 = sum((_det(verts[a], verts[b], verts[c]) for a, b, c in faces), Decimal(0))
    if not vol6 > 0:
        raise ValueError("网格体积不为正（面朝内）")


@functools.lru_cache(maxsize=None)
def _unit_mesh_cached(family: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = _raw_mesh(solid_variant(family, index))
    _check_mesh(verts, faces)
    with decimal.localcontext(CTX):
        vol6 = Decimal(0)
        sx = sy = sz = Decimal(0)
        for a, b, c in faces:
            p0, p1, p2 = verts[a], verts[b], verts[c]
            d = _det(p0, p1, p2)
            vol6 += d
            sx += d * (p0[0] + p1[0] + p2[0])
            sy += d * (p0[1] + p1[1] + p2[1])
            sz += d * (p0[2] + p1[2] + p2[2])
        cx, cy, cz = sx / (4 * vol6), sy / (4 * vol6), sz / (4 * vol6)  # 四面体 (0, v0, v1, v2) 的质心 = (v0+v1+v2)/4
        pts = [(x - cx, y - cy, z - cz) for x, y, z in verts]
        scale = 1 / max(x * x + y * y + z * z for x, y, z in pts).sqrt()
        q = [tuple(quantize_dec(c * scale, UNIT_SHAPE_QUANTUM) for c in p) for p in pts]
    vs = np.array(q, dtype=np.float64)
    fs = np.array(faces, dtype=np.int64)
    vs.setflags(write=False)
    fs.setflags(write=False)
    return vs, fs


def unit_mesh(family: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    """(顶点 (V, 3) float64，三角面 (F, 3) int64)：原点在体积质心、外接半径 1、顶点在 2^-24 网格上，只读。"""
    return _unit_mesh_cached(family, int(index))


# ---------------------------------------------------------------------------
# 摆放姿态
# ---------------------------------------------------------------------------

# 局部 → 世界的姿态旋转（精确的 0、±1 矩阵，行主序）：直立；绕 z 转 90°（局部 x 朝上）；绕 x 转 90°（局部 z 朝上）
REST_MATRICES = {
    "upright": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "side_z": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "side_x": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}
if tuple(REST_MATRICES) != SOLID_REST_POSES:  # pragma: no cover - 常量表写错
    raise ImportError("REST_MATRICES 与 SOLID_REST_POSES 不一致")


# ---------------------------------------------------------------------------
# 对称性
# ---------------------------------------------------------------------------


def _norm2(a) -> float:
    return (a[0] * a[0] + a[1] * a[1]) + a[2] * a[2]


def _parallel(a, d) -> bool:
    cx = a[1] * d[2] - a[2] * d[1]
    cy = a[2] * d[0] - a[0] * d[2]
    cz = a[0] * d[1] - a[1] * d[0]
    return (cx * cx + cy * cy) + cz * cz <= SYMMETRY_TOL * SYMMETRY_TOL * _norm2(a) * _norm2(d)


def _perpendicular(a, d) -> bool:
    dot = (a[0] * d[0] + a[1] * d[1]) + a[2] * d[2]
    return dot * dot <= SYMMETRY_TOL * SYMMETRY_TOL * _norm2(a) * _norm2(d)


@functools.lru_cache(maxsize=None)
def discrete_symmetry_axes(family: str, index: int) -> tuple[tuple[tuple[float, float, float], int], ...]:
    """长方体与棱柱的有限旋转对称轴（局部坐标，未归一化的方向，k 重）。其余族的对称性由规则给出（见模块说明）。"""
    v = solid_variant(family, index)
    if v.family == "box":
        wx, wy, wz = v.params
        axes: list[tuple[tuple[float, float, float], int]] = [
            ((1.0, 0.0, 0.0), 4 if wy == wz else 2),
            ((0.0, 1.0, 0.0), 4 if wx == wz else 2),
            ((0.0, 0.0, 1.0), 4 if wx == wy else 2),
        ]
        if wx == wz:
            axes += [((1.0, 0.0, 1.0), 2), ((1.0, 0.0, -1.0), 2)]
        if wx == wy:
            axes += [((1.0, 1.0, 0.0), 2), ((1.0, -1.0, 0.0), 2)]
        if wy == wz:
            axes += [((0.0, 1.0, 1.0), 2), ((0.0, 1.0, -1.0), 2)]
        if wx == wy == wz:
            axes += [((1.0, 1.0, 1.0), 3), ((1.0, 1.0, -1.0), 3), ((1.0, -1.0, 1.0), 3), ((-1.0, 1.0, 1.0), 3)]
        return tuple(axes)
    if v.family == "prism":
        n = int(v.params[0])
        axes = [((0.0, 1.0, 0.0), n)]
        for j in range(n):
            c, s = rot_cs(Fraction(90) + Fraction(180 * j, n))
            axes.append(((c, 0.0, s), 2))
        return tuple(axes)
    return ()


def rotation_period_about(family: str, index: int, axis) -> float:
    """绕过中心、方向为 axis（局部坐标，不必归一化）的直线的旋转周期（度）：0 连续，360 无对称，否则 360/k。"""
    v = solid_variant(family, index)
    a = tuple(float(x) for x in axis)
    if _norm2(a) == 0.0:
        raise ValueError("旋转轴不能是零向量")
    y = (0.0, 1.0, 0.0)
    f = v.family
    if f == "sphere" and v.params[0] == 1.0:
        return 0.0
    if f in ("sphere", "cylinder", "torus"):
        if _parallel(a, y):
            return 0.0
        return 180.0 if _perpendicular(a, y) else 360.0
    if f == "cone":
        return 0.0 if _parallel(a, y) else 360.0
    if f == "l_prism":
        return 360.0
    k = 1
    for d, order in discrete_symmetry_axes(v.family, v.index):
        if _parallel(a, d):
            k = max(k, order)
    return 360.0 / k


# ---------------------------------------------------------------------------
# 身份与划分
# ---------------------------------------------------------------------------


def all_solid_identities() -> list[Identity]:
    return [(f, v.index, a.index) for f in SOLID_FAMILIES for v in SOLID_VARIANTS[f] for a in SOLID_APPEARANCES]


@functools.lru_cache(maxsize=None)
def solid_identity_split_table(holdout_family: str | None = None) -> dict[Identity, str]:
    """实体身份 → 划分（规则同 2D，见 shapes.assign_splits）；holdout_family 的全部身份进 test。"""
    if holdout_family is not None and holdout_family not in SOLID_FAMILIES:
        raise ValueError(f"未知的留出实体族 {holdout_family!r}")
    table: dict[Identity, str] = {}
    for f in SOLID_FAMILIES:
        for v in SOLID_VARIANTS[f]:
            ids = [(f, v.index, a.index) for a in SOLID_APPEARANCES]
            if f == holdout_family:
                table.update({i: "test" for i in ids})
                continue
            table.update(assign_splits(ids))
    return table


@functools.lru_cache(maxsize=None)
def solid_identity_pool(split: str, holdout_family: str | None = None) -> dict[tuple[str, int], tuple[int, ...]]:
    """该划分可用的实体身份：(族, 变体) → 可用外观序号（升序），按族、变体排序。"""
    table = solid_identity_split_table(holdout_family)
    pool: dict[tuple[str, int], list[int]] = {}
    for (f, v, a), s in table.items():
        if s == split:
            pool.setdefault((f, v), []).append(a)
    return {k: tuple(sorted(a)) for k, a in sorted(pool.items(), key=lambda kv: (SOLID_FAMILIES.index(kv[0][0]), kv[0][1]))}


def sample_solid_identity(rng: np.random.Generator, split: str, holdout_family: str | None = None, rotatable_only: bool = False) -> Identity:
    """先在可用族里均匀取族，再均匀取变体，再均匀取外观（都限定在该划分的身份池里）。

    rotatable_only：去掉任意旋转都不变的变体（球），a 用。
    """
    pool = solid_identity_pool(split, holdout_family)
    keys = [k for k in pool if not (rotatable_only and is_isotropic(*k))]
    fams = [f for f in SOLID_FAMILIES if any(k[0] == f for k in keys)]
    if not fams:
        raise ValueError(f"划分 {split} 里没有可用的实体身份")
    f = fams[int(rng.integers(len(fams)))]
    vs = [k[1] for k in keys if k[0] == f]
    v = vs[int(rng.integers(len(vs)))]
    apps = pool[(f, v)]
    return f, v, apps[int(rng.integers(len(apps)))]
