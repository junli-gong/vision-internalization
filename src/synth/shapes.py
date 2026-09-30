"""形状族、单位形状、外观、物体身份与身份划分、位姿变换（无学习；numpy + decimal）。光栅化规则在 raster.py。

单位形状
- 每个 (形状族, 变体) 有一个单位多边形：原点在它的面积质心，离原点最远的顶点距离为 1（外接半径 1），
  y 轴向下（图像约定）。曲线（圆、椭圆、月牙）按 CURVE_SEGMENTS 段/整圆折线化。
- 角度 0 时的朝向：椭圆与矩形宽不小于高；三角形、正多边形、星形顶点朝上；十字是正的加号；
  L 形的拐角在左下；月牙开口朝右。
- 旋转以质心为中心。对有 n 重旋转对称的形状，质心就是对称中心，rotation_period() 给出 360/n 度；
  圆为 0（任意旋转不变）。

位姿与坐标
- 物体位姿 = 质心 (cx, cy)（像素）、外接半径 radius（像素）、角度 angle（度，屏幕上顺时针为正）。
  画布坐标 = R(angle) · (radius · 单位顶点) + (cx, cy)，R = [[c, -s], [s, c]]；y 向下时正角度即屏幕顺时针。
  与 pymunk 的 Body.angle 约定相同（physics.py 用同一变换）。

跨机器逐位复现（DEV-0009，D24）
- 只有 +、-、×、÷、sqrt 是 IEEE 正确舍入的，任何机器上结果相同。平台 libm、numpy 的 SIMD 三角函数、编译过的 C 库
  （pycocotools、GEOS）的末位可能随平台不同，经过量化或取整后会放大成不同的样本（RUN-0006）。
- 所以数据路径上：
  * 三角函数只用 dmath（decimal，精确的参数约简）；rot_cs 量化到 TRIG_QUANTUM，90° 的整数倍给出精确值；
  * 单位形状在 decimal 里构造（顶点、质心、归一化），再量化到 UNIT_SHAPE_QUANTUM；
  * 画布顶点量化到 COORD_QUANTUM（1/256 px），之后只做逐元素的 numpy 初等运算
    （每个 ufunc 单独正确舍入，不会被融合成 FMA；不用 BLAS 矩阵乘，不依赖求和顺序）；
  * 光栅化用 raster.py 的精确整数规则。
- tests/test_synth_repro.py 的钉住摘要在服务器上跑一遍即是跨机器核对；同一文件另把 math、numpy 的三角函数、np.mean
  和 pycocotools 换成抛异常的桩再生成样本，证明它们不在数据路径上。

物体身份与划分（规则固定，写死在这里）
- 身份 = (形状族, 变体序号, 外观序号)。外观 = 图案（纯色、条纹、棋盘格）× 10 种颜色，共 30 种。
- 对每个 (形状族, 变体)，把它的 30 个身份按 sha256(IDENTITY_SPLIT_SALT + "|" + 身份键) 的十六进制串升序排列，
  依次取 n_train = n - n_val - n_test 个进 train、n_val 个进 val、其余 n_test 个进 test，
  n_val = floor(n * 0.1 + 0.5)，n_test = floor(n * 0.1 + 0.5)（n = 30 时为 24/3/3）。
  于是每个形状族、每个形状变体在三个划分里都出现（划分发生在实例层面），而具体身份两两不相交。
- 可选的"留出形状族"模式（holdout_family，默认关闭）：该族的全部身份只进 test；其余族照上面的规则划分。
  这个选项必须对三个划分一致地传入。
"""

from __future__ import annotations

import decimal
import functools
import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

import numpy as np

from src.synth.constants import (
    COLORS,
    COORD_QUANTUM,
    CRESCENT_CUTS,
    CROSS_ARM_WIDTHS,
    CURVE_SEGMENTS,
    ELLIPSE_AXIS_RATIOS,
    IDENTITY_SPLIT_FRACTIONS,
    IDENTITY_SPLIT_SALT,
    L_ARM_WIDTHS,
    L_WIDTH_RATIOS,
    PATTERN_SECOND_COLOR,
    PATTERNS,
    POLYGON_SIDES,
    RECT_ASPECTS,
    SHAPE_FAMILIES,
    SPLITS,
    STAR_INNER_RATIOS,
    STAR_POINTS,
    TRIANGLE_VARIANTS,
    UNIT_SHAPE_QUANTUM,
)
from src.synth.dmath import CTX, atan2_deg, cos_sin_deg, quantize_dec
from src.synth.dmath import rot_cs  # noqa: F401  转出给 scene、各任务模块与测试；定义只在 dmath.py
from src.synth.raster import rasterize  # noqa: F401  同上；光栅化规则只在 raster.py 定义

# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------


def quantize(x, step: float):
    """四舍五入到 step 的整数倍（step 是 2 的幂，乘除无舍入误差；np.round 取偶）。"""
    a = np.asarray(x, dtype=np.float64)
    q = np.round(a / step) * step
    return float(q) if q.ndim == 0 else q


def qcoord(x):
    return quantize(x, COORD_QUANTUM)


# ---------------------------------------------------------------------------
# 形状族与变体
# ---------------------------------------------------------------------------

_POLY_NAMES = {5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}


@dataclass(frozen=True)
class Variant:
    family: str
    index: int
    params: tuple
    noun: str  # 场景里的简称（c、d 的指令用）："hexagon"、"star"、"square"……
    phrase: str  # e 的文字里用的完整描述，含离散化的形状参数
    period_deg: float  # 绕质心的旋转周期（度）；0 = 任意旋转不变（圆）


def _f2(x: float) -> str:
    return f"{x:.2f}"


def _build_variants() -> dict[str, tuple[Variant, ...]]:
    out: dict[str, list[Variant]] = {f: [] for f in SHAPE_FAMILIES}

    def add(family, params, noun, phrase, period):
        out[family].append(Variant(family, len(out[family]), tuple(params), noun, phrase, float(period)))

    add("circle", (), "circle", "circle", 0.0)
    for q in ELLIPSE_AXIS_RATIOS:
        add("ellipse", (q,), "ellipse", f"ellipse (axis ratio {_f2(q)})", 180.0)
    for a in RECT_ASPECTS:
        if a == 1.0:
            add("rectangle", (a,), "square", "square", 90.0)
        else:
            add("rectangle", (a,), "rectangle", f"rectangle (height {_f2(a)} of width)", 180.0)
    for kind, p in TRIANGLE_VARIANTS:
        if kind == "equilateral":
            add("triangle", (kind, p), "triangle", "equilateral triangle", 120.0)
        elif kind == "isosceles":
            add("triangle", (kind, p), "triangle", f"isosceles triangle (height {_f2(p)} of base)", 360.0)
        elif kind == "right":
            add("triangle", (kind, p), "triangle", f"right triangle (bottom leg {_f2(p)} of left leg)", 360.0)
        else:  # pragma: no cover - 常量表写错
            raise ValueError(kind)
    for n in POLYGON_SIDES:
        add("polygon", (n,), _POLY_NAMES[n], f"regular {_POLY_NAMES[n]}", 360.0 / n)
    for n in STAR_POINTS:
        for rho in STAR_INNER_RATIOS:
            add("star", (n, rho), "star", f"{n}-pointed star (inner radius {_f2(rho)} of outer)", 360.0 / n)
    for w in CROSS_ARM_WIDTHS:
        add("cross", (w,), "cross", f"plus-shaped cross (arm width {_f2(w)} of span)", 90.0)
    for t in L_ARM_WIDTHS:
        for wr in L_WIDTH_RATIOS:
            add("l_shape", (t, wr), "L-shape", f"L-shape (arm width {_f2(t)} of height, width {_f2(wr)} of height)", 360.0)
    for rho, d in CRESCENT_CUTS:
        add("crescent", (rho, d), "crescent", f"crescent (cut by a circle of radius {_f2(rho)} shifted {_f2(d)} right)", 360.0)
    return {f: tuple(v) for f, v in out.items()}


VARIANTS: dict[str, tuple[Variant, ...]] = _build_variants()


def variant(family: str, index: int) -> Variant:
    if family not in VARIANTS:
        raise ValueError(f"未知形状族 {family!r}")
    vs = VARIANTS[family]
    if not 0 <= int(index) < len(vs):
        raise ValueError(f"{family} 没有变体 {index}（共 {len(vs)} 个）")
    return vs[int(index)]


def rotation_period(family: str, index: int) -> float:
    return variant(family, index).period_deg


def max_equivalent_rotation(period_deg: float) -> int:
    """按对称性约简后的旋转角上限（整数度）：|θ| 不超过半个周期（最多 180°），任意角度不变（周期 0）时为 0。

    e（绕质心的平面旋转）与 a（出平面三维旋转，周期由 solids.rotation_period_about 给出）共用这条约定：
    结果与更小的等价旋转相同时，只报告最小的等价角。
    """
    if period_deg == 0.0:
        return 0
    return int(math.floor(min(180.0, period_deg / 2.0)))


# ---------------------------------------------------------------------------
# 单位多边形
# ---------------------------------------------------------------------------


def _dec(x) -> Decimal:
    """形状常量 → Decimal：int 原样；float 取它的十进制写法（0.45 → Decimal('0.45')），即常量表里写的那个数。"""
    if isinstance(x, (int, Decimal)):
        return Decimal(x)
    return Decimal(repr(float(x)))


Point = tuple[Decimal, Decimal]


def _arc(cx, cy, r, a0, a1, include_end: bool) -> list[Point]:
    """圆心 (cx, cy)、半径 r 的圆弧点，角度从 a0 到 a1（度，a1 可小于 a0）。

    段数 n = max(4, ceil(CURVE_SEGMENTS · |a1 - a0| / 360))，第 k 个点的角度 = a0 + (a1 - a0)·k/n。全程 decimal。
    """
    with decimal.localcontext(CTX):
        a0, span = _dec(a0), _dec(a1) - _dec(a0)
        n = max(4, int((CURVE_SEGMENTS * span.copy_abs() / 360).to_integral_value(rounding=decimal.ROUND_CEILING)))
        pts = []
        for k in range(n + 1 if include_end else n):
            c, s = cos_sin_deg(a0 + span * k / n)
            pts.append((cx + r * c, cy + r * s))
        return pts


def _on_unit_circle(angles, radii=None) -> list[Point]:
    """角度（度，精确的有理数）→ 单位圆（或给定半径）上的点。"""
    out = []
    with decimal.localcontext(CTX):
        for k, a in enumerate(angles):
            c, s = cos_sin_deg(a)
            r = Decimal(1) if radii is None else radii[k]
            out.append((r * c, r * s))
    return out


def _raw_polygon(v: Variant) -> list[Point]:
    """单位形状的原始顶点（Decimal），还没有移到质心、没有归一化。朝向约定见模块说明。"""
    f, p = v.family, v.params
    half = Decimal("0.5")
    with decimal.localcontext(CTX):
        if f == "circle":
            return _arc(0, 0, 1, 0, 360, include_end=False)
        if f == "ellipse":
            q = _dec(p[0])
            return [(x, y * q) for x, y in _arc(0, 0, 1, 0, 360, include_end=False)]
        if f == "rectangle":
            a = _dec(p[0])
            return [(Decimal(-1), -a), (Decimal(1), -a), (Decimal(1), a), (Decimal(-1), a)]
        if f == "triangle":
            kind, q = p
            q = _dec(q)
            if kind == "equilateral":  # 顶点朝上
                return _on_unit_circle([-90 + 120 * k for k in range(3)])
            if kind == "isosceles":  # 底边 1、高 q，顶点朝上
                return [(Decimal(0), -q / 2), (half, q / 2), (-half, q / 2)]
            # 直角三角形：竖直直角边 1 在左、水平直角边 q 在下，直角在左下
            return [(-q / 2, -half), (q / 2, half), (-q / 2, half)]
        if f == "polygon":
            n = p[0]
            return _on_unit_circle([Fraction(-90 * n + 360 * k, n) for k in range(n)])
        if f == "star":
            n, rho = p[0], _dec(p[1])
            return _on_unit_circle(
                [Fraction(-90 * n + 180 * k, n) for k in range(2 * n)], [Decimal(1) if k % 2 == 0 else rho for k in range(2 * n)]
            )
        if f == "cross":
            w = _dec(p[0])
            one = Decimal(1)
            return [(-w, -one), (w, -one), (w, -w), (one, -w), (one, w), (w, w), (w, one), (-w, one), (-w, w), (-one, w), (-one, -w), (-w, -w)]
        if f == "l_shape":
            t, wr = _dec(p[0]), _dec(p[1])  # 高 1（y ∈ [-0.5, 0.5]），宽 wr，臂宽 t；竖臂在左、横臂在下
            x0, x1 = -wr / 2, wr / 2
            return [(x0, -half), (x0 + t, -half), (x0 + t, half - t), (x1, half - t), (x1, half), (x0, half)]
        if f == "crescent":
            rho, d = _dec(p[0]), _dec(p[1])  # 外圆半径 1、圆心原点；切割圆半径 rho、圆心 (d, 0)
            xs = (1 - rho * rho + d * d) / (2 * d)
            ys = (1 - xs * xs).sqrt()
            phi0 = atan2_deg(ys, xs)
            psi0 = atan2_deg(ys, xs - d)
            outer = _arc(0, 0, 1, phi0, 360 - phi0, include_end=True)  # 左侧外弧，端点是两交点
            inner = _arc(d, 0, rho, 360 - psi0, psi0, include_end=False)[1:]  # 左侧内弧，去掉两端交点
            return outer + inner
    raise ValueError(f"未知形状族 {f!r}")  # pragma: no cover


@functools.lru_cache(maxsize=None)
def _unit_polygon_cached(family: str, index: int) -> np.ndarray:
    """原始顶点 → 移到面积质心（鞋带公式）→ 除以最远顶点的距离 → 量化到 UNIT_SHAPE_QUANTUM。全程 decimal。"""
    raw = _raw_polygon(variant(family, index))
    n = len(raw)
    with decimal.localcontext(CTX):
        a2 = sx = sy = Decimal(0)
        for i in range(n):
            (x0, y0), (x1, y1) = raw[i], raw[(i + 1) % n]
            cr = x0 * y1 - x1 * y0
            a2 += cr
            sx += (x0 + x1) * cr
            sy += (y0 + y1) * cr
        cx, cy = sx / (3 * a2), sy / (3 * a2)
        pts = [(x - cx, y - cy) for x, y in raw]
        scale = 1 / max(x * x + y * y for x, y in pts).sqrt()
        q = [(quantize_dec(x * scale, UNIT_SHAPE_QUANTUM), quantize_dec(y * scale, UNIT_SHAPE_QUANTUM)) for x, y in pts]
    p = np.array(q, dtype=np.float64)
    p.setflags(write=False)
    return p


def unit_polygon(family: str, index: int) -> np.ndarray:
    """单位多边形 (V, 2)：原点在质心，外接半径 1（量化后误差 ≤ 2^-24），只读。"""
    return _unit_polygon_cached(family, int(index))


def pose_polygon(family: str, index: int, cx: float, cy: float, radius: float, angle_deg: float) -> np.ndarray:
    """位姿 → 画布上的多边形顶点 (V, 2)，量化到 COORD_QUANTUM。全程逐元素运算。"""
    u = unit_polygon(family, index)
    c, s = rot_cs(angle_deg)
    x = u[:, 0] * float(radius)
    y = u[:, 1] * float(radius)
    px = x * c - y * s + float(cx)
    py = x * s + y * c + float(cy)
    return qcoord(np.stack([px, py], axis=1))


# ---------------------------------------------------------------------------
# 画布内判定（光栅化见 raster.py）
# ---------------------------------------------------------------------------


def polygon_inside(poly: np.ndarray, height: int, width: int, margin: float) -> bool:
    """全部顶点在 [margin, W - margin] × [margin, H - margin] 内。"""
    p = np.asarray(poly)
    return bool(
        p[:, 0].min() >= margin and p[:, 0].max() <= width - margin and p[:, 1].min() >= margin and p[:, 1].max() <= height - margin
    )


# ---------------------------------------------------------------------------
# 外观
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Appearance:
    index: int
    pattern: str
    color_name: str
    rgb: tuple[int, int, int]

    @property
    def phrase(self) -> str:
        if self.pattern == "solid":
            return self.color_name
        return f"{self.color_name}-and-{PATTERN_SECOND_COLOR[0]} {self.pattern}"

    @property
    def colors(self) -> tuple[tuple[int, int, int], ...]:
        """这个外观在图像里可能出现的全部颜色。"""
        if self.pattern == "solid":
            return (self.rgb,)
        return (self.rgb, PATTERN_SECOND_COLOR[1])


APPEARANCES: tuple[Appearance, ...] = tuple(
    Appearance(i * len(COLORS) + j, pat, name, rgb) for i, pat in enumerate(PATTERNS) for j, (name, rgb) in enumerate(COLORS)
)


def appearance(index: int) -> Appearance:
    return APPEARANCES[int(index)]


def describe(family: str, variant_index: int, appearance_index: int) -> str:
    """c、d 指令里的简称：外观 + 名词，如 "red-and-black striped hexagon"。"""
    return f"{appearance(appearance_index).phrase} {variant(family, variant_index).noun}"


# ---------------------------------------------------------------------------
# 身份与划分
# ---------------------------------------------------------------------------

Identity = tuple[str, int, int]  # (形状族, 变体序号, 外观序号)


def identity_key(identity: Identity) -> str:
    f, v, a = identity
    return f"{f}:{int(v)}:{int(a)}"


def all_identities() -> list[Identity]:
    return [(f, v.index, a.index) for f in SHAPE_FAMILIES for v in VARIANTS[f] for a in APPEARANCES]


def _split_counts(n: int) -> dict[str, int]:
    fr = dict(IDENTITY_SPLIT_FRACTIONS)
    n_val = int(math.floor(n * fr["val"] + 0.5))
    n_test = int(math.floor(n * fr["test"] + 0.5))
    return {"train": n - n_val - n_test, "val": n_val, "test": n_test}


def assign_splits(ids: list[Identity]) -> dict[Identity, str]:
    """一个 (族, 变体) 的全部身份 → 划分：按 sha256(IDENTITY_SPLIT_SALT | 身份键) 升序，依次取 train、val、test 的个数。

    2D（本模块）与 3D（solids.py）共用这一条规则。
    """
    ordered = sorted(ids, key=lambda i: hashlib.sha256(f"{IDENTITY_SPLIT_SALT}|{identity_key(i)}".encode()).hexdigest())
    cnt = _split_counts(len(ordered))
    out: dict[Identity, str] = {}
    start = 0
    for split in SPLITS:
        for i in ordered[start : start + cnt[split]]:
            out[i] = split
        start += cnt[split]
    return out


@functools.lru_cache(maxsize=None)
def identity_split_table(holdout_family: str | None = None) -> dict[Identity, str]:
    """身份 → 划分。规则见模块说明；holdout_family 的全部身份进 test。"""
    if holdout_family is not None and holdout_family not in SHAPE_FAMILIES:
        raise ValueError(f"未知的留出形状族 {holdout_family!r}")
    table: dict[Identity, str] = {}
    for f in SHAPE_FAMILIES:
        for v in VARIANTS[f]:
            ids = [(f, v.index, a.index) for a in APPEARANCES]
            if f == holdout_family:
                table.update({i: "test" for i in ids})
                continue
            table.update(assign_splits(ids))
    return table


@functools.lru_cache(maxsize=None)
def identity_pool(split: str, holdout_family: str | None = None) -> dict[tuple[str, int], tuple[int, ...]]:
    """该划分可用的身份：(形状族, 变体) → 可用外观序号（升序）。"""
    table = identity_split_table(holdout_family)
    pool: dict[tuple[str, int], list[int]] = {}
    for (f, v, a), s in table.items():
        if s == split:
            pool.setdefault((f, v), []).append(a)
    return {k: tuple(sorted(a)) for k, a in sorted(pool.items(), key=lambda kv: (SHAPE_FAMILIES.index(kv[0][0]), kv[0][1]))}


def sample_identity(rng: np.random.Generator, split: str, families, holdout_family: str | None = None) -> Identity:
    """先在可用形状族里均匀取族，再均匀取变体，再均匀取外观（都限定在该划分的身份池里）。"""
    pool = identity_pool(split, holdout_family)
    fams = [f for f in families if any(k[0] == f for k in pool)]
    if not fams:
        raise ValueError(f"划分 {split} 里没有 {tuple(families)} 的身份")
    f = fams[int(rng.integers(len(fams)))]
    vs = [k[1] for k in pool if k[0] == f]
    v = vs[int(rng.integers(len(vs)))]
    apps = pool[(f, v)]
    return f, v, apps[int(rng.integers(len(apps)))]
