"""3D 场景规格、相机与自写的确定性光栅化（DEV-0010）。不用 pyrender、OpenGL、EGL、Blender、Kubric；无学习组件。

坐标
- 世界：右手系，y 朝上；b 的圆形地板在 y = 0。物体位姿 = 中心（世界坐标，米）+ 姿态 = Ry(偏航) · 摆放旋转（solids.REST_MATRICES）
  + 缩放（外接半径，米）。
- 相机：看向 target、方位角 az、俯角 el、距离 D，再加一个世界坐标里的平移 shift（b 的平移移动）：
  位置 C = target + D·(cos el·sin az, sin el, cos el·cos az) + shift；朝向只由 (az, el) 决定，没有滚转。
  相机坐标系 x 右、y 下、z 前（右手系），三个轴在世界里是
  right = (cos az, 0, −sin az)，down = (sin el·sin az, −cos el, sin el·cos az)，forward = −(cos el·sin az, sin el, cos el·cos az)。
  方位角增大 = 相机向它自己的右边移动。深度 = 相机 z（平面深度，米）。
- 针孔投影：u = X·f / Z + W/2，v = Y·f / Z + H/2，f = CAMERA_FOCAL_FRAC × min(H, W)，像素 (i, j) 的中心是 (j + 0.5, i + 0.5)。

确定性（D24 的标准：任何平台逐位相同）
- 三角函数只用 dmath.rot_cs（decimal、精确参数约简、量化到 2^-40）。矩阵乘、矩阵乘向量在 Python float 上按固定顺序
  ((a·b + c·d) + e·f) 计算；顶点变换用逐元素的 numpy 运算（每个 ufunc 单独正确舍入，不融合成 FMA，不用 BLAS）。
  不用 np.linalg、np.dot、np.matmul、np.einsum，也不对浮点数组求和。
- 量化点：相机坐标系里的顶点坐标 → VERTEX_QUANTUM_M；投影坐标 → 1/256 px（raster.py 的整数单位，之后全是 int64）；
  面法向分量 → NORMAL_QUANTUM；像素深度 → DEPTH_QUANTUM_M 的整数单位；着色 → 整数运算。

光栅化（同一个光栅化器同时给出物体 ID 图、深度图、Lambert 着色的 RGB，三者按构造对齐）
- 覆盖：像素中心 + 整数边函数，边界规则与 raster.py 相同（像素中心落在边上时取 (lo, hi]）。等价地说：像素 p 被三角形覆盖
  ⟺ p + δ 严格在三角形内，δ = (−ε, −ε²)，ε → 0+（先向左、再向上的无穷小扰动）。对边 a→b（三角形内部在 E > 0 一侧），
  E(p) = (b_x − a_x)(p_y − a_y) − (b_y − a_y)(p_x − a_x)；E(p) = 0 时，这条边"拥有"p ⟺ b_y > a_y，或 b_y = a_y 且 b_x < a_x。
  于是共用一条边的两个三角形把边上的像素中心分给恰好一方，不重不漏；轴对齐矩形的结果与 raster.py、box_to_mask 相同。
  实现（DEV-0028 提速）：每个三角形的每一行，用整数 floor 除法直接解出满足三条边不等式的列区间，只枚举被覆盖的像素；
  与在包围盒里逐个像素判定的旧做法选出同一个集合、同一个顺序（tests/test_synth_convex_relief.py 对照旧版本）。
- 背面剔除：投影后的有向面积 A2 = (u1−u0)(v2−v0) − (v1−v0)(u2−u0)（整数，精确）< 0 为正面（法向朝相机）。
  闭合网格的任一被覆盖点上正面数 = 背面数，所以量化后剔除不会留洞。
- 深度：1/Z 在屏幕上是仿射的：1/Z(p) = (E₁₂/Z₀ + E₂₀/Z₁ + E₀₁/Z₂) / A2，E 是精确整数；按固定顺序的浮点运算求值，
  再四舍五入（取偶）到 DEPTH_QUANTUM_M 的整数单位。
- 可见性：z-buffer 比较量化后的深度；等深时优先级小的赢（物体按 obj_id、面按网格顺序编号），所以结果与计算顺序无关。
  实现：每个 (像素, 三角形) 候选对编成一个 int64 键 (像素 << 36) | (深度 << 20) | 优先级，排序后每个像素取最小键。
- 着色：面法向 = 相机坐标里 (v1 − v0) × (v2 − v0)（朝外），归一化后量化；亮度 = (AMB + DIFF·max(0, n·l)) / 256，
  颜色 = 底色 × 亮度，整数运算。每个三角面一个颜色（平面着色）。
- 地板（b）：圆形地板折线化成 VIEW_FLOOR_SEGMENTS 边形，投影后用 raster.rasterize_units 光栅化（同一边界规则）；
  深度按平面精确求：相机没有滚转，地板上一行像素的深度相同，Z = C_y / (cos el·(v − H/2)/f + sin el)。
  相机在地板上方、物体都在地板上方，所以沿任一视线物体总在地板之前（或与地板等深），物体直接盖住地板。
- 深度量程：场景只在全部顶点的深度都在 [D_MIN + DEPTH_RANGE_MARGIN_M, D_MAX − DEPTH_RANGE_MARGIN_M] 内时才渲染
  （vertices_ok；像素深度是顶点深度的调和插值，落在同一区间里），所以深度图里的深度都在 src.constants 的 D_MIN–D_MAX 内。
"""

from __future__ import annotations

import dataclasses
import decimal
import math
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from src.constants import D_MAX, D_MIN
from src.synth.constants import (
    CAMERA_FOCAL_FRAC,
    DEPTH_QUANTUM_M,
    FLOOR_ID,
    LIGHT_DIR_CAM,
    NORMAL_QUANTUM,
    SHADE_AMBIENT_256,
    SHADE_DIFFUSE_256,
    VERTEX_QUANTUM_M,
)
from src.synth.dmath import CTX, quantize_dec, rot_cs
from src.synth.raster import HALF, MAX_ABS_COORD_PX, UNITS, rasterize_units
from src.synth.relief import cell_depth_units
from src.synth.shapes import appearance
from src.synth.solids import REST_MATRICES, describe_solid, unit_mesh

Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vec3 = tuple[float, float, float]

DEPTH_SCALE = 1.0 / DEPTH_QUANTUM_M  # 1 m = DEPTH_SCALE 个深度单位（2 的幂）
DEPTH_RANGE_MARGIN_M = 0.01  # 顶点深度离 D_MIN、D_MAX 至少这么远（像素中心取在量化后的多边形里，留一点余量）
_KEY_PIX_SHIFT, _KEY_Z_SHIFT = 36, 20
_MAX_DEPTH_UNITS = 1 << (_KEY_PIX_SHIFT - _KEY_Z_SHIFT)
_MAX_PRIO = 1 << _KEY_Z_SHIFT
if D_MAX * DEPTH_SCALE >= _MAX_DEPTH_UNITS:  # pragma: no cover - 常量写错
    raise ImportError("DEPTH_QUANTUM_M 太细：深度单位放不进 z-buffer 的键")


def _light() -> Vec3:
    with decimal.localcontext(CTX):
        v = [Decimal(repr(float(x))) for x in LIGHT_DIR_CAM]
        n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        return tuple(quantize_dec(x / n, NORMAL_QUANTUM) for x in v)


LIGHT: Vec3 = _light()  # 相机坐标系里指向光源的单位向量（量化）


# ---------------------------------------------------------------------------
# 3×3 矩阵（Python float，固定运算顺序）
# ---------------------------------------------------------------------------

IDENTITY3: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rot3(axis: str, deg) -> Mat3:
    """绕坐标轴 x / y / z 的右手旋转（从轴的正方向看过去逆时针），角度为度。"""
    c, s = rot_cs(deg)
    if axis == "x":
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == "y":
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    if axis == "z":
        return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    raise ValueError(f"未知的旋转轴 {axis!r}")


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple((a[i][0] * b[0][j] + a[i][1] * b[1][j]) + a[i][2] * b[2][j] for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def mat_vec(a: Mat3, v) -> Vec3:
    return tuple((a[i][0] * v[0] + a[i][1] * v[1]) + a[i][2] * v[2] for i in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 场景规格
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraSpec:
    azimuth_deg: int
    elevation_deg: int
    distance: float  # 米
    target: Vec3 = (0.0, 0.0, 0.0)
    shift: Vec3 = (0.0, 0.0, 0.0)  # 世界坐标里的平移（米）

    def basis(self) -> Mat3:
        """世界 → 相机的旋转：三行依次是相机 x（右）、y（下）、z（前）在世界里的方向。"""
        ca, sa = rot_cs(self.azimuth_deg)
        ce, se = rot_cs(self.elevation_deg)
        cs, cc = ce * sa, ce * ca
        return ((ca, 0.0, -sa), (se * sa, -ce, se * ca), (-cs, -se, -cc))

    def position(self) -> Vec3:
        ca, sa = rot_cs(self.azimuth_deg)
        ce, se = rot_cs(self.elevation_deg)
        d = (ce * sa, se, ce * ca)
        return tuple((self.target[i] + self.distance * d[i]) + self.shift[i] for i in range(3))  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return {
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "distance": self.distance,
            "target": list(self.target),
            "shift": list(self.shift),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraSpec":
        return cls(d["azimuth_deg"], d["elevation_deg"], d["distance"], tuple(d["target"]), tuple(d["shift"]))


@dataclass(frozen=True)
class SolidSpec:
    obj_id: int
    family: str
    variant: int
    appearance: int
    rest: str  # SOLID_REST_POSES 之一
    yaw_deg: int  # 绕世界 y 轴的右手旋转
    radius: float  # 外接半径（米）
    center: Vec3  # 中心（体积质心）的世界坐标（米）

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.family, int(self.variant), int(self.appearance))

    @property
    def description(self) -> str:
        return describe_solid(self.family, self.variant, self.appearance)

    def rotation(self) -> Mat3:
        """局部 → 世界：Ry(偏航) · 摆放旋转。"""
        return mat_mul(rot3("y", self.yaw_deg), REST_MATRICES[self.rest])

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["center"] = list(self.center)
        return d


@dataclass(frozen=True)
class FloorSpec:
    radius: float
    segments: int
    color: tuple[int, int, int]


@dataclass(frozen=True)
class Scene3D:
    height: int
    width: int
    background: tuple[int, int, int]
    camera: CameraSpec
    objects: tuple[SolidSpec, ...]
    floor: FloorSpec | None = None

    def __post_init__(self) -> None:
        if [o.obj_id for o in self.objects] != list(range(len(self.objects))):
            raise ValueError("物体的 obj_id 必须依次是 0, 1, 2, ...")
        if len(self.objects) >= FLOOR_ID:
            raise ValueError(f"物体太多：ID 图最多 {FLOOR_ID - 1} 个物体")

    @property
    def focal(self) -> float:
        return CAMERA_FOCAL_FRAC * min(self.height, self.width)

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "background": list(self.background),
            "camera": self.camera.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
            "floor": None if self.floor is None else {"radius": self.floor.radius, "segments": self.floor.segments, "color": list(self.floor.color)},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene3D":
        fl = d.get("floor")
        objs = tuple(
            SolidSpec(o["obj_id"], o["family"], o["variant"], o["appearance"], o["rest"], o["yaw_deg"], o["radius"], tuple(o["center"]))
            for o in d["objects"]
        )
        return cls(
            int(d["height"]),
            int(d["width"]),
            tuple(d["background"]),
            CameraSpec.from_dict(d["camera"]),
            objs,
            None if fl is None else FloorSpec(fl["radius"], int(fl["segments"]), tuple(fl["color"])),
        )


def qscene(x: float, step: float) -> float:
    """标量四舍五入（取偶）到 step（2 的负整数次幂）的整数倍；0 给出 +0.0。"""
    return round(x / step) * step


def _qarr(a: np.ndarray, step: float) -> np.ndarray:
    return np.rint(a / step) * step


# ---------------------------------------------------------------------------
# 顶点变换与投影
# ---------------------------------------------------------------------------


def camera_vertices(obj: SolidSpec, camera: CameraSpec, extra: Mat3 | None = None) -> np.ndarray:
    """物体网格顶点在相机坐标系里的位置 (V, 3)，量化到 VERTEX_QUANTUM_M。

    extra：在相机坐标系里、绕物体中心再做的旋转（a 的出平面旋转）：p = extra · B · R_obj · (s·v) + B · (c − C)。
    """
    b = camera.basis()
    m = mat_mul(b, obj.rotation())
    if extra is not None:
        m = mat_mul(extra, m)
    cpos = camera.position()
    t = mat_vec(b, tuple(obj.center[i] - cpos[i] for i in range(3)))
    v, _ = unit_mesh(obj.family, obj.variant)
    s = float(obj.radius)
    x, y, z = v[:, 0] * s, v[:, 1] * s, v[:, 2] * s
    out = np.empty((v.shape[0], 3), dtype=np.float64)
    for i in range(3):
        out[:, i] = ((m[i][0] * x + m[i][1] * y) + m[i][2] * z) + t[i]
    return _qarr(out, VERTEX_QUANTUM_M)


def floor_vertices(floor: FloorSpec, camera: CameraSpec) -> np.ndarray:
    """地板多边形顶点（世界 (R cos a_k, 0, R sin a_k)，a_k = 360°k/n）在相机坐标系里的位置 (n, 3)，量化到 VERTEX_QUANTUM_M。"""
    b = camera.basis()
    cpos = camera.position()
    pts = []
    for k in range(floor.segments):
        c, s = rot_cs(360.0 * k / floor.segments)
        pts.append(mat_vec(b, (floor.radius * c - cpos[0], 0.0 - cpos[1], floor.radius * s - cpos[2])))
    return _qarr(np.array(pts, dtype=np.float64), VERTEX_QUANTUM_M)


def intrinsics(height: int, width: int) -> tuple[float, float, float]:
    """(f, cx, cy)：f = CAMERA_FOCAL_FRAC × min(H, W)，主点在画布中心。"""
    return CAMERA_FOCAL_FRAC * min(height, width), width / 2.0, height / 2.0


def project_float(p: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """相机坐标 (N, 3) → 像素坐标 (u, v)，浮点（未量化）。"""
    f, cx, cy = intrinsics(height, width)
    return (p[:, 0] * f) / p[:, 2] + cx, (p[:, 1] * f) / p[:, 2] + cy


def project_units(p: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """相机坐标 (N, 3) → 投影坐标，四舍五入（取偶）到 1/256 px 的 int64 整数单位（raster.py 的单位）。"""
    u, v = project_float(p, height, width)
    return np.rint(u * UNITS).astype(np.int64), np.rint(v * UNITS).astype(np.int64)


def vertices_ok(p: np.ndarray, height: int, width: int) -> bool:
    """全部顶点的深度在 [D_MIN + 余量, D_MAX − 余量] 内，投影坐标在 ±MAX_ABS_COORD_PX 内。"""
    z = p[:, 2]
    if not (float(z.min()) >= D_MIN + DEPTH_RANGE_MARGIN_M and float(z.max()) <= D_MAX - DEPTH_RANGE_MARGIN_M):
        return False
    u, v = project_float(p, height, width)
    return bool(np.all(np.abs(u) <= MAX_ABS_COORD_PX) and np.all(np.abs(v) <= MAX_ABS_COORD_PX))


def inside_canvas(p: np.ndarray, height: int, width: int, margin_px: float) -> bool:
    """全部顶点投影后在 [margin, W − margin] × [margin, H − margin] 内（于是物体的轮廓完整在画布里）。"""
    u, v = project_float(p, height, width)
    return bool(
        float(u.min()) >= margin_px and float(u.max()) <= width - margin_px and float(v.min()) >= margin_px and float(v.max()) <= height - margin_px
    )


# ---------------------------------------------------------------------------
# 着色
# ---------------------------------------------------------------------------


def shade_rgb(base_rgb, lam256: np.ndarray) -> np.ndarray:
    """Lambert：lam256 = round(max(0, n·l) · 256) ∈ [0, 256] → (N, 3) uint8。全程整数运算。"""
    lam = np.asarray(lam256, dtype=np.int64)
    shade = SHADE_AMBIENT_256 + (SHADE_DIFFUSE_256 * lam + 128) // 256
    base = np.asarray(base_rgb, dtype=np.int64)
    return ((base[None, :] * shade[:, None] + 128) // 256).astype(np.uint8)


def _lambert256(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray) -> np.ndarray:
    """未归一化的法向 → 归一化、量化到 NORMAL_QUANTUM → max(0, n·l) 量化到 1/256 的整数。"""
    ln = np.sqrt((nx * nx + ny * ny) + nz * nz)
    qx, qy, qz = (_qarr(c / ln, NORMAL_QUANTUM) for c in (nx, ny, nz))
    dot = (qx * LIGHT[0] + qy * LIGHT[1]) + qz * LIGHT[2]
    return np.rint(np.maximum(dot, 0.0) * 256.0).astype(np.int64)


# ---------------------------------------------------------------------------
# 光栅化
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """一次渲染的结果。id_map：0 = 没有表面，k + 1 = 物体 k，FLOOR_ID = 地板；depth_units：深度（DEPTH_QUANTUM_M 的整数，0 = 没有）。"""

    id_map: np.ndarray  # (H, W) uint8
    depth_units: np.ndarray  # (H, W) int64
    rgb: np.ndarray | None  # (H, W, 3) uint8
    amodal: np.ndarray | None  # (K, H, W) bool：每个物体的完整轮廓（不管遮挡）

    def depth_m(self) -> np.ndarray:
        """(H, W) float32 米（DEPTH_QUANTUM_M 的整数倍，精确），0 = 没有表面。"""
        return (self.depth_units.astype(np.float64) * DEPTH_QUANTUM_M).astype(np.float32)

    def mask(self, obj_id: int) -> np.ndarray:
        return self.id_map == obj_id + 1


def _edge_coeffs(ax, ay, bx, by):
    """有向边 a→b 的边函数 E(p) = A·p_x + B·p_y + C（整数），以及 E = 0 时是否计入（bias = 0 计入，1 不计入）。"""
    dy, dx = by - ay, bx - ax
    own = (dy > 0) | ((dy == 0) & (dx < 0))
    return -dy, dx, dy * ax - dx * ay, (~own).astype(np.int64)


def rasterize_triangles(
    u: np.ndarray, v: np.ndarray, inv_z: np.ndarray, tris: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """正面三角形的覆盖、深度与 z-buffer。

    u, v：顶点投影坐标（int64，1/256 px 单位）；inv_z：顶点的 1/Z（float64）；tris：(T, 3) 顶点下标，已排成 A2 > 0 的顺序，
    行号即优先级（等深时小的赢）。返回四项：每个被覆盖像素的赢家 (像素下标, 深度单位, 三角形下标) 三个数组，
    以及全部被覆盖的 (三角形, 像素下标) 对，形状 (2, M)（不管遮挡，给完整轮廓用）。像素下标 = 行 · W + 列。
    """
    h, w = int(height), int(width)
    if h * w >= 1 << (63 - _KEY_PIX_SHIFT):
        raise ValueError("画布太大：像素下标放不进 z-buffer 的键")
    n_tri = tris.shape[0]
    if n_tri >= _MAX_PRIO:
        raise ValueError("三角形太多")
    empty = np.zeros(0, dtype=np.int64)
    if n_tri == 0:
        return empty, empty, empty, np.zeros((2, 0), dtype=np.int64)
    x0, y0 = u[tris[:, 0]], v[tris[:, 0]]
    x1, y1 = u[tris[:, 1]], v[tris[:, 1]]
    x2, y2 = u[tris[:, 2]], v[tris[:, 2]]
    a2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if np.any(a2 <= 0):
        raise ValueError("rasterize_triangles 只接受 A2 > 0 的三角形")
    # 边 k 是顶点 k 的对边：e0 = 1→2，e1 = 2→0，e2 = 0→1
    coef = [_edge_coeffs(x1, y1, x2, y2), _edge_coeffs(x2, y2, x0, y0), _edge_coeffs(x0, y0, x1, y1)]
    # 包围盒：中心在 (min, max] 内的行、列
    xmin, xmax = np.minimum(np.minimum(x0, x1), x2), np.maximum(np.maximum(x0, x1), x2)
    ymin, ymax = np.minimum(np.minimum(y0, y1), y2), np.maximum(np.maximum(y0, y1), y2)
    j_lo = np.maximum((xmin - HALF) // UNITS + 1, 0)
    j_hi = np.minimum((xmax - HALF) // UNITS, w - 1)
    i_lo = np.maximum((ymin - HALF) // UNITS + 1, 0)
    i_hi = np.minimum((ymax - HALF) // UNITS, h - 1)
    ni = np.maximum(i_hi - i_lo + 1, 0)
    n_rows = int(ni.sum())
    if n_rows == 0 or not np.any(j_hi >= j_lo):
        return empty, empty, empty, np.zeros((2, 0), dtype=np.int64)
    # 逐行求被覆盖的列区间（DEV-0028 提速；与在包围盒里逐个像素判 E ≥ bias 选出的集合、顺序都相同）：
    # 行 r 上 E(col) = A·(col·U + H) + B·py + C 是 col 的整数线性函数，E ≥ bias ⟺
    # A > 0：col ≥ ⌈(bias − A·H − B·py − C) / (A·U)⌉；A < 0：col ≤ ⌊同一个比⌋；A = 0：右边 ≤ 0 时整行，否则整行都不在。
    # 全用整数的 floor 除法，没有舍入。三条边取交，再与包围盒的列范围取交。
    tr = np.repeat(np.arange(n_tri, dtype=np.int64), ni)
    r_start = np.cumsum(ni) - ni
    rr = i_lo[tr] + (np.arange(n_rows, dtype=np.int64) - r_start[tr])
    py_r = rr * UNITS + HALF
    lo, hi = j_lo[tr].copy(), j_hi[tr].copy()
    for a_, b_, c_, bias in coef:
        at = a_[tr]
        rhs = (bias[tr] - b_[tr] * py_r - c_[tr]) - at * HALF
        au = at * UNITS
        pos, neg = au > 0, au < 0
        lo = np.where(pos, np.maximum(lo, -((-rhs) // np.where(pos, au, 1))), lo)
        hi = np.where(neg, np.minimum(hi, rhs // np.where(neg, au, -1)), hi)
        hi = np.where((au == 0) & (rhs > 0), lo - 1, hi)
    cnt = np.maximum(hi - lo + 1, 0)
    total = int(cnt.sum())
    if total == 0:
        return empty, empty, empty, np.zeros((2, 0), dtype=np.int64)
    row_of = np.repeat(np.arange(n_rows, dtype=np.int64), cnt)
    c_start = np.cumsum(cnt) - cnt
    t_sel, r_sel = tr[row_of], rr[row_of]
    c_sel = lo[row_of] + (np.arange(total, dtype=np.int64) - c_start[row_of])
    px, py = c_sel * UNITS + HALF, r_sel * UNITS + HALF
    es = [a_[t_sel] * px + b_[t_sel] * py + c_[t_sel] for a_, b_, c_, _bias in coef]
    e0, e1, e2 = (e.astype(np.float64) for e in es)  # 被覆盖时 0 ≤ E ≤ A2 < 2^53：转换精确
    iz = inv_z[tris[t_sel]]
    num = (e0 * iz[:, 0] + e1 * iz[:, 1]) + e2 * iz[:, 2]
    z = a2[t_sel].astype(np.float64) / num
    zq = np.rint(z * DEPTH_SCALE).astype(np.int64)
    if zq.size and (int(zq.min()) <= 0 or int(zq.max()) >= _MAX_DEPTH_UNITS):
        raise ValueError("像素深度超出 z-buffer 的键能表示的范围")
    pix = r_sel * w + c_sel
    key = np.sort((pix << _KEY_PIX_SHIFT) | (zq << _KEY_Z_SHIFT) | t_sel)
    kp = key >> _KEY_PIX_SHIFT
    first = np.ones(key.size, dtype=bool)
    first[1:] = kp[1:] != kp[:-1]
    win = key[first]
    return win >> _KEY_PIX_SHIFT, (win >> _KEY_Z_SHIFT) & (_MAX_DEPTH_UNITS - 1), win & (_MAX_PRIO - 1), np.stack([t_sel, pix])


def floor_layer(floor: FloorSpec, camera: CameraSpec, height: int, width: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """地板的 (mask (H, W) bool, 每行的深度单位 (H,) int64, 颜色)。调用方先用 floor_vertices + vertices_ok 核对量程。"""
    p = floor_vertices(floor, camera)
    uu, vv = project_units(p, height, width)
    mask = rasterize_units(np.stack([uu, vv], axis=1), height, width)
    f, _cx, cy = intrinsics(height, width)
    ce, se = rot_cs(camera.elevation_deg)
    cam_y = camera.position()[1]
    b = ((np.arange(height, dtype=np.float64) + 0.5) - cy) / f
    den = ce * b + se
    # 只有地平线以下的行（den > 0）才可能有地板；地板像素的深度 ≤ 顶点深度的最大值 < D_MAX，
    # 截到 2·D_MAX 只影响用不到的行，并让转换成整数时不会越界
    zq = np.zeros(height, dtype=np.int64)
    below = den > 0.0
    zq[below] = np.rint(np.minimum(cam_y / den[below], 2.0 * D_MAX) * DEPTH_SCALE).astype(np.int64)
    # 地板法向（世界 y）在相机坐标系里 = 相机三个轴的 y 分量
    bas = camera.basis()
    lam = _lambert256(np.array([bas[0][1]]), np.array([bas[1][1]]), np.array([bas[2][1]]))
    color = tuple(int(c) for c in shade_rgb(floor.color, lam)[0])
    return mask, zq, color


def render_posed(
    scene: Scene3D,
    camera: CameraSpec,
    verts: list[np.ndarray],
    *,
    want_rgb: bool = True,
    want_amodal: bool = False,
) -> Frame:
    """用已算好的相机坐标顶点（camera_vertices，与 scene.objects 一一对应）渲染一帧。顶点必须通过 vertices_ok。"""
    h, w = scene.height, scene.width
    if len(verts) != len(scene.objects):
        raise ValueError("顶点数组与物体个数不同")
    for p in verts:
        if not vertices_ok(p, h, w):
            raise ValueError("有顶点的深度或投影坐标超出量程；先用 vertices_ok 拒绝这个场景")
    bg = np.asarray(scene.background, dtype=np.uint8)
    rgb = None
    if scene.floor is None:
        id_map = np.zeros((h, w), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.int64)
        if want_rgb:
            rgb = np.empty((h, w, 3), dtype=np.uint8)
            rgb[:] = bg
    else:
        fp = floor_vertices(scene.floor, camera)
        if not vertices_ok(fp, h, w):
            raise ValueError("地板顶点的深度或投影坐标超出量程；先用 floor_vertices + vertices_ok 拒绝这个场景")
        fmask, frow, fcolor = floor_layer(scene.floor, camera, h, w)
        # 地板层用 np.where 一次写成（DEV-0028 提速；与先填背景、再按 mask 赋值逐位相同，b 的钉住摘要核对）
        id_map = np.where(fmask, np.uint8(FLOOR_ID), np.uint8(0))
        depth = np.where(fmask, frow[:, None], np.int64(0))
        if want_rgb:  # 两色调色板按 mask 取色（0 = 背景、1 = 地板）
            rgb = np.take(np.stack([bg, np.asarray(fcolor, dtype=np.uint8)]), fmask.view(np.uint8), axis=0)

    # 全部物体的正面三角形，按 (obj_id, 面序号) 编优先级
    us, vs, izs, tris, owner, colors = [], [], [], [], [], []
    offset = 0
    for k, (obj, p) in enumerate(zip(scene.objects, verts)):
        _, faces = unit_mesh(obj.family, obj.variant)
        uu, vv = project_units(p, h, w)
        a2 = (uu[faces[:, 1]] - uu[faces[:, 0]]) * (vv[faces[:, 2]] - vv[faces[:, 0]]) - (vv[faces[:, 1]] - vv[faces[:, 0]]) * (
            uu[faces[:, 2]] - uu[faces[:, 0]]
        )
        front = faces[a2 < 0]
        us.append(uu)
        vs.append(vv)
        izs.append(1.0 / p[:, 2])
        tris.append(front[:, [0, 2, 1]] + offset)  # 交换后两个顶点：A2 > 0
        owner.append(np.full(front.shape[0], k, dtype=np.int64))
        if want_rgb:
            e1 = p[front[:, 1]] - p[front[:, 0]]
            e2 = p[front[:, 2]] - p[front[:, 0]]
            nx = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
            ny = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
            nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            colors.append(shade_rgb(appearance(obj.appearance).rgb, _lambert256(nx, ny, nz)))
        offset += p.shape[0]
    amodal = np.zeros((len(scene.objects), h, w), dtype=bool) if want_amodal else None
    if scene.objects:
        u_all, v_all, iz_all = np.concatenate(us), np.concatenate(vs), np.concatenate(izs)
        t_all = np.concatenate(tris)
        own = np.concatenate(owner)
        pix, zq, tri, pairs = rasterize_triangles(u_all, v_all, iz_all, t_all, h, w)
        id_map.reshape(-1)[pix] = (own[tri] + 1).astype(np.uint8)
        depth.reshape(-1)[pix] = zq
        if rgb is not None:
            rgb.reshape(-1, 3)[pix] = np.concatenate(colors)[tri]
        if amodal is not None:
            amodal.reshape(len(scene.objects), -1)[own[pairs[0]], pairs[1]] = True
    return Frame(id_map, depth, rgb, amodal)


def posed_vertices(scene: Scene3D, camera: CameraSpec | None = None, extra: dict[int, Mat3] | None = None) -> list[np.ndarray]:
    cam = scene.camera if camera is None else camera
    extra = extra or {}
    return [camera_vertices(o, cam, extra.get(o.obj_id)) for o in scene.objects]


def render_scene(
    scene: Scene3D,
    camera: CameraSpec | None = None,
    *,
    extra: dict[int, Mat3] | None = None,
    want_rgb: bool = True,
    want_amodal: bool = False,
) -> Frame:
    """渲染场景（默认用 scene.camera）。extra：obj_id → 相机坐标系里绕物体中心的附加旋转。"""
    cam = scene.camera if camera is None else camera
    return render_posed(scene, cam, posed_vertices(scene, cam, extra), want_rgb=want_rgb, want_amodal=want_amodal)


# ---------------------------------------------------------------------------
# 深度网格
# ---------------------------------------------------------------------------


def depth_grid(depth_units: np.ndarray, mask: np.ndarray, n: int, grid: np.ndarray | None = None) -> np.ndarray:
    """(n, n) float32 米：mask 的网格（mask_to_grid）为 1 的格子里，mask 像素深度的平均（整数单位精确求和，
    四舍五入取偶到整数单位）；其余格子为 0。grid：可选的 mask_to_grid(mask, n)（省掉重算）。"""
    _g, units = cell_depth_units(depth_units, mask, n, grid)  # 整数规则只在 relief.cell_depth_units 定义一处
    return (units.astype(np.float64) * DEPTH_QUANTUM_M).astype(np.float32)


def depth_range_m(depth_units: np.ndarray) -> list[float] | None:
    """非零深度的 [最小, 最大]（米，精确），没有表面时为 None。"""
    nz = depth_units[depth_units > 0]
    if nz.size == 0:
        return None
    return [int(nz.min()) * DEPTH_QUANTUM_M, int(nz.max()) * DEPTH_QUANTUM_M]


def ceil_to(x: float, step: float) -> float:
    """向上取到 step 的整数倍（step 是 2 的负整数次幂）。"""
    return math.ceil(x / step) * step
