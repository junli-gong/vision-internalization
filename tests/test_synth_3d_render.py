"""E5 第二部分的 3D 基础（DEV-0010）：程序化网格、对称性规则、自写光栅化（覆盖、深度、z-buffer、着色）。

- 网格：闭合、定向一致、体积为正、质心在原点、外接半径 1、顶点在 2^-24 网格上；钉住全部网格的摘要。
- 对称性：rotation_period_about 的规则逐条核对；有限对称轴在网格顶点上精确成立；连续 / 垂直 2 重对称在渲染的 mask 上近似成立。
- 光栅化：与独立的纯 Python 有理数（Fraction）参考逐位相同（覆盖用射线穿越规则、深度用精确重心坐标、z-buffer 按量化深度与优先级）；
  共用边的三角形不重不漏；立方体正视的像素数与解析值相同；轴对齐的正面与 box_to_mask 逐位相同。
- 深度：正对相机的面等于解析值；斜面与地板与独立的射线求交相差不超过 1 个深度单位；深度都在 D_MIN–D_MAX 内。
- 对齐：物体 ID 图、深度图、RGB 按构造对齐；钉住一组渲染结果的摘要（在服务器上跑一遍即是跨平台核对）。
"""

import hashlib
import math
from fractions import Fraction

import numpy as np
import pytest

from src.constants import D_MAX, D_MIN
from src.executors.raster import box_to_mask
from src.synth import shapes as S
from src.synth import solids as SO
from src.synth.constants import (
    DEPTH_QUANTUM_M,
    FLOOR_ID,
    SOLID_FAMILIES,
    SOLID_REST_POSES,
    UNIT_SHAPE_QUANTUM,
    VIEW_FLOOR_COLOR,
    VIEW_FLOOR_SEGMENTS,
)
from src.synth.raster import HALF, UNITS, rasterize
from src.synth.render3d import (
    CameraSpec,
    FloorSpec,
    Scene3D,
    SolidSpec,
    camera_vertices,
    depth_grid,
    intrinsics,
    mat_mul,
    rasterize_triangles,
    render_scene,
    rot3,
)
from src.synth.sample import iou

ALL_VARIANTS = [(f, v.index) for f in SOLID_FAMILIES for v in SO.SOLID_VARIANTS[f]]
FACE_COUNTS = {"box": 12, "sphere": 320, "cylinder": 64, "cone": 32, "torus": 256, "l_prism": 20}


# 渲染测试自己的地板半径：DEV-0010 时 b 的 VIEW_FLOOR_RADIUS_M。DEV-0018 起 b 的地板放大到 2.4 m，渲染测试（含钉住的
# 渲染摘要）固定用原值，与 b 的布局参数脱钩。
RENDER_TEST_FLOOR_RADIUS_M = 1.7


# ---------------------------------------------------------------------------
# 网格
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,index", ALL_VARIANTS)

def test_meshes_are_closed_oriented_and_normalized(family, index):
    v, f = SO.unit_mesh(family, index)
    assert not v.flags.writeable and not f.flags.writeable
    assert np.array_equal(v / UNIT_SHAPE_QUANTUM, np.round(v / UNIT_SHAPE_QUANTUM))  # 顶点在 2^-24 网格上
    edges = {}
    for a, b, c in f.tolist():
        for e in ((a, b), (b, c), (c, a)):
            edges[e] = edges.get(e, 0) + 1
    assert all(n == 1 and edges.get((b, a)) == 1 for (a, b), n in edges.items())  # 闭合、定向一致
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    det = np.einsum("ij,ij->i", p0, np.cross(p1, p2))  # 测试里可以用平台函数
    vol = det.sum() / 6
    assert vol > 0.05
    centroid = (det[:, None] * (p0 + p1 + p2)).sum(axis=0) / (4 * det.sum())
    assert np.abs(centroid).max() < 1e-6  # 原点在体积质心
    assert abs(np.sqrt((v * v).sum(axis=1)).max() - 1.0) < 1e-6
    n = SO.solid_variant(family, index).params[0] if family == "prism" else None
    assert f.shape[0] == (4 * n if family == "prism" else FACE_COUNTS[family])


def test_unit_meshes_are_pinned():
    """网格在 decimal 里构造，与平台无关。开发机上红 = 网格变了（该任务版本要加一）；只在服务器上红 = 跨平台不能逐位复现。"""
    h = hashlib.sha256()
    for fam, idx in ALL_VARIANTS:
        v, f = SO.unit_mesh(fam, idx)
        h.update(f"{fam}:{idx}:{v.shape}:{f.shape}".encode())
        h.update(v.tobytes())
        h.update(f.tobytes())
    assert h.hexdigest() == PINNED_UNIT_MESHES


PINNED_UNIT_MESHES = "7b0868ee5ba68297c9f69d9b8229d65fe0297d1fe5a684cdce42f3b4a57afb85"  # 开发机 arm64 macOS，DEV-0010


def test_identity_split_rule_matches_part_one():
    table = SO.solid_identity_split_table()
    assert set(table) == set(SO.all_solid_identities())
    assert len(table) == sum(len(v) for v in SO.SOLID_VARIANTS.values()) * 10
    for fam, idx in ALL_VARIANTS:
        ids = [(fam, idx, a.index) for a in SO.SOLID_APPEARANCES]
        assert {i: table[i] for i in ids} == S.assign_splits(ids)
        assert sorted(table[i] for i in ids).count("train") == 8
    held = SO.solid_identity_split_table("torus")
    assert all(s == "test" for i, s in held.items() if i[0] == "torus")
    assert all(held[i] == table[i] for i in table if i[0] != "torus")
    with pytest.raises(ValueError):
        SO.solid_identity_split_table("hexagon")
    rng = np.random.default_rng(0)
    for _ in range(40):
        ident = SO.sample_solid_identity(rng, "train", "torus", rotatable_only=True)
        assert ident[0] != "torus" and not SO.is_isotropic(*ident[:2]) and table[ident] == "train"


# ---------------------------------------------------------------------------
# 对称性
# ---------------------------------------------------------------------------


def _vid(family, params):
    """参数前缀为 params 的第一个变体序号。"""
    for v in SO.SOLID_VARIANTS[family]:
        if tuple(v.params[: len(params)]) == tuple(params):
            return v.index
    raise KeyError(params)


def test_rotation_period_rules():
    P = SO.rotation_period_about
    cube = _vid("box", (1.0, 1.0, 1.0))
    assert P("box", cube, (1, 0, 0)) == 90.0 and P("box", cube, (0, 0, 2)) == 90.0
    assert P("box", cube, (1, 1, 1)) == 120.0 and P("box", cube, (-1, 1, 1)) == 120.0
    assert P("box", cube, (1, -1, 0)) == 180.0 and P("box", cube, (0, 1, 1)) == 180.0
    assert P("box", cube, (1, 1e-3, 0)) == 360.0 and P("box", cube, (1, 2, 3)) == 360.0
    slab = _vid("box", (1.0, 0.5, 0.75))
    assert P("box", slab, (0, 1, 0)) == 180.0 and P("box", slab, (1, 0, 1)) == 360.0
    column = _vid("box", (0.6, 1.5, 0.6))  # x = z：y 轴 4 重，xz 平面对角线 2 重
    assert P("box", column, (0, 1, 0)) == 90.0 and P("box", column, (1, 0, 1)) == 180.0 and P("box", column, (1, 0, 0)) == 180.0
    ball = _vid("sphere", (1.0,))
    assert P("sphere", ball, (0.3, -2, 7)) == 0.0
    for fam in ("sphere", "cylinder", "torus"):
        idx = 1 if fam == "sphere" else 0
        assert P(fam, idx, (0, 1, 0)) == 0.0
        assert P(fam, idx, (1, 0, 0)) == 180.0 and P(fam, idx, (math.cos(0.3), 0, math.sin(0.3))) == 180.0
        assert P(fam, idx, (1, 1, 0)) == 360.0
    assert P("cone", 0, (0, -1, 0)) == 0.0 and P("cone", 0, (1, 0, 0)) == 360.0
    hexa, tri = _vid("prism", (6, 0.6)), _vid("prism", (3, 0.6))
    assert P("prism", hexa, (0, 1, 0)) == 60.0 and P("prism", tri, (0, 1, 0)) == 120.0
    for deg, want_hex, want_tri in ((90, 180.0, 180.0), (120, 180.0, 360.0), (150, 180.0, 180.0), (105, 360.0, 360.0), (30, 180.0, 180.0)):
        c, s = S.rot_cs(deg)
        assert P("prism", hexa, (c, 0, s)) == want_hex, deg
        assert P("prism", tri, (c, 0, s)) == want_tri, deg
    for idx in range(len(SO.SOLID_VARIANTS["l_prism"])):
        assert P("l_prism", idx, (0, 0, 1)) == 360.0 and P("l_prism", idx, (1, 1, 0)) == 360.0
    # 容差：整数度姿态下的"对齐"偏差 ~1e-12，仍判为对齐；1e-6 的偏差判为不对齐
    assert P("box", cube, (1, 1e-12, 0)) == 90.0 and P("box", cube, (1, 1e-6, 0)) == 360.0


def _axis_angle(axis, deg):
    """测试用的 Rodrigues 旋转（平台三角函数只在测试里用）。"""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


@pytest.mark.parametrize("family", ["box", "prism"])
def test_discrete_symmetries_hold_exactly_on_the_mesh_vertices(family):
    for v in SO.SOLID_VARIANTS[family]:
        verts, _ = SO.unit_mesh(family, v.index)
        for d, k in SO.discrete_symmetry_axes(family, v.index):
            r = _axis_angle(d, 360.0 / k)
            moved = verts @ r.T
            dist = np.sqrt(((moved[:, None, :] - verts[None, :, :]) ** 2).sum(-1)).min(axis=1)
            assert dist.max() < 1e-6, (family, v.index, d, k)


def _mask_after_local_rotation(obj, camera, local_axis, deg, hw=(256, 256)):
    """把实体绕局部坐标里的轴旋转 deg 度后渲染的 mask（旋转放进相机坐标系：extra = M R M^T）。"""
    m = np.array(mat_mul(camera.basis(), obj.rotation()))
    r = m @ _axis_angle(local_axis, deg) @ m.T
    scene = Scene3D(hw[0], hw[1], (255, 255, 255), camera, (obj,))
    extra = tuple(tuple(float(x) for x in row) for row in r)
    return render_scene(scene, extra={0: extra}, want_rgb=False).mask(0)


@pytest.mark.parametrize("family", ["sphere", "cylinder", "cone", "torus"])
def test_continuous_and_perpendicular_symmetries_hold_on_rendered_masks(family):
    """理想实体的连续对称与垂直 2 重对称：折线化网格旋转后的 mask 与原来的 IoU ≥ 0.97（只差在折线的棱上）。"""
    rng = np.random.default_rng(7)
    camera = CameraSpec(0, 25, 2.4)
    for v in SO.SOLID_VARIANTS[family]:
        obj = SolidSpec(0, family, v.index, 0, "upright", 17, 0.6, (0.0, 0.0, 0.0))
        base = _mask_after_local_rotation(obj, camera, (0, 1, 0), 0.0)
        for _ in range(3):
            deg = float(rng.uniform(10, 350))
            if SO.rotation_period_about(family, v.index, (0, 1, 0)) == 0.0:
                assert iou(_mask_after_local_rotation(obj, camera, (0, 1, 0), deg), base) > 0.97, (family, v.index, deg)
            phi = float(rng.uniform(0, 360))
            perp = (math.cos(math.radians(phi)), 0.0, math.sin(math.radians(phi)))
            if SO.rotation_period_about(family, v.index, perp) == 180.0:
                assert iou(_mask_after_local_rotation(obj, camera, perp, 180.0), base) > 0.97, (family, v.index, phi)


# ---------------------------------------------------------------------------
# 光栅化：与独立的有理数参考逐位相同
# ---------------------------------------------------------------------------


def _crossing_inside(pts, cx, cy):
    """raster.py 的规则（射线穿越、(lo, hi]），纯 Fraction：与边函数写法独立。"""
    count = 0
    n = len(pts)
    for k in range(n):
        (xa, ya), (xb, yb) = pts[k], pts[(k + 1) % n]
        if ya == yb:
            continue
        if ya > yb:
            xa, ya, xb, yb = xb, yb, xa, ya
        if ya < cy <= yb and xa + (cy - ya) * (xb - xa) / (yb - ya) < cx:
            count += 1
    return count % 2 == 1


def _cross2(ax, ay, bx, by):
    return ax * by - ay * bx


def fraction_zbuffer(tri_pts, tri_z, h, w):
    """每个像素：覆盖它的三角形里 (量化深度, 三角形序号) 最小的那个；深度用精确的重心坐标插值 1/z。"""
    win = np.full((h, w), -1, np.int64)
    dep = np.zeros((h, w), np.int64)
    scale = Fraction(1) / Fraction(DEPTH_QUANTUM_M)
    for i in range(h):
        cy = Fraction(2 * i + 1, 2)
        for j in range(w):
            cx = Fraction(2 * j + 1, 2)
            best = None
            for t, (pts, zs) in enumerate(zip(tri_pts, tri_z)):
                if not _crossing_inside(pts, cx, cy):
                    continue
                (x0, y0), (x1, y1), (x2, y2) = pts
                area = _cross2(x1 - x0, y1 - y0, x2 - x0, y2 - y0)
                l0 = _cross2(x1 - cx, y1 - cy, x2 - cx, y2 - cy) / area
                l1 = _cross2(x2 - cx, y2 - cy, x0 - cx, y0 - cy) / area
                l2 = 1 - l0 - l1
                z = 1 / (l0 / zs[0] + l1 / zs[1] + l2 / zs[2])
                key = (round(z * scale), t)  # Fraction 的 round：四舍五入取偶
                if best is None or key < best:
                    best = key
            if best is not None:
                dep[i, j], win[i, j] = best
    return win, dep


def _random_triangles(rng, n, h, w, grid):
    pts, zs = [], []
    while len(pts) < n:
        p = np.round(rng.uniform([-3, -3], [w + 3, h + 3], size=(3, 2)) * grid) / grid
        a2 = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0])
        if a2 == 0:
            continue
        if a2 < 0:
            p = p[[0, 2, 1]]
        z = np.round(rng.uniform(1.0, 9.0, 3) * 2**20) / 2**20  # 与 VERTEX_QUANTUM_M 同一网格
        if rng.random() < 0.4:
            z[:] = z[0]  # 平行于像平面的三角形
            if zs and rng.random() < 0.5 and np.all(zs[-1] == zs[-1][0]):
                z[:] = zs[-1][0]  # 与上一个三角形等深：重叠处按优先级（序号小的赢）
        pts.append(p)
        zs.append(z)
    return pts, zs


def _run_ours(pts, zs, h, w):
    u = np.concatenate([np.rint(p[:, 0] * UNITS) for p in pts]).astype(np.int64)
    v = np.concatenate([np.rint(p[:, 1] * UNITS) for p in pts]).astype(np.int64)
    iz = 1.0 / np.concatenate(zs)
    tris = np.arange(3 * len(pts), dtype=np.int64).reshape(-1, 3)
    pix, zq, tri, pairs = rasterize_triangles(u, v, iz, tris, h, w)
    win = np.full(h * w, -1, np.int64)
    dep = np.zeros(h * w, np.int64)
    win[pix], dep[pix] = tri, zq
    return win.reshape(h, w), dep.reshape(h, w), pairs


def test_triangles_match_exact_rational_reference():
    rng = np.random.default_rng(11)
    h, w = 13, 17
    covered = 0
    for k in range(10):
        grid = (1, 2, 256)[k % 3]  # 整数、半整数网格上大量顶点与边恰好落在像素中心上
        pts, zs = _random_triangles(rng, 6, h, w, grid)
        ours_w, ours_d, _ = _run_ours(pts, zs, h, w)
        ref_w, ref_d = fraction_zbuffer(
            [[(Fraction(float(x)), Fraction(float(y))) for x, y in p] for p in pts], [[Fraction(float(z)) for z in zz] for zz in zs], h, w
        )
        assert np.array_equal(ours_w, ref_w), k
        assert np.array_equal(ours_d, ref_d), k
        covered += int((ref_w >= 0).sum())
    assert covered > 500  # 参考确实覆盖了大量像素


def test_coverage_of_single_triangles_matches_polygon_rasterizer():
    """同一边界规则：单个三角形的覆盖与 raster.rasterize（奇偶规则、(lo, hi]）逐位相同。"""
    rng = np.random.default_rng(12)
    h, w = 40, 56
    for k in range(200):
        pts, zs = _random_triangles(rng, 1, h, w, (1, 2, 256)[k % 3])
        _, _, pairs = _run_ours(pts, zs, h, w)
        cov = np.zeros(h * w, bool)
        cov[pairs[1]] = True
        assert np.array_equal(cov.reshape(h, w), rasterize(pts[0], h, w)), k


def test_triangles_sharing_edges_partition_the_pixels():
    """星形多边形从核心点扇形剖分：各三角形两两不相交，并集等于整个多边形的光栅化（边上的像素中心不重不漏）。"""
    rng = np.random.default_rng(13)
    h, w = 48, 64
    for k in range(60):
        n = int(rng.integers(3, 12))
        c = np.round(rng.uniform([10, 10], [w - 10, h - 10]) * 2) / 2
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        if np.max(np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))) >= np.pi:
            continue  # 中心要严格在多边形的核里
        r = rng.uniform(4, 30, n)
        grid = (1, 2, 256)[k % 3]
        poly = np.round(np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], axis=1) * grid) / grid
        tris = [np.stack([c, poly[i], poly[(i + 1) % n]]) for i in range(n)]
        a2 = [(t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0]) for t in tris]
        if not (all(a > 0 for a in a2) or all(a < 0 for a in a2)):
            continue  # 取整到网格后扇形不再是合法剖分（顶点次序翻转），跳过
        tris = [t if a > 0 else t[[0, 2, 1]] for t, a in zip(tris, a2)]
        _, _, pairs = _run_ours(tris, [np.full(3, 2.0)] * len(tris), h, w)
        counts = np.bincount(pairs[1], minlength=h * w).reshape(h, w)
        assert counts.max() <= 1, k
        assert np.array_equal(counts == 1, rasterize(poly, h, w)), k


# ---------------------------------------------------------------------------
# 立方体正视、深度的解析值
# ---------------------------------------------------------------------------

CUBE = _vid("box", (1.0, 1.0, 1.0))


def _cube(half_side, center=(0.0, 0.0, 0.0), yaw=0):
    return SolidSpec(0, "box", CUBE, 0, "upright", yaw, half_side * math.sqrt(3.0), center)


@pytest.mark.parametrize("hw,half,dist", [((512, 512), 0.5, 2.5), ((512, 512), 0.25, 1.25), ((384, 640), 0.5, 3.5), ((96, 160), 0.2, 2.2)])
def test_cube_front_view_pixel_count_and_depth_are_analytic(hw, half, dist):
    """相机正对立方体的一个面：只看见这个面，投影是边长 f·2h/(D−h) 的正方形，深度处处是 D − h。"""
    h, w = hw
    scene = Scene3D(h, w, (255, 255, 255), CameraSpec(0, 0, dist), (_cube(half),))
    fr = render_scene(scene, want_amodal=True)
    f, cx, cy = intrinsics(h, w)
    side = f * 2 * half / (dist - half)
    x0, x1, y0, y1 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
    want = box_to_mask([x0, y0, x1, y1], h, w)
    m = fr.mask(0)
    assert np.array_equal(m, want) and np.array_equal(fr.amodal[0], want)
    assert m.sum() == (math.floor(x1 - 0.5) - math.floor(x0 - 0.5)) * (math.floor(y1 - 0.5) - math.floor(y0 - 0.5))
    assert np.all(fr.depth_units[m] == round((dist - half) / DEPTH_QUANTUM_M)) and np.all(fr.depth_units[~m] == 0)
    assert len({tuple(c) for c in fr.rgb[m]}) == 1  # 一个平面：一个颜色


def test_tilted_faces_and_floor_depth_match_independent_ray_casting():
    """偏航 30° 的立方体（两个面斜对相机）与地板：每个像素的深度与独立的射线—平面求交相差不超过 1 个单位。"""
    h = w = 256
    camera = CameraSpec(40, 25, 4.0)
    obj = _cube(0.4, center=(0.3, 0.4 + 1e-3, -0.2), yaw=30)
    floor = FloorSpec(RENDER_TEST_FLOOR_RADIUS_M, VIEW_FLOOR_SEGMENTS, tuple(VIEW_FLOOR_COLOR))
    scene = Scene3D(h, w, (255, 255, 255), camera, (obj,), floor)
    fr = render_scene(scene)
    f, cx, cy = intrinsics(h, w)
    b = np.array(camera.basis())
    cpos = np.array(camera.position())
    rows, cols = np.mgrid[:h, :w]
    ray = np.stack([(cols + 0.5 - cx) / f, (rows + 0.5 - cy) / f, np.ones((h, w))], -1)  # 相机坐标，z = 1
    world_dir = ray @ b  # 相机 → 世界：B^T · r
    # 地板 y = 0
    t_floor = -cpos[1] / world_dir[..., 1]
    fl = fr.id_map == FLOOR_ID
    assert fl.sum() > 1000
    assert np.abs(fr.depth_units[fl] - t_floor[fl] / DEPTH_QUANTUM_M).max() <= 1.0
    # 立方体的面：局部 ±x、±y、±z 平面（半边长 0.4），取光线最先进入的那个面
    r = np.array(obj.rotation())
    o_loc = (cpos - np.array(obj.center)) @ r  # 世界 → 局部：R^T · (C − c)
    d_loc = world_dir @ r
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-0.4 - o_loc) / d_loc
        t2 = (0.4 - o_loc) / d_loc
    t_near = np.nanmax(np.minimum(t1, t2), axis=-1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = t_near < t_far
    ob = fr.id_map == 1
    interior = ob & np.roll(ob, 1, 0) & np.roll(ob, -1, 0) & np.roll(ob, 1, 1) & np.roll(ob, -1, 1)
    assert interior.sum() > 500 and np.all(hit[interior])
    assert np.abs(fr.depth_units[interior] - t_near[interior] / DEPTH_QUANTUM_M).max() <= 1.0


def test_image_ids_and_depth_are_aligned_by_construction():
    camera = CameraSpec(200, 30, 4.5)
    floor = FloorSpec(RENDER_TEST_FLOOR_RADIUS_M, VIEW_FLOOR_SEGMENTS, tuple(VIEW_FLOOR_COLOR))
    objs = (  # 中心高度 = 外接半径：外接球整个在地板之上
        SolidSpec(0, "torus", 1, 3, "side_x", 40, 0.45, (0.5, 0.45, 0.2)),
        SolidSpec(1, "l_prism", 2, 7, "upright", 110, 0.5, (-0.4, 0.5, -0.3)),
        SolidSpec(2, "cone", 0, 1, "upright", 0, 0.4, (0.0, 0.4, 0.8)),
    )
    scene = Scene3D(200, 300, (224, 234, 246), camera, objs, floor)
    fr = render_scene(scene, want_amodal=True)
    assert np.array_equal(fr.id_map == 0, fr.depth_units == 0)
    assert np.all(fr.rgb[fr.id_map == 0] == (224, 234, 246))
    for k, o in enumerate(objs):
        m = fr.mask(k)
        assert m.any() and np.all(fr.amodal[k][m])  # 可见部分在完整轮廓里
        base = np.array(S.appearance(o.appearance).rgb)
        px = fr.rgb[m].astype(np.int64)
        assert np.all(px <= base) and np.all(px >= base * 96 // 256 - 1)  # 底色 × 亮度（环境光下限）
    assert len({tuple(c) for c in fr.rgb[fr.id_map == FLOOR_ID]}) == 1
    d = fr.depth_units[fr.depth_units > 0] * DEPTH_QUANTUM_M
    assert d.min() >= D_MIN and d.max() <= D_MAX


def test_depth_grid_is_the_rounded_cell_mean():
    rng = np.random.default_rng(3)
    mask = rng.random((64, 96)) < 0.7
    units = rng.integers(2048, 40000, (64, 96))
    g = depth_grid(units, mask, 8)
    from src.grid.codec import mask_to_grid

    grid = mask_to_grid(mask, 8)
    for i in range(8):
        for j in range(8):
            cell = (slice(8 * i, 8 * i + 8), slice(12 * j, 12 * j + 12))
            if grid[i, j]:
                want = round(Fraction(int(units[cell][mask[cell]].sum()), int(mask[cell].sum())))
                assert g[i, j] == np.float32(want * DEPTH_QUANTUM_M)
            else:
                assert g[i, j] == 0.0


# ---------------------------------------------------------------------------
# 钉住：跨平台核对
# ---------------------------------------------------------------------------


def test_pinned_render_digest():
    """全部实体变体 × 三种摆放、固定的相机，再加一个带地板的多物体场景：ID 图、深度、RGB 的摘要。

    在开发机上红 = 渲染规则或网格变了（a、b 的任务版本要加一）；只在服务器上红 = 跨平台不能逐位复现，上报管理员。
    """
    h = hashlib.sha256()
    for fam, idx in ALL_VARIANTS:
        for k, rest in enumerate(SOLID_REST_POSES):
            obj = SolidSpec(0, fam, idx, (idx + k) % 10, rest, 37 * k + 11, 0.55, (0.0, 0.0, 0.0))
            scene = Scene3D(96, 128, (236, 236, 236), CameraSpec(0, 15 + 10 * k, 2.1), (obj,))
            extra = {0: rot3("x", 23 * k - 17)}
            fr = render_scene(scene, extra=extra)
            for a in (fr.id_map, fr.depth_units, fr.rgb):
                h.update(a.tobytes())
    floor = FloorSpec(RENDER_TEST_FLOOR_RADIUS_M, VIEW_FLOOR_SEGMENTS, tuple(VIEW_FLOOR_COLOR))
    objs = tuple(
        SolidSpec(k, fam, 0, k, SOLID_REST_POSES[k % 3], 29 * k, 0.4, (math.cos(k) * 0.9, 0.45, math.sin(k) * 0.9))
        for k, fam in enumerate(SOLID_FAMILIES)
    )
    fr = render_scene(Scene3D(160, 224, (246, 240, 224), CameraSpec(123, 33, 4.4), objs, floor), want_amodal=True)
    for a in (fr.id_map, fr.depth_units, fr.rgb, fr.amodal):
        h.update(a.tobytes())
    assert h.hexdigest() == PINNED_RENDER_DIGEST


PINNED_RENDER_DIGEST = "1a1ce928ab154bcf5931acf9e2370360cc0a81f57cdd8858035f7b8f10740536"  # 开发机 arm64 macOS，DEV-0010


def test_camera_vertices_are_on_the_vertex_grid_and_entry_checks():
    obj = SolidSpec(0, "cone", 1, 0, "side_z", 77, 0.5, (0.1, 0.2, 0.3))
    p = camera_vertices(obj, CameraSpec(10, 20, 3.0))
    assert np.array_equal(p * 2**20, np.round(p * 2**20))
    with pytest.raises(ValueError):
        Scene3D(64, 64, (0, 0, 0), CameraSpec(0, 0, 3.0), (SolidSpec(1, "cone", 0, 0, "upright", 0, 0.5, (0.0, 0.0, 0.0)),))
    too_close = Scene3D(64, 64, (0, 0, 0), CameraSpec(0, 0, 0.6), (SolidSpec(0, "cone", 0, 0, "upright", 0, 0.5, (0.0, 0.0, 0.0)),))
    with pytest.raises(ValueError, match="量程"):
        render_scene(too_close)
    tri = np.array([[0, 1, 2]], np.int64)
    u = np.array([0, 0, 5 * UNITS], np.int64)
    v = np.array([0, 5 * UNITS, 0], np.int64)  # A2 < 0
    with pytest.raises(ValueError):
        rasterize_triangles(u, v, np.ones(3), tri, 8, 8)
    assert HALF == UNITS // 2
