"""E5 合成数据框架：种子与划分、形状族（含钉住的单位形状）、场景规格、渲染与目标对齐、导出、数据集接口。

光栅化规则的测试在 tests/test_synth_raster.py，确定的三角函数在 tests/test_synth_dmath.py。
"""

import hashlib
import math

import numpy as np
import pytest
import shapely
from PIL import Image

from src.constants import GRID_SIZE_E3B
from src.grid.codec import mask_to_grid
from src.synth import SynthDataset, generate, num_bins
from src.synth import shapes as S
from src.synth.constants import (
    IDENTITY_SPLIT_FRACTIONS,
    PATTERN_SECOND_COLOR,
    SCENE_SEED_STRIDE,
    SHAPE_FAMILIES,
    SPLITS,
    SYNTH_GENERATOR_VERSION,
    TASKS,
)
from src.synth.export import export_samples, load_manifest, sample_arrays
from src.synth.scene import Background, ObjectSpec, SceneSpec, object_masks, render_image, visible_masks
from src.synth.seeding import sample_rng, scene_seed

H, W = 96, 160  # 小 fixture 故意 H ≠ W


def _iou(a, b):
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 1.0


# ---------------------------------------------------------------------------
# 种子
# ---------------------------------------------------------------------------


def test_scene_seed_ranges_disjoint_by_construction():
    spans = {s: (scene_seed(s, 0), scene_seed(s, SCENE_SEED_STRIDE - 1)) for s in SPLITS}
    items = sorted(spans.values())
    assert all(a[1] < b[0] for a, b in zip(items, items[1:]))
    with pytest.raises(ValueError):
        scene_seed("train", SCENE_SEED_STRIDE)
    with pytest.raises(ValueError):
        scene_seed("dev", 0)
    with pytest.raises(TypeError):
        scene_seed("train", 1.0)


def test_rng_stream_is_pure_function_of_inputs():
    a = sample_rng("occlusion", "train", 7).random(5)
    assert np.array_equal(a, sample_rng("c", "train", 7).random(5))
    others = [
        sample_rng("occlusion", "train", 8),
        sample_rng("occlusion", "val", 7),
        sample_rng("physics", "train", 7),
        sample_rng("occlusion", "train", 7, version=SYNTH_GENERATOR_VERSION + 1),
    ]
    for r in others:
        assert not np.array_equal(a, r.random(5))


# ---------------------------------------------------------------------------
# 身份划分
# ---------------------------------------------------------------------------


def test_identity_split_partitions_every_variant():
    table = S.identity_split_table()
    assert set(table) == set(S.all_identities())
    fr = dict(IDENTITY_SPLIT_FRACTIONS)
    n = len(S.APPEARANCES)
    want = {"val": int(math.floor(n * fr["val"] + 0.5)), "test": int(math.floor(n * fr["test"] + 0.5))}
    want["train"] = n - want["val"] - want["test"]
    for f in SHAPE_FAMILIES:
        for v in S.VARIANTS[f]:
            got = {s: sum(1 for a in S.APPEARANCES if table[(f, v.index, a.index)] == s) for s in SPLITS}
            assert got == want, (f, v.index, got)
    pools = {s: S.identity_pool(s) for s in SPLITS}
    for s in SPLITS:  # 每个族、每个变体在每个划分里都有（划分发生在实例层面）
        assert {k for k in pools[s]} == {(f, v.index) for f in SHAPE_FAMILIES for v in S.VARIANTS[f]}


def test_identity_split_rule_is_pinned():
    """规则或常量一变，这个摘要就变：改规则要同时升生成器版本并更新这里。"""
    table = S.identity_split_table()
    text = "\n".join(f"{S.identity_key(i)}={table[i]}" for i in sorted(table))
    assert len(table) == 960  # 32 个形状变体 × 30 种外观
    assert hashlib.sha256(text.encode()).hexdigest() == "4fea278a004d843a91ba2d1679810cead74a2bc19fff44154417d24a34a036cd"


def test_holdout_family_goes_to_test_only():
    base = S.identity_split_table()
    held = S.identity_split_table("crescent")
    for ident, split in held.items():
        if ident[0] == "crescent":
            assert split == "test"
        else:
            assert split == base[ident]
    for s in ("train", "val"):
        assert not any(k[0] == "crescent" for k in S.identity_pool(s, "crescent"))
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert S.sample_identity(rng, "train", SHAPE_FAMILIES, "crescent")[0] != "crescent"
    with pytest.raises(ValueError):
        S.identity_split_table("hexagon")


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", SHAPE_FAMILIES)
def test_unit_polygons_are_valid_and_normalized(family):
    for v in S.VARIANTS[family]:
        p = S.unit_polygon(family, v.index)
        g = shapely.polygons(p)
        assert g.is_valid and g.area > 0.1, (family, v.index)
        c = g.centroid
        assert abs(c.x) < 1e-6 and abs(c.y) < 1e-6, (family, v.index, c)
        r = np.sqrt((p * p).sum(axis=1)).max()
        assert abs(r - 1.0) < 1e-6
        assert not p.flags.writeable


def _rot(p, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], axis=1)


def _poly_iou(a, b):
    ga, gb = shapely.polygons(a), shapely.polygons(b)
    return shapely.intersection(ga, gb).area / shapely.union(ga, gb).area


@pytest.mark.parametrize("family", SHAPE_FAMILIES)
def test_rotation_period_is_the_true_symmetry(family):
    """绕质心转一个周期与自身重合；在 (0.2, 0.8) 个周期之间都不重合（所以周期不会报小）。"""
    for v in S.VARIANTS[family]:
        p = S.unit_polygon(family, v.index)
        period = S.rotation_period(family, v.index)
        if period == 0.0:  # 圆：任意角度都重合
            for deg in (13.0, 90.0, 211.0):
                assert _poly_iou(p, _rot(p, deg)) > 0.999
            continue
        if period < 360.0:
            assert _poly_iou(p, _rot(p, period)) > 0.999, (family, v.index)
        for frac in np.linspace(0.2, 0.8, 7):
            assert _poly_iou(p, _rot(p, frac * period)) < 0.99, (family, v.index, frac)


def test_rotation_convention_is_clockwise_on_screen():
    assert S.rot_cs(90) == (0.0, 1.0) and S.rot_cs(-90) == (0.0, -1.0) and S.rot_cs(180) == (-1.0, 0.0)
    # L 形在 0° 时拐角在左下；顺时针转 90° 后拐角在左上（横臂到了上方，竖臂到了左边）
    l_idx = 0
    m0 = S.rasterize(S.pose_polygon("l_shape", l_idx, 80.0, 48.0, 40.0, 0.0), H, W)
    m90 = S.rasterize(S.pose_polygon("l_shape", l_idx, 80.0, 48.0, 40.0, 90.0), H, W)
    rows0, cols0 = np.nonzero(m0)
    rows9, cols9 = np.nonzero(m90)
    assert np.count_nonzero(rows0 > 48) > np.count_nonzero(rows0 < 48)  # 0°：下半多
    assert np.count_nonzero(cols0 < 80) > np.count_nonzero(cols0 > 80)  # 0°：左半多
    assert np.count_nonzero(rows9 < 48) > np.count_nonzero(rows9 > 48)  # 90° 顺时针：上半多
    assert np.count_nonzero(cols9 < 80) > np.count_nonzero(cols9 > 80)  # 仍是左半多


def test_unit_polygons_are_pinned():
    """单位形状在 decimal 里构造（DEV-0009），与平台无关。在开发机上红 = 单位形状变了（生成器版本要加一）；
    只在服务器上红 = 跨平台不能逐位复现，上报管理员。"""
    h = hashlib.sha256()
    for f in SHAPE_FAMILIES:
        for v in S.VARIANTS[f]:
            p = S.unit_polygon(f, v.index)
            assert np.array_equal(p * 2**24, np.round(p * 2**24))  # 在 UNIT_SHAPE_QUANTUM 网格上
            h.update(f"{f}:{v.index}:{p.shape}".encode())
            h.update(p.tobytes())
    assert h.hexdigest() == PINNED_UNIT_POLYGONS


PINNED_UNIT_POLYGONS = "24df3d3725315d776898a21a71b9695fd9971235c8ef153714df95be21746268"  # 开发机 arm64 macOS，DEV-0009


def test_quantization_puts_vertices_on_the_grid():
    p = S.pose_polygon("star", 3, 70.123, 40.987, 30.3, 17.77)
    assert np.array_equal(p * 256, np.round(p * 256))


# ---------------------------------------------------------------------------
# 场景规格与渲染
# ---------------------------------------------------------------------------

RED = S.APPEARANCES[0]  # 纯色红
BLUE = next(a for a in S.APPEARANCES if a.pattern == "solid" and a.color_name == "blue")
STRIPED = next(a for a in S.APPEARANCES if a.pattern == "striped" and a.color_name == "green")
BG = Background("solid", (236, 236, 236), (236, 236, 236))


def _square(oid, app, cx, cy, r, z, angle=0.0):
    sq = next(v.index for v in S.VARIANTS["rectangle"] if v.noun == "square")
    return ObjectSpec(oid, "rectangle", sq, app.index, cx, cy, r, angle, z)


def test_visible_masks_and_painting_follow_z_order():
    # obj 0 在上（z=2），obj 1 在下（z=1）；两者部分重叠
    scene = SceneSpec(H, W, BG, (_square(0, RED, 60.0, 50.0, 30.0, 2), _square(1, BLUE, 90.0, 50.0, 30.0, 1)))
    amodal = object_masks(scene)
    vis = visible_masks(scene, amodal)
    assert np.array_equal(vis[0], amodal[0])
    assert np.array_equal(vis[1], amodal[1] & ~amodal[0])
    assert np.any(amodal[0] & amodal[1])
    img = render_image(scene, amodal)
    assert np.all(img[vis[0]] == RED.rgb) and np.all(img[vis[1]] == BLUE.rgb)
    assert np.all(img[~(amodal[0] | amodal[1])] == BG.top)
    # z 相同时按 obj_id：后画的（obj_id 大）在上
    tie = SceneSpec(H, W, BG, (_square(0, RED, 60.0, 50.0, 30.0, 0), _square(1, BLUE, 90.0, 50.0, 30.0, 0)))
    vis_t = visible_masks(tie, object_masks(tie))
    assert np.array_equal(vis_t[1], amodal[1]) and not np.any(vis_t[0] & amodal[1])


def test_pattern_uses_both_colors_inside_the_mask():
    scene = SceneSpec(H, W, BG, (_square(0, STRIPED, 80.0, 48.0, 40.0, 0, angle=20.0),))
    m = object_masks(scene)[0]
    px = render_image(scene)[m].reshape(-1, 3)
    colors = {tuple(c) for c in np.unique(px, axis=0)}
    assert colors == {STRIPED.rgb, PATTERN_SECOND_COLOR[1]}


def test_vgradient_background_is_integer_interpolation():
    bg = Background("vgradient", (200, 100, 0), (100, 200, 50))
    img = render_image(SceneSpec(H, W, bg, ()))
    assert tuple(img[0, 0]) == (200, 100, 0) and tuple(img[-1, -1]) == (100, 200, 50)
    assert np.all(img[:, 0] == img[:, -1])


def test_scene_json_roundtrip_reproduces_masks():
    s = generate("occlusion", "val", 3, canvas_hw=(H * 3, W * 3))
    scene = SceneSpec.from_dict(s.meta["scene"])
    again = SceneSpec.from_json(scene.to_json())
    assert again == scene
    assert np.array_equal(object_masks(again), s.object_masks)
    assert np.array_equal(render_image(again), s.image)


def _check_alignment(image, scene, amodal):
    vis = visible_masks(scene, amodal)
    covered = np.zeros(image.shape[:2], bool)
    for o, v in zip(scene.objects, vis):
        cols = {tuple(c) for c in np.unique(image[v].reshape(-1, 3), axis=0)}
        assert cols <= set(S.appearance(o.appearance).colors), (o, cols)
        if S.appearance(o.appearance).pattern == "solid":
            assert cols == {S.appearance(o.appearance).rgb}
        covered |= v
    rest = image[~covered].reshape(-1, 3)
    allowed = set()
    bg = scene.background
    if bg.kind == "solid":
        allowed.add(tuple(bg.top))
    else:
        allowed |= {tuple(c) for c in np.unique(render_image(SceneSpec(scene.height, scene.width, bg, ()))[:, 0], axis=0)}
    if scene.ground is not None:
        allowed.add(tuple(scene.ground.color))
    assert {tuple(c) for c in np.unique(rest, axis=0)} <= allowed


@pytest.mark.parametrize("task", ["occlusion", "physics"])
def test_generated_masks_align_with_image_colors(task):
    for i in range(6):
        s = generate(task, "train", 40 + i, canvas_hw=(384, 448))
        scene = SceneSpec.from_dict(s.meta["scene"])
        amodal = object_masks(scene)  # d 的 object_masks 是第 k 步的；初始帧从场景规格重算
        if task == "occlusion":
            assert np.array_equal(amodal, s.object_masks)
            assert np.array_equal(amodal[s.meta["target_obj_id"]], s.target_mask)
        _check_alignment(s.image, scene, amodal)


# ---------------------------------------------------------------------------
# 目标、网格、画布
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_targets_grids_and_canvas(task):
    s = generate(task, "test", 11)
    assert s.target_mask.dtype == bool and s.target_mask.shape == (512, 512)
    assert s.target_grid.shape == (GRID_SIZE_E3B, GRID_SIZE_E3B)
    assert np.array_equal(s.target_grid, mask_to_grid(s.target_mask, GRID_SIZE_E3B))
    assert s.target_mask.any()
    for p in s.baselines.values():
        assert np.array_equal(p.grid, mask_to_grid(p.mask, GRID_SIZE_E3B))
    assert s.object_grids().shape == (len(s.object_masks), GRID_SIZE_E3B, GRID_SIZE_E3B)
    assert (s.image is None) == (task == "imagery")
    t = generate(task, "test", 11, canvas_hw=(384, 512), grid_size=16)
    assert t.target_mask.shape == (384, 512) and t.target_grid.shape == (16, 16)
    if t.image is not None:
        assert t.image.shape == (384, 512, 3) and t.image.dtype == np.uint8
    assert np.array_equal(t.target_grid, mask_to_grid(t.target_mask, 16))


def test_default_bin_rotates_with_index_and_explicit_bin_matches():
    for task in TASKS:
        n = num_bins(task)
        assert [generate(task, "train", i).meta["difficulty"]["bin"] for i in range(n)] == list(range(n))
        assert generate(task, "train", n + 1).digest() == generate(task, "train", n + 1, bin=1).digest()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate("f", "train", 0)
    with pytest.raises(ValueError):
        generate("c", "dev", 0)
    with pytest.raises(ValueError):
        generate("c", "train", -1)
    with pytest.raises(ValueError):
        generate("c", "train", 0, bin=3)
    with pytest.raises(ValueError):
        generate("c", "train", 0, grid_size=0)
    with pytest.raises(ValueError):
        generate("c", "train", 0, holdout_family="hexagon")


# ---------------------------------------------------------------------------
# 划分：大样本检查物体身份与场景种子两两不相交
# ---------------------------------------------------------------------------


def test_generated_identities_and_scene_seeds_disjoint_across_splits():
    per_task = {"occlusion": 120, "physics": 60, "imagery": 150}
    for task, n in per_task.items():
        idents = {s: set() for s in SPLITS}
        seeds = {s: set() for s in SPLITS}
        for split in SPLITS:
            for i in range(n):
                smp = generate(task, split, i)
                seeds[split].add(smp.meta["scene_seed"])
                for o in smp.meta["scene"]["objects"]:
                    ident = (o["family"], o["variant"], o["appearance"])
                    assert S.identity_split_table()[ident] == split
                    idents[split].add(ident)
        for a in range(3):
            for b in range(a + 1, 3):
                sa, sb = SPLITS[a], SPLITS[b]
                assert not idents[sa] & idents[sb], (task, sa, sb)
                assert not seeds[sa] & seeds[sb], (task, sa, sb)
        assert all(idents[s] for s in SPLITS)


# ---------------------------------------------------------------------------
# 数据集接口与导出
# ---------------------------------------------------------------------------


def test_dataset_class_matches_generate():
    ds = SynthDataset("e", "val", 5, start=10, grid_size=16)
    assert len(ds) == 5
    digests = [s.digest() for s in ds]
    assert digests == [generate("imagery", "val", 10 + i, grid_size=16).digest() for i in range(5)]
    assert ds[-1].digest() == digests[-1]
    with pytest.raises(IndexError):
        ds[5]


def test_export_roundtrip(tmp_path):
    samples = [generate("occlusion", "train", 1), generate("imagery", "train", 2), generate("physics", "train", 3)]
    samples += [generate("rotation3d", "train", 4), generate("viewpoint", "train", 5)]  # 带 extras（深度图、ID 图）
    manifest = export_samples(samples, tmp_path)
    rows = load_manifest(manifest)
    assert [r["stem"] for r in rows] == [
        "occlusion_train_000000001",
        "imagery_train_000000002",
        "physics_train_000000003",
        "rotation3d_train_000000004",
        "viewpoint_train_000000005",
    ]
    assert "extra_depth" in sample_arrays(samples[3]) and "extra_id_map" in sample_arrays(samples[4])
    assert not any(k.startswith("extra_") for k in sample_arrays(samples[0]))
    for s, r in zip(samples, rows):
        assert r["digest"] == s.digest() and r["instruction"] == s.instruction and r["meta"]["index"] == s.index
        z = np.load(tmp_path / r["arrays"])
        for k, a in sample_arrays(s).items():
            assert np.array_equal(z[k], a) and z[k].dtype == a.dtype
        if s.image is None:
            assert r["image"] is None
        else:
            assert np.array_equal(np.asarray(Image.open(tmp_path / r["image"]).convert("RGB")), s.image)
