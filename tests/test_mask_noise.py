"""mask 加噪过程：s=0 恒等、IoU 随 s 单调下降、各算子语义、确定性、噪声日程。"""

import inspect
import json
import math

import numpy as np
import pytest
from scipy import ndimage

from src.constants import (
    MASK_NOISE_BLOB_BAND,
    MASK_NOISE_BLOB_COUNT,
    MASK_NOISE_BLOB_RADIUS_MAX,
    MASK_NOISE_DILATE_MAX,
    MASK_NOISE_ERODE_MAX,
    MASK_NOISE_JITTER_AMP_MAX,
    MASK_NOISE_JITTER_SMOOTH,
    MASK_NOISE_MAX_OPS,
    MASK_NOISE_MERGE_MAX,
    MASK_NOISE_P_CLEAN,
    MASK_NOISE_SEVERITY_BETA,
    MASK_NOISE_SHIFT_MAX,
)
from src.noising import mask_noise as N
from src.noising.mask_noise import (
    OPS,
    corrupt,
    mask_scale,
    op_add_remove,
    op_dilate,
    op_erode,
    op_jitter,
    op_merge,
    op_translate,
    sample_severity,
    shift_mask,
)

H, W = 96, 160  # H ≠ W
YY, XX = np.mgrid[:H, :W]


def _iou(a, b):
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 1.0


def _batch():
    """批内样本形状、面积、位置互不相同：L 形、椭圆、细长条、两块不相连的圆。"""
    lshape = np.zeros((H, W), bool)
    lshape[20:70, 30:100] = True
    lshape[40:70, 60:100] = False
    ellipse = ((YY - 40) / 18.0) ** 2 + ((XX - 110) / 35.0) ** 2 < 1
    bar = np.zeros((H, W), bool)
    bar[10:90, 70:78] = True
    blobs = ((YY - 60) ** 2 + (XX - 40) ** 2 < 14**2) | ((YY - 30) ** 2 + (XX - 60) ** 2 < 10**2)
    return [lshape, ellipse, bar, blobs]


NEIGHBOR = (YY - 50) ** 2 + (XX - 125) ** 2 < 15**2
DISK = (YY - 48) ** 2 + (XX - 70) ** 2 < 25**2  # 标定用的圆盘：96×160 图里半径 25 px


def _others(i):
    return [[NEIGHBOR], [], [DISK], [NEIGHBOR, DISK]][i]


def test_fixture_batch_is_distinct_and_asymmetric():
    batch = _batch()
    areas = [int(m.sum()) for m in batch]
    assert len(set(areas)) == len(areas)
    for i, a in enumerate(batch):
        for b in batch[i + 1 :]:
            assert not np.array_equal(a, b)


# ---------------------------------------------------------------- 常量


def test_noise_constants_pinned():
    assert (MASK_NOISE_P_CLEAN, MASK_NOISE_SEVERITY_BETA, MASK_NOISE_MAX_OPS) == (0.1, (1.0, 1.0), 3)
    assert (MASK_NOISE_ERODE_MAX, MASK_NOISE_DILATE_MAX, MASK_NOISE_SHIFT_MAX) == (0.17, 0.24, 0.3)
    assert (MASK_NOISE_JITTER_AMP_MAX, MASK_NOISE_JITTER_SMOOTH) == (0.3, 0.1)
    assert (MASK_NOISE_BLOB_RADIUS_MAX, MASK_NOISE_BLOB_COUNT, MASK_NOISE_BLOB_BAND) == (0.5, (1, 3), 0.5)
    assert MASK_NOISE_MERGE_MAX == 1.0
    assert 1 <= MASK_NOISE_MAX_OPS <= len(OPS) and 1 <= MASK_NOISE_BLOB_COUNT[0] <= MASK_NOISE_BLOB_COUNT[1]


def test_single_op_calibration_at_full_severity():
    # constants 注释："s = 1 时单个算子作用于 96×160 图里半径 25 px 的圆盘，平均 IoU ≈ 0.5（钉住 [0.45, 0.55]）"
    scale = mask_scale(DISK)
    fns = {
        "erode": lambda rng: op_erode(DISK, 1.0, scale),
        "dilate": lambda rng: op_dilate(DISK, 1.0, scale),
        "jitter": lambda rng: op_jitter(DISK, 1.0, scale, rng),
        "add_remove": lambda rng: op_add_remove(DISK, 1.0, scale, rng),
        "translate": lambda rng: op_translate(DISK, 1.0, scale, rng),
    }
    for name, fn in fns.items():
        mean_iou = np.mean([_iou(fn(np.random.default_rng(s))[0], DISK) for s in range(40)])
        assert 0.45 <= mean_iou <= 0.55, (name, mean_iou)


# ---------------------------------------------------------------- s = 0 与单调性


def test_zero_severity_is_bitwise_identity():
    for i, m in enumerate(_batch()):
        for seed in range(12):
            out, rec = corrupt(m, 0.0, np.random.default_rng(seed), others=_others(i))
            assert out.dtype == bool and np.array_equal(out, m)
            assert len(rec["ops"]) >= 1
        scale = mask_scale(m)
        rng = np.random.default_rng(0)
        for out, _ in [
            op_erode(m, 0.0, scale),
            op_dilate(m, 0.0, scale),
            op_jitter(m, 0.0, scale, rng),
            op_add_remove(m, 0.0, scale, rng),
            op_translate(m, 0.0, scale, rng),
            op_merge(m, 0.0, [NEIGHBOR, DISK]),
        ]:
            assert np.array_equal(out, m)


def test_mean_iou_decreases_monotonically_with_severity():
    grid = np.linspace(0.0, 1.0, 11)
    means = []
    for s in grid:
        vals = [_iou(corrupt(m, s, np.random.default_rng(1000 * i + seed), others=_others(i))[0], m) for seed in range(16) for i, m in enumerate(_batch())]
        means.append(np.mean(vals))
    assert means[0] == 1.0
    assert np.all(np.diff(means) < 0), means
    assert means[-1] < 0.5


def _strip_magnitudes(rec):
    """去掉随 s 变化的幅度，只留随机抽取的部分（算子、方向、圆心、增删）。"""
    keep = []
    for op in rec["ops"]:
        if op["op"] == "add_remove":
            keep.append(("add_remove", tuple((b["kind"], tuple(b.get("center_rc", ()))) for b in op["blobs"])))
        elif op["op"] == "translate":
            keep.append(("translate", op["angle_rad"]))
        elif op["op"] == "merge":
            keep.append(("merge", op.get("neighbor"), op.get("neighbor_pixels")))
        else:
            keep.append((op["op"],))
    return keep


def test_same_seed_uses_same_random_draws_for_every_severity():
    for i, m in enumerate(_batch()):
        for seed in range(8):
            recs = [corrupt(m, s, np.random.default_rng(seed), others=_others(i))[1] for s in (0.05, 0.3, 0.8, 1.0)]
            assert all(_strip_magnitudes(r) == _strip_magnitudes(recs[0]) for r in recs)
        # 抖动：同一个随机场，s 越大翻转的像素集合越大（嵌套）
        scale = mask_scale(m)
        for seed in range(4):
            flipped = [op_jitter(m, s, scale, np.random.default_rng(seed))[0] != m for s in (0.2, 0.4, 0.7, 1.0)]
            assert all(np.all(a <= b) for a, b in zip(flipped, flipped[1:]))


# ---------------------------------------------------------------- 各算子语义


def test_dilate_never_shrinks_and_erode_never_grows():
    for m in _batch() + [DISK]:
        scale = mask_scale(m)
        prev_d, prev_e = m, m
        for s in np.linspace(0.1, 1.0, 10):
            d, rd = op_dilate(m, s, scale)
            e, re_ = op_erode(m, s, scale)
            assert np.all(d >= m) and d.sum() >= m.sum()  # 膨胀是超集，面积不减
            assert np.all(e <= m) and e.sum() <= m.sum()  # 腐蚀是子集，面积不增
            assert np.all(d >= prev_d) and np.all(e <= prev_e)  # s 越大越多
            prev_d, prev_e = d, e
            assert rd["radius_px"] == pytest.approx(s * MASK_NOISE_DILATE_MAX * scale)
            assert re_["radius_px"] == pytest.approx(s * MASK_NOISE_ERODE_MAX * scale)
    # 膨胀结果 = 到物体欧氏距离 ≤ r 的像素（独立参照：暴力算距离）
    m = _batch()[3]
    d, rec = op_dilate(m, 0.6, mask_scale(m))
    pts = np.argwhere(m)
    dist = np.sqrt(((np.argwhere(np.ones_like(m))[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(1).reshape(m.shape)
    assert np.array_equal(d, dist <= rec["radius_px"])


def test_erode_does_not_eat_from_image_border():
    full = np.ones((H, W), bool)
    assert np.array_equal(op_erode(full, 1.0, mask_scale(full))[0], full)
    edge = np.zeros((H, W), bool)
    edge[:, :20] = True  # 贴左边的竖条：左边界不是物体边界
    e, _ = op_erode(edge, 1.0, mask_scale(edge))
    assert e[:, 0].all() and not e[:, 19].any()


def test_translate_matches_record_and_does_not_wrap():
    for i, m in enumerate(_batch()):
        scale = mask_scale(m)
        for seed in range(6):
            out, rec = op_translate(m, 0.7, scale, np.random.default_rng(seed))
            dy, dx = rec["dy"], rec["dx"]
            assert np.array_equal(out, shift_mask(m, dy, dx))
            assert math.hypot(dy, dx) == pytest.approx(rec["distance_px"], abs=0.75)
            assert rec["distance_px"] == pytest.approx(0.7 * MASK_NOISE_SHIFT_MAX * scale)
            if out.sum() == m.sum():  # 没有移出图外时，质心的位移恰为 (dy, dx)
                c0, c1 = np.argwhere(m).mean(0), np.argwhere(out).mean(0)
                np.testing.assert_allclose(c1 - c0, [dy, dx], atol=1e-9)
    # 贴右边的物体往右移：移出去的像素丢弃，不会从左边绕回来
    edge = np.zeros((H, W), bool)
    edge[30:50, W - 10 :] = True
    moved = shift_mask(edge, 3, 4)
    expect = np.zeros((H, W), bool)
    expect[33:53, W - 6 :] = True  # 20 行 × 10 列右移 4：只剩 6 列
    assert np.array_equal(moved, expect)
    assert np.array_equal(shift_mask(edge, 0, 0), edge) and not shift_mask(edge, H, 0).any()


def test_merge_takes_nearest_neighbor_pixels_nearest_first():
    m = _batch()[1]  # 椭圆，中心 (40, 110)
    far = (YY - 85) ** 2 + (XX - 20) ** 2 < 8**2
    near = NEIGHBOR & ~m
    out, rec = op_merge(m, 0.5, [far, NEIGHBOR])
    assert rec["neighbor"] == 1  # 选最近的物体
    added = out & ~m
    assert np.all(out >= m) and added.any() and np.all(added <= near)  # 合并包含邻近物体的像素，不含别的
    assert rec["n_pixels"] == int(math.floor(0.5 * MASK_NOISE_MERGE_MAX * near.sum())) == added.sum()
    dist = ndimage.distance_transform_edt(~m)
    assert dist[added].max() <= dist[near & ~added].min()  # 由近到远取
    full, _ = op_merge(m, 1.0, [far, NEIGHBOR])
    assert np.array_equal(full, m | near)  # s = 1：整个邻近物体并入
    same, rec0 = op_merge(m, 0.8, [])
    assert np.array_equal(same, m) and rec0["skipped"] == "no neighbor"
    # ref：邻居与像素先后按参照物体定，并入的是当前 mask。当前 mask 是平移过的椭圆，离 far 更近，
    # 但参照（原椭圆）离 NEIGHBOR 更近，所以仍并入 NEIGHBOR 离原椭圆最近的那些像素。
    moved = shift_mask(m, 30, -60)
    out_ref, rec_ref = op_merge(moved, 0.5, [far, NEIGHBOR], ref=m)
    assert rec_ref["neighbor"] == 1 and np.array_equal(out_ref, moved | added)
    assert op_merge(moved, 0.5, [far, NEIGHBOR])[1]["neighbor"] == 0  # 不给 ref 时按当前 mask 选


def test_jitter_changes_only_near_the_boundary():
    for m in _batch():
        scale = mask_scale(m)
        for seed in range(4):
            out, rec = op_jitter(m, 0.6, scale, np.random.default_rng(seed))
            # 复现同一个随机场：变化的像素都满足 |φ| < A·max|n|
            field = ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal(m.shape), MASK_NOISE_JITTER_SMOOTH * scale)
            field /= field.std()
            phi = np.where(m, ndimage.distance_transform_edt(m), -ndimage.distance_transform_edt(~m))
            changed = out != m
            assert changed.any()
            assert np.all(np.abs(phi[changed]) < rec["amplitude_px"] * np.abs(field).max())
            assert rec["amplitude_px"] == pytest.approx(0.6 * MASK_NOISE_JITTER_AMP_MAX * scale)


def test_add_remove_blobs_follow_their_record():
    for m in _batch():
        scale = mask_scale(m)
        dist_out = ndimage.distance_transform_edt(~m)
        for seed in range(6):
            out, rec = op_add_remove(m, 0.8, scale, np.random.default_rng(seed))
            lo, hi = MASK_NOISE_BLOB_COUNT
            assert lo <= len(rec["blobs"]) <= hi
            expect = m.copy()
            for b in rec["blobs"]:
                cy, cx = b["center_rc"]
                assert 0.5 * 0.8 * MASK_NOISE_BLOB_RADIUS_MAX * scale <= b["radius_px"] <= 0.8 * MASK_NOISE_BLOB_RADIUS_MAX * scale
                disk = (YY - cy) ** 2 + (XX - cx) ** 2 < b["radius_px"] ** 2
                if b["kind"] == "add":
                    assert not m[cy, cx] and dist_out[cy, cx] <= MASK_NOISE_BLOB_BAND * scale
                    expect |= disk
                else:
                    assert m[cy, cx]
                    expect &= ~disk
            assert np.array_equal(out, expect)


# ---------------------------------------------------------------- 组合、确定性、记录、输入检查


def test_corrupt_output_contract_and_determinism():
    for i, m in enumerate(_batch()):
        for seed in range(8):
            s = 0.1 + 0.1 * seed
            out, rec = corrupt(m, s, np.random.default_rng(seed), others=_others(i))
            out2, rec2 = corrupt(m, s, np.random.default_rng(seed), others=_others(i))
            assert out.dtype == bool and out.shape == (H, W)
            assert np.array_equal(out, out2) and rec == rec2  # 同种子逐位相同
            names = [op["op"] for op in rec["ops"]]
            assert 1 <= len(names) <= MASK_NOISE_MAX_OPS
            assert names == sorted(names, key=OPS.index) and len(set(names)) == len(names)  # 按 OPS 顺序、不重复
            if not _others(i):
                assert "merge" not in names
            assert rec["severity"] == s and rec["scale_px"] == pytest.approx(math.sqrt(m.sum()))
            json.dumps(rec)  # 记录可直接写 JSON
    # 不同种子一般给出不同结果
    m = _batch()[0]
    outs = {corrupt(m, 0.6, np.random.default_rng(seed), others=[NEIGHBOR])[0].tobytes() for seed in range(6)}
    assert len(outs) > 1


def test_corrupt_does_not_modify_inputs_and_handles_empty_mask():
    m = _batch()[2]
    m0 = m.copy()
    nb0 = NEIGHBOR.copy()
    corrupt(m, 0.9, np.random.default_rng(0), others=[NEIGHBOR])
    assert np.array_equal(m, m0) and np.array_equal(NEIGHBOR, nb0)
    empty = np.zeros((H, W), bool)
    for seed in range(5):
        out, _ = corrupt(empty, 1.0, np.random.default_rng(seed), others=[NEIGHBOR])
        assert not out.any()
    # 0/1 整数 mask 与 bool mask 等价
    a, _ = corrupt(m.astype(np.uint8), 0.5, np.random.default_rng(3))
    b, _ = corrupt(m, 0.5, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_corrupt_does_not_depend_on_memory_layout():
    for i, m in enumerate(_batch()):
        ref = corrupt(m, 0.7, np.random.default_rng(i), others=_others(i))
        big = np.zeros((2 * H, 3 * W), bool)
        big[::2, ::3] = m
        for v in (np.asfortranarray(m), big[::2, ::3]):
            out, rec = corrupt(v, 0.7, np.random.default_rng(i), others=[np.asfortranarray(o) for o in _others(i)])
            assert np.array_equal(out, ref[0]) and rec == ref[1]


def test_corrupt_input_checks():
    m = _batch()[0]
    rng = np.random.default_rng(0)
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            corrupt(m, bad, rng)
    with pytest.raises(ValueError):
        corrupt(m.astype(np.float64), 0.5, rng)
    with pytest.raises(ValueError):
        corrupt(m.astype(np.int64) * 2, 0.5, rng)
    with pytest.raises(ValueError):
        corrupt(m, 0.5, rng, others=[NEIGHBOR.T])
    assert not any("task" in name for name in inspect.signature(corrupt).parameters)


# ---------------------------------------------------------------- 噪声日程


def test_sample_severity_follows_schedule():
    rng = np.random.default_rng(0)
    s = np.array([sample_severity(rng) for _ in range(20000)])
    assert s.min() >= 0.0 and s.max() <= 1.0
    p0 = np.mean(s == 0.0)
    assert abs(p0 - MASK_NOISE_P_CLEAN) < 4 * math.sqrt(MASK_NOISE_P_CLEAN * (1 - MASK_NOISE_P_CLEAN) / s.size)
    a, b = MASK_NOISE_SEVERITY_BETA
    nz = s[s > 0]
    assert abs(nz.mean() - a / (a + b)) < 0.01
    # Beta(1, 1) 即均匀：十等分直方图各格约占 10%
    hist = np.histogram(nz, bins=10, range=(0, 1))[0] / nz.size
    assert np.all(np.abs(hist - 0.1) < 0.015)
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    assert [sample_severity(r1) for _ in range(50)] == [sample_severity(r2) for _ in range(50)]


def test_module_uses_only_whitelisted_libraries():
    src = inspect.getsource(N)
    imports = [ln.split()[1] for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert set(imports) <= {"__future__", "math", "typing", "numpy", "scipy", "src.constants"}
