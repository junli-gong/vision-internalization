"""DEV-0024 第 3 条 I2：E2 按光栅位置分开的标题指标（src/e2/positions.py 的划分；tasks.position_stats 与 PositionAccumulator）。

- 划分的定义：上半 / 下半按像素中心所在的行；token 光栅下标的四分位按像素中心所在的 LLM token 格；点坐标与像素中心一致；
- 按像素划分的三个任务：每个区域的逐样本统计 = 把区域外的 GT 置为无效后调 src/metrics 的参考函数（逐项相同或在浮点求和误差内）；
  一种划分的各区域相加 = 整张图（可相加的任务）；
- 指代分割按 GT 掩码质心归入区域；空掩码不进任何区域；
- PositionAccumulator 的累计与计数。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.constants import ADE_NUM_CLASSES, ADE_REDUCE_ZERO_LABEL, NYU_EVAL_HW
from src.e2 import positions as POS
from src.e2.config import load_config, resolve_task
from src.e2.tasks import DepthTask, NormalsTask, PositionAccumulator, RefsegTask, SemsegTask, Target
from src.metrics.depth import depth_sample_metrics, depth_sample_rmse
from src.metrics.normal import normal_angular_errors, normal_image_summary
from src.metrics.seg import intersect_and_union, pixel_correct_total, reduce_zero_label


@pytest.fixture(scope="module")
def tcfg():
    cfg = load_config()
    return {t: resolve_task(cfg, t, tiny=False) for t in ("depth", "semseg", "normals", "refseg")}


# ---------------------------------------------------------------------------
# 划分的定义
# ---------------------------------------------------------------------------


def test_halves_and_raster_quartiles_by_hand():
    lab = POS.pixel_labels((480, 640), (15, 20))  # Qwen 640×480 的网格；每格 32×32 像素
    assert lab["half"].shape == (480, 640) and lab["half"][239].max() == 0 and lab["half"][240].min() == 1
    q = lab["raster_quartile"]
    # 300 个 token，每个四分位 75 个：第 3 行（k = 60..79）的前 15 格是 q1，后 5 格是 q2
    assert q[3 * 32 + 5, 14 * 32 + 31] == 0 and q[3 * 32 + 5, 15 * 32] == 1
    assert q[0, 0] == 0 and q[-1, -1] == 3 and q[7 * 32, 10 * 32] == 2  # k = 150 → q3
    cells = q[16::32, 16::32]  # 每格中心一个像素
    assert [int((cells == r).sum()) for r in range(4)] == [75, 75, 75, 75]
    odd = POS.pixel_labels((5, 3), (2, 1))["half"][:, 0].tolist()
    assert odd == [0, 0, 1, 1, 1]  # H 为奇数：正中一行（中心正好在 H/2）归下半
    # 7 个像素分 3 格：中心 (i + 0.5)·3/7 = 0.21, 0.64, 1.07, 1.50, 1.93, 2.36, 2.79
    assert POS.cell_index(480, 15).tolist()[::32] == list(range(15)) and POS.cell_index(7, 3).tolist() == [0, 0, 1, 1, 1, 2, 2]


@pytest.mark.parametrize("hw,grid", [((480, 640), (15, 20)), ((640, 480), (20, 15)), ((512, 683), (14, 19)), ((37, 53), (16, 16)),
                                     ((5, 7), (3, 4))])
def test_point_labels_equal_pixel_labels_at_pixel_centers(hw, grid):
    lab = POS.pixel_labels(hw, grid)
    rng = np.random.default_rng(0)
    for y, x in [(0, 0), (hw[0] - 1, hw[1] - 1)] + [tuple(v) for v in rng.integers(0, hw, size=(40, 2))]:
        p = POS.point_labels(y + 0.5, x + 0.5, hw, grid)
        assert p == {"half": int(lab["half"][y, x]), "raster_quartile": int(lab["raster_quartile"][y, x])}, (y, x)
    assert POS.point_labels(hw[0], hw[1], hw, grid)["raster_quartile"] == 3  # 右下角边界归最后一格
    with pytest.raises(ValueError):
        POS.point_labels(-1.0, 0.0, hw, grid)


def test_mask_centroid():
    m = np.zeros((4, 6), bool)
    m[1, 2] = m[3, 4] = True
    assert POS.mask_centroid(m) == (2.5, 3.5) and POS.mask_centroid(np.zeros((2, 2), bool)) is None


# ---------------------------------------------------------------------------
# 按像素划分：与参考函数逐项相同
# ---------------------------------------------------------------------------


def _depth_gt(seed=0):
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.5, 8.0, size=NYU_EVAL_HW)
    g[rng.uniform(size=NYU_EVAL_HW) < 0.1] = 0.0
    return g.astype(np.float32)


@pytest.mark.parametrize("grid", [(15, 20), (16, 16), (14, 19)])
def test_depth_position_stats_equal_reference_on_region_masked_gt(tcfg, grid):
    task = DepthTask(tcfg["depth"])
    gt = _depth_gt(1)
    rng = np.random.default_rng(2)
    up = torch.from_numpy(np.log(rng.uniform(0.3, 12.0, size=(2, 1) + NYU_EVAL_HW)).astype(np.float32))
    tgt = Target(gt.shape, {}, {"gt": gt}, 0)
    whole = task.sample_stats(up, tgt)
    ps = task.position_stats(up, tgt, grid, whole)
    lab = POS.pixel_labels(gt.shape, grid)
    pred = torch.exp(up[:, 0]).numpy()
    for p, per in zip(pred, ps):
        for s, regs in POS.SPLITS.items():
            for r, st in enumerate(per[s]):
                g_r = np.where(lab[s] == r, gt, 0.0)  # 区域外置为无效
                ref = depth_sample_metrics(p, g_r, align="none")
                if ref is None:
                    assert st is None
                    continue
                assert st["abs_rel"] == pytest.approx(ref["abs_rel"], rel=1e-10)
                assert st["d1"] == pytest.approx(ref["d1"], rel=1e-12)
                assert st["rmse"] == pytest.approx(depth_sample_rmse(p, g_r, align="none"), rel=1e-10)
    # 区域里没有有效像素：None（上半全部置无效）
    gt2 = gt.copy()
    gt2[:240] = 0.0
    ps2 = task.position_stats(up, Target(gt2.shape, {}, {"gt": gt2}, 0), grid, whole)
    assert ps2[0]["half"][0] is None and ps2[0]["half"][1] is not None


def test_semseg_position_stats_equal_reference_and_partition(tcfg):
    task = SemsegTask(tcfg["semseg"])
    rng = np.random.default_rng(3)
    H, W, grid = 37, 53, (5, 7)
    lab = rng.integers(0, ADE_NUM_CLASSES + 1, size=(H, W)).astype(np.uint8)
    lab[:3] = 0  # 一些忽略的像素
    tr = reduce_zero_label(lab) if ADE_REDUCE_ZERO_LABEL else lab.astype(np.int64)
    tgt = Target((H, W), {}, {"label": lab, "label_reduced": tr}, 0)
    up = torch.from_numpy(rng.normal(size=(2, ADE_NUM_CLASSES, H, W)).astype(np.float32))
    whole = task.sample_stats(up, tgt)
    ps = task.position_stats(up, tgt, grid, whole)
    labels = POS.pixel_labels((H, W), grid)
    pred = up.argmax(dim=1).numpy()
    for p, w, per in zip(pred, whole, ps):
        for s, regs in POS.SPLITS.items():
            for r, st in enumerate(per[s]):
                l_r = np.where(labels[s] == r, lab, 0)  # 区域外置为 0（reduce 之后忽略）
                inter, union = intersect_and_union(p, l_r, reduce_zero=ADE_REDUCE_ZERO_LABEL)
                c, t = pixel_correct_total(p, l_r, reduce_zero=ADE_REDUCE_ZERO_LABEL)
                assert np.array_equal(st["inter"], inter) and np.array_equal(st["union"], union) and (st["correct"], st["total"]) == (c, t)
            # 一种划分的各区域相加 = 整张图（交、正确、总数；并不可加，不比）
            assert np.array_equal(sum(x["inter"] for x in per[s]), w["inter"])
            assert sum(x["correct"] for x in per[s]) == w["correct"] and sum(x["total"] for x in per[s]) == w["total"]


def test_normals_position_stats_equal_reference_and_partition(tcfg):
    task = NormalsTask(tcfg["normals"])
    rng = np.random.default_rng(4)
    H, W, grid = 24, 32, (3, 4)
    n = rng.normal(size=(H, W, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    valid = rng.uniform(size=(H, W)) > 0.3
    n[~valid] = 0.0
    tgt = Target((H, W), {}, {"gt": n.astype(np.float32), "valid": valid}, int(valid.sum()))
    up = torch.from_numpy(rng.normal(size=(2, 3, H, W)).astype(np.float32))
    whole = task.sample_stats(up, tgt)
    ps = task.position_stats(up, tgt, grid, whole)
    w2, p2 = task.eval_stats(up, tgt, grid)  # 角误差只算一次的路径：结果逐项相同
    for a, b in zip(whole + [x for per in ps for s in POS.SPLITS for x in per[s]],
                    w2 + [x for per in p2 for s in POS.SPLITS for x in per[s]]):
        assert a["sum"] == b["sum"] and a["n"] == b["n"] and np.array_equal(a["hist"], b["hist"])
    labels = POS.pixel_labels((H, W), grid)
    u = F.normalize(up, dim=1, eps=1e-12).permute(0, 2, 3, 1).numpy()
    for p, w, per in zip(u, whole, ps):
        for s, regs in POS.SPLITS.items():
            for r, st in enumerate(per[s]):
                ref = normal_image_summary(normal_angular_errors(p, n.astype(np.float32), valid & (labels[s] == r)))
                assert st["n"] == ref["n"] and st["sum"] == pytest.approx(ref["sum"], rel=1e-12)
                assert np.array_equal(st["within"], ref["within"]) and np.array_equal(st["hist"], ref["hist"])
            assert sum(x["n"] for x in per[s]) == w["n"] and np.array_equal(sum(x["hist"] for x in per[s]), w["hist"])
            assert sum(x["sum"] for x in per[s]) == pytest.approx(w["sum"], rel=1e-12)


# ---------------------------------------------------------------------------
# 指代分割：按质心归入区域
# ---------------------------------------------------------------------------


def test_refseg_position_by_target_centroid(tcfg):
    task = RefsegTask(tcfg["refseg"])
    H, W, grid = 40, 60, (4, 6)  # 24 格，每个四分位 6 格 = 一行
    m = np.zeros((H, W), bool)
    m[22:30, 5:20] = True  # 质心 (26.5, 12.5)：下半；第 2 行 → k = 2·6 + 1 = 13 → q3
    tgt = Target((H, W), {}, {"mask": m}, m.size)
    up = torch.randn(2, 1, H, W, generator=torch.Generator().manual_seed(0))
    whole = task.sample_stats(up, tgt)
    ps = task.position_stats(up, tgt, grid, whole)
    assert task.position_labels(tgt, grid) == {"half": 1, "raster_quartile": 2}
    for w, per in zip(whole, ps):
        assert per["half"] == [None, w] and per["raster_quartile"] == [None, None, w, None]
    empty = Target((H, W), {}, {"mask": np.zeros((H, W), bool)}, m.size)
    assert task.position_unassigned(empty) and not task.position_unassigned(tgt)
    pe = task.position_stats(up, empty, grid, task.sample_stats(up, empty))
    assert all(x is None for per in pe for s in POS.SPLITS for x in per[s])


# ---------------------------------------------------------------------------
# 累计
# ---------------------------------------------------------------------------


def test_accumulator_sums_additive_tasks_and_counts(tcfg):
    task = SemsegTask(tcfg["semseg"])
    rng = np.random.default_rng(5)
    H, W, grid = 16, 24, (4, 4)
    acc = PositionAccumulator(task, ["a"])
    per_sample = []
    for k in range(3):
        lab = rng.integers(0, 6, size=(H, W)).astype(np.uint8)
        if k == 2:
            lab[: H // 2] = 0  # 第 3 张图的上半全部忽略：上半里没有参与评测的像素
        tgt = Target((H, W), {}, {"label": lab, "label_reduced": reduce_zero_label(lab)}, 0)
        up = torch.from_numpy(rng.normal(size=(1, ADE_NUM_CLASSES, H, W)).astype(np.float32))
        whole = task.sample_stats(up, tgt)
        (ps,) = task.position_stats(up, tgt, grid, whole)
        acc.add_sample(tgt)
        acc.add("a", ps)
        per_sample.append(ps)
    res = acc.result()["a"]
    top = res["half"]["top"]
    assert top["n"] == 3 and top["n_used"] == 2 and set(top) == {"miou", "pixel_acc", "n", "n_used", "n_classes_present"}
    ref = task.aggregate([ps["half"][0] for ps in per_sample], n_resamples=0, ci=False)
    assert top["miou"] == pytest.approx(ref["miou"]) and top["pixel_acc"] == pytest.approx(ref["pixel_acc"])
    assert acc.n_samples == 3 and acc.n_unassigned == 0


def test_accumulator_lists_for_depth_and_refseg(tcfg):
    depth = DepthTask(tcfg["depth"])
    acc = PositionAccumulator(depth, ["d"])
    gt = _depth_gt(6)
    up = torch.from_numpy(np.log(gt.clip(1e-3) * 1.2)[None, None].astype(np.float32))
    tgt = Target(gt.shape, {}, {"gt": gt}, 0)
    (ps,) = depth.position_stats(up, tgt, (15, 20), depth.sample_stats(up, tgt))
    gt0 = gt.copy()
    gt0[240:] = 0.0  # 下半没有有效像素
    t0 = Target(gt.shape, {}, {"gt": gt0}, 0)
    (ps0,) = depth.position_stats(up, t0, (15, 20), depth.sample_stats(up, t0))
    for t, p in ((tgt, ps), (t0, ps0)):
        acc.add_sample(t)
        acc.add("d", p)
    bottom = acc.result()["d"]["half"]["bottom"]
    assert (bottom["n"], bottom["n_used"], bottom["n_skipped_no_valid"]) == (2, 1, 1)
    assert set(bottom) == {"abs_rel", "d1", "rmse", "n", "n_used", "n_skipped_no_valid"}
    assert bottom["abs_rel"] == pytest.approx(ps["half"][1]["abs_rel"])

    ref = RefsegTask(tcfg["refseg"])
    acc = PositionAccumulator(ref, ["r"])
    for rows in ((2, 5), (30, 35), None):
        m = np.zeros((40, 60), bool)
        if rows:
            m[rows[0]:rows[1], 10:20] = True
        t = Target((40, 60), {}, {"mask": m}, m.size)
        up = torch.randn(1, 1, 40, 60, generator=torch.Generator().manual_seed(1))
        acc.add_sample(t)
        acc.add("r", ref.position_stats(up, t, (4, 6), ref.sample_stats(up, t))[0])
    res = acc.result()["r"]
    assert res["half"]["top"]["n"] == res["half"]["bottom"]["n"] == 1 and acc.n_unassigned == 1
    assert res["raster_quartile"]["q2"]["n"] == 0 and res["raster_quartile"]["q2"]["ciou"] is None
