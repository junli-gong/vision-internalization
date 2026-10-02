"""DEV-0023：E2 新增的指标（depth_sample_rmse、pixel_correct_total / pixel_accuracy、法向的逐图汇总）与四个任务的损失、
逐样本统计、数据集级聚合，在构造的例子上与手算一致。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.constants import ADE_IGNORE_INDEX, NYU_EIGEN_CROP, NYU_EVAL_HW
from src.metrics.depth import depth_sample_metrics, depth_sample_rmse, fit_scale_shift, nyu_eval_mask, clip_pred
from src.metrics.normal import (
    NORMAL_HIST_BIN_DEG,
    hist_median,
    normal_angular_errors,
    normal_image_summary,
    normal_metrics,
    normal_metrics_from_summaries,
)
from src.metrics.seg import intersect_and_union, mean_iou, pixel_accuracy, pixel_correct_total
from src.e2.config import load_config, resolve_task
from src.e2.tasks import DepthTask, NormalsTask, RefsegTask, SemsegTask, Target, depth_train_valid, upsample

H, W = NYU_EVAL_HW


def _gt(seed=0):
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.5, 8.0, size=(H, W))
    g[rng.uniform(size=(H, W)) < 0.1] = 0.0
    return g


# ---------------------------------------------------------------------------
# 深度 RMSE
# ---------------------------------------------------------------------------


def test_depth_rmse_by_hand_on_eigen_crop():
    gt = np.zeros((H, W))
    r0, r1, c0, c1 = NYU_EIGEN_CROP
    gt[r0:r1, c0:c1] = 2.0
    gt[0, 0] = 5.0  # crop 外，不计
    pred = np.full((H, W), 3.0)
    assert depth_sample_rmse(pred, gt, align="none") == pytest.approx(1.0)
    pred[r0, c0] = 12.0  # 预测先 clip 到 NYU_DEPTH_MAX=10
    n = (r1 - r0) * (c1 - c0)
    want = math.sqrt(((n - 1) * 1.0 + 8.0**2) / n)
    assert depth_sample_rmse(pred, gt, align="none") == pytest.approx(want)
    assert depth_sample_rmse(pred, np.zeros((H, W)), align="none") is None


def test_depth_rmse_uses_the_same_prediction_as_depth_sample_metrics():
    """同一组有效像素、同一个（对齐后的）预测：由 rmse 的规则复算的 AbsRel 与 depth_sample_metrics 逐位相同。"""
    gt = _gt(1)
    rng = np.random.default_rng(2)
    pred = gt * 1.3 + 0.2 + rng.normal(0, 0.3, size=gt.shape)
    pred[5, 7] = np.inf
    mask = nyu_eval_mask(gt)
    for align in ("none", "scale_shift"):
        pv, gv = pred[mask], gt[mask]
        if align == "none":
            pv = clip_pred(pv)
        else:
            fin = np.where(np.isfinite(pv), pv, clip_pred(pv))
            s, t = fit_scale_shift(fin, gv)
            pv = clip_pred(s * fin + t)
        m = depth_sample_metrics(pred, gt, align=align)
        assert m["abs_rel"] == float(np.mean(np.abs(gv - pv) / gv))
        assert depth_sample_rmse(pred, gt, align=align) == float(np.sqrt(np.mean((gv - pv) ** 2)))
    exact = 2.0 * gt + 1.0
    assert depth_sample_rmse(exact, gt, align="scale_shift") == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 像素准确率
# ---------------------------------------------------------------------------


def test_pixel_correct_total_by_hand_and_equals_intersection_sum():
    label = np.array([[0, 1, 1], [2, 2, 3]], dtype=np.uint8)  # 原始 ADE 标注：0 忽略，1→0、2→1、3→2
    pred = np.array([[5, 0, 1], [1, 1, 2]])
    c, t = pixel_correct_total(pred, label, reduce_zero=True)
    assert (c, t) == (4, 5)
    inter, _ = intersect_and_union(pred, label, reduce_zero=True)
    assert c == int(inter.sum())
    rng = np.random.default_rng(0)
    for _ in range(5):
        lab = rng.integers(0, 151, size=(17, 23))
        pr = rng.integers(-1, 152, size=(17, 23))
        c, t = pixel_correct_total(pr, lab, reduce_zero=True)
        inter, _ = intersect_and_union(pr, lab, reduce_zero=True)
        assert c == int(inter.sum()) and t == int(np.count_nonzero(lab > 0))
    assert pixel_accuracy([(4, 5), (1, 5)]) == pytest.approx(0.5)
    assert pixel_accuracy([]) is None and pixel_accuracy([(0, 0)]) is None


# ---------------------------------------------------------------------------
# 法向的逐图汇总
# ---------------------------------------------------------------------------


def test_hist_median_by_hand():
    h = np.zeros(1800)
    h[100] = 3  # 10.0–10.1°
    assert hist_median(h) == pytest.approx(10.0 + 0.5 * NORMAL_HIST_BIN_DEG)
    h2 = np.zeros(1800)
    h2[10], h2[20] = 1, 1
    assert hist_median(h2) == pytest.approx(1.1)  # 累计首次达到 n/2 的箱是第 10 箱，箱内插值到上沿
    assert hist_median(np.zeros(1800)) is None


def _median_bound(errs) -> float:
    """hist_median 与 np.median 之差的上界：一个箱宽 + 两个中间值之差的一半（偶数个时）。"""
    a = np.sort(np.concatenate(errs))
    n = a.size
    gap = 0.0 if n % 2 else (a[n // 2] - a[n // 2 - 1]) / 2
    return NORMAL_HIST_BIN_DEG + gap + 1e-12


def test_normal_summaries_match_normal_metrics():
    rng = np.random.default_rng(3)
    for sizes in ((500, 1, 3000, 0, 777), (2, 2), (3,), (40_000, 1)):
        errs = [rng.gamma(2.0, 8.0, size=n).clip(0, 180) for n in sizes]
        errs[-1][:] = 180.0
        ref = normal_metrics(errs)
        got = normal_metrics_from_summaries([normal_image_summary(e) for e in errs])
        assert got["n_pixels"] == ref["n_pixels"]
        assert got["mean"] == pytest.approx(ref["mean"], rel=1e-12)
        for k in ("within_11.25", "within_22.5", "within_30"):
            assert got[k] == ref[k]
        assert abs(got["median"] - ref["median"]) <= _median_bound(errs)
    assert normal_metrics_from_summaries([normal_image_summary(np.zeros(0))]) is None


# ---------------------------------------------------------------------------
# 任务：损失
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tcfg():
    cfg = load_config()
    return {t: resolve_task(cfg, t, tiny=False) for t in ("depth", "semseg", "normals", "refseg")}


def _depth_target(gt: np.ndarray) -> Target:
    valid = depth_train_valid(gt)
    return Target(gt.shape, {"idx": torch.from_numpy(np.flatnonzero(valid)), "log_gt": torch.from_numpy(np.log(gt[valid]).astype(np.float32))},
                  {"gt": gt.astype(np.float32)}, int(valid.sum()))


def test_silog_by_hand_and_vectorized(tcfg):
    task = DepthTask(tcfg["depth"])
    assert (task.lam, task.scale) == (0.85, 10.0)
    gt = np.array([[1.0, 2.0], [4.0, 0.0]])  # 0 无效
    pred_log = torch.tensor([[[[0.0, 0.0], [0.0, 9.0]]], [[[math.log(2.0), math.log(4.0)], [math.log(8.0), 0.0]]]])
    loss = task.loss(pred_log, _depth_target(gt))
    d0 = np.array([0.0, -math.log(2), -math.log(4)])
    want0 = math.sqrt(np.mean(d0**2) - 0.85 * np.mean(d0) ** 2) * 10
    d1 = np.full(3, math.log(2.0))  # 预测是 GT 的两倍：d 恒定，只剩 (1 − λ)·d²
    want1 = math.sqrt((1 - 0.85) * math.log(2.0) ** 2) * 10
    assert loss.shape == (2,)
    assert loss[0].item() == pytest.approx(want0, rel=1e-5) and loss[1].item() == pytest.approx(want1, rel=1e-5)


def test_depth_train_valid_matches_eval_range():
    gt = np.array([[0.0, 1e-3, 0.0011, 9.99, 10.0, np.nan]])
    assert depth_train_valid(gt).tolist() == [[False, False, True, True, False, False]]


def test_cross_entropy_ignores_label_zero(tcfg):
    task = SemsegTask(tcfg["semseg"])
    lab = np.array([[0, 1], [2, 150]], dtype=np.uint8)
    tr = np.array([[ADE_IGNORE_INDEX, 0], [1, 149]])
    tgt = Target(lab.shape, {"label": torch.from_numpy(tr)}, {"label": lab}, 3)
    logits = torch.randn(3, 150, 2, 2, generator=torch.Generator().manual_seed(0))
    loss = task.loss(logits, tgt)
    for p in range(3):
        ref = F.cross_entropy(logits[p:p + 1], torch.from_numpy(tr)[None], ignore_index=ADE_IGNORE_INDEX)
        assert loss[p].item() == pytest.approx(ref.item(), rel=1e-6)


def test_cosine_loss_by_hand(tcfg):
    task = NormalsTask(tcfg["normals"])
    gt = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).T  # 两个有效像素 [3, 2]
    tgt = Target((1, 3), {"idx": torch.tensor([0, 2]), "gt": gt}, {}, 2)
    same = torch.tensor([[[[0.0, 5.0, 2.0]], [[0.0, 5.0, 0.0]], [[3.0, 5.0, 0.0]]]])  # 像素 0 = (0,0,3)，像素 2 = (2,0,0)
    opp = -same
    loss = task.loss(torch.cat([same, opp]), tgt)
    assert loss[0].item() == pytest.approx(0.0, abs=1e-6) and loss[1].item() == pytest.approx(2.0, abs=1e-6)


def test_bce_dice_by_hand(tcfg):
    task = RefsegTask(tcfg["refseg"])
    m = np.array([[1.0, 0.0]], dtype=np.float32)
    tgt = Target((1, 2), {"mask": torch.from_numpy(m)}, {"mask": m.astype(bool)}, 2)
    x = torch.tensor([[[[2.0, -1.0]]]])
    loss = task.loss(x, tgt)
    p = 1 / (1 + np.exp(-np.array([2.0, -1.0])))
    bce = np.mean([-math.log(p[0]), -math.log(1 - p[1])])
    dice = 1 - (2 * p[0] + 1) / (p.sum() + 1 + 1)
    assert loss.item() == pytest.approx(bce + dice, rel=1e-5)


def test_upsample_is_bilinear_align_corners_false():
    x = torch.tensor([[[[0.0, 1.0]]]])
    up = upsample(x, (1, 4), {"mode": "bilinear", "align_corners": False})
    assert up.flatten().tolist() == pytest.approx([0.0, 0.25, 0.75, 1.0])
    with pytest.raises(ValueError):
        upsample(x, (1, 4), {"mode": "nearest", "align_corners": False})


# ---------------------------------------------------------------------------
# 任务：逐样本统计与聚合
# ---------------------------------------------------------------------------


def test_semseg_stats_and_aggregate_by_hand(tcfg):
    task = SemsegTask(tcfg["semseg"])
    lab = np.array([[0, 1, 1, 2]], dtype=np.uint8)
    tgt = Target(lab.shape, {}, {"label": lab}, 3)
    logits = torch.full((1, 150, 1, 4), -5.0)
    for c, k in enumerate([7, 0, 1, 0]):  # 预测 = [7, 0, 1, 0]；GT（reduce 之后）= [忽略, 0, 0, 1]
        logits[0, k, 0, c] = 5.0
    st = task.sample_stats(logits, tgt)
    agg = task.aggregate(st, n_resamples=20)
    # 类 0：交 1，并 = 预测 2 + 标签 2 − 1 = 3；类 1：交 0，并 = 1 + 1 = 2；mIoU = (1/3 + 0)/2
    assert agg["miou"] == pytest.approx((1 / 3 + 0.0) / 2)
    assert agg["pixel_acc"] == pytest.approx(1 / 3)
    assert agg["miou"] == pytest.approx(mean_iou([(st[0]["inter"], st[0]["union"])]))
    assert agg["miou_ci"]["low"] == agg["miou_ci"]["high"] == pytest.approx(agg["miou"])


def test_refseg_stats_and_aggregate_by_hand(tcfg):
    task = RefsegTask(tcfg["refseg"])
    out = []
    for gt, logit in ((np.array([[1, 1, 0, 0]], bool), [1.0, -1.0, 1.0, -1.0]), (np.array([[0, 0, 0, 1]], bool), [-1, -1, -1, 2.0])):
        tgt = Target(gt.shape, {}, {"mask": gt}, gt.size)
        out += task.sample_stats(torch.tensor(logit).reshape(1, 1, 1, 4), tgt)
    assert out == [{"inter": 1, "union": 3}, {"inter": 1, "union": 1}]
    agg = task.aggregate(out, n_resamples=50)
    assert agg["ciou"] == pytest.approx(2 / 4) and agg["giou"] == pytest.approx((1 / 3 + 1) / 2)
    assert agg["ciou_ci"]["low"] <= agg["ciou"] <= agg["ciou_ci"]["high"]


def test_depth_aggregate_skips_images_without_valid_pixels(tcfg):
    task = DepthTask(tcfg["depth"])
    gt = _gt(5).astype(np.float32)
    up = torch.from_numpy(np.log(np.stack([gt.clip(1e-3), gt.clip(1e-3) * 1.1]))[:, None].astype(np.float32))
    st = task.sample_stats(up, _depth_target(gt))
    assert st[0]["abs_rel"] == pytest.approx(0.0, abs=1e-6) and st[0]["d1"] == 1.0
    assert st[1]["abs_rel"] == pytest.approx(0.1, rel=1e-4) and st[1]["abs_rel_ss"] == pytest.approx(0.0, abs=1e-5)
    empty = task.sample_stats(up[:1], _depth_target(np.zeros_like(gt)))
    agg = task.aggregate([st[1], empty[0]], n_resamples=10)
    assert empty == [None] and agg["n_used"] == 1 and agg["n_skipped_no_valid"] == 1
    assert agg["abs_rel"] == pytest.approx(st[1]["abs_rel"]) and agg["rmse"] == pytest.approx(st[1]["rmse"])
    packed = task.pack([st[1], None])
    assert np.isnan(packed["abs_rel"][1]) and packed["abs_rel"][0] == st[1]["abs_rel"]


def test_normals_stats_match_normal_metrics(tcfg):
    task = NormalsTask(tcfg["normals"])
    rng = np.random.default_rng(7)
    n = rng.normal(size=(6, 8, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    valid = rng.uniform(size=(6, 8)) > 0.3
    pred = torch.from_numpy(rng.normal(size=(2, 3, 6, 8)).astype(np.float32))
    tgt = Target((6, 8), {}, {"gt": n.astype(np.float32), "valid": valid}, int(valid.sum()))
    st = task.sample_stats(pred, tgt)
    for p, s in zip(pred, st):
        e = normal_angular_errors(p.permute(1, 2, 0).numpy(), n.astype(np.float32), valid)
        ref = normal_metrics([e])
        agg = task.aggregate([s], n_resamples=10)
        assert agg["mean"] == pytest.approx(ref["mean"], rel=1e-6)
        assert agg["within_30"] == ref["within_30"]
        assert abs(agg["median"] - ref["median"]) <= _median_bound([e]) + 1e-4  # 预测经 fp32 单位化，误差在 1e-5 度量级
