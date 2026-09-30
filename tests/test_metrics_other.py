import numpy as np
import pytest

from src.constants import NYU_DEPTH_MAX, NYU_DEPTH_MIN, NYU_EIGEN_CROP, NYU_EVAL_HW
from src.metrics.box import box_acc_at_iou, box_sample_iou
from src.metrics.depth import depth_dataset_metrics, depth_sample_metrics, nyu_eval_mask
from src.metrics.misc import count_accuracy, multiple_choice_accuracy, normalize_choice, pair_accuracy
from src.metrics.normal import normal_angular_errors, normal_metrics

# ---------------------------------------------------------------- 框


def test_box_metrics_hand():
    gts = [[0, 0, 10, 10], [0, 0, 4, 2], [5, 5, 9, 9]]
    preds = [[0, 0, 10, 5], [2, 1, 6, 5], None]
    ious = [box_sample_iou(p, g) for p, g in zip(preds, gts)]
    assert ious == pytest.approx([0.5, 2 / 22, 0.0])
    assert box_acc_at_iou(ious) == pytest.approx(1 / 3)  # 0.5 恰好记对
    assert box_acc_at_iou([]) is None


# ---------------------------------------------------------------- 深度


def test_eigen_crop_mask():
    gt = np.full(NYU_EVAL_HW, 3.0)
    r0, r1, c0, c1 = NYU_EIGEN_CROP
    m = nyu_eval_mask(gt)
    assert m.sum() == (r1 - r0) * (c1 - c0)
    assert m[r0, c0] and not m[r0 - 1, c0] and not m[r0, c1]
    with pytest.raises(ValueError):
        nyu_eval_mask(np.ones((640, 480)))


def test_valid_interval_open_at_both_ends():
    """D10(1)：有效 GT 为 (1e-3, 10)，与 BTS/AdaBins 一致；GT 恰为 10.0 的像素被排除。"""
    r0, _, c0, _ = NYU_EIGEN_CROP
    gt = np.zeros(NYU_EVAL_HW)
    gt[r0, c0 : c0 + 5] = [NYU_DEPTH_MIN, 2.0, np.nextafter(NYU_DEPTH_MAX, 0), NYU_DEPTH_MAX, NYU_DEPTH_MAX + 0.01]
    m = nyu_eval_mask(gt)
    assert m[r0, c0 : c0 + 5].tolist() == [False, True, True, False, False]
    assert m.sum() == 2


def _nyu_gt():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.7, 8.0, NYU_EVAL_HW)  # ×1.2 后仍 ≤ 10，不触发裁剪
    gt[:, :200] *= 0.5  # 非均匀
    gt[100:110, 300:310] = 0.0  # 无效
    return gt


def test_depth_metric_hand():
    gt = _nyu_gt()
    m = nyu_eval_mask(gt)
    pred = gt * 1.2  # AbsRel = 0.2，δ1：1.2 < 1.25 → 1
    r = depth_sample_metrics(pred, gt, align="none")
    assert r["abs_rel"] == pytest.approx(0.2) and r["d1"] == 1.0 and r["n_valid"] == m.sum()
    pred2 = gt * 1.3  # 部分像素 > 10 被裁剪：逐像素对照手算
    r2 = depth_sample_metrics(pred2, gt, align="none")
    clipped = np.clip(pred2[m], NYU_DEPTH_MIN, NYU_DEPTH_MAX)
    assert r2["abs_rel"] == pytest.approx(np.mean(np.abs(gt[m] - clipped) / gt[m]))
    assert r2["d1"] == pytest.approx(np.mean(np.maximum(gt[m] / clipped, clipped / gt[m]) < 1.25))


def test_depth_pred_clipping_and_nonfinite():
    gt = _nyu_gt()
    pred = gt.copy()
    r0, _, c0, _ = NYU_EIGEN_CROP
    pred[r0, c0] = np.inf  # → 10
    pred[r0, c0 + 1] = np.nan  # → 1e-3
    pred[r0, c0 + 2] = -5.0  # → 1e-3
    r = depth_sample_metrics(pred, gt, align="none")
    g = gt[r0, c0 : c0 + 3]
    p = np.array([NYU_DEPTH_MAX, NYU_DEPTH_MIN, NYU_DEPTH_MIN])
    n = nyu_eval_mask(gt).sum()
    assert r["abs_rel"] == pytest.approx(np.sum(np.abs(g - p) / g) / n)


def test_scale_shift_alignment_recovers_affine():
    gt = _nyu_gt()
    pred = (gt - 0.3) / 2.5  # gt = 2.5 * pred + 0.3
    r = depth_sample_metrics(pred, gt, align="scale_shift")
    assert r["scale"] == pytest.approx(2.5) and r["shift"] == pytest.approx(0.3)
    assert r["abs_rel"] == pytest.approx(0.0, abs=1e-10) and r["d1"] == 1.0
    assert depth_sample_metrics(pred, gt, align="none")["abs_rel"] > 0.5
    with pytest.raises(ValueError):
        depth_sample_metrics(pred, gt, align="median")


def test_scale_shift_against_independent_reference():
    """M4：预测不是 GT 的精确仿射；无效 GT 与 crop 外像素带极端值；对齐后部分越界需要裁剪；
    拟合必须只用有效像素上的未裁剪预测。期望值用 np.polyfit 独立算出。"""
    rng = np.random.default_rng(11)
    gt = rng.uniform(0.7, 8.0, NYU_EVAL_HW)
    r0, r1, c0, c1 = NYU_EIGEN_CROP
    gt[r0 + 5 : r0 + 25, c0 + 5 : c0 + 25] = 9.9  # 对齐后会越过 10 的区域
    pred = 3.0 * gt + 0.5 + rng.normal(0, 0.8, gt.shape)  # 未裁剪的预测大量 > 10：先裁剪再拟合会错
    pred[r0 + 5 : r0 + 25, c0 + 5 : c0 + 25] += 6.0
    # 无效 GT 与 crop 外：极端预测值（在这些像素上拟合会把 s、t 拉偏）
    gt[r0 + 50 : r0 + 60, c0 + 50 : c0 + 60] = 0.0
    pred[r0 + 50 : r0 + 60, c0 + 50 : c0 + 60] = 1e4
    pred[:r0, :] = -5e3
    pred[:, c1:] = 7e3
    m = nyu_eval_mask(gt)
    s, t = np.polyfit(pred[m], gt[m], 1)
    aligned = s * pred[m] + t
    assert (aligned > NYU_DEPTH_MAX).sum() > 50  # 确实有需要裁剪的像素
    aligned = np.clip(aligned, NYU_DEPTH_MIN, NYU_DEPTH_MAX)
    g = gt[m]
    exp_absrel = np.mean(np.abs(g - aligned) / g)
    exp_d1 = np.mean(np.maximum(g / aligned, aligned / g) < 1.25)
    r = depth_sample_metrics(pred, gt, align="scale_shift")
    assert r["scale"] == pytest.approx(s, rel=1e-9) and r["shift"] == pytest.approx(t, rel=1e-9)
    assert r["abs_rel"] == pytest.approx(exp_absrel, rel=1e-9)
    assert r["d1"] == pytest.approx(exp_d1, rel=1e-9)
    assert 0 < exp_d1 < 1


def test_depth_dataset_and_empty():
    gt = _nyu_gt()
    empty = np.zeros(NYU_EVAL_HW)
    s = [depth_sample_metrics(gt * 1.1, gt, align="none"), depth_sample_metrics(gt, empty, align="none")]
    assert s[1] is None
    d = depth_dataset_metrics(s)
    assert d["abs_rel"] == pytest.approx(0.1) and d["n_images"] == 1 and d["n_skipped"] == 1
    assert depth_dataset_metrics([]) is None and depth_dataset_metrics([None]) is None


# ---------------------------------------------------------------- 法向


def test_normal_errors_hand():
    gt = np.zeros((2, 3, 3))
    gt[..., 2] = 1.0  # 全部朝 +z
    pred = np.zeros((2, 3, 3))
    pred[0, 0] = [0, 0, 5]  # 0°（未单位化也行）
    pred[0, 1] = [1, 0, 0]  # 90°
    pred[0, 2] = [1, 0, 1]  # 45°
    pred[1, 0] = [0, 0, -1]  # 180°
    pred[1, 1] = [0, 0, 0]  # 零向量 → 90°
    pred[1, 2] = [0, 1, 1]  # 45°，但 GT 无效（零向量，叠加掩掉）
    gt[1, 2] = 0.0
    e = normal_angular_errors(pred, gt, np.ones((2, 3), bool))
    assert e == pytest.approx([0, 90, 45, 180, 90])
    m = normal_metrics([e, np.array([10.0])])
    assert m["mean"] == pytest.approx((0 + 90 + 45 + 180 + 90 + 10) / 6)
    assert m["median"] == pytest.approx(67.5)
    assert m["within_11.25"] == pytest.approx(2 / 6)
    assert m["within_22.5"] == pytest.approx(2 / 6)
    assert m["within_30"] == pytest.approx(2 / 6)
    assert normal_metrics([]) is None and normal_metrics([np.zeros(0)]) is None


def test_normal_valid_is_required_and_applied():
    gt = np.zeros((2, 4, 3))
    gt[..., 0] = 1
    gt[0, 0] = [-1, -1, -1]  # uint8 映射后无效像素的典型样子：范数非零，只能由 valid 掩掉
    pred = gt.copy()
    pred[0, 0] = [1, 0, 0]
    pred[1, 3] = [0, 1, 0]
    valid = np.ones((2, 4), bool)
    valid[0, 0] = False
    valid[1, 3] = False
    assert normal_angular_errors(pred, gt, valid).max() == pytest.approx(0.0)
    assert normal_angular_errors(pred, gt, np.ones((2, 4), bool)).max() == pytest.approx(125.26438968, rel=1e-6)
    with pytest.raises(TypeError):
        normal_angular_errors(pred, gt)
    with pytest.raises(ValueError):
        normal_angular_errors(pred, gt, valid.astype(np.uint8))


# ---------------------------------------------------------------- 计数、点对、多选


def test_count_and_pair_accuracy():
    assert count_accuracy([3, 5, None, 2], [3, 4, 1, 2]) == 0.5
    assert count_accuracy([], []) is None
    with pytest.raises(TypeError):
        count_accuracy([3.0], [3])
    with pytest.raises(TypeError):
        count_accuracy([True], [1])
    assert pair_accuracy(["A", "B", None], ["A", "A", "B"]) == pytest.approx(1 / 3)
    assert pair_accuracy([], []) is None


@pytest.mark.parametrize(
    "raw,letter",
    [("A", "A"), ("b", "B"), (" (C) ", "C"), ("[d]", "D"), ("A.", "A"), ("B:", "B"), ("C)", "C"),
     ("(A) a red cup", "A"), ("B. the left one", "B"), ("D) none", "D"), ("(A) apple (A)", "A"),
     ("The answer is A", None), ("AB", None), ("", None), (None, None), ("A cat", None),
     # 审查 M2 的三个含糊回答：前缀后面还出现别的选项标记 → None
     ("(A) or (B)", None), ("A. No, the answer is B.", None), ("A) and C) are both right", None),
     ("(a) cat (b) dog", None), ("B: not D:", None)],
)
def test_normalize_choice(raw, letter):
    assert normalize_choice(raw) == letter


def test_multiple_choice_accuracy():
    assert multiple_choice_accuracy(["(A)", "b.", "The answer is C", None], ["A", "(B)", "C", "D"]) == 0.5
    assert multiple_choice_accuracy(["(A) or (B)", "A"], ["A", "A"]) == 0.5
    assert multiple_choice_accuracy([], []) is None
    with pytest.raises(ValueError):
        multiple_choice_accuracy(["A"], ["??"])
