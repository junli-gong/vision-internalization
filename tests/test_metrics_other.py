import numpy as np
import pytest

from src.constants import NYU_DEPTH_MAX, NYU_DEPTH_MIN, NYU_EIGEN_CROP, NYU_EVAL_HW
from src.metrics.box import box_acc_at_iou, box_sample_iou
from src.metrics.depth import depth_dataset_metrics, depth_sample_metrics, nyu_eval_mask
from src.metrics.detection import detection_map
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


def test_valid_interval_bounds():
    r0, _, c0, _ = NYU_EIGEN_CROP
    gt = np.zeros(NYU_EVAL_HW)
    gt[r0, c0 : c0 + 4] = [NYU_DEPTH_MIN, 2.0, NYU_DEPTH_MAX, NYU_DEPTH_MAX + 0.01]
    m = nyu_eval_mask(gt)
    assert m[r0, c0 : c0 + 4].tolist() == [False, True, True, False]  # (1e-3, 10.0]，见 constants 待裁
    assert m.sum() == 2


def _nyu_pair():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.7, 8.0, NYU_EVAL_HW)  # ×1.2 后仍 ≤ 10，不触发裁剪
    gt[:, :200] *= 0.5  # 非均匀
    gt[100:110, 300:310] = 0.0  # 无效
    return gt


def test_depth_metric_hand():
    gt = _nyu_pair()
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
    gt = _nyu_pair()
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
    gt = _nyu_pair()
    pred = (gt - 0.3) / 2.5  # gt = 2.5 * pred + 0.3
    r = depth_sample_metrics(pred, gt, align="scale_shift")
    assert r["scale"] == pytest.approx(2.5) and r["shift"] == pytest.approx(0.3)
    assert r["abs_rel"] == pytest.approx(0.0, abs=1e-10) and r["d1"] == 1.0
    # 不对齐时同一预测很差
    assert depth_sample_metrics(pred, gt, align="none")["abs_rel"] > 0.5
    with pytest.raises(ValueError):
        depth_sample_metrics(pred, gt, align="median")


def test_depth_dataset_and_empty():
    gt = _nyu_pair()
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
    pred[1, 2] = [0, 1, 1]  # 45°，但 GT 无效
    gt[1, 2] = 0.0
    e = normal_angular_errors(pred, gt)
    assert e == pytest.approx([0, 90, 45, 180, 90])
    m = normal_metrics([e, np.array([10.0])])
    assert m["mean"] == pytest.approx((0 + 90 + 45 + 180 + 90 + 10) / 6)
    assert m["median"] == pytest.approx(67.5)
    assert m["within_11.25"] == pytest.approx(2 / 6)
    assert m["within_22.5"] == pytest.approx(2 / 6)
    assert m["within_30"] == pytest.approx(2 / 6)
    assert normal_metrics([]) is None and normal_metrics([np.zeros(0)]) is None


def test_normal_valid_mask():
    gt = np.zeros((2, 4, 3))
    gt[..., 0] = 1
    pred = gt.copy()
    pred[1, 3] = [0, 1, 0]
    valid = np.ones((2, 4), bool)
    valid[1, 3] = False
    assert normal_angular_errors(pred, gt, valid).max() == pytest.approx(0.0)
    assert normal_angular_errors(pred, gt).max() == pytest.approx(90.0)


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
     ("(A) a red cup", "A"), ("B. the left one", "B"), ("D) none", "D"),
     ("The answer is A", None), ("AB", None), ("", None), (None, None), ("A cat", None)],
)
def test_normalize_choice(raw, letter):
    assert normalize_choice(raw) == letter


def test_multiple_choice_accuracy():
    assert multiple_choice_accuracy(["(A)", "b.", "The answer is C", None], ["A", "(B)", "C", "D"]) == 0.5
    assert multiple_choice_accuracy([], []) is None
    with pytest.raises(ValueError):
        multiple_choice_accuracy(["A"], ["??"])


# ---------------------------------------------------------------- 检测 mAP


def _gt():
    return {
        "images": [{"id": 1, "width": 64, "height": 48}, {"id": 2, "width": 64, "height": 48}],
        "categories": [{"id": 7, "name": "cup"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 10]},
            {"id": 2, "image_id": 2, "category_id": 7, "bbox": [30, 5, 15, 25]},
        ],
    }


def test_detection_map_hand():
    perfect = [
        {"image_id": 1, "category_id": 7, "bbox_xyxy": [10, 10, 30, 20]},
        {"image_id": 2, "category_id": 7, "bbox_xyxy": [30, 5, 45, 30]},
    ]
    r = detection_map(_gt(), perfect)
    assert r == {"AP": pytest.approx(1.0), "AP50": pytest.approx(1.0)}
    # 只检出一半：召回 0.5、精度 1 → 101 点插值下 51/101
    half = detection_map(_gt(), perfect[:1])
    assert half["AP"] == pytest.approx(51 / 101) and half["AP50"] == pytest.approx(51 / 101)
    # 一个正确一个错位（IoU=0）：同分 1.0，按输入顺序排序
    wrong = [perfect[0], {"image_id": 2, "category_id": 7, "bbox_xyxy": [0, 30, 10, 40]}]
    assert detection_map(_gt(), wrong)["AP50"] == pytest.approx(51 / 101)
    # 跨图同分：按图像 id 排，与输入顺序无关
    assert detection_map(_gt(), wrong[::-1])["AP50"] == pytest.approx(51 / 101)
    # 同一图内同分：按输入顺序。错的在前 → 精度曲线 0.5 → AP 减半（docstring 所述的顺序依赖）
    same_img = [{"image_id": 1, "category_id": 7, "bbox_xyxy": [40, 30, 60, 45]}, perfect[0]]
    assert detection_map(_gt(), same_img[::-1])["AP50"] == pytest.approx(51 / 101)
    assert detection_map(_gt(), same_img)["AP50"] == pytest.approx(0.5 * 51 / 101)
    # 显式分数覆盖顺序
    scored = [dict(same_img[0], score=0.1), dict(same_img[1], score=0.9)]
    assert detection_map(_gt(), scored)["AP50"] == pytest.approx(51 / 101)
    g = _gt()
    detection_map(g, perfect)
    assert g == _gt()  # 不改调用方的 GT


def test_detection_map_empty_cases():
    assert detection_map(_gt(), []) == {"AP": 0.0, "AP50": 0.0}
    assert detection_map({"images": [], "annotations": [], "categories": []}, []) is None
