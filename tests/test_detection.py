import numpy as np
import pytest

from src.constants import DETECTION_TIEBREAK_K
from src.metrics.detection import detection_bootstrap_statistic, detection_map
from src.stats import bootstrap_ci


def _gt(ids=(1, 2)):
    a, b = ids
    return {
        "images": [{"id": a, "width": 64, "height": 48}, {"id": b, "width": 64, "height": 48}],
        "categories": [{"id": 7, "name": "cup"}],
        "annotations": [
            {"id": 1, "image_id": a, "category_id": 7, "bbox": [10, 10, 20, 10]},
            {"id": 2, "image_id": b, "category_id": 7, "bbox": [30, 5, 15, 25]},
        ],
    }


def _right_on_first_wrong_on_second(ids=(1, 2)):
    a, b = ids
    return [
        {"image_id": a, "category_id": 7, "bbox_xyxy": [10, 10, 30, 20]},  # 对
        {"image_id": b, "category_id": 7, "bbox_xyxy": [0, 30, 10, 40]},  # 错（IoU 0）
    ]


def test_scored_predictions_hand():
    perfect = [
        {"image_id": 1, "category_id": 7, "bbox_xyxy": [10, 10, 30, 20], "score": 0.9},
        {"image_id": 2, "category_id": 7, "bbox_xyxy": [30, 5, 45, 30], "score": 0.8},
    ]
    r = detection_map(_gt(), perfect)
    assert r["AP"] == pytest.approx(1.0) and r["AP50"] == pytest.approx(1.0) and r["n_tiebreaks"] == 0
    assert (r["P50"], r["R50"], r["F1_50"]) == (1.0, 1.0, 1.0)
    half = detection_map(_gt(), perfect[:1])
    assert half["AP50"] == pytest.approx(51 / 101)  # 召回 0.5、精度 1 → 101 点插值
    assert (half["P50"], half["R50"], half["F1_50"]) == (1.0, 0.5, pytest.approx(2 / 3))
    g = _gt()
    detection_map(g, perfect)
    assert g == _gt()  # 不改调用方的 GT


def test_ap_ap50_ap75_distinct():
    """M4：AP、AP50、AP75 三者可区分（对调或把 AP75 当 AP 报都会被抓住）。"""
    gt = {
        "images": [{"id": 1, "width": 128, "height": 128}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [110, 110, 10, 10]},
        ],
    }
    preds = [
        {"image_id": 1, "category_id": 1, "bbox_xyxy": [110, 110, 120, 120], "score": 0.9},  # IoU 1
        {"image_id": 1, "category_id": 1, "bbox_xyxy": [0, 0, 100, 62.5], "score": 0.8},  # IoU 0.625
    ]
    r = detection_map(gt, preds)
    half = 51 / 101
    assert r["AP50"] == pytest.approx(1.0)
    assert r["AP75"] == pytest.approx(half)
    # IoU 阈值 0.50、0.55、0.60 上两个都对（AP=1），0.65…0.95 七档上只有一个对（AP=51/101）
    assert r["AP"] == pytest.approx((3 * 1.0 + 7 * half) / 10)
    assert len({round(r["AP"], 6), round(r["AP50"], 6), round(r["AP75"], 6)}) == 3


def test_s3_scoreless_ap_independent_of_image_numbering():
    """审查复现 1：同样的图、同样的预测，只换图像编号，AP 不再变（以前是 0.505 对 0.2525）。"""
    r12 = detection_map(_gt((1, 2)), _right_on_first_wrong_on_second((1, 2)))
    r21 = detection_map(_gt((2, 1)), _right_on_first_wrong_on_second((2, 1)))
    for k in ("AP", "AP50", "AP75", "AP50_min", "AP50_max", "P50", "R50", "F1_50"):
        assert r12[k] == r21[k], k
    assert r12["n_tiebreaks"] == DETECTION_TIEBREAK_K
    # K 次平局打破里两种排序都会出现：一对一错 → AP50 在 51/101 与其一半之间取平均
    assert r12["AP50_min"] == pytest.approx(0.5 * 51 / 101) and r12["AP50_max"] == pytest.approx(51 / 101)
    assert r12["AP50_min"] < r12["AP50"] < r12["AP50_max"]
    assert (r12["P50"], r12["R50"], r12["F1_50"]) == (0.5, 0.5, 0.5)
    # 成对（antithetic）置换：两种排序各占一半，AP50 恰为两者平均
    assert r12["AP50"] == pytest.approx((51 / 101 + 0.5 * 51 / 101) / 2)


def test_s3_equivalent_models_get_equal_ap():
    """审查 check_detection.py 的第二组：两个模型各对一张、错一张，只是对的是哪张不同，AP 应相同。"""
    m1 = [{"image_id": 1, "category_id": 7, "bbox_xyxy": [10, 10, 30, 20]}, {"image_id": 2, "category_id": 7, "bbox_xyxy": [0, 30, 10, 40]}]
    m2 = [{"image_id": 1, "category_id": 7, "bbox_xyxy": [40, 30, 60, 45]}, {"image_id": 2, "category_id": 7, "bbox_xyxy": [30, 5, 45, 30]}]
    r1, r2 = detection_map(_gt(), m1), detection_map(_gt(), m2)
    assert r1["AP50"] == r2["AP50"] and r1["AP"] == r2["AP"]


def test_s3_scoreless_ap_invariant_under_id_permutation():
    """审查复现 2：500 张图，只置换图像编号，AP50 完全不变（以前在 0.42–0.44 间变动）。"""
    rng = np.random.default_rng(0)
    N = 120
    content = []
    for _ in range(N):
        gts = [[rng.uniform(0, 400), rng.uniform(0, 300), 60.0, 40.0] for _ in range(int(rng.integers(1, 4)))]
        dets = [[g[0] + 2, g[1] + 1, g[0] + g[2] + 2, g[1] + g[3] + 1] for g in gts if rng.random() < 0.55]
        for _ in range(int(rng.integers(0, 2))):
            x, y = rng.uniform(0, 400), rng.uniform(0, 300)
            dets.append([x, y, x + 30, y + 30])
        content.append((gts, dets))

    def run(perm):
        images, anns, preds = [], [], []
        for k, (gts, dets) in enumerate(content):
            iid = int(perm[k]) + 1
            images.append({"id": iid, "width": 640, "height": 480})
            for g in gts:
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": 1, "bbox": g})
            for d in dets:
                preds.append({"image_id": iid, "category_id": 1, "bbox_xyxy": d})
        return detection_map({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "x"}]}, preds, n_tiebreaks=5)

    base = run(np.arange(N))
    for _ in range(3):
        r = run(rng.permutation(N))
        assert r["AP50"] == base["AP50"] and r["AP"] == base["AP"] and r["F1_50"] == base["F1_50"]
    assert base["AP50_min"] < base["AP50_max"]


def test_mixed_scores_rejected_and_bad_image_id():
    p = _right_on_first_wrong_on_second()
    p[0] = {**p[0], "score": 0.5}
    with pytest.raises(ValueError, match="有的有分数"):
        detection_map(_gt(), p)
    with pytest.raises(ValueError, match="image_id"):
        detection_map(_gt(), [{"image_id": 99, "category_id": 7, "bbox_xyxy": [0, 0, 1, 1]}])


def test_operating_point_order_free_and_iou_boundary():
    gt = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [4, 0, 10, 10]},
        ],
    }
    # d1 与两个 GT 的 IoU 都是 2/3；d2 只与 GT1 的 IoU 恰为 0.5（边界上算匹配），与 GT2 为 0.25。
    # 贪心按"先来先得"：d1 先占 GT1 → d2 无处可去，TP=1；最大匹配给 TP=2，且与顺序无关。
    d1 = {"image_id": 1, "category_id": 1, "bbox_xyxy": [2, 0, 12, 10]}
    d2 = {"image_id": 1, "category_id": 1, "bbox_xyxy": [0, 0, 10, 5]}
    for preds in ([d1, d2], [d2, d1]):
        r = detection_map(gt, preds, n_tiebreaks=2)
        assert (r["n_tp"], r["n_fp"], r["n_gt"]) == (2, 0, 2)
        assert r["F1_50"] == 1.0


def test_operating_point_crowd_regions_ignored():
    gt = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 40, 40], "iscrowd": 1},
        ],
    }
    preds = [
        {"image_id": 1, "category_id": 1, "bbox_xyxy": [0, 0, 10, 10]},  # TP
        {"image_id": 1, "category_id": 1, "bbox_xyxy": [55, 55, 70, 70]},  # 落在 crowd 区域里：忽略
        {"image_id": 1, "category_id": 1, "bbox_xyxy": [20, 20, 30, 30]},  # FP
    ]
    r = detection_map(gt, preds, n_tiebreaks=2)
    assert (r["n_tp"], r["n_fp"], r["n_gt"]) == (1, 1, 1)
    assert r["P50"] == 0.5 and r["R50"] == 1.0


def test_detection_empty_cases():
    r = detection_map(_gt(), [])
    assert r["AP"] == r["AP50"] == r["AP75"] == 0.0
    assert r["P50"] is None and r["R50"] == 0.0 and r["F1_50"] == 0.0
    assert detection_map({"images": [], "annotations": [], "categories": []}, []) is None


def test_detection_bootstrap_by_image():
    idx, stat = detection_bootstrap_statistic(_gt(), _right_on_first_wrong_on_second(), "F1_50")
    assert idx.tolist() == [0, 1]
    assert stat(np.array([0, 1])) == 0.5
    assert stat(np.array([0, 0])) == 1.0  # 图 1 抽两次：重新编号后两张都对
    assert stat(np.array([1, 1])) == 0.0
    ci = bootstrap_ci(idx, statistic=stat, n_resamples=50)
    assert ci.estimate == 0.5 and 0.0 <= ci.low < ci.high <= 1.0


def test_operating_point_one_gt_matched_once():
    gt = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }
    dup = [{"image_id": 1, "category_id": 1, "bbox_xyxy": [0, 0, 10, 10]}, {"image_id": 1, "category_id": 1, "bbox_xyxy": [0, 0, 10, 9]}]
    r = detection_map(gt, dup, n_tiebreaks=2)
    assert (r["n_tp"], r["n_fp"]) == (1, 1)


def test_detection_bootstrap_renumbers_duplicated_images():
    """同一张图被抽中两次时要当作两张图：否则合并成一张图、预测数超过 COCOeval 的 maxDets=100，召回被截断。"""
    boxes = [[(k % 10) * 20.0, (k // 10) * 20.0, 10.0, 10.0] for k in range(60)]
    gt = {
        "images": [{"id": 5, "width": 256, "height": 256}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [{"id": k + 1, "image_id": 5, "category_id": 1, "bbox": b} for k, b in enumerate(boxes)],
    }
    preds = [{"image_id": 5, "category_id": 1, "bbox_xyxy": [x, y, x + w, y + h], "score": 1 - k / 100} for k, (x, y, w, h) in enumerate(boxes)]
    _, stat = detection_bootstrap_statistic(gt, preds, "AP50")
    assert stat(np.array([0, 0])) == pytest.approx(1.0)
