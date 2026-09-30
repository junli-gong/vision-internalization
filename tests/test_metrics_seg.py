import numpy as np
import pytest

from src.constants import ADE_IGNORE_INDEX, ADE_NUM_CLASSES, BOUNDARY_IOU_DILATION_RATIO
from src.metrics.seg import (
    MaskStats,
    boundary_iou,
    ciou,
    giou,
    intersect_and_union,
    mask_inter_union,
    mask_to_boundary,
    mean_boundary_iou,
    mean_iou,
    miou_from_totals,
    reduce_zero_label,
    sample_iou,
)

# ---------------------------------------------------------------- cIoU / gIoU


def test_ciou_giou_hand():
    # 三个样本的 (交, 并) 互不相同
    stats = [(3, 4), (1, 5), (0, 0)]
    assert ciou(stats) == pytest.approx(4 / 9)
    assert giou(stats) == pytest.approx((0.75 + 0.2 + 1.0) / 3)
    assert sample_iou(0, 0) == 1.0  # 空并集约定
    assert sample_iou(0, 7) == 0.0


def test_mask_inter_union_hand():
    p = np.zeros((3, 5), bool)
    g = np.zeros((3, 5), bool)
    p[0, 0:3] = True
    g[0:2, 2] = True
    assert mask_inter_union(p, g) == MaskStats(1, 4, False)
    assert tuple(mask_inter_union(p.astype(np.uint8), g.astype(np.int64))) == (1, 4, False)  # 0/1 整数也可
    with pytest.raises(ValueError):
        mask_inter_union(p, g.T)


def test_seg_empty_returns_none():
    assert ciou([]) is None and giou([]) is None and mean_boundary_iou([]) is None
    assert ciou([(0, 0), (0, 0)]) is None  # 累计并为 0：未定义


def test_none_prediction_is_parse_failure():
    gt = np.zeros((4, 6), bool)
    gt[1:3, 2:5] = True
    empty = np.zeros((4, 6), bool)
    s_fail = mask_inter_union(None, gt)
    assert s_fail == MaskStats(0, 6, True)
    s_fail_empty = mask_inter_union(None, empty)
    assert s_fail_empty == MaskStats(0, 0, True)
    s_ok = mask_inter_union(gt, gt)
    assert sample_iou(*s_fail) == 0.0 and sample_iou(*s_fail_empty) == 0.0  # GT 为空也记 0
    assert sample_iou(*mask_inter_union(empty, empty)) == 1.0  # 真实的空预测 + 空 GT 仍按空并集约定
    assert ciou([s_fail, s_ok]) == pytest.approx(6 / 12)  # 失败样本按 (0, |GT|) 计入
    assert giou([s_fail, s_fail_empty, s_ok]) == pytest.approx(1 / 3)
    assert boundary_iou(None, gt) == 0.0 and boundary_iou(None, empty) == 0.0
    assert boundary_iou(empty, empty) == 1.0


def test_non_binary_inputs_rejected():
    gt = np.zeros((6, 8), np.uint8)
    gt[:, :4] = 1
    prob = np.where(gt == 1, 0.9, 0.02)  # 概率图：审查 M3 的复现（以前被按 ≠0 静默二值化，IoU 算成 0.5）
    with pytest.raises(ValueError, match="dtype"):
        mask_inter_union(prob, gt)
    with pytest.raises(ValueError, match="dtype"):
        boundary_iou(prob, gt)
    with pytest.raises(ValueError, match="dtype"):
        mask_inter_union(gt.astype(np.float32), gt)  # 浮点即使只含 0/1 也拒绝
    with pytest.raises(ValueError, match="0/1"):
        mask_inter_union(gt * 2, gt)
    with pytest.raises(ValueError, match="0/1"):
        mask_to_boundary(gt - 1)


# ---------------------------------------------------------------- Boundary IoU


def _official_mask_to_boundary_naive(mask, dilation_ratio):
    """官方 mask_to_boundary 的朴素复刻：零填充 1 像素，图外视为最大值（cv2.erode 默认），逐次 3×3 最小值。"""
    h, w = mask.shape
    dilation = int(round(dilation_ratio * np.sqrt(h**2 + w**2)))
    if dilation < 1:
        dilation = 1
    cur = np.zeros((h + 2, w + 2), np.uint8)
    cur[1:-1, 1:-1] = mask
    for _ in range(dilation):
        big = np.ones((h + 4, w + 4), np.uint8)  # 图外 = 最大值
        big[1:-1, 1:-1] = cur
        nxt = np.ones_like(cur)
        for di in range(3):
            for dj in range(3):
                nxt = np.minimum(nxt, big[di : di + h + 2, dj : dj + w + 2])
        cur = nxt
    return mask.astype(np.uint8) - cur[1 : h + 1, 1 : w + 1]


def _random_blob(h, w, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    for _ in range(4):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.integers(2, h // 2 + 2), rng.integers(2, w // 2 + 2)
        yy, xx = np.mgrid[:h, :w]
        m |= (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).astype(np.uint8)
    return m


# (35,120)：0.02*对角线恰为 2.5，银行家舍入给 2（向上舍入给 3）；(36,120)：2.506，round 给 3、floor 给 2
SHAPES = [(35, 120), (36, 120), (61, 23), (7, 11), (240, 320), (3, 3)]


def test_mask_to_boundary_matches_naive_official():
    for k, (h, w) in enumerate(SHAPES):
        for seed in range(3):
            m = _random_blob(h, w, 10 * k + seed)
            m[0, :] = 1  # 贴边的 mask：零填充使图像边缘也算边界
            ours = mask_to_boundary(m, BOUNDARY_IOU_DILATION_RATIO)
            ref = _official_mask_to_boundary_naive(m, BOUNDARY_IOU_DILATION_RATIO)
            assert np.array_equal(ours, ref.astype(bool)), (h, w, seed)


def test_mask_to_boundary_matches_cv2_if_available():
    cv2 = pytest.importorskip("cv2")
    for k, (h, w) in enumerate(SHAPES):
        m = _random_blob(h, w, 100 + k)
        m[:, -1] = 1
        dilation = int(round(BOUNDARY_IOU_DILATION_RATIO * np.sqrt(h**2 + w**2)))
        dilation = max(dilation, 1)
        new = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        er = cv2.erode(new, np.ones((3, 3), np.uint8), iterations=dilation)[1 : h + 1, 1 : w + 1]
        assert np.array_equal(mask_to_boundary(m), (m - er).astype(bool)), (h, w)


@pytest.mark.parametrize("h,w,band", [(35, 120, 2), (36, 120, 3)])
def test_boundary_dilation_uses_python_round(h, w, band):
    # 全 1 mask 的边界带宽 = dilation。35×120：round(2.5)=2（不是 3）；36×120：round(2.506)=3（不是 floor 的 2）
    b = mask_to_boundary(np.ones((h, w), np.uint8))
    assert b[:, w // 2].sum() == 2 * band
    assert b[h // 2, :].sum() == 2 * band


def test_boundary_iou_hand():
    # 10×12 图，dilation = round(0.02*sqrt(244)) = round(0.31) → 0 → 取 1：边界带宽 1
    gt = np.zeros((10, 12), bool)
    gt[2:6, 3:9] = True  # 4×6 方块，边界 = 周长像素 16
    pred = np.zeros((10, 12), bool)
    pred[2:6, 3:10] = True  # 右移扩一列：4×7，边界 18
    bg, bp = mask_to_boundary(gt), mask_to_boundary(pred)
    assert bg.sum() == 16 and bp.sum() == 18
    assert (bg & bp).sum() == 14  # 上下两行的前 6 列 12 个 + 左列内部 2 个
    assert boundary_iou(pred, gt) == pytest.approx(14 / (16 + 18 - 14))
    assert boundary_iou(gt, gt) == 1.0
    assert boundary_iou(np.zeros((4, 5), bool), np.zeros((4, 5), bool)) == 1.0  # 两边都空：约定 1.0（D10(3)）
    assert mean_boundary_iou([0.5, 1.0]) == 0.75


def test_boundary_iou_pred_not_containing_gt():
    """M4：预测不包含 GT（互相错开），期望值由独立的朴素边界实现算出。"""
    gt = np.zeros((40, 70), np.uint8)
    gt[5:25, 10:40] = 1
    pred = np.zeros((40, 70), np.uint8)
    pred[12:33, 3:30] = 1
    bg = _official_mask_to_boundary_naive(gt, BOUNDARY_IOU_DILATION_RATIO).astype(bool)
    bp = _official_mask_to_boundary_naive(pred, BOUNDARY_IOU_DILATION_RATIO).astype(bool)
    expected = (bg & bp).sum() / (bg | bp).sum()
    assert 0.01 < expected < 0.5
    assert boundary_iou(pred, gt) == pytest.approx(expected, abs=0)
    assert boundary_iou(gt, pred) == pytest.approx(expected, abs=0)


# ---------------------------------------------------------------- mIoU


def test_reduce_zero_label_matches_mmseg():
    lab = np.array([[0, 1, 2], [150, 255, 0]], np.uint8)
    out = reduce_zero_label(lab)
    assert out.tolist() == [[ADE_IGNORE_INDEX, 0, 1], [149, ADE_IGNORE_INDEX, ADE_IGNORE_INDEX]]
    ref = lab.copy()
    ref[ref == 0] = 255
    ref = ref - 1
    ref[ref == 254] = 255
    assert np.array_equal(out, ref.astype(np.int64))


def test_intersect_and_union_hand():
    K = 4
    label = np.array([[0, 0, 1, ADE_IGNORE_INDEX], [2, 2, 1, 1]])
    pred = np.array([[0, 1, 1, 3], [2, 0, 1, 7]])  # 7 越界：不计入任何类
    inter, union = intersect_and_union(pred, label, K, reduce_zero=False)
    assert inter.tolist() == [1, 2, 1, 0]
    assert union.tolist() == [3, 4, 2, 0]


def test_intersect_and_union_ade_defaults():
    """ADE 常量接在默认值上：150 类、reduce_zero_label、忽略 255。"""
    raw = np.array([[0, 1, 2], [150, 150, 255]])  # 原始标注：0=其他（忽略），1..150 → 0..149
    pred = np.array([[5, 0, 1], [149, 3, 0]])
    inter, union = intersect_and_union(pred, raw)
    assert inter.shape == (ADE_NUM_CLASSES,)
    # 有效像素 4 个：(1→0,pred0) (2→1,pred1) (150→149,pred149) (150→149,pred3)
    assert inter[[0, 1, 149]].tolist() == [1, 1, 1] and inter.sum() == 3
    assert union[3] == 1 and union[149] == 2 and union.sum() == 5
    assert union[5] == 0  # 被忽略像素上的预测不计


def test_mean_iou_accumulates_over_dataset_and_nanmeans():
    K = 3
    a = intersect_and_union(np.array([[0, 0, 1]]), np.array([[0, 1, 1]]), K, reduce_zero=False)
    b = intersect_and_union(np.array([[1, 1]]), np.array([[1, 1]]), K, reduce_zero=False)
    assert mean_iou([a, b]) == pytest.approx((1 / 2 + 3 / 4) / 2)
    per_image = np.mean([mean_iou([a]), mean_iou([b])])
    assert mean_iou([a, b]) != pytest.approx(per_image)
    assert mean_iou([]) is None
    assert mean_iou([intersect_and_union(np.array([[5]]), np.array([[ADE_IGNORE_INDEX]]), K, reduce_zero=False)]) is None


def test_mean_iou_keeps_zero_iou_classes():
    """M4：并 > 0、交 = 0 的类计 IoU = 0，不剔除（剔除会抬高 mIoU）。"""
    K = 3
    # 类 2 只出现在预测里，类 1 只出现在标签里
    x = intersect_and_union(np.array([[0, 0, 2, 2]]), np.array([[0, 0, 1, 1]]), K, reduce_zero=False)
    assert x[0].tolist() == [2, 0, 0] and x[1].tolist() == [2, 2, 2]
    assert mean_iou([x]) == pytest.approx(1 / 3)
    assert miou_from_totals(np.array([2, 0, 0, 0]), np.array([2, 2, 2, 0])) == pytest.approx(1 / 3)
