import numpy as np
import pytest

from src.constants import ADE_IGNORE_INDEX, BOUNDARY_IOU_DILATION_RATIO
from src.metrics.seg import (
    boundary_iou,
    ciou,
    giou,
    intersect_and_union,
    mask_inter_union,
    mask_to_boundary,
    mean_boundary_iou,
    mean_iou,
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
    assert mask_inter_union(p, g) == (1, 4)
    with pytest.raises(ValueError):
        mask_inter_union(p, g.T)


def test_seg_empty_returns_none():
    assert ciou([]) is None and giou([]) is None and mean_boundary_iou([]) is None
    assert ciou([(0, 0), (0, 0)]) is None  # 累计并为 0：未定义


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


SHAPES = [(35, 120), (61, 23), (7, 11), (240, 320), (3, 3)]  # (35,120)：0.02*对角线恰为 2.5，银行家舍入给 2


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


def test_boundary_dilation_uses_python_round():
    # 35×120：0.02*125 = 2.5 → round() = 2（不是 3）。全 1 mask 的边界带宽 = dilation
    m = np.ones((35, 120), np.uint8)
    b = mask_to_boundary(m)
    assert b[:, 60].sum() == 2 * 2  # 上下各 2 行
    assert b[17, :].sum() == 2 * 2


def test_boundary_iou_hand():
    # 10×12 图，dilation = round(0.02*sqrt(244)) = round(0.31) → 0 → 取 1：边界带宽 1
    gt = np.zeros((10, 12), bool)
    gt[2:6, 3:9] = True  # 4×6 方块，边界 = 周长像素 16
    pred = np.zeros((10, 12), bool)
    pred[2:6, 3:10] = True  # 右移扩一列：4×7，边界 18
    bg, bp = mask_to_boundary(gt), mask_to_boundary(pred)
    assert bg.sum() == 16 and bp.sum() == 18
    inter = (bg & bp).sum()  # 上下两行共享 6+6 中的前 6 列 → 2*6=12，左列内部 2 → 14
    assert inter == 14
    assert boundary_iou(pred, gt) == pytest.approx(14 / (16 + 18 - 14))
    assert boundary_iou(gt, gt) == 1.0
    assert boundary_iou(np.zeros((4, 5)), np.zeros((4, 5))) == 1.0  # 两边都空：约定 1.0（待裁）
    assert mean_boundary_iou([0.5, 1.0]) == 0.75


# ---------------------------------------------------------------- mIoU


def test_reduce_zero_label_matches_mmseg():
    lab = np.array([[0, 1, 2], [150, 255, 0]], np.uint8)
    out = reduce_zero_label(lab)
    assert out.tolist() == [[ADE_IGNORE_INDEX, 0, 1], [149, ADE_IGNORE_INDEX, ADE_IGNORE_INDEX]]
    # 与 mmseg 的 uint8 三行写法逐位一致
    ref = lab.copy()
    ref[ref == 0] = 255
    ref = ref - 1
    ref[ref == 254] = 255
    assert np.array_equal(out, ref.astype(np.int64))


def test_intersect_and_union_hand():
    K = 4
    label = np.array([[0, 0, 1, ADE_IGNORE_INDEX], [2, 2, 1, 1]])
    pred = np.array([[0, 1, 1, 3], [2, 0, 1, 7]])  # 7 越界：不计入任何类
    inter, union = intersect_and_union(pred, label, K)
    # 忽略像素 (0,3) 被去掉；有效像素 7 个
    assert inter.tolist() == [1, 2, 1, 0]
    assert union.tolist() == [3, 4, 2, 0]  # 类 0: pred{(0,0),(1,1)} ∪ label{(0,0),(0,1)} = 3


def test_mean_iou_accumulates_over_dataset_and_nanmeans():
    K = 3
    a = intersect_and_union(np.array([[0, 0, 1]]), np.array([[0, 1, 1]]), K)
    b = intersect_and_union(np.array([[1, 1]]), np.array([[1, 1]]), K)
    # 类 0: 交 1 并 2；类 1: 交 1+2 并 2+2；类 2: 并 0 → 不计
    assert mean_iou([a, b]) == pytest.approx((1 / 2 + 3 / 4) / 2)
    # 与逐图平均不同（逐图平均会给不同的数）
    per_image = np.mean([mean_iou([a]), mean_iou([b])])
    assert mean_iou([a, b]) != pytest.approx(per_image)
    assert mean_iou([]) is None
    assert mean_iou([intersect_and_union(np.array([[5]]), np.array([[ADE_IGNORE_INDEX]]), K)]) is None
