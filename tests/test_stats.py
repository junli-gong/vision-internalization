import numpy as np
import pytest

from src.constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED
from src.metrics.normal import normal_metrics
from src.metrics.seg import ciou, intersect_and_union, mean_iou
from src.stats import (
    bootstrap_ci,
    is_significant,
    mean_statistic,
    miou_statistic,
    paired_bootstrap_ci,
    pooled_mean,
    ratio_of_sums,
)

# ---------------------------------------------------------------- bootstrap_ci


def test_bootstrap_defaults_and_constant_input():
    ci = bootstrap_ci([0.3] * 17)
    assert (ci.estimate, ci.low, ci.high) == pytest.approx((0.3, 0.3, 0.3))
    assert ci.n_resamples == BOOTSTRAP_N_RESAMPLES and ci.level == BOOTSTRAP_CI_LEVEL
    assert bootstrap_ci([]) is None


def test_bootstrap_percentile_method_matches_manual():
    v = np.random.default_rng(5).normal(0.5, 0.2, 200)
    ci = bootstrap_ci(v)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(v), size=(BOOTSTRAP_N_RESAMPLES, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    assert (ci.low, ci.high) == (lo, hi)
    se = v.std(ddof=1) / np.sqrt(len(v))
    assert ci.high - ci.low == pytest.approx(2 * 1.96 * se, rel=0.2)
    assert bootstrap_ci(v) == ci
    assert bootstrap_ci(v, seed=BOOTSTRAP_SEED + 1) != ci


def test_ratio_bootstrap_matches_loop_reference_bitwise():
    """M4：ratio bootstrap 的 CI 宽度 > 0，且与逐次循环的参考实现在同一种子下逐位相同。"""
    rng = np.random.default_rng(2)
    union = rng.integers(50, 5000, 120)
    inter = (union * rng.uniform(0.2, 0.9, 120)).astype(int)
    stats = np.stack([inter, union], axis=1)
    ci = bootstrap_ci(stats, statistic=ratio_of_sums)
    assert ci.estimate == ciou([tuple(s) for s in stats])
    assert ci.high - ci.low > 0.01
    ref_rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = ref_rng.integers(0, len(stats), size=(BOOTSTRAP_N_RESAMPLES, len(stats)))
    boot = []
    for row in idx:
        s = stats[row].astype(np.float64)
        boot.append(s[:, 0].sum() / s[:, 1].sum())
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5])
    assert (ci.low, ci.high) == (lo, hi)
    with pytest.raises(ValueError):
        bootstrap_ci(stats)  # 多列必须给 statistic


# ---------------------------------------------------------------- S1：NaN/inf


def test_nan_inf_inputs_raise_with_position():
    v = np.random.default_rng(0).uniform(0, 1, 40)
    v[17] = np.nan
    with pytest.raises(ValueError, match=r"\[\[17\]\]"):
        bootstrap_ci(v)
    b = np.random.default_rng(1).uniform(0, 1, 40)
    with pytest.raises(ValueError, match=r"a 含 1 个 NaN/inf，前几处下标 \[\[17\]\]"):
        paired_bootstrap_ci(v, b)
    with pytest.raises(ValueError, match=r"b 含 1 个 NaN/inf，前几处下标 \[\[3\]\]"):
        paired_bootstrap_ci(b, np.where(np.arange(40) == 3, np.inf, b))
    a2 = np.stack([b, b], axis=1)
    a2[17, 1] = np.nan
    with pytest.raises(ValueError, match=r"\[\[17, 1\]\]"):
        is_significant(a2, np.stack([b, b], axis=1), statistic=mean_statistic, higher_is_better=True)


def test_statistic_nan_on_resample_raises():
    # 只有一个样本的并 > 0：很多重采样里分母为 0，统计量为 nan → CI 无定义，报错而不是给 nan 区间
    stats = np.array([[0, 0]] * 30 + [[5, 10]])
    with pytest.raises(ValueError, match="重采样"):
        bootstrap_ci(stats, statistic=ratio_of_sums)


# ---------------------------------------------------------------- paired_bootstrap_ci


def test_paired_bootstrap_uses_pairing():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 1, 300)
    a = b + 0.01
    ci = paired_bootstrap_ci(a, b)
    assert (ci.low, ci.high) == pytest.approx((0.01, 0.01))
    assert not ci.contains(0.0)
    assert bootstrap_ci(a).low < bootstrap_ci(b).high  # 非配对会覆盖 0
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b[:-1])
    assert paired_bootstrap_ci([], []) is None


def test_paired_bootstrap_with_ratio_statistic():
    rng = np.random.default_rng(3)
    union = rng.integers(100, 3000, 80)
    a = np.stack([(union * 0.7).astype(int), union], axis=1)
    b = np.stack([(union * 0.5).astype(int), union], axis=1)
    ci = paired_bootstrap_ci(a, b, statistic=ratio_of_sums)
    assert ci.estimate == pytest.approx(ratio_of_sums(a) - ratio_of_sums(b))
    assert 0.19 < ci.low <= ci.high < 0.21
    # 与逐次循环参考逐位相同
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 80, size=(BOOTSTRAP_N_RESAMPLES, 80))
    ref = np.array([ratio_of_sums(a[i].astype(float)) - ratio_of_sums(b[i].astype(float)) for i in idx])
    assert (ci.low, ci.high) == tuple(np.percentile(ref, [2.5, 97.5]))


# ---------------------------------------------------------------- 统计量


def test_statistic_helpers_match_metrics():
    rng = np.random.default_rng(4)
    K = 5
    per_img = []
    for _ in range(12):
        lab = rng.integers(0, K, (6, 9))
        pred = np.where(rng.random((6, 9)) < 0.6, lab, rng.integers(0, K, (6, 9)))
        per_img.append(intersect_and_union(pred, lab, K, reduce_zero=False))
    x = np.array([[i, u] for i, u in per_img])  # (n, 2, K)
    assert miou_statistic(x) == mean_iou(per_img)
    errs = [rng.uniform(0, 40, n) for n in (5, 50, 7)]
    y = np.array([[e.sum(), e.size] for e in errs])
    assert pooled_mean(y) == pytest.approx(normal_metrics(errs)["mean"])
    assert mean_statistic(np.array([0.2, 0.4])) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        mean_statistic(np.ones((3, 2)))


# ---------------------------------------------------------------- S2：is_significant


def _ciou_systems():
    """审查的 S2 复现：A 在大目标上好、B 在小目标上好。cIoU（按面积加权）A 高，gIoU（逐样本平均）B 高。"""
    n = 400
    big = np.arange(n) < n // 2
    union = np.where(big, 20000, 200)

    def system(big_iou, small_iou, seeds):
        out = []
        for s in seeds:
            r = np.random.default_rng(s)
            iou = np.clip(np.where(big, big_iou, small_iou) + r.normal(0, 0.02, n), 0, 1)
            out.append(np.stack([np.round(iou * union), union], axis=1))
        return np.stack(out, axis=1)  # (n, n_seeds, 2)

    return system(0.80, 0.30, (1, 2, 3)), system(0.70, 0.50, (11, 12, 13))


def test_s2_same_statistic_both_rules_cIoU():
    A, B = _ciou_systems()
    r = is_significant(A, B, statistic=ratio_of_sums, higher_is_better=True, n_resamples=300)
    assert r.mean_diff > 0.05  # cIoU：A 高
    assert r.paired_ci.estimate == r.mean_diff
    assert r.paired_ci.low > 0  # 规则 2 的 CI 在同一统计量上，方向与规则 1 一致
    assert r.significant and r.significantly_better and not r.significantly_worse
    assert r.seed_metrics_a == tuple(ratio_of_sums(A[:, s]) for s in range(3))
    # 规则 2 与逐次循环参考逐位相同：同一组样本下标同时作用于 A、B，各 seed 分别算统计量再平均
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, A.shape[0], size=(300, A.shape[0]))
    ref = [np.mean([ratio_of_sums(A[i, s]) for s in range(3)]) - np.mean([ratio_of_sums(B[i, s]) for s in range(3)]) for i in idx]
    assert (r.paired_ci.low, r.paired_ci.high) == tuple(np.percentile(ref, [2.5, 97.5]))
    # 同一数据按 gIoU（逐样本 IoU 的均值）看，方向相反：B 显著更好
    giou_a, giou_b = A[..., 0] / A[..., 1], B[..., 0] / B[..., 1]
    r2 = is_significant(giou_a, giou_b, statistic=mean_statistic, higher_is_better=True, n_resamples=300)
    assert r2.mean_diff < 0 and r2.significantly_worse and not r2.significantly_better


def test_rule3_opposite_signs_not_significant():
    """构造一个在全集与重采样上符号相反的统计量，单独检验规则 3：
    输入有序（原样本）时取 +均值，被重采样打乱后取 −均值。"""

    def odd_stat(x):
        return float(x.mean()) if np.all(np.diff(x) >= 0) else -float(x.mean())

    n = 60
    a = np.stack([np.linspace(0.5, 0.6, n), np.linspace(0.5, 0.61, n)], axis=1)
    b = np.zeros((n, 2))
    r = is_significant(a, b, statistic=odd_stat, higher_is_better=True, n_resamples=200)
    assert r.exceeds_std_rule and r.ci_excludes_zero  # 规则 1、2 各自满足
    assert r.mean_diff > 0 and r.paired_ci.high < 0  # 但方向相反
    assert not r.same_sign and not r.significant and not r.significantly_better and not r.significantly_worse


def test_lower_is_better_absrel():
    rng = np.random.default_rng(7)
    n = 150
    base = rng.uniform(0.08, 0.2, n)
    a = np.stack([base * 0.8 + rng.normal(0, 0.002, n) for _ in range(3)], axis=1)  # A 的 AbsRel 更低
    b = np.stack([base + rng.normal(0, 0.002, n) for _ in range(3)], axis=1)
    r = is_significant(a, b, statistic=mean_statistic, higher_is_better=False, n_resamples=300)
    assert r.mean_diff < 0 and r.significant
    assert r.significantly_better and not r.significantly_worse
    r_rev = is_significant(b, a, statistic=mean_statistic, higher_is_better=False, n_resamples=300)
    assert r_rev.significantly_worse and not r_rev.significantly_better


def test_std_rule_and_ci_rule_each_required():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 0.7, 100)
    # seed 方差大（B 的一个 seed 整体偏高）：规则 1 不满足
    a = np.stack([base + 0.01] * 3, axis=1)
    b = np.stack([base, base + 0.1, base - 0.1], axis=1)
    r = is_significant(a, b, statistic=mean_statistic, higher_is_better=True, n_resamples=300)
    assert r.pooled_std == pytest.approx(np.sqrt((0 + 0.1**2) / 2))
    assert not r.exceeds_std_rule and not r.significant
    # 两方完全一样：CI 含 0
    same = np.stack([base, base + 1e-3], axis=1)
    r = is_significant(same, same, statistic=mean_statistic, higher_is_better=True, n_resamples=100)
    assert not r.ci_excludes_zero and not r.significant


def test_is_significant_input_checks():
    x = np.zeros((10, 1))
    with pytest.raises(ValueError, match="2 个 seed"):
        is_significant(x, np.zeros((10, 2)), statistic=mean_statistic, higher_is_better=True)
    with pytest.raises(ValueError):
        is_significant(np.zeros((10, 2)), np.zeros((9, 2)), statistic=mean_statistic, higher_is_better=True)
    with pytest.raises(ValueError):
        is_significant(np.zeros(10), np.zeros(10), statistic=mean_statistic, higher_is_better=True)
    with pytest.raises(TypeError):
        is_significant(np.zeros((10, 2)), np.zeros((10, 2)), statistic=mean_statistic)  # 方向必填
