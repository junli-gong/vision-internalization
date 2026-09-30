import numpy as np
import pytest

from src.constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED
from src.metrics.seg import ciou
from src.stats import bootstrap_ci, is_significant, paired_bootstrap_ci, ratio_of_sums


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
    assert (ci.low, ci.high) == pytest.approx((lo, hi), abs=0)
    # 粗略合理性：接近正态近似 ±1.96 SE
    se = v.std(ddof=1) / np.sqrt(len(v))
    assert ci.high - ci.low == pytest.approx(2 * 1.96 * se, rel=0.2)
    # 固定种子可复现；换种子会变
    assert bootstrap_ci(v) == ci
    assert bootstrap_ci(v, seed=BOOTSTRAP_SEED + 1) != ci


def test_bootstrap_ratio_of_sums_for_ciou():
    stats = np.array([(3, 4), (1, 5), (7, 9), (0, 2)])
    ci = bootstrap_ci(stats, statistic=ratio_of_sums)
    assert ci.estimate == pytest.approx(ciou([tuple(s) for s in stats]))
    assert ci.low <= ci.estimate <= ci.high
    with pytest.raises(ValueError):
        bootstrap_ci(stats)  # 多列必须给 statistic


def test_paired_bootstrap_uses_pairing():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 1, 300)  # 样本间方差很大
    a = b + 0.01  # 但逐样本差恒为 0.01
    ci = paired_bootstrap_ci(a, b)
    assert (ci.low, ci.high) == pytest.approx((0.01, 0.01))
    assert not ci.contains(0.0)
    # 非配对（各自重采样）会覆盖 0 —— 说明配对是必要的
    assert bootstrap_ci(a).low < bootstrap_ci(b).high
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b[:-1])
    assert paired_bootstrap_ci([], []) is None


def test_is_significant_both_conditions():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 0.7, 100)
    # 1) 两条都满足
    r = is_significant([0.60, 0.62, 0.61], [0.50, 0.52, 0.51], base + 0.1, base)
    assert r.significant and r.exceeds_std_rule and r.ci_excludes_zero
    assert r.mean_diff == pytest.approx(0.1)
    assert r.pooled_std == pytest.approx(0.01)  # sqrt((0.01² + 0.01²)/2)，ddof=1
    # 2) seed 方差大：均值差 0.01 < 2 × 0.0711
    r = is_significant([0.60, 0.62, 0.61], [0.50, 0.70, 0.60], base + 0.1, base)
    assert r.pooled_std == pytest.approx(np.sqrt((0.01**2 + 0.1**2) / 2))
    assert not r.exceeds_std_rule and r.ci_excludes_zero and not r.significant
    # 3) seed 规则满足但逐样本无差：CI 含 0
    r = is_significant([0.60, 0.62, 0.61], [0.50, 0.52, 0.51], base, base)
    assert r.exceeds_std_rule and not r.ci_excludes_zero and not r.significant
    # 4) 负向差同样判定（绝对值）
    r = is_significant([0.50, 0.52, 0.51], [0.60, 0.62, 0.61], base, base + 0.1)
    assert r.significant and r.mean_diff == pytest.approx(-0.1)


def test_is_significant_needs_two_seeds():
    with pytest.raises(ValueError):
        is_significant([0.6], [0.5, 0.4], [1.0], [0.0])
