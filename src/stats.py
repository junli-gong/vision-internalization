"""统计：逐样本 bootstrap CI、配对 bootstrap、简报 §8.4 的"显著"判定。

默认设置（重采样次数、置信水平、种子）来自 src/constants.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    SIGNIFICANCE_STD_MULTIPLIER,
)


@dataclass(frozen=True)
class CI:
    estimate: float  # 在原样本上的统计量
    low: float
    high: float
    level: float
    n_resamples: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def _resample_stats(
    values: np.ndarray, statistic: Callable[[np.ndarray], float] | None, n_resamples: int, seed: int
) -> np.ndarray:
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    if statistic is None:
        return values[idx].mean(axis=1)
    return np.array([statistic(values[i]) for i in idx], dtype=np.float64)


def _percentile_ci(point: float, boot: np.ndarray, level: float, n_resamples: int) -> CI:
    alpha = (1.0 - level) / 2.0
    low, high = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return CI(float(point), float(low), float(high), level, n_resamples)


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> CI | None:
    """逐样本 bootstrap 置信区间（percentile 法）。空集合返回 None。

    values 形状 (n,) 或 (n, k)，按第 0 维（样本）有放回重采样。statistic 默认是均值（要求一维）；
    不能写成逐样本均值的指标（如 cIoU = Σ交/Σ并）传 values=(n, 2) 与 statistic=ratio_of_sums。
    """
    v = np.asarray(values, dtype=np.float64)
    if v.shape[0] == 0:
        return None
    if statistic is None and v.ndim != 1:
        raise ValueError("默认统计量是均值，values 须为一维；多列请给 statistic")
    point = float(v.mean()) if statistic is None else float(statistic(v))
    return _percentile_ci(point, _resample_stats(v, statistic, n_resamples, seed), level, n_resamples)


def ratio_of_sums(v: np.ndarray) -> float:
    """(n, 2) 数组 → Σv[:,0] / Σv[:,1]。用于 cIoU 的 bootstrap（列为 交、并）。并为 0 时返回 nan。"""
    den = v[:, 1].sum()
    return float(v[:, 0].sum() / den) if den > 0 else float("nan")


def paired_bootstrap_ci(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> CI | None:
    """配对 bootstrap：两个系统在同一批样本上的逐样本值 a_i、b_i，给出 mean(a − b) 的 CI。

    两者必须等长且按同一样本顺序排列（同一组重采样下标同时作用于两者）。空集合返回 None。
    """
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape != bv.shape or av.ndim != 1:
        raise ValueError(f"a {av.shape} 与 b {bv.shape} 须为等长一维")
    return bootstrap_ci(av - bv, n_resamples=n_resamples, level=level, seed=seed)


@dataclass(frozen=True)
class SignificanceResult:
    significant: bool
    mean_diff: float  # mean(seed_a) − mean(seed_b)
    pooled_std: float
    exceeds_std_rule: bool
    paired_ci: CI
    ci_excludes_zero: bool


def is_significant(
    seed_metrics_a: Sequence[float],
    seed_metrics_b: Sequence[float],
    per_sample_a: Sequence[float],
    per_sample_b: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> SignificanceResult:
    """简报 §8.4 "显著"的判定。

    这是对简报措辞"差值超过合并标准差的 2 倍，且 bootstrap 95% 置信区间不含 0"的实现解释，待 G0 确认：
    1. |mean(seed_a) − mean(seed_b)| > 2 × pooled_std，pooled_std = sqrt((s_A² + s_B²) / 2)，
       s 为逐 seed 指标的样本标准差（ddof=1）；每种方法至少 2 个 seed。
    2. 配对 bootstrap（paired_bootstrap_ci）在逐样本值上的差值 CI 不含 0。
       逐样本值怎么取（某个 seed 的、还是各 seed 平均后的）由调用方决定；建议各 seed 平均后传入。
    两条都满足才算显著。
    """
    sa = np.asarray(seed_metrics_a, dtype=np.float64)
    sb = np.asarray(seed_metrics_b, dtype=np.float64)
    if sa.size < 2 or sb.size < 2:
        raise ValueError("每种方法至少需要 2 个 seed 才能算 ddof=1 的标准差")
    mean_diff = float(sa.mean() - sb.mean())
    pooled = float(np.sqrt((sa.var(ddof=1) + sb.var(ddof=1)) / 2.0))
    exceeds = abs(mean_diff) > SIGNIFICANCE_STD_MULTIPLIER * pooled
    ci = paired_bootstrap_ci(per_sample_a, per_sample_b, n_resamples=n_resamples, level=level, seed=seed)
    if ci is None:
        raise ValueError("逐样本值为空，无法做配对 bootstrap")
    excludes = not ci.contains(0.0)
    return SignificanceResult(exceeds and excludes, mean_diff, pooled, exceeds, ci, excludes)
