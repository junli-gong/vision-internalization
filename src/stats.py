"""统计：逐样本 bootstrap CI、配对 bootstrap、简报 §8.4 的"显著"判定，以及常用统计量。

默认设置（重采样次数、置信水平、种子）来自 src/constants.py。
输入里有 NaN/inf 时一律抛 ValueError 并报出位置：对齐与过滤是调用方的事，这里不静默丢弃。
统计量在某次重采样上得到 NaN/inf（如 ratio_of_sums 的分母为 0）时同样抛错，CI 此时无定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    SIGNIFICANCE_STD_MULTIPLIER,
)
from src.metrics.seg import miou_from_totals

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class CI:
    estimate: float  # 在原样本上的统计量（配对时为差值）
    low: float
    high: float
    level: float
    n_resamples: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


# ---------------------------------------------------------------------------
# 统计量
# ---------------------------------------------------------------------------


def mean_statistic(x: np.ndarray) -> float:
    """一维逐样本值的均值。用于 gIoU（逐样本 IoU）、逐图 AbsRel/δ1、准确率（0/1）。"""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"mean_statistic 要求一维输入，收到形状 {x.shape}")
    return float(x.mean())


def ratio_of_sums(x: np.ndarray) -> float:
    """(n, 2) → Σx[:,0] / Σx[:,1]。用于 cIoU（列为 交、并）。分母为 0 时返回 nan。"""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"ratio_of_sums 要求形状 (n, 2)，收到 {x.shape}")
    den = x[:, 1].sum()
    return float(x[:, 0].sum() / den) if den > 0 else float("nan")


def pooled_mean(x: np.ndarray) -> float:
    """(n, 2) 逐图 (误差和, 像素数) → 逐像素汇总的平均。用于法向平均角误差（与 normal_metrics 的 mean 一致）。"""
    return ratio_of_sums(x)


def miou_statistic(x: np.ndarray) -> float:
    """(n, 2, K) 逐图逐类 (交, 并) → mIoU，与 src.metrics.seg.mean_iou 相同（miou_from_totals）。

    所有类并集都为 0 时返回 nan。
    """
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"miou_statistic 要求形状 (n, 2, K)，收到 {x.shape}")
    v = miou_from_totals(x[:, 0].sum(axis=0), x[:, 1].sum(axis=0))
    return float("nan") if v is None else v


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def _check_finite(name: str, arr: np.ndarray) -> None:
    bad = ~np.isfinite(arr)
    if bad.any():
        where = np.argwhere(bad)[:5].tolist()
        raise ValueError(f"{name} 含 {int(bad.sum())} 个 NaN/inf，前几处下标 {where}；对齐与过滤由调用方负责")


def _check_boot(boot: np.ndarray) -> None:
    if not np.all(np.isfinite(boot)):
        raise ValueError(f"统计量在 {int((~np.isfinite(boot)).sum())} 次重采样上为 NaN/inf（如分母为 0），CI 无定义")


def _resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples, n) 的重采样下标，一次生成；逐次循环的参考实现用同一调用可逐位复现。"""
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def _percentile_ci(point: float, boot: np.ndarray, level: float, n_resamples: int) -> CI:
    alpha = (1.0 - level) / 2.0
    low, high = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return CI(float(point), float(low), float(high), level, n_resamples)


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: Statistic | None = None,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> CI | None:
    """逐样本 bootstrap 置信区间（percentile 法）。空集合返回 None。

    values 形状 (n, …)，按第 0 维（样本）有放回重采样。statistic 默认是一维均值；
    不能写成逐样本均值的指标用对应的统计量（ratio_of_sums、miou_statistic、pooled_mean）。
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 0 or v.shape[0] == 0:
        return None
    _check_finite("values", v)
    if statistic is None:
        if v.ndim != 1:
            raise ValueError("默认统计量是均值，values 须为一维；多列请给 statistic")
        idx = _resample_indices(v.shape[0], n_resamples, seed)
        point, boot = float(v.mean()), v[idx].mean(axis=1)
    else:
        idx = _resample_indices(v.shape[0], n_resamples, seed)
        point = float(statistic(v))
        boot = np.array([statistic(v[i]) for i in idx], dtype=np.float64)
    _check_finite("原样本上的统计量", np.asarray(point))
    _check_boot(boot)
    return _percentile_ci(point, boot, level, n_resamples)


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    statistic: Statistic | None = None,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> CI | None:
    """配对 bootstrap：a、b 按样本对齐，形状相同 (n, …)；同一组重采样下标同时作用于两者，
    返回 statistic(A*) − statistic(B*) 的 CI（estimate 为原样本上的差）。

    statistic 为 None 时是一维均值。空集合返回 None。
    """
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape != bv.shape or av.ndim == 0:
        raise ValueError(f"a {av.shape} 与 b {bv.shape} 须形状相同且按样本对齐")
    if av.shape[0] == 0:
        return None
    _check_finite("a", av)
    _check_finite("b", bv)
    idx = _resample_indices(av.shape[0], n_resamples, seed)
    if statistic is None:
        if av.ndim != 1:
            raise ValueError("默认统计量是均值，a、b 须为一维；多列请给 statistic")
        point = float(av.mean() - bv.mean())
        boot = av[idx].mean(axis=1) - bv[idx].mean(axis=1)
    else:
        point = float(statistic(av) - statistic(bv))
        boot = np.array([statistic(av[i]) - statistic(bv[i]) for i in idx], dtype=np.float64)
    _check_finite("原样本上的统计量差", np.asarray(point))
    _check_boot(boot)
    return _percentile_ci(point, boot, level, n_resamples)


# ---------------------------------------------------------------------------
# "显著"判定
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceResult:
    significant: bool  # 三条规则都满足
    significantly_better: bool  # A 显著优于 B（方向按 higher_is_better）
    significantly_worse: bool  # A 显著劣于 B
    higher_is_better: bool
    seed_metrics_a: tuple[float, ...]
    seed_metrics_b: tuple[float, ...]
    mean_diff: float  # mean(seed_metrics_a) − mean(seed_metrics_b)
    pooled_std: float
    exceeds_std_rule: bool  # 规则 1
    paired_ci: CI  # 规则 2 的 CI（estimate == mean_diff）
    ci_excludes_zero: bool  # 规则 2
    same_sign: bool  # 规则 3


def is_significant(
    per_sample_a: np.ndarray,
    per_sample_b: np.ndarray,
    *,
    statistic: Statistic,
    higher_is_better: bool,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> SignificanceResult:
    """简报 §8.4 "显著"的判定：两条规则检验同一个统计量，并且分方向。

    这是对简报措辞"差值超过合并标准差的 2 倍，且 bootstrap 95% 置信区间不含 0"的实现解释（D10(9)），
    G0 报告里请项目负责人确认。

    输入形状 (n_samples, n_seeds, …)：A、B 在同一批样本上按样本对齐（第 0 维），seed 数可以不同，
    每方至少 2 个 seed；尾部维度是 statistic 需要的（如 cIoU 用 (…, 2) 的 (交, 并)）。
    seed 级指标 m_s = statistic(x[:, s])，在函数内部对每个 seed 的全集计算。
    1. |mean(m_A) − mean(m_B)| > 2 × sqrt((s_A² + s_B²) / 2)，s 为 m 的样本标准差（ddof=1）。
    2. 按样本配对重采样：每次对各 seed 分别算 statistic 再取 seed 平均，A 减 B；差值的 95% CI 不含 0。
    3. 规则 1 的差值与规则 2 的 CI 同号。
    三条都满足才算显著；significantly_better / significantly_worse 按 higher_is_better 给出方向。
    """
    a = np.asarray(per_sample_a, dtype=np.float64)
    b = np.asarray(per_sample_b, dtype=np.float64)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(f"输入须为 (n_samples, n_seeds, …)，收到 {a.shape} 与 {b.shape}")
    if a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:]:
        raise ValueError(f"A {a.shape} 与 B {b.shape} 的样本数或尾部维度不一致")
    if a.shape[0] == 0:
        raise ValueError("样本为空")
    if a.shape[1] < 2 or b.shape[1] < 2:
        raise ValueError("每种方法至少需要 2 个 seed 才能算 ddof=1 的标准差")
    _check_finite("per_sample_a", a)
    _check_finite("per_sample_b", b)

    seed_a = np.array([statistic(a[:, s]) for s in range(a.shape[1])], dtype=np.float64)
    seed_b = np.array([statistic(b[:, s]) for s in range(b.shape[1])], dtype=np.float64)
    _check_finite("seed 级指标 A", seed_a)
    _check_finite("seed 级指标 B", seed_b)
    mean_diff = float(seed_a.mean() - seed_b.mean())
    pooled = float(np.sqrt((seed_a.var(ddof=1) + seed_b.var(ddof=1)) / 2.0))
    exceeds = abs(mean_diff) > SIGNIFICANCE_STD_MULTIPLIER * pooled

    idx = _resample_indices(a.shape[0], n_resamples, seed)
    boot = np.array(
        [
            np.mean([statistic(a[i, s]) for s in range(a.shape[1])])
            - np.mean([statistic(b[i, s]) for s in range(b.shape[1])])
            for i in idx
        ],
        dtype=np.float64,
    )
    _check_boot(boot)
    ci = _percentile_ci(mean_diff, boot, level, n_resamples)
    excludes = not ci.contains(0.0)
    sign = np.sign(mean_diff)
    same_sign = bool(sign != 0 and np.sign(ci.low) == sign and np.sign(ci.high) == sign)

    significant = bool(exceeds and excludes and same_sign)
    a_ahead = mean_diff > 0 if higher_is_better else mean_diff < 0
    return SignificanceResult(
        significant=significant,
        significantly_better=significant and a_ahead,
        significantly_worse=significant and not a_ahead,
        higher_is_better=higher_is_better,
        seed_metrics_a=tuple(float(v) for v in seed_a),
        seed_metrics_b=tuple(float(v) for v in seed_b),
        mean_diff=mean_diff,
        pooled_std=pooled,
        exceeds_std_rule=bool(exceeds),
        paired_ci=ci,
        ci_excludes_zero=excludes,
        same_sign=same_sign,
    )
