"""表面法向指标：单位化后的角误差（度）。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# 角误差阈值（度），简报 §7.3 与 DEV-0001 任务书 §6 指定的三档
NORMAL_ANGLE_THRESHOLDS = (11.25, 22.5, 30.0)


def normal_angular_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """逐像素角误差（度），返回有效像素上的一维数组。

    pred、gt 形状 (H, W, 3)。两者先单位化，点积裁剪到 [-1, 1] 后取 arccos。
    valid (H, W) 必填：数据集的无效像素约定各不相同（如 uint8 映射到 [-1,1] 后无效像素成了 (-1,-1,-1)），
    由调用方按数据集显式给出，这里不推断。GT 范数为 0 或含非有限值的像素另外叠加掩掉。
    预测范数为 0 或含非有限值的像素记 90°（与任何单位向量的点积按 0 处理），不丢弃——丢弃会让坏预测变好看。
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape or p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError(f"pred {p.shape} 与 gt {g.shape} 必须同为 (H, W, 3)")
    g_norm = np.linalg.norm(g, axis=-1)
    mask = np.isfinite(g_norm) & (g_norm > 0)
    v = np.asarray(valid)
    if v.dtype != np.bool_:
        raise ValueError(f"valid 必须是 bool 数组，收到 dtype {v.dtype}")
    if v.shape != p.shape[:2]:
        raise ValueError(f"valid 形状 {v.shape} 与 {p.shape[:2]} 不一致")
    mask &= v
    pv, gv = p[mask], g[mask] / g_norm[mask][:, None]
    p_norm = np.linalg.norm(pv, axis=-1)
    ok = np.isfinite(p_norm) & (p_norm > 0)
    cos = np.zeros(len(pv))
    cos[ok] = np.sum(pv[ok] / p_norm[ok][:, None] * gv[ok], axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def normal_metrics(errors: Sequence[np.ndarray]) -> dict[str, float] | None:
    """数据集级：所有图的有效像素角误差拼在一起统计（逐像素汇总，不是逐图平均）。

    对齐 baegwangbin/surface_normal_uncertainty utils/utils.py compute_normal_errors：
    mean = np.average、median = np.median，阈值比例 = 误差 < 阈值 的像素占比（这里给 0–1 比例，不乘 100）。
    空集合（没有任何有效像素）返回 None。
    """
    arrs = [np.asarray(e, dtype=np.float64).reshape(-1) for e in errors]
    allerr = np.concatenate(arrs) if arrs else np.zeros(0)
    if allerr.size == 0:
        return None
    out = {"mean": float(np.mean(allerr)), "median": float(np.median(allerr)), "n_pixels": int(allerr.size)}
    for t in NORMAL_ANGLE_THRESHOLDS:
        out[f"within_{t:g}"] = float(np.mean(allerr < t))
    return out


# ---------------------------------------------------------------------------
# 逐图汇总（DEV-0023，E2 探针）：评测集全部像素的误差放不进内存时，逐图记可相加的汇总，再算数据集级指标与 bootstrap
# ---------------------------------------------------------------------------

# 逐图直方图的箱宽（度），只用于数据集级中位角误差。箱内线性插值，与 np.median 之差 ≤ 一个箱宽 + 两个中间值之差的一半
# （像素数为偶数时 np.median 取两个中间值的平均；评测集有上亿个像素，后一项可以忽略；tests/test_e2_metrics.py 钉住）。
# mean 与 within_* 不经过直方图，逐像素精确。
NORMAL_HIST_BIN_DEG = 0.1


def normal_image_summary(errors: np.ndarray, bin_deg: float = NORMAL_HIST_BIN_DEG) -> dict:
    """一张图的角误差（normal_angular_errors 的输出）→ 可逐图相加的汇总：

    sum（误差和，度）、n（像素数）、within（各阈值 NORMAL_ANGLE_THRESHOLDS 下 误差 < 阈值 的像素数，与 normal_metrics 同一个比较）、
    hist（[0, 180] 上箱宽 bin_deg 的计数，180° 落在最后一箱）。
    """
    e = np.asarray(errors, dtype=np.float64).reshape(-1)
    n_bins = int(round(180.0 / bin_deg))
    idx = np.clip((e / bin_deg).astype(np.int64), 0, n_bins - 1)
    return {
        "sum": float(e.sum()),
        "n": int(e.size),
        "within": np.array([int(np.count_nonzero(e < t)) for t in NORMAL_ANGLE_THRESHOLDS], dtype=np.int64),
        "hist": np.bincount(idx, minlength=n_bins).astype(np.int64),
    }


def hist_median(hist: np.ndarray, bin_deg: float = NORMAL_HIST_BIN_DEG) -> float | None:
    """由直方图求中位数：取累计计数首次达到 n/2 的那一箱，箱内按计数线性插值。空直方图返回 None。"""
    h = np.asarray(hist, dtype=np.float64)
    n = float(h.sum())
    if n <= 0:
        return None
    cum = np.cumsum(h)
    k = int(np.searchsorted(cum, n / 2.0, side="left"))
    before = float(cum[k - 1]) if k > 0 else 0.0
    frac = (n / 2.0 - before) / h[k] if h[k] > 0 else 0.0
    return float((k + frac) * bin_deg)


def normal_metrics_from_summaries(summaries: Sequence[dict], bin_deg: float = NORMAL_HIST_BIN_DEG) -> dict[str, float] | None:
    """数据集级法向指标（逐像素汇总，与 normal_metrics 同口径）：mean、within_* 与把全部误差拼在一起算的结果相同（浮点求和顺序
    不同，差在 1e-12 量级）；median 由合计直方图插值（误差界见 NORMAL_HIST_BIN_DEG）。没有任何有效像素返回 None。"""
    n = int(sum(int(s["n"]) for s in summaries))
    if n == 0:
        return None
    total = float(sum(float(s["sum"]) for s in summaries))
    within = np.sum([np.asarray(s["within"]) for s in summaries], axis=0)
    hist = np.sum([np.asarray(s["hist"]) for s in summaries], axis=0)
    out = {"mean": total / n, "median": hist_median(hist, bin_deg), "n_pixels": n}
    for t, w in zip(NORMAL_ANGLE_THRESHOLDS, within):
        out[f"within_{t:g}"] = float(w) / n
    return out
