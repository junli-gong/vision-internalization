"""表面法向指标：单位化后的角误差（度）。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# 角误差阈值（度），简报 §7.3 与 DEV-0001 任务书 §6 指定的三档
NORMAL_ANGLE_THRESHOLDS = (11.25, 22.5, 30.0)


def normal_angular_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """逐像素角误差（度），返回有效像素上的一维数组。

    pred、gt 形状 (H, W, 3)。两者先单位化，点积裁剪到 [-1, 1] 后取 arccos。
    GT 无效像素：范数为 0 或含非有限值；另可用 valid (H, W) 进一步掩掉。
    预测范数为 0 或含非有限值的像素记 90°（与任何单位向量的点积按 0 处理），不丢弃——丢弃会让坏预测变好看。
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape or p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError(f"pred {p.shape} 与 gt {g.shape} 必须同为 (H, W, 3)")
    g_norm = np.linalg.norm(g, axis=-1)
    mask = np.isfinite(g_norm) & (g_norm > 0)
    if valid is not None:
        v = np.asarray(valid).astype(bool)
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
