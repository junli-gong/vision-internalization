"""深度指标（NYUv2）：AbsRel、δ1，度量深度与尺度-平移对齐两种。

有效区域与裁剪区间来自 src/constants.py。
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from src.constants import (
    NYU_DEPTH_MAX,
    NYU_DEPTH_MAX_INCLUSIVE,
    NYU_DEPTH_MIN,
    NYU_EIGEN_CROP,
    NYU_EVAL_HW,
)

DELTA1_THRESHOLD = 1.25  # δ1 定义本身的常数（max(gt/pred, pred/gt) < 1.25），不是跨模块约定


def nyu_eval_mask(gt: np.ndarray) -> np.ndarray:
    """NYUv2 评测区域：Eigen crop ∧ GT 深度有效。

    Eigen crop 对齐 cleinc/bts pytorch/bts_eval.py 里 eigen_crop 分支的 eval_mask（区间见
    constants.NYU_EIGEN_CROP），只定义在 constants.NYU_EVAL_HW 上，其他尺寸抛 ValueError。有效区间见 constants（NYU_DEPTH_MIN 开，NYU_DEPTH_MAX 由 NYU_DEPTH_MAX_INCLUSIVE 决定）。
    """
    g = np.asarray(gt, dtype=np.float64)
    if g.shape != NYU_EVAL_HW:
        raise ValueError(f"Eigen crop 只定义在 {NYU_EVAL_HW} 上，收到 {g.shape}")
    r0, r1, c0, c1 = NYU_EIGEN_CROP
    crop = np.zeros(g.shape, dtype=bool)
    crop[r0:r1, c0:c1] = True
    upper = g <= NYU_DEPTH_MAX if NYU_DEPTH_MAX_INCLUSIVE else g < NYU_DEPTH_MAX
    valid = np.isfinite(g) & (g > NYU_DEPTH_MIN) & upper
    return crop & valid


def clip_pred(pred: np.ndarray) -> np.ndarray:
    """预测裁剪到 [NYU_DEPTH_MIN, NYU_DEPTH_MAX]；inf → 上限，NaN → 下限。

    对齐 cleinc/bts pytorch/bts_eval.py 中对 pred_depth 的四行处理。
    """
    p = np.array(pred, dtype=np.float64, copy=True)
    p[np.isposinf(p)] = NYU_DEPTH_MAX
    p[np.isneginf(p)] = NYU_DEPTH_MIN
    p[np.isnan(p)] = NYU_DEPTH_MIN
    return np.clip(p, NYU_DEPTH_MIN, NYU_DEPTH_MAX)


def fit_scale_shift(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """在给定像素上做深度空间最小二乘：argmin_{s,t} Σ (s·pred + t − gt)²。输入是一维有效像素。"""
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    a = np.stack([p, np.ones_like(p)], axis=1)
    (s, t), *_ = np.linalg.lstsq(a, g, rcond=None)
    return float(s), float(t)


def depth_sample_metrics(
    pred: np.ndarray, gt: np.ndarray, *, align: Literal["none", "scale_shift"]
) -> dict[str, float] | None:
    """单张图的 AbsRel 与 δ1。有效像素为空时返回 None。

    指标公式对齐 cleinc/bts pytorch/bts_eval.py compute_errors：
    abs_rel = mean(|gt − pred| / gt)，d1 = mean(max(gt/pred, pred/gt) < 1.25)。
    align="none"：度量深度，预测先裁剪（clip_pred）再算。
    align="scale_shift"：在有效像素上用未裁剪的预测拟合 s·pred + t（fit_scale_shift），
    对齐后的预测再裁剪、计算。预测含非有限值时在拟合前按 clip_pred 的规则替换。
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"pred 形状 {p.shape} 与 gt 形状 {g.shape} 不一致")
    mask = nyu_eval_mask(g)
    if not np.any(mask):
        return None
    pv, gv = p[mask], g[mask]
    if align == "none":
        pv = clip_pred(pv)
        scale, shift = 1.0, 0.0
    elif align == "scale_shift":
        finite = np.where(np.isfinite(pv), pv, clip_pred(pv))
        scale, shift = fit_scale_shift(finite, gv)
        pv = clip_pred(scale * finite + shift)
    else:
        raise ValueError(f"未知 align={align!r}")
    thresh = np.maximum(gv / pv, pv / gv)
    return {
        "abs_rel": float(np.mean(np.abs(gv - pv) / gv)),
        "d1": float(np.mean(thresh < DELTA1_THRESHOLD)),
        "scale": scale,
        "shift": shift,
        "n_valid": int(mask.sum()),
    }


def depth_dataset_metrics(samples: Sequence[dict[str, float] | None]) -> dict[str, float] | None:
    """数据集级：逐图指标的平均（与 BTS 的逐图累加后取平均一致）。

    跳过有效像素为空的图（None），并在结果里报告 n_images 与 n_skipped。全部为空或空集合返回 None。
    """
    used = [s for s in samples if s is not None]
    if not used:
        return None
    return {
        "abs_rel": float(np.mean([s["abs_rel"] for s in used])),
        "d1": float(np.mean([s["d1"] for s in used])),
        "n_images": len(used),
        "n_skipped": len(samples) - len(used),
    }
