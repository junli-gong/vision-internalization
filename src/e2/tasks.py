"""E2 的四个任务：GT、损失、逐样本统计与数据集级指标（带 bootstrap CI）（DEV-0023；configs/e2_probes.json 的 tasks）。

一律在 GT 分辨率上：探针输出 [P, C, h, w]（P 个探针一块）先双线性上采样（align_corners=False）到 GT 的 (H, W)，再算损失或指标。
指标全部复用 src/metrics（DEV-0023 补了 depth_sample_rmse、pixel_correct_total / pixel_accuracy、法向的逐图汇总）。

- depth：探针输出 log 深度（米）。损失 SILog（cleinc/bts silog_loss：sqrt(mean(d²) − λ·mean(d)²)·scale），只在 GT 有效像素上
  （NYU_DEPTH_MIN < GT，GT < / ≤ NYU_DEPTH_MAX 按 NYU_DEPTH_MAX_INCLUSIVE；与评测同一组常量，不含 Eigen crop）。
  指标：depth_sample_metrics（AbsRel、δ1，度量与尺度-平移对齐两种）+ depth_sample_rmse，Eigen crop，逐图平均（BTS）。
- semseg：交叉熵，标签先 reduce_zero_label（0 → ADE_IGNORE_INDEX 忽略，其余减 1；ADE_REDUCE_ZERO_LABEL）。指标：mIoU（逐类全集累计，
  intersect_and_union / miou_from_totals）、像素准确率（pixel_correct_total / pixel_accuracy）。
- normals：预测逐像素单位化，损失 1 − cos 在有效掩码上的平均。指标：normal_angular_errors → normal_image_summary（逐图可加），
  数据集级 normal_metrics_from_summaries（mean、within_* 逐像素精确，median 由 0.1° 直方图插值）。
- refseg：BCE（logits）+ Dice（sigmoid 概率）。指标：预测 = logit > 0，mask_inter_union → cIoU（ciou）、gIoU（giou）。

bootstrap（src/stats.bootstrap_ci，按样本重采样，种子 BOOTSTRAP_SEED）：逐图均值用均值；mIoU 用 miou_statistic；cIoU、像素准确率、
法向 mean 与 within_* 用 ratio_of_sums；法向 median 用直方图合计的中位数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import (
    ADE_IGNORE_INDEX,
    ADE_NUM_CLASSES,
    ADE_REDUCE_ZERO_LABEL,
    BOOTSTRAP_SEED,
    NYU_DEPTH_MAX,
    NYU_DEPTH_MAX_INCLUSIVE,
    NYU_DEPTH_MIN,
)
from src.data.records import Sample
from src.data.train_loaders import read_ade_label, read_depth_m, read_normals
from src.e1.scoring import gt_mask
from src.metrics.depth import depth_dataset_metrics, depth_sample_metrics, depth_sample_rmse
from src.metrics.normal import (
    NORMAL_ANGLE_THRESHOLDS,
    NORMAL_HIST_BIN_DEG,
    hist_median,
    normal_angular_errors,
    normal_image_summary,
    normal_metrics_from_summaries,
)
from src.metrics.seg import (
    ciou,
    giou,
    intersect_and_union,
    mask_inter_union,
    miou_from_totals,
    pixel_accuracy,
    pixel_correct_total,
    reduce_zero_label,
    sample_iou,
)
from src.stats import CI, bootstrap_ci, mean_statistic, miou_statistic, ratio_of_sums


def upsample(out: torch.Tensor, hw: tuple[int, int], spec: Mapping[str, Any]) -> torch.Tensor:
    """[P, C, h, w] → [P, C, H, W]（configs/e2_probes.json 的 upsample：bilinear、align_corners=False）。"""
    if spec["mode"] != "bilinear":
        raise ValueError(f"只实现了双线性上采样，配置是 {spec['mode']!r}")
    return F.interpolate(out, size=(int(hw[0]), int(hw[1])), mode="bilinear", align_corners=bool(spec["align_corners"]))


@dataclass
class Target:
    hw: tuple[int, int]
    t: dict[str, torch.Tensor]
    np: dict[str, Any]
    n_valid: int


def _ci(c: CI | None) -> dict[str, float] | None:
    return None if c is None else {"low": c.low, "high": c.high, "level": c.level, "n_resamples": c.n_resamples}


def depth_train_valid(gt: np.ndarray) -> np.ndarray:
    """训练损失用的有效像素：与 nyu_eval_mask 同一组常量的有效区间，不含 Eigen crop。"""
    g = np.asarray(gt, dtype=np.float64)
    upper = g <= NYU_DEPTH_MAX if NYU_DEPTH_MAX_INCLUSIVE else g < NYU_DEPTH_MAX
    return np.isfinite(g) & (g > NYU_DEPTH_MIN) & upper


class E2Task:
    name: str = ""
    headline: tuple[tuple[str, str], ...] = ()  # (指标名, ↑/↓)

    def __init__(self, task_cfg: Mapping[str, Any]) -> None:
        self.cfg = task_cfg
        self.out_channels = int(task_cfg["out_channels"])
        self.loss_cfg = dict(task_cfg["loss"])

    def target(self, s: Sample, device: Any) -> Target:  # pragma: no cover - 子类实现
        raise NotImplementedError

    def loss(self, up: torch.Tensor, tgt: Target) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def sample_stats(self, up: torch.Tensor, tgt: Target) -> list[Any]:  # pragma: no cover
        raise NotImplementedError

    def aggregate(self, stats: Sequence[Any], *, n_resamples: int, seed: int = BOOTSTRAP_SEED, ci: bool = True) -> dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    def pack(self, stats: Sequence[Any]) -> dict[str, np.ndarray]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 深度
# ---------------------------------------------------------------------------


class DepthTask(E2Task):
    name = "depth"
    headline = (("abs_rel", "↓"), ("d1", "↑"), ("rmse", "↓"))
    FIELDS = ("abs_rel", "d1", "rmse", "abs_rel_ss", "d1_ss")

    def __init__(self, task_cfg: Mapping[str, Any]) -> None:
        super().__init__(task_cfg)
        if self.loss_cfg["kind"] != "silog":
            raise ValueError(f"depth 只实现了 silog，配置是 {self.loss_cfg['kind']!r}")
        self.lam = float(self.loss_cfg["variance_focus"])
        self.scale = float(self.loss_cfg["scale"])

    @staticmethod
    def read_gt(s: Sample) -> np.ndarray:
        if "depth_npz" in s.gt:
            with np.load(s.gt["depth_npz"]) as z:
                return np.asarray(z[s.gt["key"]], dtype=np.float32)
        return read_depth_m(s.gt["depth"])

    def target(self, s: Sample, device: Any) -> Target:
        gt = self.read_gt(s)
        valid = depth_train_valid(gt)
        idx = torch.from_numpy(np.flatnonzero(valid).astype(np.int64))
        log_gt = torch.from_numpy(np.log(gt[valid].astype(np.float64)).astype(np.float32))
        return Target(tuple(gt.shape), {"idx": idx.to(device), "log_gt": log_gt.to(device)}, {"gt": gt}, int(valid.sum()))

    def loss(self, up: torch.Tensor, tgt: Target) -> torch.Tensor:
        p = up.shape[0]
        pl = up[:, 0].reshape(p, -1).index_select(1, tgt.t["idx"])
        d = pl - tgt.t["log_gt"][None]
        v = (d * d).mean(dim=1) - self.lam * d.mean(dim=1) ** 2
        return torch.sqrt(v.clamp_min(1e-12)) * self.scale

    def sample_stats(self, up: torch.Tensor, tgt: Target) -> list[dict[str, float] | None]:
        pred = torch.exp(up[:, 0].float()).cpu().numpy()
        gt = tgt.np["gt"]
        out: list[dict[str, float] | None] = []
        for p in pred:
            m = depth_sample_metrics(p, gt, align="none")
            if m is None:
                out.append(None)
                continue
            ss = depth_sample_metrics(p, gt, align="scale_shift")
            out.append({"abs_rel": m["abs_rel"], "d1": m["d1"], "rmse": depth_sample_rmse(p, gt, align="none"),
                        "abs_rel_ss": ss["abs_rel"], "d1_ss": ss["d1"]})
        return out

    def aggregate(self, stats: Sequence[Any], *, n_resamples: int, seed: int = BOOTSTRAP_SEED, ci: bool = True) -> dict[str, Any]:
        used = [s for s in stats if s is not None]
        out: dict[str, Any] = {"n": len(stats), "n_used": len(used), "n_skipped_no_valid": len(stats) - len(used)}
        if not used:
            return {**out, **{f: None for f in self.FIELDS}}
        ds = depth_dataset_metrics(list(stats))
        for f in self.FIELDS:
            v = np.array([s[f] for s in used], dtype=np.float64)
            out[f] = ds[f] if f in ("abs_rel", "d1") else float(v.mean())
            if ci:
                out[f"{f}_ci"] = _ci(bootstrap_ci(v, n_resamples=n_resamples, seed=seed))
        return out

    def pack(self, stats: Sequence[Any]) -> dict[str, np.ndarray]:
        return {f: np.array([np.nan if s is None else s[f] for s in stats], dtype=np.float64) for f in self.FIELDS}


# ---------------------------------------------------------------------------
# 语义分割
# ---------------------------------------------------------------------------


class SemsegTask(E2Task):
    name = "semseg"
    headline = (("miou", "↑"), ("pixel_acc", "↑"))

    def __init__(self, task_cfg: Mapping[str, Any]) -> None:
        super().__init__(task_cfg)
        if self.loss_cfg["kind"] != "cross_entropy":
            raise ValueError(f"semseg 只实现了 cross_entropy，配置是 {self.loss_cfg['kind']!r}")
        if self.out_channels != ADE_NUM_CLASSES:
            raise ValueError(f"semseg 的 out_channels {self.out_channels} ≠ ADE_NUM_CLASSES {ADE_NUM_CLASSES}")

    def target(self, s: Sample, device: Any) -> Target:
        lab = read_ade_label(s.gt["label"])
        tr = reduce_zero_label(lab) if ADE_REDUCE_ZERO_LABEL else lab.astype(np.int64)
        n_valid = int(np.count_nonzero(tr != ADE_IGNORE_INDEX))
        return Target(tuple(lab.shape), {"label": torch.from_numpy(tr).to(device)}, {"label": lab}, n_valid)

    def loss(self, up: torch.Tensor, tgt: Target) -> torch.Tensor:
        p = up.shape[0]
        lab = tgt.t["label"][None].expand(p, -1, -1).contiguous()
        ce = F.cross_entropy(up, lab, ignore_index=ADE_IGNORE_INDEX, reduction="none")
        return ce.reshape(p, -1).sum(dim=1) / tgt.n_valid

    def sample_stats(self, up: torch.Tensor, tgt: Target) -> list[dict[str, Any]]:
        pred = up.argmax(dim=1).to(torch.int32).cpu().numpy()
        lab = tgt.np["label"]
        out = []
        for p in pred:
            inter, union = intersect_and_union(p, lab, reduce_zero=ADE_REDUCE_ZERO_LABEL)
            c, t = pixel_correct_total(p, lab, reduce_zero=ADE_REDUCE_ZERO_LABEL)
            out.append({"inter": inter, "union": union, "correct": c, "total": t})
        return out

    def aggregate(self, stats: Sequence[Any], *, n_resamples: int, seed: int = BOOTSTRAP_SEED, ci: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(stats)}
        if not stats:
            return {**out, "miou": None, "pixel_acc": None}
        ti = np.sum([s["inter"] for s in stats], axis=0)
        tu = np.sum([s["union"] for s in stats], axis=0)
        out["miou"] = miou_from_totals(ti, tu)
        out["pixel_acc"] = pixel_accuracy([(s["correct"], s["total"]) for s in stats])
        out["n_classes_present"] = int(np.count_nonzero(tu > 0))
        if ci:
            iu = np.stack([np.stack([s["inter"], s["union"]]) for s in stats]).astype(np.float64)
            ct = np.array([[s["correct"], s["total"]] for s in stats], dtype=np.float64)
            out["miou_ci"] = _ci(bootstrap_ci(iu, statistic=miou_statistic, n_resamples=n_resamples, seed=seed)) if out["miou"] is not None else None
            out["pixel_acc_ci"] = _ci(bootstrap_ci(ct, statistic=ratio_of_sums, n_resamples=n_resamples, seed=seed)) if out["pixel_acc"] is not None else None
        return out

    def pack(self, stats: Sequence[Any]) -> dict[str, np.ndarray]:
        return {"inter": np.stack([s["inter"] for s in stats]).astype(np.uint32),
                "union": np.stack([s["union"] for s in stats]).astype(np.uint32),
                "correct": np.array([s["correct"] for s in stats], dtype=np.int64),
                "total": np.array([s["total"] for s in stats], dtype=np.int64)}


# ---------------------------------------------------------------------------
# 法向
# ---------------------------------------------------------------------------


class NormalsTask(E2Task):
    name = "normals"
    headline = (("mean", "↓"), ("median", "↓"), ("within_11.25", "↑"))

    def __init__(self, task_cfg: Mapping[str, Any]) -> None:
        super().__init__(task_cfg)
        if self.loss_cfg["kind"] != "cosine" or self.out_channels != 3:
            raise ValueError("normals 只实现了 3 通道 + cosine 损失")

    def target(self, s: Sample, device: Any) -> Target:
        n, valid = read_normals(s.gt["normal"])
        idx = np.flatnonzero(valid).astype(np.int64)
        gt = np.ascontiguousarray(n.reshape(-1, 3)[idx].T)
        return Target(tuple(valid.shape), {"idx": torch.from_numpy(idx).to(device), "gt": torch.from_numpy(gt).to(device)},
                      {"gt": n, "valid": valid}, int(idx.size))

    def loss(self, up: torch.Tensor, tgt: Target) -> torch.Tensor:
        p = up.shape[0]
        u = F.normalize(up, dim=1, eps=1e-12).reshape(p, 3, -1).index_select(2, tgt.t["idx"])
        cos = (u * tgt.t["gt"][None]).sum(dim=1)
        return (1.0 - cos).mean(dim=1)

    def sample_stats(self, up: torch.Tensor, tgt: Target) -> list[dict[str, Any]]:
        u = F.normalize(up.float(), dim=1, eps=1e-12).permute(0, 2, 3, 1).cpu().numpy()
        return [normal_image_summary(normal_angular_errors(p, tgt.np["gt"], tgt.np["valid"])) for p in u]

    def aggregate(self, stats: Sequence[Any], *, n_resamples: int, seed: int = BOOTSTRAP_SEED, ci: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(stats)}
        m = normal_metrics_from_summaries(list(stats)) if stats else None
        keys = ["mean", "median"] + [f"within_{t:g}" for t in NORMAL_ANGLE_THRESHOLDS]
        if m is None:
            return {**out, **{k: None for k in keys}}
        out.update({k: m[k] for k in keys}, n_pixels=m["n_pixels"], median_bin_deg=NORMAL_HIST_BIN_DEG)
        if ci:
            n = np.array([s["n"] for s in stats], dtype=np.float64)
            sums = np.array([s["sum"] for s in stats], dtype=np.float64)
            out["mean_ci"] = _ci(bootstrap_ci(np.stack([sums, n], 1), statistic=ratio_of_sums, n_resamples=n_resamples, seed=seed))
            for j, t in enumerate(NORMAL_ANGLE_THRESHOLDS):
                w = np.array([s["within"][j] for s in stats], dtype=np.float64)
                out[f"within_{t:g}_ci"] = _ci(bootstrap_ci(np.stack([w, n], 1), statistic=ratio_of_sums,
                                                           n_resamples=n_resamples, seed=seed))
            hist = np.stack([s["hist"] for s in stats]).astype(np.float64)

            def med(x: np.ndarray) -> float:
                v = hist_median(x.sum(axis=0))
                return float("nan") if v is None else v

            out["median_ci"] = _ci(bootstrap_ci(hist, statistic=med, n_resamples=n_resamples, seed=seed))
        return out

    def pack(self, stats: Sequence[Any]) -> dict[str, np.ndarray]:
        return {"sum": np.array([s["sum"] for s in stats], dtype=np.float64),
                "n": np.array([s["n"] for s in stats], dtype=np.int64),
                "within": np.stack([s["within"] for s in stats]).astype(np.int64)}


# ---------------------------------------------------------------------------
# 指代分割
# ---------------------------------------------------------------------------


class RefsegTask(E2Task):
    name = "refseg"
    headline = (("ciou", "↑"), ("giou", "↑"))

    def __init__(self, task_cfg: Mapping[str, Any]) -> None:
        super().__init__(task_cfg)
        if self.loss_cfg["kind"] != "bce_dice" or self.out_channels != 1:
            raise ValueError("refseg 只实现了 1 通道 + bce_dice 损失")
        self.wb = float(self.loss_cfg["bce_weight"])
        self.wd = float(self.loss_cfg["dice_weight"])
        self.smooth = float(self.loss_cfg["dice_smooth"])

    def target(self, s: Sample, device: Any) -> Target:
        m = gt_mask(s)
        return Target(tuple(m.shape), {"mask": torch.from_numpy(m.astype(np.float32)).to(device)}, {"mask": m}, int(m.size))

    def loss(self, up: torch.Tensor, tgt: Target) -> torch.Tensor:
        p = up.shape[0]
        x = up[:, 0].reshape(p, -1)
        y = tgt.t["mask"].reshape(1, -1).expand(p, -1)
        bce = F.binary_cross_entropy_with_logits(x, y, reduction="none").mean(dim=1)
        prob = torch.sigmoid(x)
        dice = 1.0 - (2.0 * (prob * y).sum(dim=1) + self.smooth) / (prob.sum(dim=1) + y.sum(dim=1) + self.smooth)
        return self.wb * bce + self.wd * dice

    def sample_stats(self, up: torch.Tensor, tgt: Target) -> list[dict[str, int]]:
        pred = (up[:, 0] > 0).cpu().numpy()
        out = []
        for p in pred:
            st = mask_inter_union(p, tgt.np["mask"])
            out.append({"inter": st.inter, "union": st.union})
        return out

    def aggregate(self, stats: Sequence[Any], *, n_resamples: int, seed: int = BOOTSTRAP_SEED, ci: bool = True) -> dict[str, Any]:
        pairs = [(s["inter"], s["union"]) for s in stats]
        out: dict[str, Any] = {"n": len(stats), "ciou": ciou(pairs), "giou": giou(pairs)}
        if ci and stats:
            iu = np.array(pairs, dtype=np.float64)
            out["ciou_ci"] = _ci(bootstrap_ci(iu, statistic=ratio_of_sums, n_resamples=n_resamples, seed=seed)) if out["ciou"] is not None else None
            ious = np.array([sample_iou(i, u) for i, u in pairs], dtype=np.float64)
            out["giou_ci"] = _ci(bootstrap_ci(ious, statistic=mean_statistic, n_resamples=n_resamples, seed=seed))
        return out

    def pack(self, stats: Sequence[Any]) -> dict[str, np.ndarray]:
        return {"inter": np.array([s["inter"] for s in stats], dtype=np.int64),
                "union": np.array([s["union"] for s in stats], dtype=np.int64)}


TASK_CLASSES: dict[str, type[E2Task]] = {"depth": DepthTask, "semseg": SemsegTask, "normals": NormalsTask, "refseg": RefsegTask}


def make_task(task_cfg: Mapping[str, Any]) -> E2Task:
    return TASK_CLASSES[task_cfg["task"]](task_cfg)


__all__ = [
    "DepthTask",
    "E2Task",
    "NormalsTask",
    "RefsegTask",
    "SemsegTask",
    "TASK_CLASSES",
    "Target",
    "depth_train_valid",
    "make_task",
    "upsample",
]
