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

按光栅位置分开的标题指标（DEV-0024；D53-3 的 I2；划分的定义在 src/e2/positions.py）：position_stats 给每个探针、每种划分、每个
区域一份与 sample_stats 同格式的逐样本统计（深度只含标题指标的三项），PositionAccumulator 在评测集上累计，再用同一个 aggregate
（不重采样、无 CI）算数据集级指标。深度、语义分割、法向按像素划分：逐像素的量（深度的相对误差、δ1 命中、平方误差；分割的交并计数；
法向角误差）只算一次，再按区域下标用 bincount 分组——与"把 GT 在区域外置为无效后调 src/metrics 的函数"逐项相同（tests 钉住），
但不用对每个区域重算一遍。指代分割按 GT 掩码质心把整条样本归入一个区域，统计就是 sample_stats 的那一份。
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
from src.e2.positions import SPLITS, mask_centroid, pixel_labels, point_labels
from src.metrics.depth import (
    DELTA1_THRESHOLD,
    clip_pred,
    depth_dataset_metrics,
    depth_sample_metrics,
    depth_sample_rmse,
    nyu_eval_mask,
)
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


PositionStats = dict[str, list[Any]]  # {划分: [每个区域的逐样本统计或 None]}（positions.SPLITS 的顺序）


class E2Task:
    name: str = ""
    headline: tuple[tuple[str, str], ...] = ()  # (指标名, ↑/↓)
    # 按位置分开（positions.py）：position_unit = "pixel"（每个区域都有一份统计，None = 区域里没有参与评测的像素）
    # 或 "sample"（整条样本归入每种划分的一个区域，其余区域为 None）；additive = 逐样本统计可以直接相加（累计时求和，不留列表）
    position_unit: str = "pixel"
    additive: bool = False
    # 区域指标里留下的计数项（标题指标之外）
    position_count_keys: tuple[str, ...] = ("n",)

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

    def position_stats(self, up: torch.Tensor, tgt: Target, grid_hw: tuple[int, int],
                       whole: Sequence[Any]) -> list[PositionStats]:  # pragma: no cover
        """每个探针一份 {划分: [各区域的逐样本统计]}；whole 是同一块探针的 sample_stats（按样本划分的任务直接复用）。"""
        raise NotImplementedError

    def eval_stats(self, up: torch.Tensor, tgt: Target, grid_hw: tuple[int, int]) -> tuple[list[Any], list[PositionStats]]:
        """评测集上一块探针的 (sample_stats, position_stats)。默认分别算；子类可以共用中间量（法向的角误差）。"""
        whole = self.sample_stats(up, tgt)
        return whole, self.position_stats(up, tgt, grid_hw, whole)

    def position_has_pixels(self, st: Any) -> bool:
        """additive 的任务：这份区域统计里有没有参与评测的像素（数 n_used）。"""
        raise NotImplementedError  # pragma: no cover

    def position_unassigned(self, tgt: Target) -> bool:
        """按样本划分的任务：这条样本归不进任何区域（例如 GT 掩码为空）。"""
        return False

    def position_metrics(self, m: Mapping[str, Any]) -> dict[str, Any]:
        """aggregate 的结果里只留标题指标与计数项。"""
        keep = [k for k, _ in self.headline] + [k for k in self.position_count_keys]
        return {k: m.get(k) for k in keep if k in m}


def _bincount(idx: np.ndarray, n: int, weights: np.ndarray | None = None) -> np.ndarray:
    return np.bincount(idx, weights=weights, minlength=n)[:n]


class PositionAccumulator:
    """评测集上按位置分开的统计（positions.SPLITS）：add 逐样本累计每个探针的 position_stats，result 给
    {探针: {划分: {区域: 标题指标与计数}}}（aggregate 不重采样，无 CI）。

    additive 的任务（semseg、normals）逐区域求和，不留逐样本的列表（ADE 的 ADE_NUM_CLASSES 类 × 6 个区域 × 全部评测图 × 全部探针放不下），
    另数区域里有像素的样本数 n_used；其余任务留列表：深度含 None（区域里没有有效像素，aggregate 记 n_skipped_no_valid），
    指代分割只留归入该区域的样本。"""

    def __init__(self, task: E2Task, names: Sequence[str]) -> None:
        self.task = task
        self.n_samples = 0
        self.n_unassigned = 0
        self.data: dict[str, dict[str, list[Any]]] = {
            n: {s: [None if task.additive else [] for _ in regs] for s, regs in SPLITS.items()} for n in names}
        self.n_used: dict[str, dict[str, list[int]]] = {n: {s: [0] * len(regs) for s, regs in SPLITS.items()} for n in names}

    def add_sample(self, tgt: Target) -> None:
        self.n_samples += 1
        self.n_unassigned += int(self.task.position_unassigned(tgt))

    def add(self, name: str, ps: PositionStats) -> None:
        for s, regs in SPLITS.items():
            for r, st in enumerate(ps[s]):
                if self.task.additive:
                    if st is None:
                        continue
                    cur = self.data[name][s][r]
                    self.data[name][s][r] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()} \
                        if cur is None else {k: cur[k] + st[k] for k in cur}
                    self.n_used[name][s][r] += int(self.task.position_has_pixels(st))
                elif st is not None or self.task.position_unit == "pixel":
                    self.data[name][s][r].append(st)

    def result(self) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
        out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for name, by_split in self.data.items():
            out[name] = {}
            for s, regs in SPLITS.items():
                out[name][s] = {}
                for r, reg in enumerate(regs):
                    acc = by_split[s][r]
                    if self.task.additive:
                        m = self.task.aggregate([] if acc is None else [acc], n_resamples=0, ci=False)
                        m = {**self.task.position_metrics(m), "n": self.n_samples, "n_used": self.n_used[name][s][r]}
                    else:
                        m = self.task.position_metrics(self.task.aggregate(acc, n_resamples=0, ci=False))
                    out[name][s][reg] = m
        return out


# ---------------------------------------------------------------------------
# 深度
# ---------------------------------------------------------------------------


class DepthTask(E2Task):
    name = "depth"
    headline = (("abs_rel", "↓"), ("d1", "↑"), ("rmse", "↓"))
    FIELDS = ("abs_rel", "d1", "rmse", "abs_rel_ss", "d1_ss")
    position_count_keys = ("n", "n_used", "n_skipped_no_valid")

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
        """逐图平均。逐样本统计只含 FIELDS 的一部分时（按位置分开的统计只有标题指标三项）只算这几项。"""
        used = [s for s in stats if s is not None]
        out: dict[str, Any] = {"n": len(stats), "n_used": len(used), "n_skipped_no_valid": len(stats) - len(used)}
        if not used:
            return {**out, **{f: None for f in self.FIELDS}}
        ds = depth_dataset_metrics(list(stats))
        for f in (f for f in self.FIELDS if f in used[0]):
            v = np.array([s[f] for s in used], dtype=np.float64)
            out[f] = ds[f] if f in ("abs_rel", "d1") else float(v.mean())
            if ci:
                out[f"{f}_ci"] = _ci(bootstrap_ci(v, n_resamples=n_resamples, seed=seed))
        return out

    def pack(self, stats: Sequence[Any]) -> dict[str, np.ndarray]:
        return {f: np.array([np.nan if s is None else s[f] for s in stats], dtype=np.float64) for f in self.FIELDS}

    def position_stats(self, up: torch.Tensor, tgt: Target, grid_hw: tuple[int, int],
                       whole: Sequence[Any]) -> list[PositionStats]:
        """按像素划分：评测像素（nyu_eval_mask：Eigen crop ∧ GT 有效）上逐像素的 |g − p|/g、[max(g/p, p/g) < 1.25]、(g − p)²
        （p 先 clip_pred；与 depth_sample_metrics / depth_sample_rmse 的 align="none" 同一公式），按区域取平均（RMSE 再开方）。
        区域里没有评测像素时为 None。"""
        pred = torch.exp(up[:, 0].float()).cpu().numpy()
        gt = np.asarray(tgt.np["gt"], dtype=np.float64)
        mask = nyu_eval_mask(gt)
        gv = gt[mask]
        labs = {s: lab[mask] for s, lab in pixel_labels(gt.shape, grid_hw).items()}
        out = []
        for p in pred:
            pv = clip_pred(np.asarray(p, dtype=np.float64)[mask])
            terms = (np.abs(gv - pv) / gv, (np.maximum(gv / pv, pv / gv) < DELTA1_THRESHOLD).astype(np.float64), (gv - pv) ** 2)
            per: PositionStats = {}
            for s, regs in SPLITS.items():
                R, g = len(regs), labs[s]
                cnt = _bincount(g, R)
                a, d, q = (_bincount(g, R, t) for t in terms)
                per[s] = [None if cnt[r] == 0 else {"abs_rel": float(a[r] / cnt[r]), "d1": float(d[r] / cnt[r]),
                                                    "rmse": float(np.sqrt(q[r] / cnt[r]))} for r in range(R)]
            out.append(per)
        return out


# ---------------------------------------------------------------------------
# 语义分割
# ---------------------------------------------------------------------------


class SemsegTask(E2Task):
    name = "semseg"
    headline = (("miou", "↑"), ("pixel_acc", "↑"))
    additive = True
    position_count_keys = ("n", "n_used", "n_classes_present")

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
        return Target(tuple(lab.shape), {"label": torch.from_numpy(tr).to(device)}, {"label": lab, "label_reduced": tr}, n_valid)

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

    def position_stats(self, up: torch.Tensor, tgt: Target, grid_hw: tuple[int, int],
                       whole: Sequence[Any]) -> list[PositionStats]:
        """按像素划分：参与评测的像素（reduce_zero_label 之后不是忽略值、在 [0, C) 里；与 intersect_and_union /
        pixel_correct_total 相同）上按 (区域, 类) 计交、预测面积、标签面积，并 = 预测 + 标签 − 交；correct = 交之和，total = 像素数。"""
        pred = up.argmax(dim=1).to(torch.int64).cpu().numpy()
        tr = np.asarray(tgt.np["label_reduced"], dtype=np.int64)
        C = ADE_NUM_CLASSES
        keep = (tr != ADE_IGNORE_INDEX) & (tr >= 0) & (tr < C)
        t = tr[keep]
        labs = {s: lab[keep] for s, lab in pixel_labels(tr.shape, grid_hw).items()}
        out = []
        for p in pred:
            pk = p[keep]
            ok = (pk >= 0) & (pk < C)
            hit = pk == t
            per: PositionStats = {}
            for s, regs in SPLITS.items():
                R, g = len(regs), labs[s]
                inter = _bincount(g[hit] * C + t[hit], R * C).reshape(R, C)
                area_p = _bincount(g[ok] * C + pk[ok], R * C).reshape(R, C)
                area_l = _bincount(g * C + t, R * C).reshape(R, C)
                tot = _bincount(g, R)
                per[s] = [{"inter": inter[r], "union": area_p[r] + area_l[r] - inter[r], "correct": int(inter[r].sum()),
                           "total": int(tot[r])} for r in range(R)]
            out.append(per)
        return out

    def position_has_pixels(self, st: Any) -> bool:
        return int(st["total"]) > 0


# ---------------------------------------------------------------------------
# 法向
# ---------------------------------------------------------------------------


class NormalsTask(E2Task):
    name = "normals"
    headline = (("mean", "↓"), ("median", "↓"), ("within_11.25", "↑"))
    additive = True
    position_count_keys = ("n", "n_used", "n_pixels")

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

    @staticmethod
    def _errors(up: torch.Tensor, tgt: Target) -> list[np.ndarray]:
        """每个探针的逐像素角误差（预测逐像素单位化之后，normal_angular_errors）。"""
        u = F.normalize(up.float(), dim=1, eps=1e-12).permute(0, 2, 3, 1).cpu().numpy()
        return [normal_angular_errors(p, tgt.np["gt"], tgt.np["valid"]) for p in u]

    def sample_stats(self, up: torch.Tensor, tgt: Target) -> list[dict[str, Any]]:
        return [normal_image_summary(e) for e in self._errors(up, tgt)]

    def eval_stats(self, up: torch.Tensor, tgt: Target, grid_hw: tuple[int, int]) -> tuple[list[Any], list[PositionStats]]:
        """角误差只算一次（它是评测里最贵的一步），整图的汇总与按位置分开的汇总都由它得出。"""
        errs = self._errors(up, tgt)
        return [normal_image_summary(e) for e in errs], self._position_from_errors(errs, tgt, grid_hw)

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

    def position_stats(self, up: torch.Tensor, tgt: Target, grid_hw: tuple[int, int],
                       whole: Sequence[Any]) -> list[PositionStats]:
        """按像素划分：normal_angular_errors 的逐像素角误差（整张图算一次）按区域分组，逐区域做 normal_image_summary 的四项
        （sum、n、within、hist；与对区域内像素调 normal_image_summary 相同）。角误差只在 GT 有效、范数有限且 > 0 的像素上（与
        normal_angular_errors 的掩码相同，长度不符就报错）。"""
        return self._position_from_errors(self._errors(up, tgt), tgt, grid_hw)

    @staticmethod
    def _position_from_errors(errs: Sequence[np.ndarray], tgt: Target, grid_hw: tuple[int, int]) -> list[PositionStats]:
        gt, valid = tgt.np["gt"], tgt.np["valid"]
        gn = np.linalg.norm(np.asarray(gt, dtype=np.float64), axis=-1)
        m = np.isfinite(gn) & (gn > 0) & valid
        labs = {s: lab[m] for s, lab in pixel_labels(valid.shape, grid_hw).items()}
        nb = int(round(180.0 / NORMAL_HIST_BIN_DEG))
        out = []
        for e in errs:
            if e.size != labs["half"].size:
                raise RuntimeError(f"法向角误差 {e.size} 个像素，区域下标 {labs['half'].size} 个：掩码与 normal_angular_errors 不一致")
            bins = np.clip((e / NORMAL_HIST_BIN_DEG).astype(np.int64), 0, nb - 1)
            per: PositionStats = {}
            for s, regs in SPLITS.items():
                R, g = len(regs), labs[s]
                n, sm = _bincount(g, R), _bincount(g, R, e)
                within = np.stack([_bincount(g[e < t], R) for t in NORMAL_ANGLE_THRESHOLDS], axis=1).astype(np.int64)
                hist = _bincount(g * nb + bins, R * nb).reshape(R, nb).astype(np.int64)
                per[s] = [{"sum": float(sm[r]), "n": int(n[r]), "within": within[r], "hist": hist[r]} for r in range(R)]
            out.append(per)
        return out

    def position_has_pixels(self, st: Any) -> bool:
        return int(st["n"]) > 0


# ---------------------------------------------------------------------------
# 指代分割
# ---------------------------------------------------------------------------


class RefsegTask(E2Task):
    name = "refseg"
    headline = (("ciou", "↑"), ("giou", "↑"))
    position_unit = "sample"

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

    @staticmethod
    def position_labels(tgt: Target, grid_hw: tuple[int, int]) -> dict[str, int] | None:
        """这条样本在两种划分里的区域：GT 掩码质心（positions.mask_centroid）所在的区域；掩码为空时 None。"""
        c = mask_centroid(tgt.np["mask"])
        return None if c is None else point_labels(c[0], c[1], tgt.hw, grid_hw)

    def position_stats(self, up: torch.Tensor, tgt: Target, grid_hw: tuple[int, int],
                       whole: Sequence[Any]) -> list[PositionStats]:
        """按样本划分：整条样本的统计（whole，即 sample_stats）归入质心所在的区域，其余区域为 None。"""
        lab = self.position_labels(tgt, grid_hw)
        return [{s: [st if (lab is not None and lab[s] == r) else None for r in range(len(regs))] for s, regs in SPLITS.items()}
                for st in whole]

    def position_unassigned(self, tgt: Target) -> bool:
        return mask_centroid(tgt.np["mask"]) is None


TASK_CLASSES: dict[str, type[E2Task]] = {"depth": DepthTask, "semseg": SemsegTask, "normals": NormalsTask, "refseg": RefsegTask}


def make_task(task_cfg: Mapping[str, Any]) -> E2Task:
    return TASK_CLASSES[task_cfg["task"]](task_cfg)


__all__ = [
    "DepthTask",
    "E2Task",
    "NormalsTask",
    "PositionAccumulator",
    "RefsegTask",
    "SemsegTask",
    "TASK_CLASSES",
    "Target",
    "depth_train_valid",
    "make_task",
    "upsample",
]
