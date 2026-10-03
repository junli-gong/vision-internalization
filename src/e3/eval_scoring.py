"""E3b 评测的逐样本评分（DEV-0030 §1）：网格的紧凑编码、网格 → 原图分辨率（三种上采样）、逐样本指标。

**网格编码**（predictions 里存解析出的网格，不存 1057 个 token id）：
- 分割（packbits_hex）：N×N 的 0/1 按行主序 np.packbits（高位在前）后写十六进制，N = 32 时 256 个字符；
- 深度（base64_alphabet）：每格一个字符，档号 k → B64_ALPHABET[k]（标准 base64 字母表的第 k 个），行主序，N = 32 时 1024 个字符。
往返逐个相同（tests 钉住）。

**上采样到原图**（configs/e3b_train.json 的 eval.upsample；简报 §8.4、§8.5）：
- guided（默认）：src/executors/upsample.guided_upsample（先双线性放大到原图，再以原图为引导做引导滤波）；
- bilinear：只用双线性（bilinear_upsample）；
- cells：按 cell_edges 原样回填（src/e3/feedback.upsample_cells，与叠加反馈图同一规则），只作诊断。
分割把 0/1 网格当作 [0, 1] 的场，放大后 ≥ threshold 记前景；深度把档号换成档的几何中心（codec.bins_to_depth）取 log，在 log
空间放大，再取 exp。

**逐样本指标**：
- 分割：每种上采样一组 (inter, union, iou, biou)：src/metrics/seg 的 mask_inter_union、sample_iou、boundary_iou；解析失败
  （网格为 None）按 metrics.seg 的约定：IoU 0、Boundary IoU 0、cIoU 计 (0, |GT|)。网格层面的诊断：与 GT 网格（mask_to_grid）的
  格子 IoU（两者都空记 1）与是否逐格相同。
- 深度：每种上采样一组 src/metrics/depth.depth_sample_metrics（度量深度 align = none：abs_rel、d1；尺度-平移对齐 align =
  scale_shift：abs_rel_ss、d1_ss，只作诊断）；解析失败为 None。网格层面：可监督格子（src/e3/targets.depth_grid，与训练目标同一规则）
  上的平均档差与逐格相同的比例。
score_payload 是评分进程池（spawn）的入口：子进程自己读图与 GT。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from src.constants import NYU_EVAL_HW
from src.e3.feedback import upsample_cells
from src.executors.upsample import bilinear_upsample, guided_upsample
from src.grid.codec import bins_to_depth, mask_to_grid
from src.metrics.depth import depth_sample_metrics, nyu_eval_mask
from src.metrics.seg import boundary_iou, mask_inter_union, sample_iou

UPSAMPLE_METHODS: tuple[str, ...] = ("guided", "bilinear", "cells")
GRID_CODES = {"refseg": "packbits_hex", "depth": "base64_alphabet"}
B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_B64_INDEX = {c: k for k, c in enumerate(B64_ALPHABET)}


# ---------------------------------------------------------------------------
# 网格编码
# ---------------------------------------------------------------------------


def encode_grid(task: str, grid: np.ndarray) -> str:
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"网格必须是 N×N，收到形状 {g.shape}")
    if task == "refseg":
        if g.size % 8 or (g.size and (int(g.min()) < 0 or int(g.max()) > 1)):
            raise ValueError("分割网格必须是 0/1，且格数是 8 的倍数")
        return np.packbits(g.astype(np.uint8).reshape(-1)).tobytes().hex()
    if task == "depth":
        if g.size and (int(g.min()) < 0 or int(g.max()) >= len(B64_ALPHABET)):
            raise ValueError(f"深度档号越界：[{g.min()}, {g.max()}]")
        return "".join(B64_ALPHABET[int(v)] for v in g.reshape(-1))
    raise ValueError(f"未知任务 {task!r}")


def decode_grid(task: str, code: str, n: int) -> np.ndarray:
    if task == "refseg":
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(code), dtype=np.uint8))
        if bits.size != n * n:
            raise ValueError(f"分割网格编码的长度与 {n}×{n} 不符")
        return bits.reshape(n, n).astype(np.int64)
    if task == "depth":
        if len(code) != n * n:
            raise ValueError(f"深度网格编码的长度与 {n}×{n} 不符")
        return np.asarray([_B64_INDEX[c] for c in code], dtype=np.int64).reshape(n, n)
    raise ValueError(f"未知任务 {task!r}")


# ---------------------------------------------------------------------------
# 上采样
# ---------------------------------------------------------------------------


def upsample_field(low: np.ndarray, image: np.ndarray, method: str) -> np.ndarray:
    """N×N 的场 → 原图分辨率 (H, W) float64（规则见模块说明）。"""
    h, w = image.shape[:2]
    lo = np.asarray(low, dtype=np.float64)
    if method == "guided":
        return guided_upsample(lo, image)
    if method == "bilinear":
        return bilinear_upsample(lo, h, w)
    if method == "cells":
        return upsample_cells(lo, h, w).astype(np.float64)
    raise ValueError(f"未知的上采样 {method!r}（可用：{UPSAMPLE_METHODS}）")


def seg_mask(grid: np.ndarray, image: np.ndarray, method: str, threshold: float) -> np.ndarray:
    return upsample_field(np.asarray(grid, dtype=np.float64), image, method) >= float(threshold)


def depth_map(grid: np.ndarray, image: np.ndarray, method: str) -> np.ndarray:
    low = np.log(bins_to_depth(np.asarray(grid, dtype=np.int64)))
    return np.exp(upsample_field(low, image, method))


# ---------------------------------------------------------------------------
# 逐样本指标
# ---------------------------------------------------------------------------


def score_refseg(image: np.ndarray, gt: np.ndarray, grids: Mapping[str, np.ndarray | None], *, methods: Sequence[str],
                 threshold: float, n: int) -> dict[str, Any]:
    gt = np.asarray(gt, dtype=bool)
    if gt.shape != image.shape[:2]:
        raise ValueError(f"GT mask {gt.shape} 与图像 {image.shape[:2]} 尺寸不同")
    gt_grid = mask_to_grid(gt, n).astype(bool)
    cache: dict[bytes, dict[str, Any]] = {}
    out: dict[str, Any] = {}
    for proc, g in grids.items():
        if g is None:
            st = mask_inter_union(None, gt)
            row = {m: {"inter": st.inter, "union": st.union, "iou": 0.0, "biou": 0.0} for m in methods}
            out[proc] = {"valid": False, **row, "grid": {"cell_iou": 0.0, "exact": False}}
            continue
        key = np.asarray(g, dtype=np.int64).tobytes()
        if key not in cache:
            pg = np.asarray(g).astype(bool)
            u = int((pg | gt_grid).sum())
            rec: dict[str, Any] = {"valid": True,
                                   "grid": {"cell_iou": 1.0 if u == 0 else float((pg & gt_grid).sum()) / u,
                                            "exact": bool(np.array_equal(pg, gt_grid))}}
            for m in methods:
                pm = seg_mask(g, image, m, threshold)
                st = mask_inter_union(pm, gt)
                rec[m] = {"inter": st.inter, "union": st.union, "iou": sample_iou(st.inter, st.union),
                          "biou": boundary_iou(pm, gt)}
            cache[key] = rec
        out[proc] = cache[key]
    return out


def score_depth(image: np.ndarray, gt: np.ndarray, grids: Mapping[str, np.ndarray | None], *, methods: Sequence[str],
                n: int, min_valid_fraction: float) -> dict[str, Any]:
    from src.e3.targets import NoSupervision, depth_grid

    gt = np.asarray(gt, dtype=np.float32)
    if gt.shape != image.shape[:2]:
        raise ValueError(f"GT 深度 {gt.shape} 与图像 {image.shape[:2]} 尺寸不同")
    try:
        gt_bins, gt_sup, _ = depth_grid(gt, min_valid_fraction=min_valid_fraction, n=n)
    except NoSupervision:
        gt_bins, gt_sup = None, None
    cache: dict[bytes, dict[str, Any]] = {}
    out: dict[str, Any] = {}
    for proc, g in grids.items():
        if g is None:
            out[proc] = {"valid": False, **{m: None for m in methods}, "grid": None}
            continue
        key = np.asarray(g, dtype=np.int64).tobytes()
        if key not in cache:
            gg = np.asarray(g, dtype=np.int64)
            grid_rec = None
            if gt_bins is not None:
                d = np.abs(gg - gt_bins)[gt_sup]
                grid_rec = {"mean_abs_bin_error": float(d.mean()), "exact_cell_rate": float((d == 0).mean()),
                            "n_supervised_cells": int(gt_sup.sum())}
            rec: dict[str, Any] = {"valid": True, "grid": grid_rec}
            for m in methods:
                pred = depth_map(gg, image, m)
                a = depth_sample_metrics(pred, gt, align="none")
                b = depth_sample_metrics(pred, gt, align="scale_shift")
                rec[m] = None if a is None else {"abs_rel": a["abs_rel"], "d1": a["d1"], "abs_rel_ss": b["abs_rel"],
                                                 "d1_ss": b["d1"], "n_valid_px": a["n_valid"]}
            cache[key] = rec
        out[proc] = cache[key]
    return out


def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def read_gt(task: str, sample: Any) -> np.ndarray:
    """评测样本的 GT：分割 = E1 的 gt_mask（src/e1/scoring，与 E1、E3 训练同一套解码）；深度 = depth_raw（米，0 = 无效）。"""
    if task == "refseg":
        from src.e1.scoring import gt_mask

        return gt_mask(sample)
    if task == "depth":
        g = sample.gt
        if "depth_npz" in g:
            with np.load(g["depth_npz"]) as z:
                return np.asarray(z[g["key"]], dtype=np.float32)
        raise ValueError(f"{sample.id}：深度 GT 只认 depth_npz（E1 的 nyuv2 落地）")
    raise ValueError(f"未知任务 {task!r}")


def score_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """一个样本的全部评分（进程池的入口）。payload：task、sample（src.data.records.Sample）、codes（流程 → 网格编码或 None）、
    n、methods、threshold、min_valid_fraction。返回 id、GT 的规模（分割：前景像素数；深度：Eigen crop 内的有效像素数，0 = 这张图
    不能评）与各流程的指标。"""
    task, s, n = payload["task"], payload["sample"], int(payload["n"])
    image = load_image(s.images[0])
    gt = read_gt(task, s)
    grids = {p: (None if c is None else decode_grid(task, c, n)) for p, c in payload["codes"].items()}
    if task == "refseg":
        procs = score_refseg(image, gt, grids, methods=payload["methods"], threshold=float(payload["threshold"]), n=n)
        info = {"area": int(np.count_nonzero(gt))}
    else:
        procs = score_depth(image, gt, grids, methods=payload["methods"], n=n,
                            min_valid_fraction=float(payload["min_valid_fraction"]))
        info = {"n_eval_px": int(nyu_eval_mask(gt).sum()) if gt.shape == tuple(NYU_EVAL_HW) else 0}
    return {"id": s.id, "gt": info, "procs": procs}


__all__ = [
    "B64_ALPHABET",
    "GRID_CODES",
    "UPSAMPLE_METHODS",
    "decode_grid",
    "depth_map",
    "encode_grid",
    "load_image",
    "read_gt",
    "score_depth",
    "score_payload",
    "score_refseg",
    "seg_mask",
    "upsample_field",
]
