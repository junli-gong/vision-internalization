"""样本容器、平凡基线的预测、难度分档、样本摘要。

Sample 的字段
- image：(H, W, 3) uint8；没有图像的任务（e）为 None。
- instruction：英文指令（任务的固定模板之一，模板序号在 meta["template_id"]）。
- target_mask：(H, W) bool，全分辨率目标；target_grid：(N, N) uint8，= grid.codec.mask_to_grid(target_mask, N)。
- object_masks：(K, H, W) bool，场景里每个物体一张（顺序同 meta["scene"]["objects"]；含义见各任务模块）。
- baselines：平凡基线名 → Prediction(mask, grid)。
- meta：可 JSON 序列化的元数据（场景规格、难度、模板、种子、基线 IoU……）；meta["provenance"] 记录依赖包版本，
  不进样本摘要。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from src.grid.codec import mask_to_grid
from src.metrics.seg import mask_inter_union, sample_iou


class SynthGenerationError(RuntimeError):
    """拒绝采样在 MAX_ATTEMPTS 次内没有得到合格样本。"""


@dataclass(frozen=True)
class Prediction:
    mask: np.ndarray  # (H, W) bool
    grid: np.ndarray  # (N, N) uint8


def make_prediction(mask: np.ndarray, grid_size: int) -> Prediction:
    m = np.ascontiguousarray(mask, dtype=bool)
    return Prediction(m, mask_to_grid(m, grid_size))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """单样本 IoU（src.metrics.seg 的口径：两者都空记 1.0）。"""
    return sample_iou(*mask_inter_union(pred, gt))


@dataclass
class Sample:
    task: str
    split: str
    index: int
    image: np.ndarray | None
    instruction: str
    target_mask: np.ndarray
    target_grid: np.ndarray
    object_masks: np.ndarray
    baselines: dict[str, Prediction]
    meta: dict = field(default_factory=dict)

    @property
    def grid_size(self) -> int:
        return int(self.target_grid.shape[0])

    def object_grids(self) -> np.ndarray:
        """(K, N, N) uint8，按需计算。"""
        n = self.grid_size
        return np.stack([mask_to_grid(m, n) for m in self.object_masks]) if len(self.object_masks) else np.zeros((0, n, n), np.uint8)

    def digest(self) -> str:
        return sample_digest(self)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sample_digest(s: Sample) -> str:
    """sha256：任务/划分/序号、指令、全部数组（dtype、形状、字节）、除 provenance 外的元数据。"""
    h = hashlib.sha256()

    def add_bytes(tag: str, b: bytes) -> None:
        h.update(tag.encode())
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)

    def add_arr(tag: str, a: np.ndarray | None) -> None:
        if a is None:
            add_bytes(tag, b"None")
            return
        a = np.ascontiguousarray(a)
        add_bytes(tag + ":meta", f"{a.dtype.str}{a.shape}".encode())
        add_bytes(tag, a.tobytes())

    add_bytes("id", f"{s.task}|{s.split}|{s.index}".encode())
    add_bytes("instruction", s.instruction.encode("utf-8"))
    add_arr("image", s.image)
    add_arr("target_mask", s.target_mask)
    add_arr("target_grid", s.target_grid)
    add_arr("object_masks", s.object_masks)
    for name in sorted(s.baselines):
        add_arr(f"baseline:{name}:mask", s.baselines[name].mask)
        add_arr(f"baseline:{name}:grid", s.baselines[name].grid)
    meta = {k: v for k, v in s.meta.items() if k != "provenance"}
    add_bytes("meta", canonical_json(meta).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 难度分档
# ---------------------------------------------------------------------------


def in_half_open_bin(value: float, lo: float, hi: float) -> bool:
    return lo <= value < hi


def difficulty_meta(name: str, value, bin_index: int, bins, closed: bool = False) -> dict:
    """元数据里的难度记录。closed=False：档为 [lo, hi)；closed=True：档为单点（d 的步数）。"""
    if closed:
        rng = [bins[bin_index], bins[bin_index]]
        all_bins = [[b, b] for b in bins]
    else:
        rng = list(bins[bin_index])
        all_bins = [list(b) for b in bins]
    return {"name": name, "value": value, "bin": int(bin_index), "bin_range": rng, "bins": all_bins, "closed": bool(closed)}


def value_in_bin(d: dict) -> bool:
    lo, hi = d["bin_range"]
    return d["value"] == lo if d["closed"] else lo <= d["value"] < hi


# ---------------------------------------------------------------------------
# 生成上下文（api.generate 构造，各任务模块使用）
# ---------------------------------------------------------------------------


@dataclass
class GenContext:
    task: str
    split: str
    index: int
    rng: np.random.Generator
    bin: int
    height: int
    width: int
    grid_size: int
    holdout_family: str | None
    scene_seed: int
    version: int

    @property
    def short_side(self) -> int:
        return min(self.height, self.width)

    def base_meta(self) -> dict:
        return {
            "task": self.task,
            "split": self.split,
            "index": self.index,
            "generator_version": self.version,
            "scene_seed": self.scene_seed,
            "canvas_hw": [self.height, self.width],
            "grid_size": self.grid_size,
            "holdout_family": self.holdout_family,
        }


def baseline_iou_meta(baselines: dict[str, Prediction], target_mask: np.ndarray, target_grid: np.ndarray) -> dict:
    """各基线相对目标的 IoU：全分辨率与 N×N 网格两种口径。"""
    return {
        name: {"full": iou(p.mask, target_mask), "grid": iou(p.grid, target_grid)} for name, p in sorted(baselines.items())
    }
