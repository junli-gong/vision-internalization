"""E3b 的网格目标：GT → N×N 网格 → token 序列，以及反方向的解析（DEV-0025 §1）。

**序列格式**（N = GRID_SIZE_E3B = 32）：光栅顺序（行主序：自上而下逐行，行内自左而右），每行 N 个值 token 之后接一个行分隔
token，最后接本家族的停止 token。长度 = N × (N + 1) + 1 = 1057（SEQ_LEN）。值 token 的下标就是格子的取值：分割是 0 / 1，
深度是档号 0..DEPTH_NUM_BINS − 1。

**分割**：mask (H, W) → src/grid/codec.mask_to_grid（格内前景占比 ≥ GRID_FG_AREA_THRESHOLD 记 1）。所有格子都监督。

**深度**（无效像素的处理）：
- 有效像素 = 有限且 NYU_DEPTH_MIN < d < NYU_DEPTH_MAX（与 NYU 评测的有效区间相同，src/constants）；BTS 的 0 是无效。
- 格边界与 mask_to_grid 相同（codec.cell_edges）。一个格子是"可监督"的，当且仅当其中有效像素的个数 ≥ min_valid_fraction × 格面积
  （配置 tasks.depth.cell.min_valid_fraction，默认 0.5，与 mask 的"过半"同一口径）；格值 = 有效像素深度的几何平均
  （log 深度的均值再取 exp：分档是 log 均匀的），再按 codec.depth_to_bins 归档（越界先裁剪到 [D_MIN, D_MAX]）。
- 不可监督的格子：损失权重 0（不计入监督 token 数）。教师强制的输入里它仍要有一个 token，取离它最近的可监督格子的档号
  （格中心的欧氏距离，scipy.ndimage.distance_transform_edt 的 return_indices，结果确定）；整张图没有一个可监督的格子时报
  NoSupervision（样本跳过并计数）。
- 行分隔与停止 token 总是监督。

**解析**（parse_grid_ids）：严格，不补齐、不截断、不猜：最后一个 token 必须是停止 token（否则 TRUNCATED / NO_STOP），总长度必须是
SEQ_LEN，每行末尾必须是行分隔，其余位置必须是本任务的值 token。解析失败给出原因（GridIdsFailure）。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Sequence

import numpy as np
from scipy import ndimage

from src.constants import DEPTH_NUM_BINS, GRID_SIZE_E3B, NYU_DEPTH_MAX, NYU_DEPTH_MIN
from src.e3.tokens import GridTokens, N_MASK_VALUES
from src.grid.codec import cell_edges, depth_to_bins, mask_to_grid

GRID_N = GRID_SIZE_E3B
ROW_LEN = GRID_N + 1  # 一行 N 个值 token + 1 个行分隔
SEQ_LEN = GRID_N * ROW_LEN + 1  # + 停止 token


def seq_len(n: int = GRID_N) -> int:
    return n * (n + 1) + 1


class NoSupervision(ValueError):
    """这条样本没有可监督的格子（深度全无效），跳过并计数。"""


class GridIdsFailure(str, enum.Enum):
    TRUNCATED = "truncated"  # 没有以停止 token 结尾（被 max_new_tokens 截断）
    WRONG_LENGTH = "wrong_length"  # 以停止 token 结尾，但长度不是 SEQ_LEN
    ROW_SEPARATOR = "row_separator"  # 行末不是行分隔，或行内出现了行分隔
    FOREIGN_TOKEN = "foreign_token"  # 格子位置出现了本任务值 token 以外的 token（含别的任务的值 token、提前的停止 token）


class GridIdsParse(NamedTuple):
    grid: np.ndarray | None
    failure: GridIdsFailure | None


@dataclass(frozen=True)
class GridTarget:
    """一条网格目标：值网格（N×N int64）、每格是否监督、token 序列与逐 token 的损失权重（0 / 1）。"""

    task: str
    grid: np.ndarray
    supervised: np.ndarray
    ids: tuple[int, ...]
    weights: tuple[int, ...]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_supervised(self) -> int:
        return int(sum(self.weights))


# ---------------------------------------------------------------------------
# 网格 ↔ token 序列
# ---------------------------------------------------------------------------


def _check_grid(grid: np.ndarray, n_values: int, n: int) -> np.ndarray:
    g = np.asarray(grid)
    if g.shape != (n, n):
        raise ValueError(f"网格形状应为 ({n}, {n})，收到 {g.shape}")
    if not (np.issubdtype(g.dtype, np.integer) or g.dtype == np.bool_):
        raise ValueError(f"网格必须是整数（或 bool），收到 dtype {g.dtype}")
    g = g.astype(np.int64)
    if g.size and (int(g.min()) < 0 or int(g.max()) >= n_values):
        raise ValueError(f"网格取值越界：[{g.min()}, {g.max()}]，应在 [0, {n_values})")
    return g


def grid_to_ids(grid: np.ndarray, value_ids: Sequence[int], row_id: int, stop_id: int, *, n: int = GRID_N) -> list[int]:
    """N×N 取值网格 → token 序列（格式见模块说明）。"""
    g = _check_grid(grid, len(value_ids), n)
    vid = np.asarray([int(v) for v in value_ids], dtype=np.int64)
    rows = np.concatenate([vid[g], np.full((n, 1), int(row_id), dtype=np.int64)], axis=1)
    return [int(t) for t in rows.reshape(-1)] + [int(stop_id)]


def parse_grid_ids(ids: Sequence[int], value_ids: Sequence[int], row_id: int, stop_id: int, *,
                   n: int = GRID_N) -> GridIdsParse:
    """token 序列 → N×N 取值网格（int64）；不合格式就给出原因（规则见模块说明）。ids 可以含停止 token（generate_restricted 的
    token_ids 含停止 token）。"""
    seq = [int(t) for t in ids]
    if not seq or seq[-1] != int(stop_id):
        return GridIdsParse(None, GridIdsFailure.TRUNCATED)
    body = seq[:-1]
    if len(body) != n * (n + 1):
        return GridIdsParse(None, GridIdsFailure.WRONG_LENGTH)
    col = {int(v): k for k, v in enumerate(value_ids)}
    grid = np.zeros((n, n), dtype=np.int64)
    for r in range(n):
        row = body[r * (n + 1): (r + 1) * (n + 1)]
        if row[-1] != int(row_id) or int(row_id) in row[:-1]:
            return GridIdsParse(None, GridIdsFailure.ROW_SEPARATOR)
        for c, t in enumerate(row[:-1]):
            k = col.get(t)
            if k is None:
                return GridIdsParse(None, GridIdsFailure.FOREIGN_TOKEN)
            grid[r, c] = k
    return GridIdsParse(grid, None)


def weights_for(supervised: np.ndarray, *, n: int = GRID_N) -> list[int]:
    """逐 token 的损失权重：格子按 supervised，行分隔与停止 token 总是 1。"""
    s = np.asarray(supervised, dtype=bool)
    if s.shape != (n, n):
        raise ValueError(f"supervised 形状应为 ({n}, {n})，收到 {s.shape}")
    rows = np.concatenate([s.astype(np.int64), np.ones((n, 1), dtype=np.int64)], axis=1)
    return [int(w) for w in rows.reshape(-1)] + [1]


# ---------------------------------------------------------------------------
# 分割
# ---------------------------------------------------------------------------


def seg_grid(mask: np.ndarray, *, n: int = GRID_N) -> np.ndarray:
    """mask (H, W) → N×N 0/1 网格（int64；codec.mask_to_grid）。"""
    return mask_to_grid(mask, n).astype(np.int64)


def seg_target(mask: np.ndarray, tokens: GridTokens, *, n: int = GRID_N) -> GridTarget:
    g = seg_grid(mask, n=n)
    sup = np.ones((n, n), dtype=bool)
    ids = grid_to_ids(g, tokens.value_ids("refseg"), tokens.row_id, tokens.stop_id, n=n)
    return GridTarget("refseg", g, sup, tuple(ids), tuple(weights_for(sup, n=n)),
                      {"n_fg_cells": int(g.sum())})


# ---------------------------------------------------------------------------
# 深度
# ---------------------------------------------------------------------------


def depth_valid(depth_m: np.ndarray) -> np.ndarray:
    """有效像素：有限且 NYU_DEPTH_MIN < d < NYU_DEPTH_MAX（模块说明）。"""
    d = np.asarray(depth_m, dtype=np.float64)
    return np.isfinite(d) & (d > NYU_DEPTH_MIN) & (d < NYU_DEPTH_MAX)


def depth_grid(depth_m: np.ndarray, *, min_valid_fraction: float, n: int = GRID_N) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """深度图（米，H×W）→ (N×N 档号 int64，N×N 是否监督 bool，记录)。规则见模块说明。"""
    d = np.asarray(depth_m, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"深度图必须是二维，收到形状 {d.shape}")
    if not 0.0 < float(min_valid_fraction) <= 1.0:
        raise ValueError(f"min_valid_fraction 必须在 (0, 1]，收到 {min_valid_fraction}")
    valid = depth_valid(d)
    h, w = d.shape
    re_, ce = cell_edges(h, n), cell_edges(w, n)
    logd = np.where(valid, np.log(np.where(valid, d, 1.0)), 0.0)
    cnt = np.add.reduceat(np.add.reduceat(valid.astype(np.int64), re_[:-1], axis=0), ce[:-1], axis=1)
    s = np.add.reduceat(np.add.reduceat(logd, re_[:-1], axis=0), ce[:-1], axis=1)
    area = np.diff(re_)[:, None] * np.diff(ce)[None, :]
    sup = cnt >= float(min_valid_fraction) * area
    if not sup.any():
        raise NoSupervision(f"深度图没有可监督的格子（有效像素 {int(valid.sum())} / {valid.size}）")
    mean_log = np.where(sup, s / np.maximum(cnt, 1), 0.0)
    bins = np.zeros((n, n), dtype=np.int64)
    bins[sup] = depth_to_bins(np.exp(mean_log[sup]))
    n_fill = int((~sup).sum())
    if n_fill:
        _, (ri, ci) = ndimage.distance_transform_edt(~sup, return_indices=True)
        bins = bins[ri, ci]
    return bins, sup, {"n_supervised_cells": int(sup.sum()), "n_filled_cells": n_fill,
                       "valid_pixel_fraction": float(valid.mean())}


def depth_target(depth_m: np.ndarray, tokens: GridTokens, *, min_valid_fraction: float, n: int = GRID_N) -> GridTarget:
    bins, sup, info = depth_grid(depth_m, min_valid_fraction=min_valid_fraction, n=n)
    ids = grid_to_ids(bins, tokens.value_ids("depth"), tokens.row_id, tokens.stop_id, n=n)
    return GridTarget("depth", bins, sup, tuple(ids), tuple(weights_for(sup, n=n)), info)


def target_from_grid(task: str, grid: np.ndarray, supervised: np.ndarray, tokens: GridTokens, *,
                     n: int = GRID_N) -> GridTarget:
    """已有网格（例如修正对的干净网格）→ 目标。"""
    ids = grid_to_ids(grid, tokens.value_ids(task), tokens.row_id, tokens.stop_id, n=n)
    return GridTarget(task, np.asarray(grid, dtype=np.int64), np.asarray(supervised, dtype=bool), tuple(ids),
                      tuple(weights_for(supervised, n=n)))


def parse_target(task: str, ids: Sequence[int], tokens: GridTokens, *, n: int = GRID_N) -> GridIdsParse:
    """任务的 token 序列 → 网格（parse_grid_ids，值 token 取本任务的）。"""
    return parse_grid_ids(ids, tokens.value_ids(task), tokens.row_id, tokens.stop_id, n=n)


N_VALUES = {"refseg": N_MASK_VALUES, "depth": DEPTH_NUM_BINS}


__all__ = [
    "GRID_N",
    "GridIdsFailure",
    "GridIdsParse",
    "GridTarget",
    "NoSupervision",
    "N_VALUES",
    "ROW_LEN",
    "SEQ_LEN",
    "depth_grid",
    "depth_target",
    "depth_valid",
    "grid_to_ids",
    "parse_grid_ids",
    "parse_target",
    "seg_grid",
    "seg_target",
    "seq_len",
    "target_from_grid",
    "weights_for",
]
