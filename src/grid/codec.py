"""网格编解码：mask ↔ N×N 网格、深度 ↔ 分档、E1 数字字符网格 文本 ↔ 数组。

规则全部来自 src/constants.py。
"""

from __future__ import annotations

import enum
import re
from typing import NamedTuple

import numpy as np

from src.constants import D_MAX, D_MIN, DEPTH_NUM_BINS, GRID_FG_AREA_THRESHOLD

# ---------------------------------------------------------------------------
# mask ↔ 网格
# ---------------------------------------------------------------------------


def cell_edges(size: int, n: int) -> np.ndarray:
    """N 格划分长度 size 的边界：edges[k] = floor(k * size / n)，k = 0..n（整数运算，无浮点误差）。"""
    if n <= 0 or size <= 0:
        raise ValueError(f"size={size}, n={n} 必须为正")
    if size < n:
        raise ValueError(f"size={size} < n={n}：会有空格子，拒绝")
    k = np.arange(n + 1, dtype=np.int64)
    return (k * size) // n


def mask_to_grid(mask: np.ndarray, n: int) -> np.ndarray:
    """mask (H, W) → N×N 0/1 网格（uint8）。格内前景像素占比 ≥ GRID_FG_AREA_THRESHOLD 记 1。

    格边界按 cell_edges，与 grid_to_mask 相同。
    """
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"mask 必须是二维，收到形状 {m.shape}")
    h, w = m.shape
    re_, ce = cell_edges(h, n), cell_edges(w, n)
    fg = np.add.reduceat(np.add.reduceat(m.astype(np.int64), re_[:-1], axis=0), ce[:-1], axis=1)
    area = np.diff(re_)[:, None] * np.diff(ce)[None, :]
    if GRID_FG_AREA_THRESHOLD == 0.5:
        return (2 * fg >= area).astype(np.uint8)  # 整数比较，避免浮点边界误差
    return (fg >= GRID_FG_AREA_THRESHOLD * area).astype(np.uint8)


def grid_to_mask(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """N×N 网格 → mask (H, W)，按 cell_edges 的格边界精确回填。"""
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"网格必须是 N×N，收到形状 {g.shape}")
    n = g.shape[0]
    rows = np.diff(cell_edges(height, n))
    cols = np.diff(cell_edges(width, n))
    return np.repeat(np.repeat(g.astype(bool), rows, axis=0), cols, axis=1)


# ---------------------------------------------------------------------------
# 深度 ↔ 分档（暂定参数，见 constants）
# ---------------------------------------------------------------------------

_LOG_MIN = np.log(D_MIN)
_LOG_MAX = np.log(D_MAX)
_LOG_STEP = (_LOG_MAX - _LOG_MIN) / DEPTH_NUM_BINS


def depth_bin_edges() -> np.ndarray:
    """DEPTH_NUM_BINS+1 条档边界（米），对数均匀，首尾为 D_MIN、D_MAX。"""
    return np.exp(_LOG_MIN + _LOG_STEP * np.arange(DEPTH_NUM_BINS + 1))


def depth_to_bins(depth: np.ndarray) -> np.ndarray:
    """深度（米）→ 档号 0..DEPTH_NUM_BINS-1（int64）。

    先裁剪到 [D_MIN, D_MAX]；档 k 覆盖 [edge_k, edge_{k+1})，最后一档含 D_MAX。
    深度必须为正且有限；无效深度（0、负、NaN、inf）由调用方先掩掉，这里抛 ValueError。
    """
    d = np.asarray(depth, dtype=np.float64)
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("深度含非正或非有限值；请先掩掉无效像素")
    t = (np.log(np.clip(d, D_MIN, D_MAX)) - _LOG_MIN) / _LOG_STEP
    return np.clip(np.floor(t), 0, DEPTH_NUM_BINS - 1).astype(np.int64)


def bins_to_depth(bins: np.ndarray) -> np.ndarray:
    """档号 → 深度（米），取档的几何中心 sqrt(edge_k * edge_{k+1})。"""
    b = np.asarray(bins)
    if not np.issubdtype(b.dtype, np.integer):
        raise ValueError(f"档号必须是整数，收到 dtype {b.dtype}")
    if np.any(b < 0) or np.any(b >= DEPTH_NUM_BINS):
        raise ValueError(f"档号越界，须在 [0, {DEPTH_NUM_BINS})")
    return np.exp(_LOG_MIN + _LOG_STEP * (b + 0.5))


# ---------------------------------------------------------------------------
# E1 数字字符网格：文本 ↔ 数组
# ---------------------------------------------------------------------------


class GridParseFailure(str, enum.Enum):
    """解析失败原因（有限枚举，供统计解析失败率）。"""

    NO_GRID = "no_grid"  # 找不到任何 0/1 网格行
    MULTIPLE_BLOCKS = "multiple_blocks"  # 有多于一个网格块，不猜用哪个
    ILLEGAL_CHAR = "illegal_char"  # 网格块里夹杂非法字符（非 0/1、空格、逗号）
    WRONG_ROW_COUNT = "wrong_row_count"  # 行数 ≠ N
    WRONG_COL_COUNT = "wrong_col_count"  # 某行的数字个数 ≠ N


class GridParse(NamedTuple):
    grid: np.ndarray | None
    failure: GridParseFailure | None


# 纯网格行：只有 0/1，数字之间可有空格、制表符或逗号（行首尾的分隔符也允许）
_GRID_LINE = re.compile(r"[ \t,]*[01](?:[ \t,]*[01])*[ \t,]*")
_HAS_DIGIT = re.compile(r"[0-9]")
_HAS_LETTER = re.compile(r"[^\W\d_]")  # 任何字母（含非 ASCII）


def grid_to_text(grid: np.ndarray) -> str:
    """N×N 0/1 网格 → 文本：每行 N 个 0/1 字符，无分隔，行间换行，末尾无换行。"""
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"网格必须是 N×N，收到形状 {g.shape}")
    if not np.all((g == 0) | (g == 1)):
        raise ValueError("网格只能含 0/1")
    return "\n".join("".join("1" if v else "0" for v in row) for row in g.tolist())


def parse_grid_text(text: str, n: int) -> GridParse:
    """从模型输出里找恰好 N 行、每行恰好 N 个 0/1 的块。不补齐、不截断、不猜。

    行分类（逐行，去掉首尾空白后）：
    - 纯网格行：只含 0/1 与空格/逗号分隔；
    - 污染行：含数字、不含字母，但有其他字符（如 "0 1 2"、"[0,1]"、"0;1"）；
    - 文本行：空行、不含数字的行、含字母的行（说明文字、代码围栏等）。
    连续的非文本行组成一段；含至少一条纯网格行的段是候选块。
    恰好一个候选块时依次检查：含污染行 → ILLEGAL_CHAR；行数 ≠ N → WRONG_ROW_COUNT；
    某行数字个数 ≠ N → WRONG_COL_COUNT。没有候选块 → NO_GRID；多于一个 → MULTIPLE_BLOCKS。
    """
    if n <= 0:
        raise ValueError("n 必须为正")
    segments: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if _GRID_LINE.fullmatch(line):
            kind = "grid"
        elif line and _HAS_DIGIT.search(line) and not _HAS_LETTER.search(line):
            kind = "dirty"
        else:
            kind = "text"
        if kind == "text":
            if current:
                segments.append(current)
                current = []
        else:
            current.append((kind, line))
    if current:
        segments.append(current)

    candidates = [s for s in segments if any(k == "grid" for k, _ in s)]
    if not candidates:
        return GridParse(None, GridParseFailure.NO_GRID)
    if len(candidates) > 1:
        return GridParse(None, GridParseFailure.MULTIPLE_BLOCKS)
    block = candidates[0]
    if any(k == "dirty" for k, _ in block):
        return GridParse(None, GridParseFailure.ILLEGAL_CHAR)
    if len(block) != n:
        return GridParse(None, GridParseFailure.WRONG_ROW_COUNT)
    rows = [[int(c) for c in line if c in "01"] for _, line in block]
    if any(len(r) != n for r in rows):
        return GridParse(None, GridParseFailure.WRONG_COL_COUNT)
    return GridParse(np.array(rows, dtype=np.uint8), None)
