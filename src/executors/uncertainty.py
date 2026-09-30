"""按不确定度选区域（无学习；只用 numpy、scipy）。

与任务无关（简报 §4 第 11 条、N8）：输入只是逐像素分布，函数不接受任何"任务类型"参数；
mask 的前景概率和深度的 64 档分布走同一个函数，不确定度一律用熵。
窗口格式见 constants.PIXEL_WINDOW_FORMAT：(r0, r1, c0, c1)，半开。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import entr

from src.constants import (
    UNCERTAINTY_ENTROPY_QUANTUM,
    UNCERTAINTY_NUM_REGIONS,
    UNCERTAINTY_PROB_SUM_ATOL,
    UNCERTAINTY_REGION_SIZE,
)


class Region(NamedTuple):
    """选中的正方形区域：(r0, r1, c0, c1) 半开窗口，加上窗口内的平均熵（nats，由量化后的整数和换算）。"""

    r0: int
    r1: int
    c0: int
    c1: int
    mean_entropy: float


def entropy_map(dist: np.ndarray) -> np.ndarray:
    """逐像素熵（nats），形状 (H, W)，float64。0·log0 记 0。

    - 二维 (H, W)：二值情形，值是前景概率 p，熵 = −p·log p − (1−p)·log(1−p)；
    - 三维 (H, W, C)：多类情形，最后一维是分布（C ≥ 2），熵 = −Σ p_c·log p_c。
      (H, W) 的 p 与 (H, W, 2) 的 [1−p, p] 给出逐位相同的熵。
    概率必须有限且在 [0, 1] 内；多类的和须在 1 ± UNCERTAINTY_PROB_SUM_ATOL 内，否则报错，不静默归一化。
    输入先转成 C 连续：numpy 沿连续轴求和用成对求和、沿非连续轴不用，不转的话多类熵的末位随内存布局变。
    """
    p = np.asarray(dist)
    if p.dtype.kind != "f":
        raise ValueError(f"分布必须是浮点数组，收到 dtype {p.dtype}")
    p = np.ascontiguousarray(p, dtype=np.float64)
    if p.ndim not in (2, 3) or p.shape[0] < 1 or p.shape[1] < 1:
        raise ValueError(f"分布必须是 (H, W) 或 (H, W, C)，收到形状 {p.shape}")
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("概率必须是有限值且在 [0, 1] 内")
    if p.ndim == 2:
        return entr(p) + entr(1.0 - p)
    if p.shape[2] < 2:
        raise ValueError(f"多类分布至少要 2 类，收到 C={p.shape[2]}")
    err = np.abs(p.sum(axis=2) - 1.0)
    if err.max() > UNCERTAINTY_PROB_SUM_ATOL:
        i, j = np.unravel_index(int(np.argmax(err)), err.shape)
        raise ValueError(f"像素 ({i}, {j}) 的概率和偏离 1 达 {err[i, j]:.3g}，超过容差 {UNCERTAINTY_PROB_SUM_ATOL}")
    return entr(p).sum(axis=2)


def select_uncertain_regions(dist: np.ndarray, k: int = UNCERTAINTY_NUM_REGIONS, size: int = UNCERTAINTY_REGION_SIZE) -> list[Region]:
    """选出至多 k 个互不重叠、完全在图内的 size×size 正方形区域，按窗口内熵的总和从大到小贪心选（非极大抑制）。

    - 得分：逐像素熵量化为 UNCERTAINTY_ENTROPY_QUANTUM 的整数倍（四舍六入五成双），窗口内求整数和。
      整数和精确，内容相同的窗口得分严格相等。
    - 候选：左上角 (r0, c0) 取遍 0..H−size、0..W−size 的所有位置，所以区域不越界。
    - 每轮取得分最大的候选；平局取左上角 (r0, c0) 按行优先（先比 r0 再比 c0）最小的那个。
      选中后，与它有任何像素重叠的候选（|Δr0| < size 且 |Δc0| < size）全部作废。
    - 得分为 0（完全确定）的窗口照样可选：只要还有不重叠的位置，就选满 k 个，每张图的细化开销固定。
    - k 大于可选区域数：所有候选都作废后提前停止，返回的区域少于 k 个，不报错。k=0 返回空列表。
    - size 大于图的高或宽时报错。
    返回按选中顺序排列的 Region 列表。
    """
    e = entropy_map(dist)
    if isinstance(k, bool) or int(k) != k or k < 0:
        raise ValueError(f"k 必须是非负整数，收到 {k}")
    if isinstance(size, bool) or int(size) != size or size < 1:
        raise ValueError(f"size 必须是正整数，收到 {size}")
    k, size = int(k), int(size)
    h, w = e.shape
    if size > h or size > w:
        raise ValueError(f"区域边长 {size} 大于图的尺寸 ({h}, {w})")
    # 整数和的溢出上界：每像素熵 ≤ log(类数)
    n_classes = 2 if np.ndim(dist) == 2 else np.shape(dist)[2]
    if h * w * (np.log(n_classes) / UNCERTAINTY_ENTROPY_QUANTUM + 1.0) >= 2.0**62:
        raise ValueError("图太大，量化后的熵之和可能溢出 int64")
    q = np.rint(e / UNCERTAINTY_ENTROPY_QUANTUM).astype(np.int64)
    sat = np.zeros((h + 1, w + 1), dtype=np.int64)
    sat[1:, 1:] = q.cumsum(axis=0).cumsum(axis=1)
    score = sat[size:, size:] - sat[:-size, size:] - sat[size:, :-size] + sat[:-size, :-size]  # (h−size+1, w−size+1)
    score = score.copy()
    regions: list[Region] = []
    for _ in range(k):
        flat = int(np.argmax(score))  # 第一个最大值，即行优先最小的 (r0, c0)
        best = int(score.flat[flat])
        if best < 0:
            break
        r0, c0 = divmod(flat, score.shape[1])
        regions.append(Region(r0, r0 + size, c0, c0 + size, best * UNCERTAINTY_ENTROPY_QUANTUM / (size * size)))
        score[max(0, r0 - size + 1) : r0 + size, max(0, c0 - size + 1) : c0 + size] = -1
    return regions
