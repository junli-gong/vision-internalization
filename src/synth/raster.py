"""E5 合成数据的精确整数光栅化（DEV-0009，D24）：多边形 → (H, W) bool mask，结果与平台无关。

规则（唯一定义在这里）
- 像素 (i, j) 的中心是 (j + 0.5, i + 0.5)（src.constants.PIXEL_CENTER_OFFSET）。像素在多边形内 ⟺ 中心在多边形内。
- "在内"用奇偶规则：从中心向左的水平射线与多边形的边相交奇数次。多边形不自交时，奇偶规则与非零规则相同。
- 中心恰好落在边上时，边界归属与 src.constants.BOX_RASTER_RULE 相同，取 (lo, hi]：
  * 一条边只在 min(y_a, y_b) < y ≤ max(y_a, y_b) 的扫描线上计数。水平边从不计数。扫描线过顶点时，
    穿过的顶点恰好计一次，擦过的顶点计零次或两次。
  * 交点 X 严格在中心左边（X < x）才计数；中心恰好在边上时，这条边不计数。
  * 于是轴对齐矩形 [x0, x1] × [y0, y1] 的结果是 x0 < cx ≤ x1 且 y0 < cy ≤ y1，与 src.executors.raster.box_to_mask
    逐位相同。两个共用一条边的多边形，把边上的像素中心分给恰好一方，不重不漏。
  * 结果只取决于多边形围成的区域：与顶点顺序、起点、方向、共线顶点的增删都无关。
- 顶点必须落在 COORD_QUANTUM = 1/256 px 的网格上（shapes.pose_polygon 的输出都是），否则抛 ValueError。
  坐标乘 256 以后是精确整数。交点判定 X < x ⟺ (x - x_a)(y_b - y_a) > (y - y_a)(x_b - x_a)（y_a < y_b）全部用 int64 计算，
  整条路径上没有浮点运算。结果是这些整数的数学函数，与平台、编译器、numpy 版本都无关。
  坐标限制在 |x|, |y| ≤ MAX_ABS_COORD_PX，保证中间量 < 2^53，不溢出。
- 只在多边形的包围盒窗口里计算，与在整幅画布上逐像素判定逐位相同（tests/test_synth_raster.py 钉住）。
  多边形可以部分或全部在画布外。

算法
- 对每条边，精确求出它穿过的像素行 i（中心 y = 256·i + 128，单位 1/256 px）。
- 在每个穿过点上，求出第一个中心严格在交点右边的列 j* = floor(T / (256·Δy)) + 1，
  其中 T = (y - y_a)(x_b - x_a) + (x_a - 128)·Δy。从 (i, j*) 起，这一行右边的像素都多计一次。
- 把窗口按行展平，翻转位置 f = 行·窗口宽 + clip(j* - 窗口左端, 0, 窗口宽)，排序后相邻两个翻转之间交替为外、内，
  一次 np.repeat 写出整个窗口。每一行的翻转次数都是偶数（闭合多边形），所以奇偶状态不会串到下一行；
  交点在窗口右边的翻转落在下一行行首，正好把状态复位。

与 pycocotools 的关系
- src.executors.raster.polygons_to_mask 走 pycocotools：先把坐标放大 5 倍取整，沿边描点，再下采样。
- 它是编译过的 C 代码：(int)(ys + s*t + .5) 这类表达式在 arm64 上可能被编译器融合成 FMA，在 x86_64 上不会，
  取整前的末位差异就会改变边界像素。RUN-0006 的 occlusion/test/2 场景规格相同而光栅化不同，与此相符。
- 两者只在离边界很近的像素上不同，统计见 tests/test_synth_raster.py。
- src/constants.py 的 POLYGON_RASTERIZER = "pycocotools" 仍适用于 COCO/RefCOCO 真值；E5 合成数据一律用这里的规则。
"""

from __future__ import annotations

import numpy as np

from src.constants import PIXEL_CENTER_OFFSET
from src.synth.constants import COORD_QUANTUM

# 1 px = UNITS 个整数单位。由 COORD_QUANTUM 导出，不另写数字。
UNITS = int(round(1.0 / COORD_QUANTUM))
if UNITS * COORD_QUANTUM != 1.0 or UNITS & (UNITS - 1):  # pragma: no cover - 常量写错
    raise ImportError(f"COORD_QUANTUM 必须是 2 的负整数次幂，收到 {COORD_QUANTUM!r}")
# 像素中心相对像素左上角的偏移，整数单位。
HALF = int(round(PIXEL_CENTER_OFFSET * UNITS))
if HALF != PIXEL_CENTER_OFFSET * UNITS:  # pragma: no cover
    raise ImportError("PIXEL_CENTER_OFFSET 乘以 UNITS 必须是整数")

# 坐标上限（像素）。2^17 px 时整数单位 < 2^25，中间量 |T| < 2^53，int64 远不会溢出。
MAX_ABS_COORD_PX = float(1 << 17)
_MAX_ABS_UNITS = int(MAX_ABS_COORD_PX) * UNITS


def to_units(poly) -> np.ndarray:
    """多边形 (V, 2) 像素坐标 → int64 整数单位（精确）。不在 1/256 px 网格上、非有限或超出范围时抛 ValueError。"""
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
        raise ValueError(f"多边形必须是形状 (V, 2)、V ≥ 3 的数组，收到形状 {p.shape}")
    u = p * float(UNITS)  # 乘 2 的幂：没有舍入
    if not np.all(np.isfinite(u)):
        raise ValueError("多边形坐标含非有限值")
    if np.any(np.abs(u) > _MAX_ABS_UNITS):
        raise ValueError(f"多边形坐标超出 ±{MAX_ABS_COORD_PX:g} px")
    q = u.astype(np.int64)
    if not np.array_equal(q, u):
        bad = p[np.any(q != u, axis=1)][0]
        raise ValueError(f"多边形顶点必须落在 1/{UNITS} px 的网格上（先用 shapes.qcoord 量化），例如 {bad.tolist()}")
    return q


def rasterize(poly, height: int, width: int) -> np.ndarray:
    """多边形（像素坐标，顶点在 1/256 px 网格上）→ (H, W) bool mask。规则见模块说明。"""
    return rasterize_units(to_units(poly), height, width)


def rasterize_units(q: np.ndarray, height: int, width: int) -> np.ndarray:
    """多边形（int64 整数单位，形状 (V, 2)）→ (H, W) bool mask。只接受 int64：入口处拒绝浮点输入。"""
    if not isinstance(q, np.ndarray) or q.dtype != np.int64:
        raise TypeError(f"rasterize_units 只接受 int64 数组（整数单位），收到 {getattr(q, 'dtype', type(q))}")
    if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] < 3:
        raise ValueError(f"多边形必须是形状 (V, 2)、V ≥ 3 的数组，收到形状 {q.shape}")
    if int(np.abs(q).max()) > _MAX_ABS_UNITS:
        raise ValueError(f"多边形坐标超出 ±{MAX_ABS_COORD_PX:g} px")
    h, w = int(height), int(width)
    if h <= 0 or w <= 0 or h != height or w != width:
        raise ValueError(f"非法画布尺寸 H={height}, W={width}")
    out = np.zeros((h, w), dtype=bool)

    ax, ay = q[:, 0], q[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    keep = ay != by  # 水平边从不计数
    ax, ay, bx, by = ax[keep], ay[keep], bx[keep], by[keep]
    swap = ay > by  # 让每条边 y0 < y1（交点与方向无关）
    x0 = np.where(swap, bx, ax)
    y0 = np.where(swap, by, ay)
    x1 = np.where(swap, ax, bx)
    y1 = np.where(swap, ay, by)

    # 这条边穿过的像素行：y0 < UNITS·i + HALF ≤ y1，并裁到画布内
    i_lo = np.maximum((y0 - HALF) // UNITS + 1, 0)
    i_hi = np.minimum((y1 - HALF) // UNITS, h - 1)
    cnt = np.maximum(i_hi - i_lo + 1, 0)
    total = int(cnt.sum())
    if total == 0:
        return out

    # 窗口列：中心在 (x_min, x_max] 内的列才可能在内
    j0 = max(int((q[:, 0].min() - HALF) // UNITS) + 1, 0)
    j1 = min(int((q[:, 0].max() - HALF) // UNITS), w - 1)
    if j1 < j0:
        return out
    ww = j1 - j0 + 1

    e = np.repeat(np.arange(cnt.size), cnt)  # 每个穿过点所属的边
    start = np.cumsum(cnt) - cnt
    rows = i_lo[e] + (np.arange(total, dtype=np.int64) - start[e])
    dy = (y1 - y0)[e]
    t = (rows * UNITS + HALF - y0[e]) * (x1 - x0)[e] + (x0[e] - HALF) * dy
    jstar = t // (UNITS * dy) + 1  # 第一个中心严格在交点右边的列（floor 除法，精确）

    r0 = int(rows.min())
    nr = int(rows.max()) - r0 + 1
    # 展平窗口里的翻转位置。交点在窗口左边：从行首翻转；在窗口右边：落在下一行行首（该行翻转次数为偶数，复位）。
    f = np.sort((rows - r0) * ww + np.clip(jstar - j0, 0, ww))
    lengths = np.diff(f, prepend=0, append=nr * ww)
    runs = (np.arange(f.size + 1) & 1).astype(bool)  # 外、内、外、……
    out[r0 : r0 + nr, j0 : j0 + ww] = np.repeat(runs, lengths).reshape(nr, ww)
    return out
