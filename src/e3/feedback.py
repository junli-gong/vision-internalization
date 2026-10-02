"""叠加反馈图（简报 §3.8、§8.5；§12"训练和评测用同一套渲染参数"）：把一个网格画回原图，作为下一轮的第二张输入图。

训练的修正对（src/e3/data.py）与评测的叠加反馈迭代（第二部分）都只调 feedback_image，渲染参数没有入口可改：
- 分割：网格先按 codec.grid_to_mask 精确回填到原图分辨率（格边界 cell_edges，最近邻，不做引导上采样——反馈图要显示"模型实际
  输出的网格"本身），再交给 src/executors/overlay.render_overlay（颜色、alpha、轮廓都取自 src/constants，第 0 种颜色）。
- 深度：网格同样按 cell_edges 回填（每格一个档号），档号经 DEPTH_COLORMAP 换成伪彩色，与原图逐像素做 alpha 混合，alpha 与舍入规则
  与 render_overlay 的填充相同（OVERLAY_ALPHA_NUM / OVERLAY_ALPHA_DEN，整数运算，恰为 .5 时进位）；整张图都覆盖（深度网格没有背景），
  不画轮廓。
- DEPTH_COLORMAP（64 × RGB，uint8）：HSV 色相环上的等距色带，档 0（最近）为蓝（色相 240°），末档（最远）为红（0°），饱和度与明度都是 1；
  按 HSV → RGB 的标准公式算（colorsys 同一公式），通道值 round(255·c)（.5 进位）。相邻档的色差（RGB 的 L1）≥ 16、64 色两两不同，
  测试钉住；表的 sha256 也钉住。暂定，E3b 预注册前定稿（D14-3 的同类项）。
"""

from __future__ import annotations

import numpy as np

from src.constants import DEPTH_NUM_BINS, OVERLAY_ALPHA_DEN, OVERLAY_ALPHA_NUM
from src.executors.overlay import render_overlay
from src.grid.codec import cell_edges, grid_to_mask

DEPTH_HUE_NEAR_DEG = 240.0  # 档 0 的色相（蓝）
DEPTH_HUE_FAR_DEG = 0.0  # 末档的色相（红）


def _colormap(n: int) -> np.ndarray:
    """HSV（S = V = 1）色相从 DEPTH_HUE_NEAR_DEG 等距走到 DEPTH_HUE_FAR_DEG 的 n 色表（标准 HSV → RGB）。"""
    h = np.linspace(DEPTH_HUE_NEAR_DEG, DEPTH_HUE_FAR_DEG, n, dtype=np.float64) / 60.0
    x = 1.0 - np.abs(np.mod(h, 2.0) - 1.0)
    sector = np.minimum(np.floor(h).astype(np.int64), 5)
    one, zero = np.ones_like(h), np.zeros_like(h)
    table = {0: (one, x, zero), 1: (x, one, zero), 2: (zero, one, x), 3: (zero, x, one), 4: (x, zero, one), 5: (one, zero, x)}
    rgb = np.zeros((n, 3), dtype=np.float64)
    for k, (r, g, b) in table.items():
        m = sector == k
        rgb[m] = np.stack([r[m], g[m], b[m]], axis=1)
    return np.floor(rgb * 255.0 + 0.5).astype(np.uint8)


DEPTH_COLORMAP = _colormap(DEPTH_NUM_BINS)
DEPTH_COLORMAP.setflags(write=False)


def _check_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image 必须是 (H, W, 3) uint8，收到 dtype {img.dtype}、形状 {img.shape}")
    return img


def upsample_cells(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """N×N 网格 → (H, W)：每个格子按 cell_edges 的边界原样铺满（与 grid_to_mask 同一规则，取值不限于 0/1）。"""
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"网格必须是 N×N，收到形状 {g.shape}")
    n = g.shape[0]
    rows = np.diff(cell_edges(int(height), n))
    cols = np.diff(cell_edges(int(width), n))
    return np.repeat(np.repeat(g, rows, axis=0), cols, axis=1)


def seg_feedback_image(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """分割网格（N×N 0/1）画回原图（规则见模块说明）。"""
    img = _check_image(image)
    mask = grid_to_mask(np.asarray(grid), img.shape[0], img.shape[1])
    return render_overlay(img, [mask])


def depth_feedback_image(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """深度档号网格（N×N，0..DEPTH_NUM_BINS-1）画回原图（规则见模块说明）。"""
    img = _check_image(image)
    g = np.asarray(grid)
    if not np.issubdtype(g.dtype, np.integer):
        raise ValueError(f"深度网格必须是整数档号，收到 dtype {g.dtype}")
    if g.size and (int(g.min()) < 0 or int(g.max()) >= DEPTH_NUM_BINS):
        raise ValueError(f"深度档号越界：[{g.min()}, {g.max()}]")
    color = DEPTH_COLORMAP[upsample_cells(g.astype(np.int64), img.shape[0], img.shape[1])].astype(np.int64)
    num, den = int(OVERLAY_ALPHA_NUM), int(OVERLAY_ALPHA_DEN)
    out = (img.astype(np.int64) * (den - num) + color * num + den // 2) // den
    return out.astype(np.uint8)


def feedback_image(task: str, image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """训练与评测共用的唯一入口：task = refseg / depth。"""
    if task == "refseg":
        return seg_feedback_image(image, grid)
    if task == "depth":
        return depth_feedback_image(image, grid)
    raise ValueError(f"任务 {task!r} 没有叠加反馈图")


__all__ = ["DEPTH_COLORMAP", "depth_feedback_image", "feedback_image", "seg_feedback_image", "upsample_cells"]
