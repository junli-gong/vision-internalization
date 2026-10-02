"""token 序列 ↔ token 网格 [H_t, W_t] 的映射（DEV-0023）。

约定（三个家族一致，D21-4 审查核实"视觉 token 都是行主序"）：一张图的视觉 token 在序列里按行主序排列，第 k 个 token 在网格的
第 k // W_t 行、第 k % W_t 列；网格形状取适配器给的 ImageTokenInfo.grid_hw（合并 / 下采样之后的网格，见 src/adapters/base.py）：

- Qwen3.5：(H'/32, W'/32)，H'、W' 是 smart resize 之后的尺寸（E1 口径 640×480 → (15, 20)）；
- Gemma 4：由 image_position_ids 推出（行 patch 数 / 3, 列 patch 数 / 3），E1 口径 640×480 → (14, 19)；
- InternVL3.5：单 tile 恒为 (16, 16)（整图缩放到 448×448）；多 tile 时 grid_hw 是 tile 的纵向堆叠、不是整图网格，
  适配器的 hidden_states 本身就拒绝多 tile（D19）。

E2 的输入一律一张图。len(span) 必须恰为 H_t·W_t（某些家族若在行间插分隔 token 会不等，这里直接报错，不猜）。
"""

from __future__ import annotations

from typing import Any

from src.adapters.base import ImageTokenInfo, ModelInputs


def single_image(inputs: ModelInputs) -> ImageTokenInfo:
    """E2 的输入只有一张图；返回它的 ImageTokenInfo，并核对 len(span) = h·w。"""
    if len(inputs.images) != 1:
        raise ValueError(f"E2 的输入必须恰好一张图，收到 {len(inputs.images)} 张")
    info = inputs.images[0]
    h, w = info.grid_hw
    if h * w != len(info.span):
        raise ValueError(f"图像 token 区间长度 {len(info.span)} ≠ 网格 {h}×{w}：序列里有非网格 token，不能直接排成网格")
    return info


def tokens_to_grid(x: Any, grid_hw: tuple[int, int]) -> Any:
    """[n, D] → [h, w, D]（行主序：第 k 个 token → (k // w, k % w)）。n 必须等于 h·w。"""
    h, w = (int(v) for v in grid_hw)
    if x.dim() != 2 or x.shape[0] != h * w:
        raise ValueError(f"token 张量形状 {tuple(x.shape)} 与网格 {h}×{w} 不符")
    return x.reshape(h, w, x.shape[1])


def grid_to_chw(g: Any) -> Any:
    """[h, w, D] → [D, h, w]（卷积探针的输入布局）。"""
    if g.dim() != 3:
        raise ValueError(f"网格张量必须是三维 [h, w, D]，收到 {tuple(g.shape)}")
    return g.permute(2, 0, 1).contiguous()


__all__ = ["grid_to_chw", "single_image", "tokens_to_grid"]
