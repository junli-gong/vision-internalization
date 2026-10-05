"""C1′ 的 T 文本评测：解码时屏蔽网格 token（DEV-0036；D72-4，"C1′：T 文本解码时屏蔽网格 token，之后同理"引用基座的读数）。

run_e1.py --checkpoint … --mask-grid-tokens：自由生成（text 读法）时挂一个 logits 处理器，把检查点登记的网格 token
（adapter.grid_token_ids）的 logits 置为 −inf，其余不变。受限字母读数本来只在字母上取，不挂。检查点没有网格 token 时不挂处理器，
输出与不加这个选项逐字节相同；base 运行不能用这个选项（没有网格 token）。

C1′ 只训了网格 token 的嵌入行（LLM 全部冻结），屏蔽它们之后自由生成的分布就是基座的分布——这是"冻结 LLM 的对照引用基座的 T 文本读数、
抽查几百条确认相同"（D72-4）的前提。
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch

try:  # 与适配器相同的基类（HF generate 的处理器列表只认它）
    from transformers import LogitsProcessor as _LogitsProcessorBase
except ImportError:  # pragma: no cover — 开发机与服务器都装了 transformers
    _LogitsProcessorBase = object

RULE = ("自由生成（text 读法）的每一步把检查点登记的网格 token（adapter.grid_token_ids）的 logits 置为 −inf，其余 logits 不变"
        "（HF 的 logits 处理器，排在 greedy 的 argmax 之前）；受限字母读数不挂；检查点没有网格 token 时不挂（输出与不加选项逐字节相同）")


class GridTokenMask(_LogitsProcessorBase):
    """把给定的 token 的 logits 置为 −inf（无状态，可跨批复用）。"""

    def __init__(self, ids: Sequence[int]):
        self.ids = sorted({int(i) for i in ids})
        if not self.ids:
            raise ValueError("要屏蔽的 token 为空：没有网格 token 时不挂处理器")
        self._idx: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.ids[-1] >= scores.shape[-1]:
            raise RuntimeError(f"网格 token {self.ids[-1]} 超出 logits 宽度 {scores.shape[-1]}")
        if self._idx is None or self._idx.device != scores.device:
            self._idx = torch.tensor(self.ids, dtype=torch.long, device=scores.device)
        out = scores.clone()
        out[:, self._idx] = float("-inf")
        return out


def grid_ids_of(adapter: Any) -> list[int]:
    """适配器登记的网格 token（训练后的 E3b / C1′ 检查点才有；base 为空），升序。"""
    return sorted(int(v) for v in (getattr(adapter, "grid_token_ids", None) or {}).values())


def mask_record(adapter: Any) -> dict[str, Any]:
    """config.yaml 的 mask_grid_tokens 一节（运行身份）：屏蔽了多少个、是哪些（id 的 sha256）、规则。"""
    ids = grid_ids_of(adapter)
    return {"enabled": True, "n_ids": len(ids),
            "ids_sha256": hashlib.sha256(",".join(str(i) for i in ids).encode("utf-8")).hexdigest(),
            "rule": RULE, "basis": "D72-4（DEV-0036）"}


__all__ = ["GridTokenMask", "RULE", "grid_ids_of", "mask_record"]
