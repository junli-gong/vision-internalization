"""与真 fla 相同的导出位置：transformers 从这里取两个 GDN 内核（说明见 fla/__init__.py）。"""

from .chunk import chunk_gated_delta_rule
from .fused_recurrent import fused_recurrent_gated_delta_rule

__all__ = ["chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"]
