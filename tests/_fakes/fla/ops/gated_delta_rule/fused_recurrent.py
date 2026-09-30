"""假的 fla fused_recurrent 内核（生成时逐 token 解码走它）。签名取 fla 的参数名，transformers 按名过滤关键字。"""

from ._gpu_only import reject_non_cuda


def fused_recurrent_gated_delta_rule(q, k, v, g=None, beta=None, scale=None, initial_state=None,
                                     output_final_state=False, cu_seqlens=None, use_qk_l2norm_in_kernel=False,
                                     **kwargs):
    reject_non_cuda(q, k, v, g, beta, initial_state)
