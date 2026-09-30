"""假的 fla chunk 内核（prefill 与 use_cache=False 的整段前向走它）。签名取 fla 的参数名，transformers 按名过滤关键字。"""

from ._gpu_only import reject_non_cuda


def chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                           cu_seqlens=None, use_qk_l2norm_in_kernel=False, **kwargs):
    reject_non_cuda(q, k, v, g, beta, initial_state)
