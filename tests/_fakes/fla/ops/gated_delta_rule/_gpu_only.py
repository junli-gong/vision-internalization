"""两个假内核共用的检查：与 fla 的 Triton 内核一样只收 GPU 张量。"""

import torch

# RUN-0003 服务器上 Triton 抛出的原话（experiments/P0/0003_g0_smoke_qwen/pytest.txt）
TRITON_CPU_ERROR = "Pointer argument cannot be accessed from Triton (cpu tensor?)"


def reject_non_cuda(*tensors):
    for t in tensors:
        if torch.is_tensor(t) and t.device.type != "cuda":
            raise ValueError(TRITON_CPU_ERROR)
    raise AssertionError("假 fla 内核收到了 GPU 张量：它没有数值实现，只该在开发机上用")
