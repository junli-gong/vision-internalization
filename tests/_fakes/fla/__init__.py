"""假的 flash-linear-attention（fla）：让开发机按服务器的内核绑定口径跑测试（DEV-0003-a1）。

模拟的是什么
    服务器的 venv 装了 fla 0.5.2（RUN-0001）。transformers 5.17 在 import `modeling_qwen3_5` 时执行
    `importlib.import_module("fla")`，把 Qwen3.5 的两个 Gated DeltaNet 模块级函数
    `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` 绑到
    `fla.ops.gated_delta_rule.{chunk_gated_delta_rule, fused_recurrent_gated_delta_rule}`（docs/model_notes.md §5.6.3）。
    这是 import 时一次决定的，与张量在哪个设备无关；而 fla 的 Triton 内核只收 GPU 张量，见到 CPU 张量就抛
    `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`。
    本包提供同名、同模块路径的两个函数，行为只模拟这一点：见到非 CUDA 张量就抛同一句错。

为什么要有它
    开发机没装 fla，transformers 退回 torch 实现，CPU 上的小模型测试全绿；服务器上同一批测试走进 Triton 内核，
    RUN-0003 因此 16 个红（reports/0003-g0-smoke-qwen.md）。把本目录的上一级（tests/_fakes）放进 PYTHONPATH，
    开发机上 transformers 的绑定就与服务器相同：那 16 个失败用它在开发机上逐条复现过。适配器在 CPU 上把 GDN
    显式换成 torch 实现（src/adapters/qwen35.py 的 torch_gdn / gdn_scope）之后，这个口径下也应全绿。

怎么用
    只通过 scripts/pytest_server_parity.sh 使用。普通的 `pytest tests/` 不会加载本包：它不在 sys.path 上，
    pytest 也只 import test_*.py；tests/test_adapter_qwen35.py 里有测试断言这一点。

不模拟什么
    不是真内核，没有任何数值实现；收到 GPU 张量时直接报错（开发机上不该发生）。causal-conv1d 两台机器都没装，
    这里也不提供。
"""

__version__ = "0.0.0+fake"
