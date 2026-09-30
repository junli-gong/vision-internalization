[DEV-TASK 0003 addendum-1] Qwen3.5 适配器：CPU 上显式走 GDN 的 torch 实现（修"开发机绿、算力机红"）

## 背景

RUN-0003 在服务器上跑全量测试，结果是 16 failed / 295 passed / 1 skipped，而开发机是 312 passed。详见 `reports/0003-g0-smoke-qwen.md` 和 `experiments/P0/0003_g0_smoke_qwen/pytest.txt`。

原因已经由服务器 CC 诊断清楚：
- 服务器的 venv 装了 flash-linear-attention（fla）0.5.2。transformers 5.17 在 **import 时** 就把 `modeling_qwen3_5` 模块里的 `torch_chunk_gated_delta_rule` 和 `torch_recurrent_gated_delta_rule` 绑定到 fla 的 Triton 内核，与张量在哪个设备无关（model_notes §5.6.3）。
- 测试和 `smoke --tiny` 里的小型随机 Qwen3.5 在 CPU 上跑，CPU 张量进了 Triton 内核，报 `Pointer argument cannot be accessed from Triton (cpu tensor?)`。
- 开发机没装 fla，transformers 退回 torch 实现，所以开发机上是绿的。
- 服务器上屏蔽 `import fla` 后是 311 passed + 1 skipped（cv2 未装），条数与开发机对得上。

这是 lessons"跨机器差异"一类的问题：同一套测试在两台机器上走了不同的内核路径。

先读：
- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `lessons.md` 的"跨机器差异"
- 上面的 report 与 pytest.txt
- `docs/model_notes.md` §5.6
- `src/adapters/qwen35.py`

## 任务

1. 在 `src/adapters/qwen35.py` 里，让适配器**在 CPU 设备上的每次前向与生成都显式使用 GDN 的 torch 实现**，GPU 上保持现状（fla）。
   - 做法：一个上下文管理器。进入时，把 `transformers.models.qwen3_5.modeling_qwen3_5` 里这两个模块级函数临时换成它们的 torch 原函数（`__wrapped__`，见 model_notes §5.6.2 的做法）；退出时一定恢复，包括异常退出。
   - 适配器的 `_run`、`vision_features`、`generate` 在 `self.device.type == "cpu"` 时套上它。
   - 只替换这两个函数。`causal_conv1d_fn` 等在两台机器上本来就都走 torch，不动。
   - 取 `__wrapped__` 失败，或者换完后不是预期的 torch 函数，就报错，不要静默继续。
   - 这是全局状态：写明线程安全上的限制（本项目单进程单线程使用），并确认嵌套进入时的行为正确。
2. 新增一个只读方法：给出当前前向会走的 GDN 实现，包括函数的模块路径与名字。冒烟脚本的检查 0 已经记录实现名，确认它在 CPU 路径上记的是切换之后的实现；如需改冒烟脚本，只动检查 0 里的这一处。
3. tests：
   - 加一个测试，在开发机上**模拟"装了 fla"**：先把模块级函数替换成一个"不接受 CPU 张量就抛错"的桩，再跑一次 CPU 前向，证明适配器的上下文管理器确实把它换走了，并且退出后恢复原绑定。
   - 原有的 69 条 Qwen 测试照常全绿。
   - 用变异确认：删掉上下文管理器，或者只在 `_run` 里套、`generate` 里不套，新测试就会失败。

## 允许改动的文件

- `src/adapters/qwen35.py`
- `tests/test_adapter_qwen35.py`
- `scripts/smoke_adapter.py`：只动检查 0 里记录实现名的地方；并行的 DEV-0004/0005 会在分派处加入口，**不要碰分派那部分**。

**不许碰**：`src/adapters/base.py`、其他适配器、`src/constants.py`、`requirements.txt`。

## 验收

- `conda run -n visint pytest tests/ -q` 全绿，报告测试数与耗时。
- `smoke_adapter.py --tiny` 退出码 0，检查 0 里记录的 GDN 实现是 torch 版。
- 附变异结果。逐条列出偏离。
