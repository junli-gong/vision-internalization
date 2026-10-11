[DEV-TASK 0045] 合并核对的病态检查点测试在 HPC（x86_64）上确定失败：查清是测试构造依赖平台舍入，还是核对本身在 x86 上变钝；修成跨平台成立

## 背景

- RUN-0044 在 ITSC HPC 上跑全量测试，有一条非计时的测试 4 次都失败：`tests/test_check_training_interfaces.py::test_grid_checkpoint_split_across_merge_fails_restricted`。
  - 环境：x86_64 Linux CPU，torch 2.13.0+cu130，numpy 2.5.2，transformers 5.17.0，peft 0.21.1；没有 causal_conv1d；fla 的 Triton 回退到 CPU。
  - 报错：`assert (0 == 1)`。病态检查点的 restricted 核对返回 0（通过），测试期望 1。
  - 同一条测试在开发机（Apple Silicon）和 B200 上都通过。
- 原始输出在 `experiments/E3/0044_hpc_bootstrap/raw/`：
  - `failtest_grid_checkpoint_split_check.json`：HPC 上那次核对的完整 JSON；
  - `failtest_grid_checkpoint_split_locals.txt`：pytest 输出。
  - 要点：across_merge 的教师强制 3171 个 token，argmax 一致率 1.0，0 处不一致；`max_abs_logit` 8.57，`max_abs_hidden` 3.45；by_thinking 也全部一致；样本 n = 3。
- 病态检查点的构造在 `tests/test_check_training_interfaces.py` 的 `_grid_checkpoint(pathological=True)`：
  - A 的两行几乎相同（差 1e-6），B 的两列相反并放大 1e7；
  - 真实的 ΔW 很小，但合并与不合并两条路径的中间量都很大、舍入不同，合并前后网格 token 的 logits 应当分家。
  - 是否分家取决于 BLAS 在大数相消时的舍入，可能因平台而异。
- **这件事为什么要紧**：后面 0.8B、4B 的评测和 E5 都改在 A100 上做（D91），起点与评测检查点的合并核对（判据 v1，D50-3）都在 HPC 上跑。要确认核对在 x86 上照样能查出合并前后分家，不是变钝了。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D50、D91
- `scripts/check_training_interfaces.py`、`src/adapters/checkpoint.py`（`compare_merge_sides`）
- `tests/test_check_training_interfaces.py` 第 394 行起的 `_grid_checkpoint` 与相关测试；`tests/test_e3_eval.py` 的 `bad_checkpoint`（同一种构造）

## 任务

1. **定性**：在开发机上量出病态构造的分家幅度。
   - 量什么：合并前后网格 token 的 logits 差、allowed 集合上的 argmax 不一致数、中间量的量级。
   - 再模拟 x86 的情形：例如改 BLAS 线程数、换 matmul 的累加顺序或精度，或用纯 numpy 的参考实现，看分家是不是只靠特定的舍入才出现。
   - 由此判断：失败的原因是测试构造依赖平台，还是核对的逻辑在 x86 上漏检。后者是 bug，要修核对本身。
2. **写一个 CPU 上一分钟内跑完的诊断脚本**，放在 `scripts/diag/`，由 A100 的 CC 在 HPC 上跑：
   - 打印病态构造在本机的分家幅度；
   - 打印一个"确定分家"的对照：不依赖舍入，例如合并侧与不合并侧的 token 行或 LoRA 明确不同。核对必须判不过它；
   - 输出 JSON，可以拿来和开发机的结果逐项对照。
3. **把测试修成跨平台成立**：
   - 病态检查点改成不依赖特定舍入也一定分家的构造，同时仍然是"合并后的模型内部自洽、只有 across_merge 查得出"的那一类；
   - 或者在依赖舍入的测试里先断言分家确实发生了（幅度超过判据阈值），没发生就 skip 并写明平台原因。另加一条确定分家的测试，任何平台都必须判不过。
   - `tests/test_e3_eval.py` 的 `bad_checkpoint` 是同一种构造，一并检查。
4. 不改判据 v1 的阈值与检查项。如果你认为核对本身需要改，先停下，写进完成总结，由管理员定。

## 验收

- `conda run -n visint pytest tests/ -q` 全过（计时类测试在 CPU 争用下失败的，单独重跑确认）。
- 完成总结写明：
  - 定性结论与依据（数字）；
  - 诊断脚本的用法与开发机上的输出；
  - 测试改了什么，为什么在 x86 上也成立；
  - 给 A100 的 CC 的一行运行命令：在 hpcmgt 上用 `$VI/env/bin/python`；或在节点上经 `srun --overlap`，不用 GPU。

只在自己的 worktree 里改，不 commit、不 push。
