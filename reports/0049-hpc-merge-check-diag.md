# RUN-0049 — 在 HPC 上复核合并核对（DEV-0045 诊断 + 两组测试，只用 CPU）

服务器 CC（HPC）。只贴事实，不做解读。文件在 `experiments/E3/0049_hpc_merge_check_diag/`。

## hpcgpu07（2026-10-12 10:36–10:43 UTC，RUN-0045 的 07 队列 09:32:06Z 跑完之后，RUN-0048 之前）

### 准备（`r49_prep.out`）

- `flock … git -C $VI/repo fetch -q origin main`：rc=0，4 s。origin/main `59f086f8`，包含 `71e2ab3`（DEV-0045 合入）与 `e374d70`（本任务书）。
- `git -C $VI/repo worktree add --detach $VI/wt/wt0049 origin/main`：HEAD `59f086f8f02731531a2e903491eedc404c07fde4`，`status --short` 为空。
- 命令照任务书 §3 写进 `r49_node.sh`：环境变量一行一个，`CUDA_VISIBLE_DEVICES=` 空，`PATH=$VI/env/bin:/usr/bin:/bin`。在 hpcgpu07 上经 `srun --jobid=157297 --overlap -n1` 运行，没有用 GPU，没有写暂停文件。

### squeue

```
## squeue 2026-10-12T10:12:44Z [0049 before]
157296 RUNNING 57-21:38:30 hpcgpu06
157297 RUNNING 59-19:10:39 hpcgpu07
## squeue 2026-10-12T10:43:15Z [0049-07 after]
157296 RUNNING 57-22:09:01 hpcgpu06
157297 RUNNING 59-19:41:10 hpcgpu07
```

### 诊断：`diag_rc=0`

- 平台：`x86_64 Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz，torch 2.13.0+cu130（mkl，AVX512，8 线程）`。对照是开发机：`arm64 Apple M4，torch 2.14.0（accelerate，DEFAULT）`。
- **verdict：** `{"split_splits_here": true, "control_refused": true, "ok": true}`，用时 126.5 s，输出 `msp_0049_hpcgpu07.json`。
- **四个情形的 across_merge**（`--compare` 打印的原文，本机 | 开发机，全文见 `0049_hpcgpu07.log`）：

  | 情形 | argmax 不同的位置数（共 3171，control 共 1057） | max_abs_logit |
  |---|---|---|
  | split_run0044 | **0** \| 31 | **8.574** \| 12.32 |
  | split | 3149 \| 2714 | 30.56 \| 29.06 |
  | conditioned | 0 \| 0 | 1.335e-05 \| 1.049e-05 |
  | control（check） | 1057 \| 1057；exit_code 1 \| 1；pass False \| False；failures_only_across_merge True \| True | 22.28 \| 22.28 |

- **与任务书预期不符之处（原样贴出）：** 任务书写"`split_run0044` 的输入与 LoRA 指纹和开发机相同"。`--compare` 的输出是：

  ```
  split_run0044.construction.base_fingerprint | de634628a9a7 | 707f89df27e5  ≠
  split_run0044.construction.lora_sha256 | a8aea28c6f17 | c4b5558a5066  ≠
  split_run0044.construction.token_rows_sha256 | c7b5895b28f5 | b5703127dda9  ≠
  split_run0044.inputs_sha256 | ca3c97634867 | ca3c97634867  =
  split_run0044.targets.sha256 | 4b02a409d25f | 4b02a409d25f  =
  ```

  - 也就是：输入与目标相同；基座指纹、LoRA 与 token 行的 sha256 和开发机不同。
  - `split`、`conditioned` 两个情形的这三项同样是 ≠，输入与目标同样是 =。
- 关于 hpcgpu06：任务书预期 06 会复现 RUN-0044 的读数（0 处不同，max_abs_logit 8.57）。hpcgpu07 这次就是 0 与 8.574。06 那一次等它的 4B 队列跑完再做。

### pytest：`pytest_rc=0`

- 命令：`-rfEs tests/test_check_training_interfaces.py "tests/test_e3_eval.py::test_bad_checkpoint_refused"`。
- 汇总行：`28 passed, 15 warnings in 210.52s (0:03:30)`，没有 skip、没有失败。

## hpcgpu06

等 RUN-0045 的 06 队列跑完再跑，结果追加到这里。
