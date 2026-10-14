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

## hpcgpu06（2026-10-14 13:06–13:24 UTC，RUN-0045 的 06 队列 12:09:57Z 跑完之后，RUN-0052 之前）

- 用的是同一个工作树 `$VI/wt/wt0049`（`59f086f8`）与同一个 `r49_node.sh`，经 `srun --jobid=157296 --overlap -n1` 在 hpcgpu06 上运行，没有用 GPU。命令 rc=0，868 s，日志 `0049_hpcgpu06.log`。
- squeue：`13:06:48Z 157296 RUNNING 60-00:32:34 / 157297 RUNNING 61-22:04:43`；跑完之后 `13:21:23Z 157296 RUNNING 60-00:47:09 / 157297 RUNNING 61-22:19:18`。

### 诊断：`diag_rc=0`

- 平台：`x86_64 Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz，torch 2.13.0+cu130（mkl，AVX512，8 线程）`，与 hpcgpu07 相同。
- **verdict：** `{"split_splits_here": true, "control_refused": true, "ok": true}`，输出 `msp_0049_hpcgpu06.json`。
- **四个情形的 across_merge**（本机 | 开发机）：

  | 情形 | argmax 不同的位置数 | max_abs_logit |
  |---|---|---|
  | split_run0044 | **0** \| 31 | **8.574** \| 12.32 |
  | split | 3149 \| 2714 | 30.56 \| 29.06 |
  | conditioned | 0 \| 0 | 1.335e-05 \| 1.049e-05 |
  | control（check） | 1057 \| 1057；exit_code 1 \| 1 | 22.28 \| 22.28 |

- **复现了 RUN-0044 的读数**：split_run0044 在 06 上 0 处不同，max_abs_logit 8.574，与任务书的预期相同。
- 两台节点 JSON 里的 `cases`：**除 `timing_s` 外逐项相同**（脚本比对）。不同的只有 6 个耗时字段（07 \| 06，单位 s）：
  - split：total 21.40 \| 21.49，decode 10.02 \| 10.05；
  - conditioned：total 4.13 \| 4.17；
  - control：total 13.92 \| 14.31；
  - split_run0044：total 86.59 \| 377.22，decode 74.29 \| 363.31。
  - 更正：commit 5361d8b7 的说明写成了 "cases identical to 07"，不准确，准确的说法是上面这句。
- 指纹的情况与 07 相同：输入与目标的 sha256 和开发机相同，基座、LoRA、token 行的指纹不同。

### pytest

- **第一次：`1 failed, 27 passed`（`pytest_rc=1`）。** 失败的是 `tests/test_check_training_interfaces.py::test_v1_rejudges_run0021_all_pass`：

  ```
  FileNotFoundError: [Errno 2] No such file or directory: '/home/mail/chenyuma/visint/wt/wt0049/experiments/E3/0021_train_iface_check/gemma-4-e2b-it.json'
  ```

  - 原因是 RUN-0053 已把 wt0049 改成稀疏签出，只保留了 `experiments/P0/`；这条测试要读的 `experiments/E3/0021_train_iface_check/` 不在工作树里。07 那次是在 RUN-0053 之前跑的。
- **补跑**（`r49b.sh`）：
  - 在同一个 commit `59f086f8` 上临时建了 `wt0049b`，稀疏规则多保留 `/experiments/E3/0021_train_iface_check/`（11 个文件），status 为空；
  - 在 06 上经 srun 只重跑这一条：`1 passed in 5.01s`，`pytest_rc=0`；
  - 跑完删掉 wt0049b 和它的 basetemp。前后 squeue 都是 RUNNING（`13:23:55Z`、`13:24:31Z`）。
- 两次合起来，06 上 28 条都通过。

### 收尾

wt0049 已删除（`git worktree remove`，删前 status 为空，没有进程在用）。
