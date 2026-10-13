# RUN-0053 — 共享盘清理第一步：稀疏签出、清 tmp、git gc（D94；顺带执行 D95 的工作树更新）

服务器 CC（HPC）。2026-10-13 22:38–22:47 UTC。只记事实。原始输出：`experiments/E3/0053_hpc_disk_cleanup/step12.txt`、`step345.txt`；脚本是同目录的 `c1.sh`、`c2.sh`，在 hpcmgt 上以 `setsid nohup` 运行。

## 0. 结果

- **份额差值**（`lfs quota -u chenyuma` 第 3 行第 2 列减 7,059,403,936 KB）：清理前 **68,446,692 KB**，清理后 **56,503,580 KB**，减少 11,943,112 KB（约 11.9 GB）。
- 没有碰训练或评测驱动，也没有碰它们在用的工作树 wt0045、wt0050 与 wt/push。
- 前后 squeue 都是两条 RUNNING：

```
## squeue 2026-10-13T22:38:21Z [0053 before]
157296 RUNNING 59-10:04:07 hpcgpu06
157297 RUNNING 61-07:36:16 hpcgpu07
## squeue 2026-10-13T22:46:53Z [0053 after]
157296 RUNNING 59-10:12:39 hpcgpu06
157297 RUNNING 61-07:44:48 hpcgpu07
```

## 1. 稀疏签出（第 2 步）

命令是 `git -C $W sparse-checkout set --no-cone '/*' '!/experiments/*/' '/experiments/P0/'`。

| 工作树 | 使用中的进程 | status 前 / 后 | 大小前（B） | 大小后（B） | `experiments/` | P0 清单 sha256 |
|---|---|---|---|---|---|---|
| wt0049 | 无（hpcmgt、06、07） | 0 / 0 | 2,734,533,286 | 94,128,462 | P0 | `60267210d97f8145` |
| wt0051 | 无 | 0 / 0 | 2,972,526,227 | 91,403,904 | P0 | `60267210d97f8145` |
| wt0052 | 无 | 0 / 0 | 2,994,689,613 | 91,426,399 | P0 | `60267210d97f8145` |
| wt/sync | v2 循环 3044664（0050）、3811513（0045），只读 `sync_push.sh` | 0 / 0 | 2,663,937,071 | 91,229,634 | P0 | `60267210d97f8145` |

- 稀疏之后，运行要用的脚本 sha256 不变：
  - wt0051 `queue.sh` 2e2f013c…；
  - wt0052 `run.sh` a4ae09a0…、`eval.sh` 2f501585…、`queue.sh` 4c7ff900…；
  - wt/sync `sync_push.sh` 4d50e7a1…。
- **D95 的工作树更新：** fetch 之后，origin/main 是 `07d677df`，包含 `2c67380d`。wt0051、wt0052 都执行 `checkout --detach origin/main`：
  - HEAD `07d677df0f0cd79c5f04e58b67e557e2ea18d6c1`，status 为空；
  - `core.sparseCheckout=true` 保留，`experiments/` 只有 P0，P0 清单 `60267210…`，大小 91,447,025 B。
  - 按更新后的任务书核对 sha256，**12 项全部 OK**：
    - 0051：queue.sh `8f33ad28…`、train_queue.sh `fcb10f47…`、e5_train.json `e7f28997…`、e5_prompts.json `c0e59d35…`；
    - 0052：run.sh `a4ae09a0…`、eval.sh `51b0db5a…`、queue.sh `5daa6013…`、eval_lib.sh `2380464b…`、train_queue.sh `fcb10f47…`、e3b_train.json `b265cd87…`、e1_prompts.json `f3353de0…`、e1_posthoc.json `9a560782…`。

## 2. `$VI/tmp`（第 3 步）

**删掉的**：pytest 的 basetemp，都没有最近 2 小时内修改过的文件。

| 目录 | 字节 |
|---|---|
| pt_0049_hpcgpu07 | 1,879,989 |
| pt_failtest | 123,196 |
| pt_parity_r3 | 441,840,279 |
| pt_plain_r3 | 441,847,567 |
| pt_rerun3_1 … pt_rerun3_4 | 123,195 + 123,196 + 123,196 + 123,194 |
| pt_skips | 441,829,539 |
| **合计** | **1,328,136,547** |

**没删，问管理员**：

| 项 | 字节 | 来历 |
|---|---|---|
| `repo.bundle` | 248,903,095 | RUN-0044 送仓库用的 git bundle |
| `delta.bundle` | 57,861,080 | RUN-0045 送增量用的 bundle |
| `b200_files.txt`、`b200_list.txt`、`sha256_b200_nologs.txt` | 763,172 + 763,172 + 1,829,930 | RUN-0047 核对用的清单 |
| `sha_0051_starts.txt` | 12,516 | RUN-0051 起点核对用的清单 |
| `visint_nyu_pairs_rendered/` | 1,276,928,859 | 10-12 22:41 由评测生成（NYU 点对的渲染图）；4B 的评测可能还会用 |
| `xet/`、`pip/`、`torchinductor_chenyuma/` | 101,550 + 4,096 + 4,096 | 缓存 |
| `msp_0049_hpcgpu07.json` | 9,660 | RUN-0049 的诊断输出，任务书说留着 |

## 3. git gc（第 4 步）

- 命令：`flock $VI/logs/git.lock -c 'git -C $VI/repo gc --prune=now'`，rc=0，277 s。
- `count-objects -vH` 前后对照：

| 项 | gc 前 | gc 后 |
|---|---|---|
| count / size（松散对象） | 6594 / 354.73 MiB | 0 / 0 bytes |
| in-pack / packs | 19858 / 2 | 26389 / 1 |
| size-pack | 293.24 MiB | 454.73 MiB |
| prune-packable | 6 | 0 |
| garbage / size-garbage | 2（`tmp_pack_ELWMRD`、`tmp_pack_RBlcbB`）/ 63.58 MiB | 0 / 0 bytes |

## 4. 第 5 步的核对

- 两台驱动的心跳在更新：06 `22:43:18Z running e3b_c3_s2`，07 `22:42:36Z running e3b_s1`。
- 推送循环：0045 `22:42:09Z ok`，0050 `22:39:25Z ok`。
