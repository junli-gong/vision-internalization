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

## 5. 管理员对 tmp 的答复之后（22:49Z，`bundles_rm.txt`）

- 删掉 `repo.bundle`（248,903,095 B）和 `delta.bundle`（57,861,080 B）。
- 核对清单、`visint_nyu_pairs_rendered/`、三个小缓存按答复留着。

## 6. 补充 1：推送循环换成 v3（`addendum1_v3.txt`、`addendum1_step67.txt`，脚本 `d1.sh`）

- **第一次运行（22:48Z）在第 2 步停下**：
  - 任务书的命令是 `worktree add --no-checkout` 之后接 `sparse-checkout set`；这样索引是空的，`sync3` 里一个文件都没签出（status 16,246 行），`sync_push.sh` 不存在，sha256 核对不过；
  - 脚本按设计停下，v2 循环没有被动过。
  - 补了一步 `git -C $VI/wt/sync3 read-tree -mu HEAD` 之后：HEAD `f12e4927`，status 为空，`experiments/` 只有 P0（P0 清单 `60267210…`），大小 91,492,284 B，`sync_push.sh` 的 sha256 `21e5892ef539b5095b21974978a36d44f28f197c7bd000ab6ddf1240a16092d4` 与任务书相同。
  - 照这个写法，以后的任务书建 sparse 工作树要么不加 `--no-checkout`，要么补 `read-tree -mu HEAD`（只陈述）。
- **第二次运行（22:49Z）：**
  - 停 v2：都是在 sleep、`git.lock` 没人拿着时按 PID 停掉：
    - `22:49:55Z v2 0045 stopped (pid 3811513, child 924888)`；
    - `22:50:58Z v2 0050 stopped (pid 3044664, child 810833)`。
  - 起 v3：命令与任务书相同，TAG 与路径不变，没有 `SYNC_WT`：
    - `v3 0045 started pid 975340`；
    - `v3 0050 started pid 988275`。
  - 第一轮：`sync_0045` 是 `22:50:50Z ok`，`sync_0050` 是 `22:52:41Z ok`；origin 上有 `9b12f635 [results] 0050: sync 2026-10-13T22:52:36Z`。
    - 脚本里判断"第一轮已完成"的那一段，对 0050 读到的还是 v2 留下的 22:39 那条 status。这是脚本的问题，我事后另读确认了上面这两行。
- **核对第一轮**（origin/main 的树）：

  | 目录 | 树里大于 5 MB 的文件 | `LARGE_FILES.manifest.tsv` | 清单行数 |
  |---|---|---|---|
  | 0045_hpc_train_hpcgpu06 | 0 | 有 | 3 |
  | 0045_hpc_train_hpcgpu07 | 0 | 有 | 4 |
  | 0050_hpc_eval_08b_hpcgpu07 | 0 | 有 | 32 |
  | 0050_hpc_e3b_hpcgpu07 | 0 | 没有。目录里没有大于 5 MB 的文件，最大的是 2,295,325 B 的 train_log.jsonl | 0 |

  - 抽查：`0050_hpc_eval_08b_hpcgpu07/eval/e3b_c3/qwen3.5-0.8b/seed0/refseg__structured/predictions.jsonl`，8,957,246 B。Lustre 上重算的 sha256 是 `1fab29d5…87ec4`，与清单相同。
  - `$VI/repo` 的 `git status --short`：切换前 74 行，切换后 76 行。多出的两行是 `?? experiments/E3/0045_hpc_train_hpcgpu0{6,7}/LARGE_FILES.manifest.tsv`，即 v3 写进源目录的清单；其余行相同。
- **第 6 步：** `wt/push`（3,421,357,048 B）和 `wt/sync`（91,229,634 B）在没有进程使用后，用 `worktree remove --force` 删掉，然后 `worktree prune`。
  - 剩下的工作树：repo、wt/sync3、wt0045、wt0049、wt0050、wt0051、wt0052。
- **第 7 步：** 份额差值 **54,401,328 KB**（22:53Z）。squeue 两条都是 RUNNING：

```
## squeue 2026-10-13T22:53:37Z [0053a step6 before]
157296 RUNNING 59-10:19:23 hpcgpu06
157297 RUNNING 61-07:51:32 hpcgpu07
## squeue 2026-10-13T22:53:52Z [0053a after]
157296 RUNNING 59-10:19:38 hpcgpu06
157297 RUNNING 61-07:51:47 hpcgpu07
```
