# RUN-0055 报告：hpcgpu06 上 9B 的实测（只贴耗时、显存与事实，不贴指标读数）

任务书 `prompts/server_hpc/0055-hpc-06-9b-measure.md`（origin 23f5ad20；脚本 commit e496ea0b）。

## 第一部分：下载并核对 9B 权重

### 工作树

- 10-19 08:39:46Z squeue：157296、157297 都是 RUNNING。
- `wt0055`：fetch rc=0；`worktree add --no-checkout --detach … e496ea0b9bd368b29240cdb153b1312cfbf984ac` rc=0；`sparse-checkout set` rc=0；`read-tree -mu HEAD` rc=0。HEAD `e496ea0b9bd368b29240cdb153b1312cfbf984ac`，status 空，`experiments/` 只有 P0。
- sha256，7 个都与任务书相同：

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0055-hpc-06-9b-measure/fetch_9b.sh` | `8ef9a46511010e4989f1b6c61200c5606f963a9c802bad5834d23df472ed303b` |
| `prompts/server_hpc/0055-hpc-06-9b-measure/measure_9b.sh` | `d680172ee26d606b472fe88b810da430c753652b5e9c7bf7997cde58868a0fc1` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |
| `configs/e3b_train.json` | `b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375` |
| `src/data/train_loaders.py` | `76535736def11a13a19a94a930ead215684ee4a697db3c05f6fd8c35d54d10cb` |
| `experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py` | `0fcd224eb3b0025e7d34045a5243af8acd8eafc14daef98aa7007f3452504550` |
| `experiments/P0/0009_ifeval_rescore/weights_manifest.json` | `8535b8cb89e6def1f8a79feff669b315ca0578c75d4b7ce310d4ff8aa86efa46` |

- 启动前：`$VI/hf/hub/` 下没有 `models--Qwen--Qwen3.5-9B`，没有 `fetch_0055*` 日志。

### 下载

- 08:40:34Z 按任务书命令在 06 上启动 fetch_9b.sh，step 157296.163；之后 squeue 两条 RUNNING，big_s0 的 step 157296.162 照常。
- 心跳：`2026-10-19T08:40:42Z running download qwen3.5-9b (attempt 1)` → `2026-10-19T09:05:09Z done fetch_0055 (attempt 1)`。没有重启。
- 日志里核对那一行原文：`qwen3.5-9b @ c202236235762e1c871ad0ccb60c8ee5ba337b9a: 16/16 files verified (/home/mail/chenyuma/visint/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a)`
- `blobs/` 下 16 个文件，`ls -l` 字节数合计 19,329,393,661 B（19.33 GB）；最大四个：5,368,717,440、5,335,161,512、5,276,436,216、3,325,995,712 B。没有剩下的 `.incomplete`。
- 用时 08:40:42Z – 09:05:09Z，1,467 s；平均 13.2 MB/s（合计字节 / 用时）。
- 我这边每 10 分钟的速度采样脚本有引号错误，取到的字节数一直是 0，没有起到监测作用；下载在第一个采样周期内正常完成，没有因此误停或漏停。
- 09:13:50Z squeue：两条 RUNNING；step 157296.163 已不在；hpcmgt 上没有 fetch_9b 进程。

## 第二部分：实测

### 启动

- 10-20 04:27:43Z（经 hpcmgt 上脱离会话的 `$VI/tmp/start0055b.sh`，输出 `$VI/logs/start0055b.txt`）核对四条：
  - `train_0051-hpcgpu06.status` 为 `2026-10-20T04:14:16Z done 0051-hpcgpu06`，`train_0051-hpcgpu06.done` 存在；
  - `e5_claims_0051/big_s0/done`：`hpcgpu06 2026-10-20T04:14:15Z`；
  - 06 上没有 0051、0052、0055 的进程，GPU 上没有计算进程；没有 06 的暂停文件；
  - `fetch_0055.status` `2026-10-19T09:05:09Z done fetch_0055 (attempt 1)`，日志有 16/16 那一行。
  - wt0055 HEAD e496ea0b、status 空；measure_9b.sh sha256 `d680172e…` 与任务书相同。
- 04:28:11Z 按任务书命令启动（`EV_CAP_H=5`），step 157296.164；前后 squeue 两条 RUNNING。
- 推送循环 0055：04:29:22Z 启动，实际 PID 2911629（`$VI/logs/sync_0055.pid`）。
- `weights_verify.log`：`qwen3.5-9b @ c202236235762e1c871ad0ccb60c8ee5ba337b9a: 16/16 files verified (…)`。

### steps.log 原文

```
2026-10-20T04:29:22Z start commit e496ea0b9bd368b29240cdb153b1312cfbf984ac node hpcgpu06 model qwen3.5-9b
2026-10-20T04:34:59Z train20_e3b_c3_micro4 rc=0 wall_s=337
2026-10-20T04:41:25Z train20_e3b_micro4 rc=0 wall_s=386
2026-10-20T04:44:35Z train20_c3_micro4 rc=0 wall_s=190
2026-10-20T04:44:35Z 网格评测用 e3b_c3 的检查点 /home/mail/chenyuma/visint/ckpt/hpcgpu06/tmp0055/e3b_c3/qwen3.5-9b/seed0/latest/model
2026-10-20T04:53:21Z grid_e3b_c3_refseg100 rc=0 wall_s=526
2026-10-20T04:54:45Z grid_e3b_c3_depth50 rc=0 wall_s=84
2026-10-20T04:58:42Z e1_base_dev20 rc=0 wall_s=237
2026-10-20T05:01:37Z e1_base_free10 rc=0 wall_s=175
2026-10-20T05:11:10Z ttext_e3b_c3_dev20 rc=0 wall_s=573
2026-10-20T05:11:11Z done; failed steps: 0
```

（另有三行 `（不推送，EV_NO_GIT=1：…）` 未贴。）心跳 `2026-10-20T05:11:11Z done 0055 (failed steps: 0)`。从启动到结束 04:28:11Z – 05:11:11Z，2,580 s（0.72 h），上限 5 h。

以下各表由 `experiments/E3/0055_hpc_measure_9b_probe/p55_tables.py` 从结果文件读出。显存峰值都是各行 `peak_mem_bytes` 的最大值（torch 的 allocated 峰值）；reserved 结果文件与日志里都没有记。网格评测与 T 文本的结果文件里没有每批耗时与显存峰值。

### 训练 20 步（种子 0，微批 4 × 累积 4）

| 臂 | 步 rc | 命令墙钟 s | 命令开始到训练开始 s | 第 1 步 step_s | 20 步每步中位 s | 第 2–20 步每步中位 s | 输入构建中位 s | 前向中位 s | 反向中位 s | peak_mem_bytes 最大（allocated）GB | reserved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 | 0 | 337 | 66 | 24.41 | 11.950 | 11.909 | 0.091 | 3.686 | 8.084 | 36.35 | 日志里没有 |
| e3b | 0 | 386 | 48 | 16.85 | 15.885 | 15.866 | 0.170 | 4.817 | 10.897 | 30.49 | 日志里没有 |
| c3 | 0 | 190 | 29 | 8.85 | 7.883 | 7.879 | 0.202 | 2.502 | 5.101 | 28.61 | 日志里没有 |

- 三臂都在微批 4 下跑完，没有走微批 2 的补测（`steps.log` 里没有 micro2 的步）。
- e3b_c3 的 `accounting.json`：peak_mem_bytes 36,354,630,144，wall_s 269.9，n_samples 320，GPU NVIDIA A100-PCIE-40GB。

### 网格评测（E3b+C3 的 20 步检查点，结构强制）

- 合并核对（两条共用，`eval/e3b_c3/qwen3.5-9b/seed0/merge_check.json`，写于 2026-10-20T04:50:45Z）：pass = True，failures = `{"restricted": [], "merge": []}`，criteria_version = v1；耗时 merge_before_s 162.8 s、merge_after_s 122.8 s、restricted_s 34.9 s、restricted_across_merge_s 24.7 s、total_s 346.0 s。

| 步 | rc | 命令墙钟 s | 条数 | 解码 call_time_s | 解码 wall_time_s | 每条 wall_time_s | 批大小 | 每批耗时 | 显存峰值 |
|---|---|---|---|---|---|---|---|---|---|
| grid_e3b_c3_refseg100 | 0 | 526 | 100 | 113.2 | 114.2 | 1.142 | 64 | 日志里没有 | 日志里没有 |
| grid_e3b_c3_depth50 | 0 | 84 | 50 | 41.2 | 43.3 | 0.866 | 64 | 日志里没有 | 日志里没有 |

### T 文本

- **e1_base_dev20**：rc=0，命令墙钟 237 s；各任务评测运行 wall_s 合计 165.6 s；加载等 71 s（命令墙钟 − 各任务合计）；显存峰值日志里没有。

| 任务 | 条数 | 批大小 | 批数 | 评测运行 wall_s | 每条 s | 生成 token |
|---|---|---|---|---|---|---|
| count__dev | 20 | 64 | 1 | 2.6 | 0.056 | 42 |
| grid16__dev | 20 | 64 | 1 | 13.0 | 0.588 | 5492 |
| ovd__dev | 20 | 64 | 1 | 93.0 | 4.208 | 11564 |
| refexp_box__dev | 20 | 64 | 1 | 6.4 | 0.141 | 894 |
| refexp_poly__dev | 20 | 64 | 1 | 38.0 | 1.839 | 1769 |
| rel_depth_blink__dev | 12 | 64 | 1 | 2.1 | 0.056 | 12 |
| rel_depth_nyu__dev | 20 | 64 | 1 | 2.4 | 0.051 | 20 |
| spatial_cvbench__dev | 20 | 64 | 11 | 4.5 | 0.123 | 20 |
| spatial_embspatial__dev | 20 | 64 | 1 | 3.5 | 0.059 | 20 |

- **e1_base_free10**：rc=0，命令墙钟 175 s；各任务合计 64.4 s；加载等 111 s；显存峰值日志里没有。

| 任务 | 条数 | 批大小 | 批数 | 评测运行 wall_s | 每条 s | 生成 token |
|---|---|---|---|---|---|---|
| rel_depth_blink__free | 10 | 64 | 1 | 9.4 | 0.828 | 1593 |
| rel_depth_nyu__free | 10 | 64 | 1 | 22.6 | 2.055 | 1788 |
| spatial_cvbench__free | 10 | 64 | 1 | 15.2 | 1.399 | 1591 |
| spatial_embspatial__free | 10 | 64 | 1 | 17.2 | 1.613 | 1278 |

- **ttext_e3b_c3_dev20**：rc=0，命令墙钟 573 s；各任务合计 226.2 s；加载与核对等 347 s；合并核对 pass = True，failures = `{"merge": []}`，耗时 merge_before_s 153.4 s、merge_after_s 122.9 s、total_s 276.6 s；显存峰值日志里没有。

| 任务 | 条数 | 批大小 | 批数 | 评测运行 wall_s | 每条 s | 生成 token |
|---|---|---|---|---|---|---|
| count__dev | 20 | 64 | 1 | 2.4 | 0.056 | 42 |
| grid16__dev | 20 | 64 | 1 | 34.2 | 0.598 | 5715 |
| ovd__dev | 20 | 64 | 1 | 87.8 | 4.234 | 11534 |
| refexp_box__dev | 20 | 64 | 1 | 30.4 | 0.142 | 895 |
| refexp_poly__dev | 20 | 64 | 1 | 60.9 | 1.976 | 2629 |
| rel_depth_blink__dev | 12 | 64 | 1 | 1.6 | 0.058 | 12 |
| rel_depth_nyu__dev | 20 | 64 | 1 | 2.5 | 0.052 | 20 |
| spatial_cvbench__dev | 20 | 64 | 11 | 3.8 | 0.070 | 20 |
| spatial_embspatial__dev | 20 | 64 | 1 | 2.5 | 0.061 | 20 |

- spatial_cvbench__dev 的 `run_001.json` 记 n_batches 11（20 条、批大小 64），照录。

### 收尾

- `$VI/ckpt/hpcgpu06/tmp0055` 已不存在（05:1xZ 查）。
- 用量：各步命令墙钟合计 2,508 s；从启动到结束 2,580 s。
- 推送循环 0055：跑完后的一轮 origin `15a14409 [results] 0055: sync 2026-10-20T05:36:32Z`（含 `steps.log` 的 `done; failed steps: 0`），`sync_0055.status` `2026-10-20T05:36:38Z ok`；05:37:25Z 在 sleep 时、`flock -n git.lock` 下按 PID 停掉（2911629 与 sleep 3808403）。0054 的循环（2836160）照常。
- 06 空着，等管理员的下一份任务书（N38）。
