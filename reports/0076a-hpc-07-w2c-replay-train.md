# RUN-0076 A 段报告：hpcgpu07，W2-c 加自蒸馏回放：回放数据与训练（9B，2,000 步）

任务书 `prompts/server_hpc/0076-hpc-w2c-replay.md`（A 段）+ `0076-addendum-1.md`（env 加 `TINY=0`）；D137（越过通用回放截断率 30% 的核对线）。只贴数字，不解读。

## 工作树与原始文件

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0076a.sh`（10-26 04:19Z 起）。前 squeue：157296、157297 RUNNING。
- wt0076（稀疏）HEAD `237fbf453a6936381d5da699cec52fb7f6d53895`，status 空，`experiments/` 只有 P0；15 个 sha256 与任务书逐字相同。
- 原始文件经 hf-mirror（任务书的命令原样）：`build_w2c_replay.py --stage fetch --raw-dir $VI/data/raw/w2c_replay`，rc=0，04:19:41Z – 04:20:16Z，4 个文件都与固定 sha256 相同：

| 文件 | 字节 | sha256 |
|---|---:|---|
| `databricks__databricks-dolly-15k/databricks-dolly-15k.jsonl` | 13,085,339 | `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec` |
| `TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet` | 4,144,185 | `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8` |
| `openai__gsm8k/main/test-00000-of-00001.parquet` | 419,088 | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` |
| `google__IFEval/ifeval_input_data.jsonl` | 207,111 | `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2` |

## 启动经过

- **data 第 1 次**（04:20:16Z）：任务书的命令；驱动立即退出，`eval_0076a_data.driver.log` 原文：
  ```
  /home/mail/chenyuma/visint/wt/wt0076/prompts/server_hpc/0076-hpc-w2c-replay/replay.sh: line 52: TINY: unbound variable
  srun: error: hpcgpu07: task 0: Exited with exit code 1
  ```
  没有心跳、没有暂停文件、没有占 GPU。日志另存为 `eval_0076a_data.driver.attempt1.log`。
- **data 第 2 次**（管理员选 (a)，补充 1）：`$VI/tmp/start0076a2.sh`，04:28:50Z，修正后的命令：
  `srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 NODE=hpcgpu07 STAGES=data WAIT_GLOB=eval_0073.status EV_CAP_H=2 bash …/replay.sh > $VI/logs/eval_0076a_data.driver.log`
  `-c 8` 被接受；前后 squeue 两条 RUNNING；等 RUN-0073 期间心跳 "waiting for gpu (running: eval_0073.status …)"；17:33:36Z 开始。
- **train**（D137：通用回放截断率 38.2% 超过 30%，管理员越线放行）：`$VI/tmp/start0076a3.sh`，18:47:23Z：
  `srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 NODE=hpcgpu07 STAGES=train WAIT_GLOB=eval_0073.status EV_CAP_H=11 bash …/replay.sh > $VI/logs/eval_0076a_train.driver.log`
  启动前核对 `done 0073`、`done 0076 [data]`、15/15 sha256；前后 squeue 两条 RUNNING。
- 推送循环 TAG 0076a（PID 3663637，1800 s），从 data 的第一步起。

## steps.log 原文

```
2026-10-26T17:33:36Z start commit 237fbf453a6936381d5da699cec52fb7f6d53895 node hpcgpu07 model qwen3.5-9b stages [data] replay cfg a775861041c6bf4ebb8c05a7d02505ef50407727b6c2c91e072e885e76b0f920
2026-10-26T18:02:23Z data rc=0 wall_s=1727
2026-10-26T18:02:23Z done; failed steps: 0
2026-10-26T18:02:23Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2-c replay 0076 [data] done (failed steps: 0)
2026-10-26T18:47:25Z start commit 237fbf453a6936381d5da699cec52fb7f6d53895 node hpcgpu07 model qwen3.5-9b stages [train] replay cfg a775861041c6bf4ebb8c05a7d02505ef50407727b6c2c91e072e885e76b0f920
2026-10-27T01:57:16Z train rc=0 wall_s=25791
2026-10-27T01:57:16Z done; failed steps: 0
2026-10-27T01:57:16Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2-c replay 0076 [train] done (failed steps: 0)
```

- data 1,727 s；train 25,791 s（7.16 h，上限 11 h）。

## 回放数据（`0076_replay_data/metrics.json` 与 `replay/manifest.json`）

- metrics.json：complete true；manifest_sha256 `1a68646e415d6ad4cb42b0e71c903d7b52ba407662203650863370c4e742ca8c`。

| 类 | n | 截断 | 截断率 | 可用 | 生成 token 中位 / 最大 | prompt token 中位 |
|---|---:|---:|---:|---:|---|---:|
| 通用（general） | 2,000 | 763 | 0.3815 | 1,237 | 344.5 / 512（可用的中位 185） | 27 |
| 计数（count） | 2,000 | 0 | 0.0 | 2,000 | 2 / 4 | 331 |

- 计数回放的回答：解析 2,000、未解析 0；取值中位 2，最大 100（只作诊断）。
- 通用部分（dolly-15k @ `bdd27f4d…`，15,011 行）的剔除：超长 157、结构性标记 0、与评测重叠 15（health_check 3、mmlu_pro 12）、重复 209；保留 14,630；选 2,000（sha256 `09b7a108…`），带 context 的 562。
- 计数部分：总体 10,438 张图、13,108 对；CountBench 去重 same_key 0、phash_near 0、去掉的图 0；选 2,000（sha256 `e3e37ff8…`），1,934 张图；类别 person 841、chair 78、dining table 59 …；object fallback 0。
- 核对线：complete true；两类的池合计 1,237 + 2,000 = 3,237 ≥ 1,500；通用截断率 0.3815 > 0.30（D137 放行），计数 0.0。

## 训练（`0076_train/w2c/qwen3.5-9b/seed0`）

逐 500 步（由 `experiments/W2/0076_w2c_replay_probe/p76_train.py` 从 `train_log.jsonl` 与 `logs/timing_run_001.jsonl` 读出）：

| 步 | loss 均值 | 段末 loss | propose | select | replay_general | replay_count | step_s 中位 | 显存峰值（字节） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–499 | 1.2668 | 1.0951 | 0.3018 | 0.6103 | 0.1514 | 0.5580 | 12.61 | 28,287,325,696 |
| 500–999 | 1.1051 | 0.5604 | 0.2970 | 0.5129 | 0.0816 | 0.5089 | 12.62 | 28,441,752,576 |
| 1000–1499 | 0.9377 | 0.7917 | 0.2933 | 0.4817 | 0.0432 | 0.2823 | 12.64 | 28,492,822,016 |
| 1500–1999 | 0.8555 | 0.6183 | 0.2902 | 0.4418 | 0.0150 | 0.2317 | 12.62 | 28,435,249,664 |

- train_summary：complete True，steps 2000，variant w2c，seed 0，final_loss 0.6183（propose 0.2889，select 0.2529，replay_general 0.0047，replay_count 0.1482），final_checkpoint_bytes 346,305,781，LoRA r 32 alpha 64.0 n_parameters 86,556,672，w2_config_sha256 c4d08348…，identity_sha256 1494de31…
- accounting：n_runs 1，n_steps_timed 2000，n_samples 42,000（propose 16,000，select 16,000，replay_general 6,000，replay_count 4,000），wall_s 25,444，gpu_hours 7.07，peak_mem_bytes 28,492,822,016
- 检查点 `ckpt/final/model`：`manifest.json` sha256 `ecad7667b4588f4840ba13f79b71b8c52d475e26a99c79a0e6d3999c99dcecfd`；目录里是 `lora/adapter_config.json`、`lora/adapter_model.safetensors`、`manifest.json`、`token_rows.safetensors`、`tokens.json`。adapter 没有被压缩（新版 `eval_lib.sh` 的 `shrink_large` 跳过 `*/ckpt/*`，D127）。检查点不入库。

## 显存与 traceback

- `gpu_mem.csv`（data 与 train 两段，27,509 行）全程最大 34,207 MiB；训练段每步 `peak_mem_bytes` 最大 28,492,822,016。
- `logs/data.log`、`logs/train.log` grep `Traceback`、`out of memory`：都是 0。

## 收尾

- 推送循环 01:38:56Z 一轮推送成功（在 train done 之前）；02:01:08Z 在 git.lock 下按 PID 停掉循环 3663637（当时在 sleep），随后 `ONCE=1` 推一轮：`2026-10-27T02:01:19Z ok`。
- 后 squeue：157296、157297 RUNNING。07 接着跑 RUN-0080（01:57:27Z running train_probe）；B 段在 06 上排在 RUN-0078 之后（02:02:26Z 启动，等 GPU）。
