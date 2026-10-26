[RUN-TASK 0076] N44 (b)：W2-c 加自蒸馏回放。A 段在 hpcgpu07 上生成数据并训练（约 9 GPU 小时，上限 12 小时，排在 RUN-0073 之后，现在就发）；B 段在 hpcgpu06 上评测（约 5.5 GPU 小时，上限 9 小时，等 A 段训练完、我回复之后再发）

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 每次启动前后各跑一次 `squeue`。
- 不碰其他在跑的 RUN。
- 只用本节点自己的暂停文件。

## 为什么

- 负责人要求（D125-2 (b)）：在 W2-c 的训练里混入约 25% 的基座自蒸馏回放，其余不变，1 个种子。
- 这是事后的描述，与 W2-c 并列，不在 devcheck2 上判定。
- 设计与预测写在 W2 预注册 §12、§12a。合入的代码见 D132。
- 回放分两条流：每步 8（提出）+ 8（挑选）+ 3（通用回放）+ 2（计数回放），权重 1 : 1 : 0.5 : 0.5。

## 文件（commit 见发出时的启动语；工作树 wt0076 建在 origin/main，A、B 两段用同一个工作树）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0076-hpc-w2c-replay/replay.sh` | `23c21db3aa5e98e8cef9b9b6c7d0e982617b70d9efc3f9d1f553165e8c6e47c6` |
| `scripts/build_w2c_replay.py` | `8cdcb1cca6abf02f065f29ebe4615afc72e22cef1e484b16b163ded4dc50ddd5` |
| `src/w2/w2c_replay.py` | `99f419e60f1c8d2102c25c0076bf8067d06fba6a03e83f7717e5fa5d376d1f70` |
| `scripts/train_w2c.py` | `354ec7ef5258a99077bded2f85e898bdcaf01ba2ba32118ccd20dc0b09d4f346` |
| `src/w2/w2c_train.py` | `96dc8db7a64b2b7a5531401db4482783e3b503dd62bd995ece75693292ca2c5b` |
| `configs/w2c_replay_train.json` | `a775861041c6bf4ebb8c05a7d02505ef50407727b6c2c91e072e885e76b0f920` |
| `configs/w2c_train.json` | `709b24aec5ba9e0d0449e47d3391bdca5f79dc2673ae497c1b907582962b668f` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `scripts/run_w2.py` | `3093c94d96bf29647fc24b436e85b0cea034e1f3d6d2833d3b7e12cf1628ec0f` |
| `src/w2/runner.py` | `d6a2b4bb38e785bb598303e0bd79563f3b26efef2b1b94a7bd569e17e7894f48` |
| `src/w2/w2c.py` | `61b5c9b166c98c7e9a6693763b33c0243bf7dd86eb1334a9c199be0115a8dcf9` |
| `scripts/lang_battery.py` | `c6cee3ff3b610f1203c6a7e00ca7c85fbeb5b1f2a9c5d3f844641fce28d062e7` |
| `scripts/health_check.py` | `4728a0ca26433f5c02b8b3c064b637a437a32de4c1a1847056614aee2ec5c926` |
| `scripts/run_e1.py` | `5f46ed7c28b4653221fa5ef6eb6b8dc8eec952a53360b30ffa444fa1e72bfd4d` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## A 段：07，数据与训练（现在发）

1. hpcmgt 上 `squeue`。建工作树 wt0076（稀疏，规则同 RUN-0072），建在 origin/main，核对上表 15 个 sha256。
2. **原始文件**（hpcmgt，一次，约 18 MB，不进仓库）：
   ```bash
   cd $VI/wt/wt0076 && env -u HF_HUB_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=$VI/hf \
     $VI/env/bin/python scripts/build_w2c_replay.py --stage fetch --raw-dir $VI/data/raw/w2c_replay
   ```
   - 内容：dolly-15k 以及三个评测来源（MMLU-Pro、GSM8K、IFEval），后者用于去重。脚本按固定 revision 核对 sha256。
   - 出错就贴原文，不要换来源，等我回复。
   - 数据只限研究用，不再分发。
3. **启动**（方案 B）：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin NODE=hpcgpu07 STAGES=data WAIT_GLOB=eval_0073.status EV_CAP_H=2 \
     bash /home/mail/chenyuma/visint/wt/wt0076/prompts/server_hpc/0076-hpc-w2c-replay/replay.sh \
     > /home/mail/chenyuma/visint/logs/eval_0076a_data.driver.log 2>&1 < /dev/null &'
   ```
   - 先只跑 data，约 30 分钟。驱动先等 `eval_0073.status` 不再是 running，且 GPU 上没有进程，才开始。
   - RUN-0077 是纯 CPU 任务，可以同时在 07 上跑。
   - 心跳在 `$VI/logs/eval_0076.status`。
   - 驱动开始第一步之后，起一个 v3 推送循环，TAG 0076a，路径 `experiments/W2/0076_w2c_replay_hpcgpu07`。检查点在 `ckpt/` 下，不入库，`shrink_large` 也不碰它（D127）。
4. **data 跑完报一次**：
   - `0076_replay_data/metrics.json` 的 `complete`；
   - 两类各自的条数、截断数、池的大小；
   - `replay/manifest.json` 里通用部分的剔除计数（超长、结构性标记、与评测重叠、重复）与 CountBench 去重数；
   - 墙钟与显存峰值。
   - **核对线**：`complete = true`、两类的池合计 ≥ 1,500 条、两类的截断率都 ≤ 30%。都满足，就不必等我，用同一条命令改成 `STAGES=train EV_CAP_H=11`、日志 `eval_0076a_train.driver.log` 启动训练。任何一条不满足，就停下报告。
5. **train**：
   - 每 500 步报一次：步数、loss、各流的损失、每步墙钟、显存峰值。
   - 跑完报 `train_summary.json` 的要点，以及检查点 `manifest.json` 的 sha256。
   - 中途失败时，同一条命令再启动即可，驱动在有 `latest` 时自动 `--resume`。

## B 段：06，评测（A 段的 train 跑完、我回复之后再发）

- 启动命令同 A 段第 3 步，改为：
  - `--jobid=157296`、`NODE=hpcgpu06 STAGES=eval EV_CAP_H=9`；
  - `WAIT_GLOB` 用我回复里给的值，取决于那时 06 上排着什么；
  - 驱动日志写 `eval_0076b.driver.log`；
  - 推送循环 TAG 0076b，路径 `experiments/W2/0076_w2c_replay_eval_hpcgpu06`。
- 内容：
  - 5 个 test 划分各 1,000 句的 `w2c` 设置，用回放变体的检查点，输出到新目录；
  - 健康检查，与 RUN-0069 的 `health_base` 配对；
  - 定位与计数的 T 文本读数；
  - 大题量语言读数，与 RUN-0074 的 `lb_base` 配对。
  - 这些都与 06 上已有的基座、W2-c 读数在同一节点上。
- 每个划分跑完报 cIoU，健康检查与 lb 的 `--compare` 原文照贴。

## 报告

- A 段跑完推 `reports/0076a-hpc-07-w2c-replay-train.md`：
  - `steps.log` 原文；
  - 回放数据的计数与剔除；
  - 训练的逐 500 步摘要、`train_summary` 要点；
  - 检查点 manifest 的 sha256；
  - 显存峰值、traceback。
- B 段跑完推 `reports/0076b-hpc-06-w2c-replay-eval.md`：
  - `steps.log` 原文；
  - 各划分的 cIoU、gIoU；
  - 健康检查与 lb 的 `--compare` 原文；
  - T 文本读数的标题数字。
- 只贴数字，不解读。
