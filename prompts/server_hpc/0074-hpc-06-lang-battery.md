[RUN-TASK 0074] hpcgpu06：9B 的语言代价读数：大题量的语言读数加 E3b+C3 的健康检查（N44 (c)、D125-3、D129）。约 5.5 GPU 小时，上限 8 小时。现在就发，驱动自动排在 RUN-0072 之后

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 RUN-0072 与 07 上的 RUN-0073。
- 只用 06 自己的暂停文件。

## 为什么

- 负责人要求（D125-2 (c)、D125-3）：
  - 用更大的题量把 W2-c 的语言读数测准；
  - 在 9B 上用同一套健康检查与读数，把 E3b+C3、W2-c、W2-c 加回放三者的语言代价并排。
- RUN-0056 没有给 9B 的 E3b+C3 做健康检查，这里补上。W2-c 的健康检查已在 RUN-0069 做过，不重跑。
- 只报告、不判定。语言读数不涉及 test 划分，不需要盲法。

## 文件（commit 见发出时的启动语；工作树 wt0074 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0074-hpc-06-lang-battery/battery.sh` | `c522d1317a02b8f75a28cf1673fc4de78e27b13a163371eb04241c3b5ce18419` |
| `scripts/lang_battery.py` | `c6cee3ff3b610f1203c6a7e00ca7c85fbeb5b1f2a9c5d3f844641fce28d062e7` |
| `scripts/health_check.py` | `4728a0ca26433f5c02b8b3c064b637a437a32de4c1a1847056614aee2ec5c926` |
| `scripts/build_health_items.py` | `c1577dea3e0cbd98f1897c358786ecaa0c93da739887d71e429b20d02e49065f` |
| `src/health/run.py` | `ca4a995857d69fa080ba6a01aac4225c076e69ef0b4150c6f84cf3a5dfad1437` |
| `src/health/scoring.py` | `05d11f230fe66b67abcabdff2064bc1cb72bb8401cbdc042e0515378871111be` |
| `src/health/protocol.py` | `60c6bc53edd5455c27becf2ab71174144bd523788965904b70e56b5ad84605bf` |
| `configs/health_check_items.json` | `0239675f9c226f17963419e2e12a282dd4ced8ed42452b53a939145372a525df` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

- `eval_lib.sh` 是 D127 之后的新版：`shrink_large` 不再压路径含 `/ckpt/` 的文件。RUN-0072、0073 用的旧版不受影响。

## 内容（`battery.sh`）

- **模型**：
  - 基座 Qwen3.5-9B；
  - W2-c：RUN-0069 的检查点，已在 RUN-0072 开跑前恢复，manifest `c605408f…`；
  - E3b+C3：RUN-0056 种子 0，`$VI/ckpt/hpcgpu06/0056/e3b_c3/qwen3.5-9b/seed0/ckpt/final/model`。
- **步骤**：
  1. `lang_battery.py` 依次跑基座、W2-c、E3b+C3，即 MMLU-Pro 字母作答 2,000 题与 IFEval 541 条；然后 `--compare` 基座对 W2-c、基座对 E3b+C3。
  2. `health_check.py` 跑基座、E3b+C3，再 `--compare`。另把这次的基座与 RUN-0069 的 `health/health_base.json` 配对，核对节点上的可复现性，预期差值为 0。
- **排队**：`claim_gpu` 等 `$LOGS/eval_0072.status` 不再以 running 开头，并且 GPU 上没有进程，才开始。等待上限 24 小时，跑满 `EV_CAP_H` 即止。
- **结果目录**：`experiments/W2/0074_lang_battery_hpcgpu06/`。
- **预计**：
  - lang_battery 每个模型约 1.5 小时，三个约 4.5 小时；
  - 健康检查两次约 40 分钟；
  - 合计约 5.5 小时。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0074（稀疏，规则同 RUN-0072），建在 origin/main。fetch 失败就重试。核对上表 9 个 sha256。
3. **数据**（hpcmgt 上，一次，不进仓库）：
   ```bash
   cd $VI/wt/wt0074 && env -u HF_HUB_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=$VI/hf \
     $VI/env/bin/python scripts/lang_battery.py --fetch --raw-dir $VI/data/lang_battery_raw
   ```
   - `--fetch` 按固定 revision 下载 MMLU-Pro 与 IFEval 的原始文件，并逐个核对 sha256，共几 MB。
   - 经 hf-mirror 下数据集文件在本项目还没试过。出错就贴报错原文，**不要换来源**，等我回复。
   - 数据只限非商业研究，不再分发；不放进仓库，不放进结果目录。
4. **E3b+C3 检查点的核对**：
   - 算 `manifest.json` 的 sha256，记进报告；
   - 按它的训练叶目录 `train_summary.json` 的 `final_checkpoint_files`，或 RUN-0056 的台账，逐个文件核对字节数。对不上就不启动，报告。
   - 核对通过后，把 manifest 的 sha256 作为 `E3B_MANIFEST` 传给驱动。
   - 驱动开头会再核对两个检查点的 manifest。
5. **启动**（方案 B）：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=8 E3B_MANIFEST=<第 4 步的 sha256> \
     bash /home/mail/chenyuma/visint/wt/wt0074/prompts/server_hpc/0074-hpc-06-lang-battery/battery.sh \
     > /home/mail/chenyuma/visint/logs/eval_0074.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `$VI/logs/eval_0074.status`。RUN-0072 跑完之前，心跳应是 "waiting for gpu"。
   - 注意：等待期间 `claim_gpu` 会改写 06 的暂停文件，退出时会删掉它。06 上现在没有训练队列，不影响 RUN-0072。
   - 驱动开始跑第一步之后，再起一个 v3 推送循环，TAG 0074，路径 `experiments/W2/0074_lang_battery_hpcgpu06`。跑完、推完后按 PID 停掉。
6. 每个 `lb_*` 与 `health_*` 步骤跑完报一次：rc、墙钟、合并核对的 pass 与 failures。`--compare` 的结果原样贴出：各读数的差值、CI、丢题与得题数。
7. 全部跑完：写报告，再跑一次 `squeue`。

## 报告

推送 `reports/0074-hpc-06-lang-battery.md`，内容：
- 数据下载：文件、sha256、来源，以及是否经 hf-mirror；
- 两个检查点的核对；
- `steps.log` 原文与每步墙钟；
- 各 `lang_battery.json` 的读数与 CI，三个 `--compare` 与两个健康检查 `--compare` 的原文；
- 基座可复现核对的结果；
- 显存峰值、任何 traceback。
