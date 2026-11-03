[RUN-TASK 0104] hpcgpu06：X-M 训练（D165；预注册 XM-4a；DEV-0081）。9B 从基座开始，2,000 步；先 50 步探测，按实测决定是否续跑。约 13–15 GPU 小时，上限 34 小时。**排在 RUN-0103 的训练目标生成之后**

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 只用 06 自己的暂停文件。

## 为什么

- 负责人的 X-M 指令（D165），X-M 排在所有运行之前。配方是 `configs/xm_train.json`：等于 X-2 的配方，另加 COCO 的列框与每实例样条两条流，每步 45 条，2,000 步，seed 0。
- 驱动的步骤：
  1. 从 RUN-0103 的训练目标建 COCO 训练样本，剔除带 crowd 的对，提示用冻结的 xm-prompts-v1；
  2. 跑 50 步，取第 5 步以后 step_s 的中位数；
  3. 中位数 ≤ 54 s（2,000 步在 30 小时内）就续跑到 2,000 步；否则停下报告，由管理员按 XM-4a 定退路。
- 用量在 D165 的 100 GPU 小时之内，不再逐个报批。

## 文件（工作树 wt0104 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0104-hpc-06-xm-train/xm_train.sh` | `ad743d0b83e4da83718a5cb01003de98f77f7dd8ebaa5084835f2d19dc662abe` |
| `scripts/build_xm_data.py` | `2f10dcc486d31da62ab66351d8f227c9a753b97cdd1530ad06b7696dded92872` |
| `scripts/train_w2c.py` | `f33994e72dca082b3ffa33792c90b27bd873d02416aaa3cbf043a9e461bf7f84` |
| `src/w2/w2c_train.py` | `4086a0015dcf87319a540bc72ff4b03250ea7b1bf9b7275ef6243d10cb424ec1` |
| `src/xm/train_data.py` | `f37e2710e3500d90bdb903cc7a525b28e180a4e2b2580fbfa29d412f5d3ad32c` |
| `src/xm/prompts.py` | `89208d257aa277ac910f23dfb4b40538b5c8f940ef3dc57be464372eb752cb7a` |
| `src/xm/targets.py` | `c88d94a7d7530bf1cb7040e1160c372f71ae551700fcf6d31e15a47f7a9d3eae` |
| `src/xm/data.py` | `f51759702f8dc5b67c4775859df97d64334fb0b9df03e5c7a430c01ba7a73870` |
| `configs/xm_train.json` | `45c35566c37cd8f187c4fe67fd602d24f85a5b335836917eb9193b1ec0a6c1a0` |
| `configs/x2_train.json` | `c80a4b0b39ac8275dafd6be7af83ec05dc0d5f4fbf21335d5e67e2ab829b5228` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0104（稀疏，含 `src/xm/`、`src/w2/`、`configs/`、`scripts/`），核对上表 12 个 sha256。
   - 确认 `$VI/repo/experiments/XM/0103_xm_targets/xm_targets/manifest.json` 在，即 RUN-0103 第 2 步已完成。（补充 1：原写的少了一层 `xm_targets/`。）
2. 启动：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 12 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=34 \
     bash /home/mail/chenyuma/visint/wt/wt0104/prompts/server_hpc/0104-hpc-06-xm-train/xm_train.sh \
     > /home/mail/chenyuma/visint/logs/eval_0104.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0104.status`。推送循环 TAG 0104，路径 `experiments/XM/0104_xm_hpcgpu06`。
   - 训练样本较大的部分只留在 HPC，记进 `LARGE_FILES.manifest.tsv`。检查点放 `ckpt/` 下，不压缩。
3. **探测跑完立即报**：`steps.log` 里 "probe median step_s" 那一行，以及 `timing_summary` 的全文（含各流的用时、显存峰值）。
   - 如果驱动按 54 s 的门槛停了，就停在那里等管理员，不要自己改。
4. 训练每 500 步报一次：各流的平均损失、step_s 中位数、显存峰值。跑完报检查点 manifest 的 sha256。
5. 跑完写 `reports/0104-hpc-06-xm-train.md`，再跑一次 `squeue`。

管理员在开发机上用 tiny 彩排过：建样本、探测、续跑、写检查点都是 rc=0；重启时各步按已有结果跳过；门槛判断在 12.3 s、60 s、nan 三种情形下都对。
