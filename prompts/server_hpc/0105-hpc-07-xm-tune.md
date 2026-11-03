[RUN-TASK 0105] hpcgpu07：基座零样本在 X-M 调参集上（检测与实例分割的起点），以及 X-2 在 ReasonSeg val 上的指代分割（描述性）（D165 第六节，D168）。约 1.5 GPU 小时，上限 4 小时；现在就发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 07 上 RUN-0103 第 2 步在借 CPU（`-c 14`）；这里用 GPU 与 `-c 8`，互不影响。不碰 06。
- 只用 07 自己的暂停文件。

## 为什么

- **调参集 `xm_tune`**（300 张，923 对）：基座零样本的 `base_det` 与 `base_box_grabcut`，是检测与实例分割的起点，用来调推理设置（XM-4a）。
  - **不碰判定集**：基座在判定集上的运行推迟到判定时，在 06 上跑（D168）。
- **ReasonSeg val**（196 条）：X-2 seed 0（RUN-0080 的检查点）按 X-2 原协议做指代分割，描述性（D165 第六节 D1–D3）。
- 用量在 D165 的 100 GPU 小时之内。

## 文件（工作树 wt0105 建在 origin/main，含 61abf1e39）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0105-hpc-07-xm-tune/xm_tune.sh` | `1c2e44a69c632d0e6a9d39c527e7c715c7aa6b7b34289263283b9d84fb866e8a` |
| `scripts/run_xm.py` | `9a8cb1047db3e133b3970404d1f9c85b905818fd4755e85c6f885ea836632802` |
| `src/xm/runner.py` | `5551d9cf430f74d3046b5aa81366db53fd96ae6bb2f06af3abb5644b6691f8d9` |
| `src/xm/prompts.py` | `89208d257aa277ac910f23dfb4b40538b5c8f940ef3dc57be464372eb752cb7a` |
| `src/xm/post.py` | `799f3f2cef9f69707d49a1d4a8b8a903af706cc63af5681c4bb16668f757e721` |
| `src/xm/metrics.py` | `448ebb26139d0df2f3555d67108b0316254cda77021ce526105e852b5bd85df9` |
| `src/xm/gate.py` | `dc9bef5506f8247980a76572f05d95c35f4df883685938beba69df982fc54b45` |
| `src/xm/data.py` | `f51759702f8dc5b67c4775859df97d64334fb0b9df03e5c7a430c01ba7a73870` |
| `configs/xm_splits.json` | `995345f470111e3ac73c9681754dd46eca66da7d9ca0b648873cc1429219db8d` |
| `configs/x2_run.json` | `c92ad838a9e698236e8a34d17f6ce187ee185ffaa7df96df5ee99a04f47dc061` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0105（稀疏，含 `src/`、`scripts/`、`configs/`），核对上表 12 个 sha256。
2. 启动：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=4 \
     bash /home/mail/chenyuma/visint/wt/wt0105/prompts/server_hpc/0105-hpc-07-xm-tune/xm_tune.sh \
     > /home/mail/chenyuma/visint/logs/eval_0105.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0105.status`。推送循环 TAG 0105，路径 `experiments/XM/0105_xm_tune_hpcgpu07`。
3. 第一组 64 个对跑完后先报 `logs/timing.jsonl` 的每对用时。
4. 跑完报：
   - `base_det` 与 `base_box_grabcut` 的 `f1_rexomni`、F1@mIoU、微平均 F1、P、R、失败数；
   - ReasonSeg 的 gIoU 与 cIoU：W2 格式一份，以及 `reasonseg_readout` 里扣掉忽略区域的一份。
5. 写 `reports/0105-hpc-07-xm-tune.md`，再跑一次 `squeue`。

管理员在开发机上用 tiny 彩排过：基座调参的首次与重启都是 rc=0；ReasonSeg 一步的命令行核对过。
