[RUN-TASK 0106] X-M 判定集的评测：X-M 的 4 个设置与基座零样本，都在密封的判定集上，三个叶目录同一节点（D165 第五节；XM-4b、XM-5）。约 3 GPU 小时，上限 6 小时；**排在 RUN-0104 训完之后**

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰别的项目（lly-mle）的进程。
- 只用所在节点自己的暂停文件。

## 为什么

- 判定第 1、2 条要三个叶目录：X-M 的 `xm_spline_trimap` 与 `xm_box_grabcut`、基座的 `base_det`。它们必须同一节点（XM-4b）。
- 驱动先在调参集上跑 X-M，只作检查、不改任何推理设置；再在判定集（`--gate G-XM`）上跑 X-M 与基座。
- **判定集的读数在门报告出来之前不看**：服务器照常推送，报告里只贴 complete、失败数与用时，**不贴任何 F1**。

## 节点（XM-4b、D175）

- X-M 训完时（`eval_0104.status` 以 `done 0104` 开头），06 的 nvidia-smi 若没有别的计算进程，就用 `NODE=hpcgpu06`、`srun --jobid=157296`，与 07 上的 RUN-0107 同时跑。
- 06 仍被占时，用 `NODE=hpcgpu07`、`srun --jobid=157297`，在 07 上先跑本任务，RUN-0107 排在它之后。
- 同一时间只启动一次。

## 文件（工作树 wt0106 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0106-hpc-xm-decision/xm_decision.sh` | `70d2cb3ded1d06bf33d855d77bc4ca203b926f11f8471c8be4b66105b77d21d1` |
| `scripts/run_xm.py` | `9a8cb1047db3e133b3970404d1f9c85b905818fd4755e85c6f885ea836632802` |
| `src/xm/runner.py` | `5551d9cf430f74d3046b5aa81366db53fd96ae6bb2f06af3abb5644b6691f8d9` |
| `src/xm/prompts.py` | `89208d257aa277ac910f23dfb4b40538b5c8f940ef3dc57be464372eb752cb7a` |
| `src/xm/post.py` | `799f3f2cef9f69707d49a1d4a8b8a903af706cc63af5681c4bb16668f757e721` |
| `src/xm/metrics.py` | `448ebb26139d0df2f3555d67108b0316254cda77021ce526105e852b5bd85df9` |
| `src/xm/gate.py` | `dc9bef5506f8247980a76572f05d95c35f4df883685938beba69df982fc54b45` |
| `src/xm/data.py` | `f51759702f8dc5b67c4775859df97d64334fb0b9df03e5c7a430c01ba7a73870` |
| `configs/xm_splits.json` | `995345f470111e3ac73c9681754dd46eca66da7d9ca0b648873cc1429219db8d` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0106（稀疏，含 `src/`、`scripts/`、`configs/`），核对上表 11 个 sha256。
2. 启动（驱动自己等 `done 0104`）：
   ```bash
   ssh -n <节点> 'setsid nohup srun --jobid=<157296 或 157297> --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 NODE=<hpcgpu06 或 hpcgpu07> EV_CAP_H=6 \
     bash /home/mail/chenyuma/visint/wt/wt0106/prompts/server_hpc/0106-hpc-xm-decision/xm_decision.sh \
     > /home/mail/chenyuma/visint/logs/eval_0106.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0106.status`。推送循环 TAG 0106，路径 `experiments/XM/0106_xm_decision_<节点>`。
3. 调参集上 X-M 的读数可以报，用来检查是否正常：f1_rexomni（检测与实例分割）、失败数。判定集一律不报读数，只报 complete、失败数与用时。
4. 跑完写 `reports/0106-hpc-xm-decision.md`，同样不含判定集的读数；再跑一次 `squeue`。

管理员在开发机上用 tiny 彩排过：首次与重启都是 rc=0；NODE 写错时拒绝。正式分支的参数逐条对过 run_xm.py 的限制：xm_* 带检查点，base_* 不带，判定集带门名。
