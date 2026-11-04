# RUN-0104 报告：X-M 训练（9B，从基座开始，2,000 步）——最终在 hpcgpu07 上（D165、XM-4a；补充 1 D174、补充 2 D175）

任务书 `prompts/server_hpc/0104-hpc-06-xm-train.md`。只贴数字，不解读。

## 经过

1. **排队**：hpcmgt 上的 `start0104.sh` 先等 RUN-0103 第 2 步（训练目标的 manifest 与 rc=0 标记），再等预注册门（origin/main 的 `PREREGISTRATION.md` 含 XM-2a、`configs/xm_train_exclusions.json` 与 `configs/xm_splits.json` 已提交），再核对两份配置与 RUN-0103 的输出逐字节相同（不同就写 `blocked: exclusions mismatch` 并等管理员）。
2. **补充 1（D174）**：`build_targets` 把 manifest 写在 `0103_xm_targets/xm_targets/manifest.json`，原驱动查 `0103_xm_targets/manifest.json`，会被挡住。启动前报给管理员；驱动改成查 `xm_targets/manifest.json`（sha256 `ad743d0b…`），启动器的等待条件同步改。
3. **06 被别的项目占着**：11-03 07:43:05Z 在 06 上启动（`srun --jobid=157296 -c 12`、`EV_CAP_H=34`；wt0104 `52bc9331`…，12 个 sha256 相同，配置逐字节相同）。06 的 GPU 上有别的项目（lly-mle）的两个进程（`VLLM::EngineCore` 约 1.9 GB、`mle.main` 约 12.7 GB），驱动在 `claim_gpu` 里等，08:44:10Z 记 `blocked: gpu not free after 3600s`，未开始训练。没碰这两个进程。
4. **补充 2（D175）**：驱动可选节点（sha256 `1992b75a20c68e161ef77655ba41ae8fc8e37f513ce454738490715e96e2b984`）。`start0104b.sh` 等旧的 06 实例退出后，看哪台先空：RUN-0105 于 09:54:56Z 在 07 上完成、07 的 GPU 空出，选 07。确认两台上都没有别的 RUN-0104 驱动或训练在跑后，把 wt0104 更新到 `11909d94ad86c0db102145474fdb8f73f7b932c9`（含 55ae563f），12 个 sha256 相同，XM-2a 在，`configs/xm_train_exclusions.json`（`b5039567…`）与 `configs/xm_splits.json`（`995345f4…`）与 RUN-0103 的输出逐字节相同，训练目标的 manifest 在。
5. 09:59:19Z 在 07 上启动（`srun --jobid=157297 --overlap -n1 -c 12 env PATH=… TINY=0 NODE=hpcgpu07 EV_CAP_H=34 bash …/xm_train.sh`）。前后 squeue：157296、157297 RUNNING。
   - 启动器在新驱动写出心跳之前读到旧的 `blocked` 心跳，判成"第一步之前就结束"，没有起推送循环。另起一个只看 steps.log 的小脚本补上：10:02:57Z 起推送循环 TAG 0104（PID 2276517，路径 `experiments/XM/0104_xm_hpcgpu07`）。

## steps.log 原文（`experiments/XM/0104_xm_hpcgpu07/steps.log`）

```
2026-11-03T10:02:07Z start commit 11909d94ad86c0db102145474fdb8f73f7b932c9 node hpcgpu07 model qwen3.5-9b xm cfg 45c35566c37cd8f187c4fe67fd602d24f85a5b335836917eb9193b1ec0a6c1a0
2026-11-03T10:14:33Z xm_data rc=0 wall_s=746
2026-11-03T10:51:21Z xm_probe rc=0 wall_s=2208
2026-11-03T10:51:25Z probe median step_s 23.953751776367426 (max 54 for 2,000 steps within 30 h)
2026-11-04T00:16:42Z xm_train rc=0 wall_s=48317
2026-11-04T00:16:42Z checkpoint /home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e
2026-11-04T00:16:42Z done; failed steps: 0
2026-11-04T00:16:42Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M train 0104 done (failed steps: 0)
```

- 心跳 `2026-11-04T00:16:42Z done 0104 (failed steps: 0)`。`gpu_mem.csv` 50,839 行，最大 29,351 MiB。`logs/` 下日志无 Traceback / OOM。
- 11-03 19:30Z 前后 Mac→a100 的连接断过一次（三条线路都不通，19:45:24Z 第一次重试即恢复）；HPC 上的训练与推送不受影响，其间的读数取自推回仓库的 `train_log.jsonl`。

## 训练样本（`logs/xm_data.log` 末行原文）

```
X-M 数据：14400/310838 对（共 320464 对，剔除带 crowd 的 9626 对；13494 张图）→ 列框 14400 条、每实例样条 14477 条（切开的对 79，只进列框流的对 2）；要 {'coco_det': 14400, 'coco_inst': 14400}，够了 = True；每对框数 {'n': 14400, 'mean': 2.156875, 'min': 1.0, 'p50': 1.0, 'p90': 5.0, 'p99': 12.0, 'max': 15.0}
```

## 探测与 timing_summary

- `probe median step_s 23.953751776367426 (max 54 for 2,000 steps within 30 h)` → 续跑到 2,000 步。
- `timing_summary`（`src.w2.w2c_train.timing_summary`，取时叶目录已到第 66 步，故是第 5–66 步）：n_steps 62，step_s 中位 23.861、均值 24.439，峰值显存 28,918,862,848 字节。

| 流 | s 中位 | n_micro | 峰值显存（字节） |
|---|---|---|---|
| propose | 2.238 | 2 | 25,415,566,336 |
| select | 8.768 | 2 | 26,440,025,600 |
| spline | 2.601 | 2 | 26,960,166,400 |
| correct | 2.256 | 2 | 25,652,561,920 |
| coco_det | 2.078 | 2 | 26,331,399,680 |
| coco_inst | 4.043 | 3 | 28,918,862,848 |
| replay_general | 1.084 | 1 | 27,958,507,008 |
| replay_count | 0.503 | 1 | 24,521,296,384 |

## 训练（`train_log.jsonl` + `logs/timing_run_*.jsonl`，每 500 步；探针 `experiments/W2/0076_w2c_replay_probe/p76_train.py`）

| 步 | loss 均值 | 末步 loss | coco_det | coco_inst | correct | propose | replay_count | replay_general | select | spline | step_s 中位 | 峰值显存（字节） |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–499 | 4.2429 | 4.0918 | 0.7309 | 0.6647 | 0.6386 | 0.2968 | 0.5548 | 0.1506 | 0.6064 | 0.9529 | 23.80 | 28,932,132,864 |
| 500–999 | 3.8906 | 3.3619 | 0.7281 | 0.6143 | 0.5714 | 0.2931 | 0.4813 | 0.0750 | 0.5091 | 0.8964 | 24.00 | 28,930,814,464 |
| 1000–1499 | 3.6387 | 3.4144 | 0.7113 | 0.5915 | 0.5477 | 0.2911 | 0.2769 | 0.0389 | 0.4674 | 0.8718 | 24.02 | 28,928,447,488 |
| 1500–1999 | 3.5363 | 3.3793 | 0.7062 | 0.5897 | 0.5368 | 0.2880 | 0.2308 | 0.0133 | 0.4336 | 0.8600 | 24.09 | 28,941,765,120 |

- 2,000 步（0–1999）全部记录；探测 2,208 s，训练（续跑）48,317 s。
- 检查点 `0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`，manifest sha256 **`29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e`**。

## 收尾

- 推送循环 0104：11-04 00:28:08Z 在睡眠中按 PID 停（2276517），ONCE 推送 00:28:20Z ok。
- 后 squeue（00:28:20Z）：157296 RUNNING 80-11:54:06 hpcgpu06；157297 RUNNING 82-09:26:15 hpcgpu07。
- 07 上随即由 RUN-0106 接上（06 仍被 lly-mle 占着），RUN-0107 排在它之后。
