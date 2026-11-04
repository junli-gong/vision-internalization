# RUN-0106 报告：X-M 判定集的评测（X-M 4 个设置与基座零样本，都在 07；首次运行与补充 1 的重跑）（D165 第五节；XM-4b、XM-5；D176）

任务书 `prompts/server_hpc/0106-hpc-xm-decision.md`（含"补充 1"）。**判定集的读数在门报告出来之前不看、不报**：本报告只有 complete、失败数与用时；调参集上 X-M 的读数可报。

## 节点

- RUN-0104 训完（11-04 00:16:42Z）时，06 的 nvidia-smi 有别的项目（lly-mle）的计算进程（`VLLM::EngineCore` 35,300 MiB、`mle.main` 908 MiB），按任务书两项都在 07：本任务先跑（`NODE=hpcgpu07`、`srun --jobid=157297 -c 8`、`EV_CAP_H=6`），RUN-0107 排在它之后。
- 启动器 `start0106_0107.sh`：wt0106 `b3e96ee4…`，11 个 sha256 相同；X-M 检查点 manifest `29e26667…`。前后 squeue：157296、157297 RUNNING。推送循环 TAG 0106（路径 `experiments/XM/0106_xm_decision_hpcgpu07`）。

## 首次运行（00:17Z 起）

- xm_tune、xm_decision 两步都在评测前的合并核对处失败（rc=1）：`拒绝评测：评测前的合并核对没有通过（退出码 2）：FileNotFoundError(2, 'No such file or directory')`。`xm_merge_check.json` 的 argv 以 `--data-root /home/mail/chenyuma/visint/data` 结尾；回溯在 `check_training_interfaces.load_dev → src/e1/tasks._refcoco_dev → load_refcoco`，即在 `$VI/data` 下找 E1 的 RefCOCO 开发样本。原因：驱动没给 `--merge-check-data-root`（RUN-0105 给了），run_xm 退回 `--data-root`。两步都没有写出读数。
- base_decision 不带检查点、不做合并核对：rc=0，2,728 s。
- 失败的 `0106_xm_tune`、`0106_xm_dec` 两个目录原样留着，不删、不用。

## 补充 1（D176）：只重跑 X-M 两步——用的是 (b)

- `start0106r2.sh`：等到首次运行写出 `done 0106 (failed steps: 2)`（01:04:35Z）、RUN-0107 已开跑、两台上都没有 0106 的驱动在跑；按 PID 停首次运行的推送循环并 ONCE 推送（01:05:58Z ok）；wt0106 更新到 `2af4a4bd9bc070e136c7c60581cad31a9d061ed2`（含 cb2a528b），11 个 sha256 相同（驱动 `60d470a6…`），驱动带 `--merge-check-data-root`。
- 01:06Z 06 仍有 lly-mle 的进程（`VLLM::EngineCore` 1,884 MiB、`mle.main` 13,328 MiB）→ **(b)**：01:06:15Z 在 07 上启动（`NODE=hpcgpu07`、`srun --jobid=157297 -c 8`、`EV_CAP_H=5`、`STAGES='tune decision'`、`SUF=_r2`、`WAIT_DONE='eval_0107.status:done 0107'`，日志 `eval_0106.r2.driver.log`），RUN-0107 结束后 05:14:27Z 开始。前后 squeue：157296、157297 RUNNING。
- 判定用 `0106_xm_dec_r2` 与首次运行的 `0106_base_dec`，两者都在 07。

## steps.log 原文

```
2026-11-04T00:17:53Z start commit b3e96ee4a090167fa2c9345719d586ca08cbf861 node hpcgpu07 model qwen3.5-9b x-m checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e
2026-11-04T00:19:02Z xm_tune rc=1 wall_s=68
2026-11-04T00:19:07Z xm_decision rc=1 wall_s=5
2026-11-04T01:04:35Z base_decision rc=0 wall_s=2728
2026-11-04T01:04:35Z done; failed steps: 2
2026-11-04T01:04:35Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M decision eval 0106 done (failed steps: 2)
2026-11-04T05:14:27Z stages [tune decision] suffix [_r2]
2026-11-04T05:14:27Z start commit 2af4a4bd9bc070e136c7c60581cad31a9d061ed2 node hpcgpu07 model qwen3.5-9b x-m checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e
2026-11-04T06:23:30Z xm_tune rc=0 wall_s=4143
2026-11-04T08:04:35Z xm_decision rc=0 wall_s=6065
2026-11-04T08:04:35Z done; failed steps: 0
2026-11-04T08:04:35Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M decision eval 0106 done (failed steps: 0)
```

- 心跳 `2026-11-04T08:04:35Z done 0106 (failed steps: 0)`。`logs/` 下无 Traceback / OOM。
- 合并核对：`0106_xm_tune_r2`、`0106_xm_dec_r2` 都 pass True，failures merge []；首次运行的两份 pass False。

## 调参集上的 X-M（`0106_xm_tune_r2`，n = 923，complete True；95% CI，1,000 次重抽）

- 检测 f1_rexomni：0.7388 [0.7149, 0.7796]（四个设置相同，同一次调用的框）。

| 设置 | 实例分割 f1_rexomni |
|---|---|
| xm_spline_trimap | 0.6517 [0.6240, 0.6913] |
| xm_spline_raster | 0.6421 [0.6160, 0.6816] |
| xm_ellipse_trimap | 0.5600 [0.5308, 0.6017] |
| xm_box_grabcut | 0.4243 [0.3910, 0.4710] |

- 失败：list_failure ok 921、truncated 2；n_raw 2,392、capped 0；nms_suppressed 25；样条设置 trace_failure count_mismatch 3、spline_failure count_mismatch 81（回退 box_grabcut 81）；tool_not_ok：box_grabcut 3，其余 0。

## 判定集（`xm_decision`；**只报 complete 与失败数**）

| 叶目录 | n | complete | list_failure | list_counts | nms_suppressed | 其余 |
|---|---|---|---|---|---|---|
| 0106_base_dec / base_det | 1437/1437 | True | ok 1368、truncated 67、malformed 1、out_of_range 1 | n_raw 4,641、capped 359 | 130 | — |
| 0106_base_dec / base_box_grabcut | 1437/1437 | True | 同上 | 同上 | 130 | fallback ok 4,152；tool_not_ok 10 |
| 0106_xm_dec_r2 / xm_spline_trimap | 1437/1437 | True | ok 1436、truncated 1 | n_raw 3,901、capped 109 | 35 | trace_failure count_mismatch 14；spline_failure count_mismatch 366（回退 box_grabcut 366）；tool_not_ok 19 |
| 0106_xm_dec_r2 / xm_spline_raster | 1437/1437 | True | 同上 | 同上 | 35 | 同上的 trace/spline 回退；tool_not_ok 0 |
| 0106_xm_dec_r2 / xm_ellipse_trimap | 1437/1437 | True | 同上 | 同上 | 35 | tool_not_ok 17 |
| 0106_xm_dec_r2 / xm_box_grabcut | 1437/1437 | True | 同上 | 同上 | 35 | tool_not_ok 21 |

## 收尾

- 推送循环 0106（重跑）：08:08:31Z 按 PID 停，ONCE 推送 ok。后 squeue：157296 RUNNING 80-19:34:17 hpcgpu06；157297 RUNNING 82-17:06:26 hpcgpu07。没碰 lly-mle 的进程。
