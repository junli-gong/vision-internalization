# RUN-0109 报告：hpcgpu07，X-M 与基座在最终报告集上，X-M 在 ReasonSeg val 上（D165 第五节；D177）

任务书 `prompts/server_hpc/0109-hpc-07-xm-final.md`。G-XM 已通过，最终报告集与 ReasonSeg 已揭盲。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0109.sh`（11-04 08:25:34Z 起）。前 squeue：157296 RUNNING 80-19:51:20 hpcgpu06；157297 RUNNING 82-17:23:29 hpcgpu07。
- fetch 第 1 次 rc=0；wt0109（稀疏）HEAD `671599d775c7ee63ecd61dc73eb5833019e35dea`，status 空；12 个工作树 sha256 与任务书相同（`xm_final.sh` `a96ee54a…`），REPO 里的 `gate_xm.json` `d2f32d22…` 也相同（13/13）；驱动用本节点暂停文件。
- X-M 检查点 manifest `29e26667…`；07 上 RUN-0108 已 `done 0108`（08:15:52Z）。
- 08:28:54Z 按任务书命令启动（`srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=10 bash …/xm_final.sh`）。后 squeue：157296、157297 RUNNING。推送循环 TAG 0109（PID 3749336）。

## steps.log 原文

```
2026-11-04T08:29:14Z stages [xm base reasonseg]
2026-11-04T08:29:14Z start commit 671599d775c7ee63ecd61dc73eb5833019e35dea node hpcgpu07 model qwen3.5-9b x-m checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e gate d2f32d22fcf11acf75e312fd93d7d2390f28c12376fe8b98a05c124ff7dfc8f0
2026-11-04T11:56:53Z xm_final rc=0 wall_s=12459
2026-11-04T13:33:21Z base_final rc=0 wall_s=5788
2026-11-04T15:29:28Z reasonseg_xm rc=0 wall_s=6967
2026-11-04T15:29:28Z done; failed steps: 0
2026-11-04T15:29:28Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M final/ReasonSeg 0109 done (failed steps: 0)
```

- 心跳 `2026-11-04T15:29:28Z done 0109 (failed steps: 0)`。`logs/` 下三份日志无 Traceback / OutOfMemory。
- 三个叶目录的 `xm_gate_unlock.jsonl` 各一行：`gate_xm.json` sha256 `d2f32d22…`，verdict 通过，candidate manifest `29e26667…`；`checkpoint_compared` 在 xm_final 与 reasonseg_val 为 true，在 base_final 为 false。unlocked_utc：xm_final 08:38:37Z，base_final 12:03:03Z，reasonseg_val 13:40:26Z。

## 最终报告集 `xm_final`（1,000 张图，2,940 对；`<叶>/qwen3.5-9b/xm/<设置>/xm_final/metrics.json`；f1_rexomni，95% CI）

| 设置 | n | complete | 检测 f1 [CI] | 检测 P / R | 实例分割 f1 [CI] | 分割 P / R |
|---|---|---|---|---|---|---|
| xm_spline_trimap | 2940 | True | 0.6953 [0.6823, 0.7254] | 0.7176 / 0.6744 | 0.5848 [0.5687, 0.6150] | 0.5969 / 0.5731 |
| xm_box_grabcut | 2940 | True | 0.6953 [0.6823, 0.7254] | 0.7176 / 0.6744 | 0.4120 [0.3949, 0.4386] | 0.4191 / 0.4052 |
| xm_ellipse_trimap | 2940 | True | 0.6953 [0.6823, 0.7254] | 0.7176 / 0.6744 | 0.5121 [0.5010, 0.5419] | 0.5244 / 0.5003 |
| xm_spline_raster | 2940 | True | 0.6953 [0.6823, 0.7254] | 0.7176 / 0.6744 | 0.5727 [0.5570, 0.6047] | 0.5853 / 0.5606 |
| base_det | 2940 | True | 0.6425 [0.6290, 0.6768] | 0.6260 / 0.6598 | – | – |
| base_box_grabcut | 2940 | True | 0.6425 [0.6290, 0.6768] | 0.6260 / 0.6598 | 0.3929 [0.3758, 0.4214] | 0.3828 / 0.4036 |

失败与计数（`failures`，不含 `n_pred_boxes`）：

| 设置 | list_failure | list_counts（n_raw / n_capped / n_clipped / n_inverted） | nms_suppressed | trace_failure | spline_failure | fallback | tool_not_ok |
|---|---|---|---|---|---|---|---|
| xm_spline_trimap | ok 2933，truncated 7 | 7729 / 99 / 0 / 0 | 39 | ok 2914，count_mismatch 19 | ok 7155，count_mismatch 436 | ok 7155，box_grabcut 436 | 15 |
| xm_box_grabcut | 同上 | 同上 | 39 | – | – | ok 7591 | 6 |
| xm_ellipse_trimap | 同上 | 同上 | 39 | – | – | ok 7591 | 0 |
| xm_spline_raster | 同上 | 同上 | 39 | ok 2914，count_mismatch 19 | ok 7155，count_mismatch 436 | ok 7155，box_grabcut 436 | 0 |
| base_det | ok 2837，truncated 99，out_of_range 2，inverted 1，no_box 1 | 10244 / 865 / 0 / 0 | 290 | – | – | – | – |
| base_box_grabcut | 同上 | 同上 | 290 | – | – | ok 9089 | 56 |

- 合并核对（`0109_xm_final/qwen3.5-9b/*merge_check.json`）：pass True，failures merge []。基座叶目录不带检查点，没有合并核对文件。

## ReasonSeg val（X-M 检查点，`--system xm`，缺省设置 x2_spline_trimap；`0109_reasonseg_xm/qwen3.5-9b/`）

- 合并核对 `w2c_merge_check.json`：pass True，failures merge []。n = 196（n_expected 196）；complete True；`failures`、`fallback`、`spline_failure` 都是 `{"ok": 196}`。

| x2_spline_trimap | gIoU | cIoU |
|---|---|---|
| W2 格式（`refseg/x2_spline_trimap/reasonseg_val/metrics.json` 的 headline，不扣忽略区域）[95% CI] | 0.5448 [0.5017, 0.5883] | 0.4624 [0.3749, 0.5584] |
| `reasonseg_readout/reasonseg_val.json`，扣掉忽略区域（`meta.ignore_rle` 的像素不进交与并，同 LISA 的 ignore_index；真值空且预测空记 1） | 0.5804 | 0.5191 |

- 196 条里有忽略区域的 76 条。readout 的 `without_ignore`：gIoU 0.5448、cIoU 0.4624（与 headline 相同）。
- 每条：模型调用 2、工具调用 2（196 条全同）；prompt token 中位 754（664–814），生成中位 219（190–251），视觉中位 594（532–660），inserted 0。
- `compute.json`：n_calls 392，generated_tokens 43,111，prompt_tokens 147,911，visual_tokens 115,540，call_time_s 115.7，wall_time_s 10,497.1。

## 用时

- xm_final 12,459 s；base_final 5,788 s；reasonseg_xm 6,967 s；合计 25,214 s（7.00 h），上限 EV_CAP_H=10。

## 收尾

- 推送循环 0109：15:55:58Z 在睡眠中按 PID 停（3749336，sleep 1647604），ONCE 推送 15:56:07Z ok。
- 后 squeue（15:56:07Z）：157296 RUNNING 81-03:21:53 hpcgpu06；157297 RUNNING 83-00:54:02 hpcgpu07。
- 07 上随后由 RUN-0112 接上（15:30:07Z running xm_lvis，推送循环 0112 PID 1585060）。
