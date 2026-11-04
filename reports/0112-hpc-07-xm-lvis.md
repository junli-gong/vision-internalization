# RUN-0112 报告：hpcgpu07，X-M 与基座零样本在 LVIS 子集上（描述性；D165 第五节；XM-2b，D179）

任务书 `prompts/server_hpc/0112-hpc-07-xm-lvis.md`。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0111_0112.sh`（11-04 12:17:17Z 起）。前 squeue：157296 RUNNING 80-23:43:03 hpcgpu06；157297 RUNNING 82-21:15:12 hpcgpu07。
- fetch 第 1 次 rc=0；wt0112（稀疏）HEAD `7269aa98fd8ea0785419bed5420ccb41d77b1aba`，status 空；11 个工作树 sha256 与任务书相同，REPO 里的 `gate_xm.json` `d2f32d22…` 也相同（12/12）；驱动用本节点暂停文件。
- 12:17:35Z 按任务书命令启动（`srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=8 bash …/xm_lvis.sh`）；驱动先等 `eval_0109.status`（12:17:37Z `waiting for eval_0109.status to start with 'done 0109' (now: running base_final)`）。
- 15:30:07Z RUN-0109 结束后开始；推送循环 TAG 0112（PID 1585060，15:30:12Z 起，由 `$VI/tmp/sync0112.sh` 在第一步出现后启动）。

## steps.log 原文

```
2026-11-04T15:30:07Z stages [xm base]
2026-11-04T15:30:07Z start commit 7269aa98fd8ea0785419bed5420ccb41d77b1aba node hpcgpu07 model qwen3.5-9b x-m checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e gate d2f32d22fcf11acf75e312fd93d7d2390f28c12376fe8b98a05c124ff7dfc8f0
2026-11-04T17:52:55Z xm_lvis rc=0 wall_s=8568
2026-11-04T18:58:15Z base_lvis rc=0 wall_s=3920
2026-11-04T18:58:15Z done; failed steps: 0
2026-11-04T18:58:15Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M LVIS 0112 done (failed steps: 0)
```

- 心跳 `2026-11-04T18:58:15Z done 0112 (failed steps: 0)`。`logs/` 下日志无 Traceback / OutOfMemory。
- 用时：xm_lvis 8,568 s，base_lvis 3,920 s，合计 12,488 s（3.47 h），上限 EV_CAP_H=8。
- `xm_gate_unlock.jsonl`：两个叶目录各一行，`gate_xm.json` `d2f32d22…`，verdict 通过，candidate manifest `29e26667…`；`checkpoint_compared` xm_lvis true、base_lvis false；unlocked_utc 15:41:27Z、17:55:14Z。

## `xm.lvis_cap`（每个设置的 `config.yaml` 相同）

| 项 | 值 |
|---|---|
| n_images | 1000 |
| max_cats_per_image | 3（salt `xm_lvis_cats_v1`） |
| n_images_capped | 327 |
| **n_pairs_before_cap** | **3134** |
| n_pairs_after_cap | 2213 |
| n_pairs_dropped | 921 |
| n_instances_before_cap / after_cap | 10163 / 7521 |
| n_not_exhaustive_pairs_after_cap | 172 |
| pairs_sha256_after_cap | `d9dfb2c8614578f05cb6747101a608dcbef367636a10af2afc7fea278a7f3de9` |

- `data.pair_ids_sha256`：所有 6 个设置（及 xm_trace、base_trace）都是 `2d938d01a59e991fd4e3228407f74475e4f9360340574d6c137abd9ddb29559b`，与 RUN-0111 的 Rex-Omni lvis_sub 两个设置相同。
- 检测口径的 n_images 1000、n_categories 446。

## 读数（`<叶>/qwen3.5-9b/xm/<设置>/lvis_sub/metrics.json`；f1_rexomni，95% CI）

| 设置 | n | complete | 检测 f1 [CI] | 检测 P / R | 检测 f1_rexomni_exact | 实例分割 f1 [CI] | 分割 P / R | 分割 f1_rexomni_exact |
|---|---|---|---|---|---|---|---|---|
| xm_spline_trimap | 2213 | True | 0.7202 [0.7093, 0.7467] | 0.8047 / 0.6518 | 0.7843 | 0.5946 [0.5773, 0.6204] | 0.6666 / 0.5367 | 0.7082 |
| xm_box_grabcut | 2213 | True | 0.7202 [0.7093, 0.7467] | 0.8047 / 0.6518 | 0.7843 | 0.3959 [0.3824, 0.4274] | 0.4474 / 0.3551 | 0.5726 |
| xm_ellipse_trimap | 2213 | True | 0.7202 [0.7093, 0.7467] | 0.8047 / 0.6518 | 0.7843 | 0.5226 [0.5063, 0.5504] | 0.5864 / 0.4712 | 0.6686 |
| xm_spline_raster | 2213 | True | 0.7202 [0.7093, 0.7467] | 0.8047 / 0.6518 | 0.7843 | 0.5619 [0.5453, 0.5898] | 0.6306 / 0.5066 | 0.6885 |
| base_det | 2213 | True | 0.6792 [0.6718, 0.7093] | 0.7210 / 0.6420 | 0.7551 | – | – | – |
| base_box_grabcut | 2213 | True | 0.6792 [0.6718, 0.7093] | 0.7210 / 0.6420 | 0.7551 | 0.3741 [0.3630, 0.4067] | 0.4039 / 0.3485 | 0.5540 |

失败与计数（`failures`，不含 `n_pred_boxes`）：

| 设置 | list_failure | list_counts（n_raw / n_capped / n_clipped / n_inverted） | nms_suppressed | trace_failure | spline_failure | fallback | tool_not_ok |
|---|---|---|---|---|---|---|---|
| xm_spline_trimap | ok 2211，truncated 2 | 5174 / 65 / 0 / 0 | 23 | ok 2206，count_mismatch 5 | ok 4977，count_mismatch 109 | ok 4977，box_grabcut 109 | 16 |
| xm_box_grabcut | 同上 | 同上 | 23 | – | – | ok 5086 | 4 |
| xm_ellipse_trimap | 同上 | 同上 | 23 | – | – | ok 5086 | 0 |
| xm_spline_raster | 同上 | 同上 | 23 | ok 2206，count_mismatch 5 | ok 4977，count_mismatch 109 | ok 4977，box_grabcut 109 | 0 |
| base_det | ok 2153，truncated 53，malformed 3，out_of_range 2，inverted 1，no_box 1 | 6159 / 316 / 0 / 0 | 125 | – | – | – | – |
| base_box_grabcut | 同上 | 同上 | 125 | – | – | ok 5718 | 19 |

- 每对模型调用：xm_spline_* 2 次（2211 对）/ 1 次（2 对）；xm_box_grabcut、xm_ellipse_trimap、base_* 1 次。
- 合并核对（`0112_xm_lvis/qwen3.5-9b/xm_merge_check.json`）：pass True，failures merge []。基座叶目录不带检查点，没有合并核对文件。

## 收尾

- 推送循环 0112：20:05:15Z 在睡眠中按 PID 停（1585060，sleep 104220），ONCE 推送 20:05:21Z ok。
- 后 squeue（20:05:21Z）：157296 RUNNING 81-07:31:07 hpcgpu06；157297 RUNNING 83-05:03:16 hpcgpu07。
