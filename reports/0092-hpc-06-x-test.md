# RUN-0092 报告：hpcgpu06，通过 G-EX 的两个候选在 W2 的 test 上（X-2 = C1、X-3 = C2；样条与椭圆两个设置）

任务书 `prompts/server_hpc/0092-hpc-06-x-test.md`（预注册 X-6、X-6a；D147、D148、D149）。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0092.sh`（10-28 23:27:51Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0092（稀疏）HEAD `e1f15977702165d88384d7bbca53a5d2c544e0ff`（含 1f0c026d），status 空，`experiments/` 只有 P0；11 个 sha256 与任务书逐字相同（`x_test.sh` `8eeec087…`）；驱动用本节点暂停文件；`configs/w2.json` frozen True。
- 候选 manifest：x2 `a49cf4e5…`、x3 `837bb679…`，与任务书一致，`read_manifest` 都通过（各 4 个文件）；gate 文件 `gate_x2/gate_explore.json`（sha `ed2b6c5c…`）、`gate_x3/gate_explore.json`（sha `dd7f7649…`）都在。`CANDS` 照任务书写入，展开后 6 个路径都存在。
- 23:28:58Z 按任务书命令启动（`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=14 CANDS=… bash …/x_test.sh`）；驱动等 RUN-0091（`waiting for eval_0091.status to start with 'done 0091'`）。后 squeue：157296、157297 RUNNING。
- 01:34:35Z RUN-0091 结束后开始；推送循环 TAG 0092（PID 3377623；三个路径 `0092_x_test_hpcgpu06`、`0092_x2_test_hpcgpu06`、`0092_x3_test_hpcgpu06`）。
- `run_x2` 对两个候选的 `--test-gate … --test-approval D147` 核对都通过（各划分 rc=0）。

## steps.log 原文

```
2026-10-29T01:34:35Z start commit e1f15977702165d88384d7bbca53a5d2c544e0ff node hpcgpu06 cands [x2|/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model|/home/mail/chenyuma/visint/repo/experiments/X/0090_gex_hpcgpu07/gate_x2/gate_explore.json x3|/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model|/home/mail/chenyuma/visint/repo/experiments/X/0090_gex_hpcgpu07/gate_x3/gate_explore.json]
2026-10-29T01:34:35Z candidate x2 checkpoint a49cf4e5a81fd8f7f57ca5c86a879cf24cdd1449daa43cc355e6d3a074a6fd0a gate ed2b6c5c28ba92b3b33e8638f01446af4c6be33fadfae18e4ebffaafbaf2b970
2026-10-29T02:37:04Z x2_test_refcoco_unc_testA rc=0 wall_s=3749
2026-10-29T03:35:20Z x2_test_refcoco_unc_testB rc=0 wall_s=3496
2026-10-29T04:32:21Z x2_test_refcoco+_unc_testA rc=0 wall_s=3421
2026-10-29T05:33:18Z x2_test_refcoco+_unc_testB rc=0 wall_s=3657
2026-10-29T06:30:55Z x2_test_refcocog_umd_test rc=0 wall_s=3457
2026-10-29T06:30:55Z candidate x3 checkpoint 837bb679cfc172ecfa4bce3af1ae21c4e1774d3d9fdf0e16fc4e5d2c34ed7f5c gate dd7f764940c6c966528b7d1636a0c577d2be5cb03c498a5a0c54b97fdfbe7dd9
2026-10-29T07:03:38Z x3_test_refcoco_unc_testA rc=0 wall_s=1963
2026-10-29T07:31:26Z x3_test_refcoco_unc_testB rc=0 wall_s=1668
2026-10-29T07:59:31Z x3_test_refcoco+_unc_testA rc=0 wall_s=1685
2026-10-29T08:27:38Z x3_test_refcoco+_unc_testB rc=0 wall_s=1687
2026-10-29T08:56:01Z x3_test_refcocog_umd_test rc=0 wall_s=1703
2026-10-29T08:56:01Z done; failed steps: 0
2026-10-29T08:56:01Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X test 0092 done (failed steps: 0)
```

- 心跳 `2026-10-29T08:56:01Z done 0092 (failed steps: 0)`。总时长约 7.4 小时（上限 14 小时）。`logs/` 下 10 个日志均无 Traceback / OOM。

## 各划分读数（每个设置的 `metrics.json`；95% CI，1,000 次重抽）

## 0092_x2_test_hpcgpu06

| 划分 | 设置 | n | cIoU [95% CI] | gIoU [95% CI] | complete | failures | spline_failure | fallback |
|---|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | x2_spline_trimap | 1000 | 0.7558 [0.7382, 0.7718] | 0.7517 [0.7405, 0.7624] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testA | x2_ellipse_trimap | 1000 | 0.5592 [0.5459, 0.5718] | 0.5764 [0.5661, 0.5870] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testB | x2_spline_trimap | 1000 | 0.7124 [0.6909, 0.7337] | 0.7245 [0.7083, 0.7392] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testB | x2_ellipse_trimap | 1000 | 0.6043 [0.5869, 0.6225] | 0.6315 [0.6182, 0.6463] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testA | x2_spline_trimap | 1000 | 0.7335 [0.7131, 0.7548] | 0.7372 [0.7244, 0.7504] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testA | x2_ellipse_trimap | 1000 | 0.5477 [0.5326, 0.5643] | 0.5686 [0.5565, 0.5805] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testB | x2_spline_trimap | 1000 | 0.6585 [0.6328, 0.6831] | 0.6765 [0.6580, 0.6935] | True | {"ok": 1000} | {"ok": 999, "wrong_count": 1} | {"ok": 999, "parse_failure": 1} |
| test_refcoco+_unc_testB | x2_ellipse_trimap | 1000 | 0.5598 [0.5398, 0.5788] | 0.5850 [0.5674, 0.6003] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcocog_umd_test | x2_spline_trimap | 1000 | 0.6937 [0.6672, 0.7168] | 0.6943 [0.6796, 0.7093] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcocog_umd_test | x2_ellipse_trimap | 1000 | 0.5402 [0.5218, 0.5575] | 0.5574 [0.5441, 0.5713] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |

## 0092_x3_test_hpcgpu06

| 划分 | 设置 | n | cIoU [95% CI] | gIoU [95% CI] | complete | failures | spline_failure | fallback |
|---|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | x2_spline_trimap | 1000 | 0.7596 [0.7418, 0.7758] | 0.7549 [0.7438, 0.7651] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testA | x2_ellipse_trimap | 1000 | 0.5599 [0.5459, 0.5738] | 0.5756 [0.5653, 0.5862] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testB | x2_spline_trimap | 1000 | 0.7148 [0.6908, 0.7376] | 0.7311 [0.7142, 0.7467] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testB | x2_ellipse_trimap | 1000 | 0.5976 [0.5792, 0.6168] | 0.6282 [0.6141, 0.6427] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testA | x2_spline_trimap | 1000 | 0.7333 [0.7120, 0.7537] | 0.7352 [0.7220, 0.7486] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testA | x2_ellipse_trimap | 1000 | 0.5449 [0.5300, 0.5611] | 0.5640 [0.5522, 0.5758] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testB | x2_spline_trimap | 1000 | 0.6658 [0.6401, 0.6884] | 0.6790 [0.6608, 0.6972] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testB | x2_ellipse_trimap | 1000 | 0.5629 [0.5430, 0.5809] | 0.5845 [0.5679, 0.6005] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcocog_umd_test | x2_spline_trimap | 1000 | 0.7112 [0.6872, 0.7329] | 0.7063 [0.6912, 0.7202] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcocog_umd_test | x2_ellipse_trimap | 1000 | 0.5464 [0.5284, 0.5631] | 0.5621 [0.5476, 0.5761] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |

- 表由 `experiments/X/0092_x_test_probe/p92_split.py` 从 `metrics.json` 生成。`failures` 一列即解析失败的口径：除 x2 的 refcoco+ testB 样条一行（`spline_failure` wrong_count 1、`fallback` parse_failure 1）外，所有行都是 `{"ok": 1000}`。
- 每句调用与生成 token（`x2_summary/<划分>.json`）：样条 2.00 次调用，约 205 token（refcocog 约 210）；椭圆 1.00 次调用，约 45 token（refcocog 约 50.5）。两个候选相同到 ±0.6 token。
- 每划分耗时：x2 3,421–3,749 s；x3 1,668–1,963 s（见 steps.log）。

## 收尾

- 推送循环 0092：09:02:24Z 在睡眠中按 PID 停（3377623 + sleep 730469），ONCE 推送 09:02:41Z ok。
- 后 squeue（09:02:41Z）：157296 RUNNING 74-20:28:27 hpcgpu06；157297 RUNNING 76-18:00:36 hpcgpu07。
- 06 现在空闲；07 上 RUN-0093 在训练。
