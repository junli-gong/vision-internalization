# RUN-0111 报告：hpcgpu06，Rex-Omni 外部参照（仅检测）与 X-M 在 W2 test 上（D165 第四、五节；D177、D178、D179；补充 1 / D180）

任务书 `prompts/server_hpc/0111-hpc-06-rexomni-xmtest.md`（含补充 1）。Rex-Omni 是单独运行的外部参照，输出不进本项目的数据或奖励。只贴数字，不解读。

## 权重（任务书第 2 步）

- hpcmgt 上 `$VI/tmp/rexomni_dl.sh`（工作树 wt_rexomni @ `c44681f9a`，`configs/rexomni_weights.json` `a4c77e86…`），11-04 12:04:49Z 起；经 hf-mirror（`HF_HUB_DISABLE_XET=1`）。
- `weights --download` rc=0，wall_s 12,079；`weights --verify` rc=0（15:27:14Z 结束）：

```
{
 "repo": "IDEA-Research/Rex-Omni",
 "revision": "0e5693d24657f6c0e091008dd6809bb1bd28988c",
 "dir": "/home/mail/chenyuma/visint/hf/hub/models--IDEA-Research--Rex-Omni/snapshots/0e5693d24657f6c0e091008dd6809bb1bd28988c",
 "n_files": 12,
 "total_bytes": 8136991717,
 "sha256_verified": true,
 "manifest_sha256": "a4c77e863d74010ea02e1feec50965ca237648decb7735ec6c710e3634752aae",
 "ok": true,
 "total_gb": 8.137
}
```

- df：前 `$VI` 可用 5,955 GB、`/` 685,884 MB；后 5,843 GB、684,372 MB。
- 每个 Rex-Omni 设置的 `config.yaml` 里 `model.weights` 同上（sha256_verified true，manifest `a4c77e86…`），bfloat16、sdpa。

## 启动

- 12:17:17Z 首次建工作树 wt0111 @ `7269aa98f`（16/16 + gate 17/17 相同）；原启动器等权重核对，12:20:01Z 按管理员要求撤下。12:20:10Z 改为先跑 xmtest 的启动器发现 06 上有别的计算进程（PID 793389，34,864 MiB，VLLM::EngineCore），未启动，报告管理员。
- 补充 1（D180）：wt0111 移到 `e209cb02e47bc1fa39bf4a51eb721e4d378d0618`，17 个 sha256 与任务书相同（驱动 `21817d27…`；`gate_xm.json` `d2f32d22…`）；`git diff --stat 7269aa98f e209cb02e -- src scripts configs` 为空。
- 启动器 `$VI/tmp/start0111c.sh`（12:26:58Z 起，每分钟查一次，只定一次）：12:27:00Z 06 仍忙；**12:33:05Z 06 上没有计算进程 → 走 (i)，NODE=hpcgpu06、`srun --jobid=157296`**。X-M 的 test 与 RUN-0092 同节点。
- 第一次：12:33:05Z `STAGES=xmtest`（`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 NODE=hpcgpu06 EV_CAP_H=14 STAGES=xmtest bash …/rexomni_xmtest.sh`，驱动日志 `eval_0111.driver.log`）。前后 squeue：157296、157297 RUNNING。
- 推送循环 TAG 0111（PID 3462362，12:34:05Z 起，路径 `experiments/XM/0111_rexomni_xmtest_hpcgpu06`），两次启动共用。
- 第二次：17:30:32Z `done 0111` 且权重 verify rc=0、节点上无 0111 驱动 → 17:31:19Z `STAGES=rexomni`（同一行，驱动日志 `eval_0111.rexomni.driver.log`）。前后 squeue：157296、157297 RUNNING。

## steps.log 原文

```
2026-11-04T12:33:05Z stages [xmtest]
2026-11-04T12:33:05Z start commit e209cb02e47bc1fa39bf4a51eb721e4d378d0618 node hpcgpu06 x-m checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e gate d2f32d22fcf11acf75e312fd93d7d2390f28c12376fe8b98a05c124ff7dfc8f0
2026-11-04T13:36:04Z xm_test_refcoco_unc_testA rc=0 wall_s=3778
2026-11-04T14:34:33Z xm_test_refcoco_unc_testB rc=0 wall_s=3509
2026-11-04T15:31:13Z xm_test_refcoco+_unc_testA rc=0 wall_s=3400
2026-11-04T16:32:22Z xm_test_refcoco+_unc_testB rc=0 wall_s=3669
2026-11-04T17:30:32Z xm_test_refcocog_umd_test rc=0 wall_s=3489
2026-11-04T17:30:32Z done; failed steps: 0
2026-11-04T17:30:32Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）Rex-Omni + X-M test 0111 done (failed steps: 0)
2026-11-04T17:31:20Z stages [rexomni]
2026-11-04T17:31:20Z start commit e209cb02e47bc1fa39bf4a51eb721e4d378d0618 node hpcgpu06 x-m checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e gate d2f32d22fcf11acf75e312fd93d7d2390f28c12376fe8b98a05c124ff7dfc8f0
2026-11-04T17:50:48Z rexomni_tune rc=0 wall_s=1168
2026-11-04T18:43:49Z rexomni_final rc=0 wall_s=3181
2026-11-04T19:28:46Z rexomni_lvis rc=0 wall_s=2697
2026-11-04T19:28:46Z done; failed steps: 0
2026-11-04T19:28:46Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）Rex-Omni + X-M test 0111 done (failed steps: 0)
```

- 心跳 `2026-11-04T19:28:46Z done 0111 (failed steps: 0)`。`logs/` 下 8 个日志无 Traceback / OutOfMemory。
- 用时（各步 wall_s 合计）：xmtest 17,845 s（4.96 h），rexomni 7,046 s（1.96 h）；各自一次启动，EV_CAP_H=14。

## A. Rex-Omni（`0111_rexomni_<阶段>/rex-omni/xm/<设置>/<划分>/metrics.json`；检测，框 IoU；f1_rexomni 95% CI，1,000 次重抽）

- 冒烟线（`smoke_line`，xm_tune / rexomni_det_joint）：statistic f1_rexomni_exact，threshold 0.6，value 0.7576，**pass True** → xm_final 与 lvis_sub 照跑。
- `xm_gate_unlock.jsonl`：xm_final、lvis_sub 各一行（`gate_xm.json` `d2f32d22…`，verdict 通过，`checkpoint_compared` false）；xm_tune 不带 gate。
- `data.pair_ids_sha256`：xm_final `2d53047c…`；lvis_sub `2d938d01a59e991fd4e3228407f74475e4f9360340574d6c137abd9ddb29559b`，与 RUN-0112 的 lvis_sub 全部设置相同。

| 划分 | 设置 | n | complete | f1_rexomni [CI] | P / R | f1_rexomni_exact (P / R) | f1_micro [CI] | matched_mean_iou | TP / FP / FN / ignored |
|---|---|---|---|---|---|---|---|---|---|
| xm_tune | rexomni_det_joint | 923 | True | 0.7414 [0.7177, 0.7812] | 0.7863 / 0.7015 | 0.7576 (0.8114 / 0.7106) | 0.6974 [0.6521, 0.7391] | 0.8291 | 1521 / 578 / 742 / 304 |
| xm_tune | rexomni_det | 923 | True | 0.7571 [0.7347, 0.7944] | 0.8110 / 0.7099 | 0.7759 (0.8423 / 0.7191) | 0.6986 [0.6438, 0.7452] | 0.8291 | 1533 / 593 / 730 / 274 |
| xm_final | rexomni_det_joint | 2940 | True | 0.7109 [0.6976, 0.7340] | 0.7627 / 0.6656 | 0.7160 (0.7747 / 0.6656) | 0.6992 [0.6804, 0.7174] | 0.8283 | 4777 / 1564 / 2546 / 599 |
| xm_final | rexomni_det | 2940 | True | 0.7078 [0.6945, 0.7303] | 0.7673 / 0.6569 | 0.7118 (0.7766 / 0.6569) | 0.7016 [0.6855, 0.7174] | 0.8286 | 4739 / 1447 / 2584 / 540 |
| lvis_sub | rexomni_det_joint | 2213 | True | 0.6932 [0.6854, 0.7227] | 0.7880 / 0.6187 | 0.7720 (0.8790 / 0.6881) | 0.5896 [0.5516, 0.6278] | 0.8196 | 3465 / 767 / 4056 / 536 |
| lvis_sub | rexomni_det | 2213 | True | 0.6926 [0.6846, 0.7223] | 0.8012 / 0.6100 | 0.7731 (0.8867 / 0.6852) | 0.5896 [0.5527, 0.6272] | 0.8201 | 3462 / 761 / 4059 / 630 |

解析、截断、未问类名、token 与用时（`metrics.json` 的 `summary`）：

| 划分 | 设置 | 图 / 类 | 调用 | failure | 解析失败 | 截断 | stop_reason | n_unasked_category | parse n_other_category | n_incomplete_group | 框：解析 / capped / 出格 / 反向 / 非框 | 预测框数 | 生成 token 均值 / 中位 / 最大 | prompt token 均值 | 视觉 token 均值 | 每次调用 s 均值 / 中位 / 最大 | 每对 s 均值 | 调用总 s | timing 缺失 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xm_tune | rexomni_det_joint | 300 / 78 | 300 | {"ok": 300} | 0 | 1 | {"eos": 299, "max_new_tokens": 1} | 0 | 0 | 0 | 2523 / 120 / 0 / 7 / 0 | 2403 (p50 1, max 100) | 73.1 / 38 / 3645 | 394.0 | 345.2 | 1.813 / 0.955 / 85.88 | 0.589 | 544 | 0 |
| xm_tune | rexomni_det | 300 / 78 | 923 | {"ok": 923} | 0 | 1 | {"eos": 922, "max_new_tokens": 1} | 0 | 0 | 1 | 3109 / 709 / 0 / 0 / 1 | 2400 (p50 1, max 100) | 22.5 / 11 / 3661 | 390.5 | 346.2 | 0.605 / 0.340 / 86.63 | 0.605 | 558 | 0 |
| xm_final | rexomni_det_joint | 1000 / 79 | 1000 | {"ok": 1000} | 0 | 1 | {"eos": 999, "max_new_tokens": 1} | 0 | 0 | 0 | 7222 / 282 / 0 / 0 / 0 | 6940 (p50 1, max 100) | 58.0 / 37 / 3647 | 393.5 | 345.0 | 1.443 / 0.949 / 85.75 | 0.491 | 1443 | 0 |
| xm_final | rexomni_det | 1000 / 79 | 2940 | {"ok": 2940} | 0 | 0 | {"eos": 2940} | 0 | 2 | 0 | 6726 / 0 / 0 / 0 / 0 | 6726 (p50 1, max 44) | 17.1 / 11 / 225 | 389.7 | 345.5 | 0.468 / 0.334 / 5.32 | 0.468 | 1377 | 0 |
| lvis_sub | rexomni_det_joint | 1000 / 446 | 1000 | {"ok": 1000} | 0 | 2 | {"eos": 998, "max_new_tokens": 2} | 15 | 0 | 2 | 6037 / 1269 / 0 / 0 / 2 | 4768 (p50 1, max 100) | 45.0 / 35 / 3702 | 394.3 | 346.4 | 1.148 / 0.892 / 88.28 | 0.519 | 1148 | 0 |
| lvis_sub | rexomni_det | 1000 / 446 | 2213 | {"ok": 2213} | 0 | 2 | {"eos": 2211, "max_new_tokens": 2} | 0 | 19 | 2 | 6153 / 1300 / 0 / 0 / 2 | 4853 (p50 1, max 100) | 20.1 / 12 / 3707 | 391.3 | 346.7 | 0.548 / 0.355 / 86.34 | 0.548 | 1213 | 0 |

- 两列"未问类名"按字段原名列出：`summary.n_unasked_category` 与 `summary.parse_counts.n_other_category`。
- joint 每次调用的对数（xm_final）：均值 2.94，p50 2，p90 5，最大 13。

## B. X-M 在 W2 test 上（`0111_xm_test/qwen3.5-9b/refseg/<设置>/<划分>/metrics.json`；95% CI，1,000 次重抽）

- 06 上，与 RUN-0092 的 X-2 同节点。合并核对 `w2c_merge_check.json`：pass True，failures merge []。
- 放行记录 `test_unlock.jsonl`：5 行，每个划分一行；approval D165，gate G-XM，`gate_xm.json` sha256 `d2f32d22…`，candidate_setting x2_spline_trimap，checkpoint manifest `29e26667…`，settings [x2_spline_trimap, x2_ellipse_trimap]；unlocked_utc 12:33:39Z、13:36:06Z、14:34:35Z、15:31:16Z、16:32:25Z。

| 划分 | 设置 | n | cIoU [95% CI] | gIoU [95% CI] | complete | failures | spline_failure | fallback |
|---|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | x2_spline_trimap | 1000 | 0.7522 [0.7342, 0.7688] | 0.7487 [0.7373, 0.7598] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testA | x2_ellipse_trimap | 1000 | 0.5564 [0.5427, 0.5699] | 0.5727 [0.5623, 0.5840] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testB | x2_spline_trimap | 1000 | 0.7267 [0.7044, 0.7493] | 0.7389 [0.7234, 0.7546] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco_unc_testB | x2_ellipse_trimap | 1000 | 0.6050 [0.5869, 0.6233] | 0.6374 [0.6239, 0.6520] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testA | x2_spline_trimap | 1000 | 0.7278 [0.7054, 0.7484] | 0.7345 [0.7211, 0.7472] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testA | x2_ellipse_trimap | 1000 | 0.5430 [0.5277, 0.5592] | 0.5642 [0.5518, 0.5760] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testB | x2_spline_trimap | 1000 | 0.6580 [0.6343, 0.6802] | 0.6783 [0.6610, 0.6959] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcoco+_unc_testB | x2_ellipse_trimap | 1000 | 0.5572 [0.5378, 0.5767] | 0.5832 [0.5674, 0.5983] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcocog_umd_test | x2_spline_trimap | 1000 | 0.7012 [0.6799, 0.7235] | 0.6960 [0.6813, 0.7108] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |
| test_refcocog_umd_test | x2_ellipse_trimap | 1000 | 0.5408 [0.5229, 0.5578] | 0.5528 [0.5385, 0.5670] | True | {"ok": 1000} | {"ok": 1000} | {"ok": 1000} |

- 表由 `experiments/X/0092_x_test_probe/p92_split.py` 生成（与 RUN-0092 同一脚本）。
- 每句调用与生成 token（`x2_summary/<划分>.json`）：样条 2.00 次调用，204.7–205.1 token（refcocog 210.2）；椭圆 1.00 次调用，45.2–45.8 token（refcocog 50.5）。
- 每划分耗时 3,400–3,778 s（见 steps.log）。

## 需管理员知道的事实

- 推送循环按任务书路径推了整个 `experiments/XM/0111_rexomni_xmtest_hpcgpu06`，其中 Rex-Omni 的 `predictions.jsonl` 有 5 个已在 git 里（xm_tune 两个设置、xm_final 的 rexomni_det、lvis_sub 两个设置），xm_final 的 rexomni_det_joint 那个按大文件只进了 `LARGE_FILES.manifest.tsv`。Rex-Omni 为 IDEA License 1.0（非商用）。是否保留在仓库里由管理员定；我没有改动。

## 收尾

- 推送循环 0111：20:04:58Z 在睡眠中按 PID 停（3462362，sleep 3953625），ONCE 推送 20:05:15Z ok。
- 后 squeue（20:05:15Z）：157296 RUNNING 81-07:31:01 hpcgpu06；157297 RUNNING 83-05:03:10 hpcgpu07。
