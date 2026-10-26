# RUN-0073 报告：hpcgpu07，W2 的 test 主实验之 W2-S（2B、4B 的 W2-a，5 个 test 划分 × 1,000 句）

任务书 `prompts/server_hpc/0073-hpc-07-w2-test-w2s.md`。test 只报告、不判定。没有碰 06。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0073.sh`（10-26 02:27:36Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0073（稀疏）HEAD `d9de1d4275301f5c126edebb7f3ed839798a0d9c`（含 d77ed1fd），status 空，`experiments/` 只有 P0。
- 12 个 sha256 与任务书逐字相同；`configs/w2.json` frozen True；驱动用本节点暂停文件；2B、4B 权重在 `$VI/hf/hub`（2B snapshot `15852e8c…`、4B snapshot `851bf6e8…`）；07 无暂停文件、GPU 空。
- 02:27:58Z 按任务书命令启动（`srun --jobid=157297 --overlap -n1 -c 8`，`EV_CAP_H=14 NODE=hpcgpu07 RUN_ID=0073 MODELS="qwen3.5-2b qwen3.5-4b"`）。`-c 8` 被接受。
- 推送循环 TAG 0073（PID 519248，1800 s）。启动后 squeue 两条 RUNNING。

## steps.log 原文

```
2026-10-26T02:27:59Z start commit d9de1d4275301f5c126edebb7f3ed839798a0d9c node hpcgpu07 models [qwen3.5-2b qwen3.5-4b] w2.json c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29
2026-10-26T02:33:31Z e1_box_qwen3.5-2b_test_refcoco_unc_testA rc=0 wall_s=332
2026-10-26T02:33:36Z ownbox_qwen3.5-2b_test_refcoco_unc_testA rc=0 wall_s=5
2026-10-26T02:39:15Z w1f_qwen3.5-2b_test_refcoco_unc_testA rc=0 wall_s=339
2026-10-26T03:44:22Z w2a_qwen3.5-2b_test_refcoco_unc_testA rc=0 wall_s=3907
2026-10-26T03:46:18Z e1_box_qwen3.5-2b_test_refcoco_unc_testB rc=0 wall_s=116
2026-10-26T03:46:22Z ownbox_qwen3.5-2b_test_refcoco_unc_testB rc=0 wall_s=4
2026-10-26T03:51:37Z w1f_qwen3.5-2b_test_refcoco_unc_testB rc=0 wall_s=315
2026-10-26T05:05:31Z w2a_qwen3.5-2b_test_refcoco_unc_testB rc=0 wall_s=4434
2026-10-26T05:09:10Z e1_box_qwen3.5-2b_test_refcoco+_unc_testA rc=0 wall_s=219
2026-10-26T05:09:21Z ownbox_qwen3.5-2b_test_refcoco+_unc_testA rc=0 wall_s=11
2026-10-26T05:16:50Z w1f_qwen3.5-2b_test_refcoco+_unc_testA rc=0 wall_s=449
2026-10-26T06:36:39Z w2a_qwen3.5-2b_test_refcoco+_unc_testA rc=0 wall_s=4789
2026-10-26T06:40:47Z e1_box_qwen3.5-2b_test_refcoco+_unc_testB rc=0 wall_s=248
2026-10-26T06:40:57Z ownbox_qwen3.5-2b_test_refcoco+_unc_testB rc=0 wall_s=10
2026-10-26T06:49:25Z w1f_qwen3.5-2b_test_refcoco+_unc_testB rc=0 wall_s=508
2026-10-26T07:59:37Z w2a_qwen3.5-2b_test_refcoco+_unc_testB rc=0 wall_s=4211
2026-10-26T08:01:22Z e1_box_qwen3.5-2b_test_refcocog_umd_test rc=0 wall_s=105
2026-10-26T08:01:27Z ownbox_qwen3.5-2b_test_refcocog_umd_test rc=0 wall_s=5
2026-10-26T08:06:31Z w1f_qwen3.5-2b_test_refcocog_umd_test rc=0 wall_s=304
2026-10-26T09:11:07Z w2a_qwen3.5-2b_test_refcocog_umd_test rc=0 wall_s=3876
2026-10-26T09:13:28Z e1_box_qwen3.5-4b_test_refcoco_unc_testA rc=0 wall_s=141
2026-10-26T09:13:32Z ownbox_qwen3.5-4b_test_refcoco_unc_testA rc=0 wall_s=4
2026-10-26T09:18:48Z w1f_qwen3.5-4b_test_refcoco_unc_testA rc=0 wall_s=316
2026-10-26T10:50:52Z w2a_qwen3.5-4b_test_refcoco_unc_testA rc=0 wall_s=5524
2026-10-26T10:55:18Z e1_box_qwen3.5-4b_test_refcoco_unc_testB rc=0 wall_s=266
2026-10-26T10:55:23Z ownbox_qwen3.5-4b_test_refcoco_unc_testB rc=0 wall_s=5
2026-10-26T11:00:32Z w1f_qwen3.5-4b_test_refcoco_unc_testB rc=0 wall_s=309
2026-10-26T12:32:45Z w2a_qwen3.5-4b_test_refcoco_unc_testB rc=0 wall_s=5533
2026-10-26T12:35:13Z e1_box_qwen3.5-4b_test_refcoco+_unc_testA rc=0 wall_s=148
2026-10-26T12:35:17Z ownbox_qwen3.5-4b_test_refcoco+_unc_testA rc=0 wall_s=4
2026-10-26T12:40:22Z w1f_qwen3.5-4b_test_refcoco+_unc_testA rc=0 wall_s=305
2026-10-26T14:11:37Z w2a_qwen3.5-4b_test_refcoco+_unc_testA rc=0 wall_s=5475
2026-10-26T14:15:00Z e1_box_qwen3.5-4b_test_refcoco+_unc_testB rc=0 wall_s=203
2026-10-26T14:15:05Z ownbox_qwen3.5-4b_test_refcoco+_unc_testB rc=0 wall_s=5
2026-10-26T14:20:14Z w1f_qwen3.5-4b_test_refcoco+_unc_testB rc=0 wall_s=309
2026-10-26T15:51:21Z w2a_qwen3.5-4b_test_refcoco+_unc_testB rc=0 wall_s=5467
2026-10-26T15:54:34Z e1_box_qwen3.5-4b_test_refcocog_umd_test rc=0 wall_s=193
2026-10-26T15:54:38Z ownbox_qwen3.5-4b_test_refcocog_umd_test rc=0 wall_s=4
2026-10-26T15:59:35Z w1f_qwen3.5-4b_test_refcocog_umd_test rc=0 wall_s=297
2026-10-26T17:33:12Z w2a_qwen3.5-4b_test_refcocog_umd_test rc=0 wall_s=5617
2026-10-26T17:33:12Z done; failed steps: 0
2026-10-26T17:33:12Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2 test 0073 done (failed steps: 0)
```

- 心跳 `2026-10-26T17:33:12Z done 0073 (failed steps: 0)`。总墙钟 02:27:59Z – 17:33:12Z（54,313 s，15.09 h），超过任务书的上限 14 h（`EV_CAP_H=14`）；steps.log 里没有因上限中止或跳过的记录，40 步全部 rc=0。
- 同节点同时期的其他任务：RUN-0075（CPU，03:42–03:44Z）、RUN-0077（CPU 12 核，04:18–06:07Z）、RUN-0079（CPU 12 核，06:21–07:17Z）、RUN-0080 的 x2_data（CPU 12 核，07:41–07:46Z）。负载检查与 2B w2a 各划分墙钟的对照见 RUN-0077、RUN-0079 报告。

## 各划分的读数（由 `experiments/W2/0073_w2_test_probe/p73_tables.py` 读出）

### qwen3.5-2b

| 划分 | E1 框 acc@0.5 | ownbox cIoU | w1f cIoU | w1f gIoU | w2a cIoU | w2a gIoU | n | complete |
|---|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 0.8980 [0.8790, 0.9160] | 0.4815 [0.4665, 0.4964] | 0.5507 [0.5288, 0.5720] | 0.5334 [0.5169, 0.5503] | 0.5502 [0.5309, 0.5691] | 0.5551 [0.5396, 0.5702] | 1000 | True |
| test_refcoco_unc_testB | 0.8200 [0.7960, 0.8440] | 0.5036 [0.4846, 0.5243] | 0.5424 [0.5165, 0.5678] | 0.5343 [0.5144, 0.5553] | 0.5395 [0.5158, 0.5632] | 0.5572 [0.5380, 0.5755] | 1000 | True |
| test_refcoco+_unc_testA | 0.8340 [0.8110, 0.8560] | 0.4580 [0.4419, 0.4754] | 0.5086 [0.4860, 0.5304] | 0.4935 [0.4755, 0.5113] | 0.5020 [0.4799, 0.5239] | 0.5127 [0.4968, 0.5293] | 1000 | True |
| test_refcoco+_unc_testB | 0.7160 [0.6900, 0.7420] | 0.4470 [0.4267, 0.4673] | 0.4739 [0.4476, 0.5011] | 0.4665 [0.4462, 0.4890] | 0.4668 [0.4412, 0.4926] | 0.4802 [0.4596, 0.5004] | 1000 | True |
| test_refcocog_umd_test | 0.8510 [0.8300, 0.8710] | 0.4603 [0.4416, 0.4791] | 0.5279 [0.5043, 0.5518] | 0.5196 [0.5012, 0.5400] | 0.5247 [0.5015, 0.5472] | 0.5324 [0.5150, 0.5510] | 1000 | True |

#### w2a 的挑选统计（分母 = 有候选的句子；都不对 = pick.none）

| 划分 | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 | 出框失败 | thinking_end | parse_failure |
|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 0.272（272/999） | 0.581（580/999） | 0.004（4/999） | 0.486（486/999） | {'multiple_boxes': 1} | {'forced': 905, 'natural': 94} | {'no_choice': 190, 'out_of_range': 6, 'truncated': 290} |
| test_refcoco_unc_testB | 0.298（294/988） | 0.572（565/988） | 0.008（8/988） | 0.472（466/988） | {'multiple_boxes': 6, 'truncated': 6} | {'forced': 872, 'natural': 116} | {'no_choice': 147, 'out_of_range': 3, 'truncated': 316} |
| test_refcoco+_unc_testA | 0.282（279/991） | 0.572（567/991） | 0.010（10/991） | 0.484（480/991） | {'multiple_boxes': 9} | {'forced': 880, 'natural': 111} | {'no_choice': 172, 'out_of_range': 5, 'truncated': 303} |
| test_refcoco+_unc_testB | 0.335（325/971） | 0.554（538/971） | 0.006（6/971） | 0.454（441/971） | {'multiple_boxes': 22, 'truncated': 7} | {'forced': 858, 'natural': 113} | {'no_choice': 113, 'out_of_range': 4, 'truncated': 324} |
| test_refcocog_umd_test | 0.301（298/989） | 0.574（568/989） | 0.005（5/989） | 0.491（486/989） | {'multiple_boxes': 11} | {'forced': 883, 'natural': 106} | {'no_choice': 172, 'out_of_range': 4, 'truncated': 310} |

### qwen3.5-4b

| 划分 | E1 框 acc@0.5 | ownbox cIoU | w1f cIoU | w1f gIoU | w2a cIoU | w2a gIoU | n | complete |
|---|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 0.9210 [0.9050, 0.9360] | 0.4931 [0.4804, 0.5052] | 0.5547 [0.5316, 0.5747] | 0.5425 [0.5255, 0.5608] | 0.5729 [0.5517, 0.5911] | 0.5852 [0.5710, 0.5987] | 1000 | True |
| test_refcoco_unc_testB | 0.8450 [0.8220, 0.8670] | 0.5241 [0.5067, 0.5428] | 0.5581 [0.5339, 0.5821] | 0.5374 [0.5175, 0.5574] | 0.5775 [0.5532, 0.6010] | 0.5958 [0.5772, 0.6136] | 1000 | True |
| test_refcoco+_unc_testA | 0.8420 [0.8190, 0.8630] | 0.4626 [0.4465, 0.4788] | 0.5101 [0.4883, 0.5322] | 0.4970 [0.4781, 0.5157] | 0.5275 [0.5062, 0.5494] | 0.5373 [0.5218, 0.5538] | 1000 | True |
| test_refcoco+_unc_testB | 0.7420 [0.7150, 0.7680] | 0.4636 [0.4422, 0.4843] | 0.4924 [0.4655, 0.5181] | 0.4777 [0.4558, 0.4979] | 0.5010 [0.4750, 0.5261] | 0.5155 [0.4957, 0.5345] | 1000 | True |
| test_refcocog_umd_test | 0.8740 [0.8550, 0.8930] | 0.4851 [0.4682, 0.5029] | 0.5488 [0.5256, 0.5724] | 0.5282 [0.5093, 0.5480] | 0.5644 [0.5430, 0.5876] | 0.5583 [0.5427, 0.5753] | 1000 | True |

#### w2a 的挑选统计（分母 = 有候选的句子；都不对 = pick.none）

| 划分 | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 | 出框失败 | thinking_end | parse_failure |
|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 0.309（306/990） | 0.583（577/990） | 0.006（6/990） | 0.213（211/990） | {'multiple_boxes': 9, 'no_box': 1} | {'forced': 858, 'natural': 132} | {'no_choice': 70, 'truncated': 141} |
| test_refcoco_unc_testB | 0.319（310/973） | 0.547（532/973） | 0.005（5/973） | 0.190（185/973） | {'multiple_boxes': 23, 'no_box': 1, 'truncated': 3} | {'forced': 838, 'natural': 135} | {'no_choice': 39, 'truncated': 146} |
| test_refcoco+_unc_testA | 0.327（317/968） | 0.586（567/968） | 0.010（10/968） | 0.220（213/968） | {'multiple_boxes': 31, 'truncated': 1} | {'forced': 842, 'natural': 126} | {'no_choice': 70, 'truncated': 143} |
| test_refcoco+_unc_testB | 0.367（343/935） | 0.540（505/935） | 0.000（0/935） | 0.233（218/935） | {'multiple_boxes': 57, 'no_box': 1, 'truncated': 7} | {'forced': 799, 'natural': 136} | {'no_choice': 57, 'truncated': 161} |
| test_refcocog_umd_test | 0.349（337/965） | 0.543（524/965） | 0.011（11/965） | 0.207（200/965） | {'multiple_boxes': 33, 'no_box': 1, 'truncated': 1} | {'forced': 816, 'natural': 149} | {'no_choice': 57, 'truncated': 143} |


## 显存与 traceback

- 显存峰值：test_w2.sh 不采样显存，结果目录里没有 gpu_mem.csv，日志里也没有。07 的 GPU 在 06:30Z 时只有本任务一个进程，9,796 MiB（RUN-0079 报告里的那次检查）。
- `logs/` 下 40 个步日志 grep `Traceback`、`out of memory`：都是 0。

## 收尾

- 推送循环 18:36:14Z 一轮推送成功（已含 done）。18:51:14Z 在 git.lock 下按 PID 停掉循环 519248（当时在 sleep），随后 `ONCE=1` 推一轮：`2026-10-26T18:51:25Z ok`。
- 后 squeue：157296、157297 RUNNING。07 接着跑 RUN-0076 A 段的 train（18:47Z 起）。
