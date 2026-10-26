# RUN-0081 报告：W2 test 汇总表（在 HPC 上补齐 W2-a 的配对差；只用 CPU）

任务书 `prompts/server_hpc/0081-hpc-w2-test-table.md`。不用 GPU，没有写暂停文件；没有碰 RUN-0076、0078、0080。只贴数字，不解读。

## 工作树与运行

- 经 hpcmgt 上脱离会话的 `$VI/tmp/run0081.sh`（10-26 19:22:46Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0081（稀疏）HEAD `d0b5f0c714134025e831389e0b527aa01e5bc2e2`（含 bc20a690），status 空；`scripts/mgr/w2_test_table.py` sha256 `1f085a7c7f1068cdd3a81bc91bbd0dd23345df291f872f4e80f6332dff6a7b02` 与任务书相同。
- 两次都在 07 上 `srun --jobid=157297 --overlap -n1 -c 1`，env `PATH=$VI/env/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0`，工作目录 wt0081：

| 顺序 | 命令 | 输出 | 退出码 |
|---|---|---|---|
| 1 | `python scripts/mgr/w2_test_table.py --root $VI/repo --out $VI/repo/experiments/W2/0081_w2_test_table/w2_test_table --check` | `check same` | 0 |
| 2 | 同上，不带 `--check` | 写出 `w2_test_table.json`、`w2_test_table.md`（原文见下） | 0（墙钟 23 s） |

- 说明：`--check` 只做"向量化 CI 与 `src.stats` 参考实现在随机数据上逐位比较"后退出，不写结果。脚本第一次只跑了 `--check`，19:23:09Z 的 ONCE 推送时目录为空；19:24:33Z 补跑第 2 行的正式命令，再推一次。
- 输出：

| 文件 | 字节 | sha256 |
|---|---:|---|
| `w2_test_table.json` | 29,159 | `1b272b912e23e5b3de3eceac0b439f9be380fd3a8cc7555aa9bd263eef67c2f2` |
| `w2_test_table.md` | 3,167 | `22a280c423ed3ae7d024eae5a4637846655620325d99634f2a2b7ba1eb1cafc9` |

- 推送：`ONCE=1`（TAG 0081）`2026-10-26T19:25:04Z ok`。后 squeue：157296、157297 RUNNING。
- 表里没有"缺逐句数据"的格子（grep 计数 0）。

## `w2_test_table.md` 原文

# W2 test 汇总（W2 预注册 §11；`scripts/mgr/w2_test_table.py` 生成；只描述，不判定）

配对 bootstrap，ratio-of-sums，95%，10,000 次，种子 20260929。合并 = 5 个划分的句子合在一起。

## qwen3.5-9b

| 划分 | ownbox | w1f | random_pick | oracle_pick | w2a | w2c | baseline_e1 |
|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 0.491 | 0.551 | 0.548 | 0.632 | 0.583 | 0.642 | 0.000 |
| test_refcoco_unc_testB | 0.530 | 0.571 | 0.563 | 0.645 | 0.595 | 0.664 | 0.000 |
| test_refcoco+_unc_testA | 0.478 | 0.542 | 0.539 | 0.617 | 0.563 | 0.605 | 0.000 |
| test_refcoco+_unc_testB | 0.471 | 0.497 | 0.494 | 0.563 | 0.521 | 0.610 | 0.000 |
| test_refcocog_umd_test | 0.481 | 0.550 | 0.557 | 0.625 | 0.575 | 0.635 | 0.000 |

配对差（cIoU）：

| 对比 | test_refcoco_unc_testA | test_refcoco_unc_testB | test_refcoco+_unc_testA | test_refcoco+_unc_testB | test_refcocog_umd_test | 合并 |
|---|---|---|---|---|---|---|
| w2c-w1f | +0.091 [+0.074, +0.109] | +0.094 [+0.073, +0.115] | +0.063 [+0.045, +0.082] | +0.114 [+0.089, +0.140] | +0.085 [+0.062, +0.109] | +0.090 [+0.080, +0.100]（n 5000） |
| w2a-w1f | +0.032 [+0.020, +0.045] | +0.025 [+0.014, +0.035] | +0.021 [+0.008, +0.034] | +0.024 [+0.011, +0.041] | +0.025 [+0.007, +0.048] | +0.025 [+0.019, +0.032]（n 5000） |
| w2c-w2a | +0.059 [+0.045, +0.075] | +0.069 [+0.050, +0.089] | +0.042 [+0.026, +0.058] | +0.090 [+0.067, +0.112] | +0.060 [+0.043, +0.077] | +0.064 [+0.056, +0.073]（n 5000） |

挑中最优的比例：w2a 0.346、0.348、0.357、0.429、0.366；w2c 0.659、0.653、0.646、0.658、0.652

## qwen3.5-4b

| 划分 | ownbox | w1f | w2a |
|---|---|---|---|
| test_refcoco_unc_testA | 0.493 | 0.555 | 0.573 |
| test_refcoco_unc_testB | 0.524 | 0.558 | 0.578 |
| test_refcoco+_unc_testA | 0.463 | 0.510 | 0.528 |
| test_refcoco+_unc_testB | 0.464 | 0.492 | 0.501 |
| test_refcocog_umd_test | 0.485 | 0.549 | 0.564 |

配对差（cIoU）：

| 对比 | test_refcoco_unc_testA | test_refcoco_unc_testB | test_refcoco+_unc_testA | test_refcoco+_unc_testB | test_refcocog_umd_test | 合并 |
|---|---|---|---|---|---|---|
| w2a-w1f | +0.018 [+0.010, +0.027] | +0.019 [+0.009, +0.030] | +0.017 [+0.006, +0.029] | +0.009 [-0.001, +0.018] | +0.016 [-0.001, +0.039] | +0.016 [+0.010, +0.021]（n 5000） |

挑中最优的比例：w2a 0.309、0.319、0.327、0.367、0.349

## qwen3.5-2b

| 划分 | ownbox | w1f | w2a |
|---|---|---|---|
| test_refcoco_unc_testA | 0.482 | 0.551 | 0.550 |
| test_refcoco_unc_testB | 0.504 | 0.542 | 0.540 |
| test_refcoco+_unc_testA | 0.458 | 0.509 | 0.502 |
| test_refcoco+_unc_testB | 0.447 | 0.474 | 0.467 |
| test_refcocog_umd_test | 0.460 | 0.528 | 0.525 |

配对差（cIoU）：

| 对比 | test_refcoco_unc_testA | test_refcoco_unc_testB | test_refcoco+_unc_testA | test_refcoco+_unc_testB | test_refcocog_umd_test | 合并 |
|---|---|---|---|---|---|---|
| w2a-w1f | -0.001 [-0.010, +0.009] | -0.003 [-0.013, +0.007] | -0.007 [-0.017, +0.004] | -0.007 [-0.015, +0.001] | -0.003 [-0.015, +0.009] | -0.004 [-0.009, +0.000]（n 5000） |

挑中最优的比例：w2a 0.272、0.298、0.282、0.335、0.301

