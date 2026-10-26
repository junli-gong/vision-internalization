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

