# W2 test 各系统的合并读数（9B，5 个划分共 5,000 句；`scripts/mgr/w2_test_pooled.py` 生成；只描述）

cIoU = Σinter / Σunion，gIoU = 逐句 IoU 的均值；95% CI，bootstrap 10,000 次，种子 20260929（D152）。

| 系统 | n | cIoU [95% CI] | gIoU [95% CI] | 来源 |
|---|---|---|---|---|
| 自己的框 | 5,000 | 0.490 [0.482, 0.497] | 0.503 [0.497, 0.509] | `experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/ownbox` |
| W1-F | 5,000 | 0.541 [0.530, 0.552] | 0.528 [0.519, 0.537] | `experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w1f` |
| 随机挑选 | 5,000 | 0.539 [0.529, 0.549] | 0.537 [0.529, 0.545] | `experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/random_pick` |
| 最优候选 | 5,000 | 0.615 [0.605, 0.625] | 0.629 [0.622, 0.636] | `experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/oracle_pick` |
| W2-a | — | 缺：test_refcoco_unc_testA, test_refcoco_unc_testB, test_refcoco+_unc_testA, test_refcoco+_unc_testB, test_refcocog_umd_test | | |
| W2-c | 5,000 | 0.631 [0.622, 0.640] | 0.646 [0.639, 0.653] | `experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c` |
| W2-c+回放 | 5,000 | 0.632 [0.623, 0.641] | 0.647 [0.640, 0.654] | `experiments/W2/0076_w2c_replay_test_hpcgpu06/qwen3.5-9b/refseg/w2c` |
| E1 零样本多边形 | 5,000 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | `experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/baseline_e1` |
| C1 = X-2（主方法） | 5,000 | 0.709 [0.699, 0.719] | 0.717 [0.710, 0.723] | `experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap` |
| C1 的椭圆消融 | 5,000 | 0.562 [0.554, 0.570] | 0.584 [0.578, 0.590] | `experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_ellipse_trimap` |
| C2 = X-3（消融） | 5,000 | 0.715 [0.705, 0.725] | 0.721 [0.715, 0.728] | `experiments/X/0092_x3_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap` |
