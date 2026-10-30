# X-2 · qwen3.5-4b · refseg · x2_oracle5 · explore_val_refcocog_umd_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.745421425093913（95% CI {'low': 0.707429164142594, 'high': 0.7824666212618024, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7207423453974582（95% CI {'low': 0.6905410302817327, 'high': 0.7494533290203895, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
