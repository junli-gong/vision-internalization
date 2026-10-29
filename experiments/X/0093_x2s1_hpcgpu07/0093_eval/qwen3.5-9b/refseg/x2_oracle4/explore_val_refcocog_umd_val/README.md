# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcocog_umd_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6674556704455795（95% CI {'low': 0.6287232865898135, 'high': 0.7064393428994298, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6477399640509401（95% CI {'low': 0.6157122519016756, 'high': 0.6790419990536689, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 1.0
