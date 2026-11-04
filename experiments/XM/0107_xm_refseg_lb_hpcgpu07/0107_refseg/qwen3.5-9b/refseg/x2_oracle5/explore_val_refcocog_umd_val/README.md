# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcocog_umd_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7414572095484387（95% CI {'low': 0.6991060889751488, 'high': 0.7845713791672465, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7357572628218676（95% CI {'low': 0.7065362568857457, 'high': 0.7654447087151968, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
