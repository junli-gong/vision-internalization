# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcocog_umd_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6611278707291981（95% CI {'low': 0.6221852079662028, 'high': 0.7019111269840774, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6478264326768698（95% CI {'low': 0.6166173754009863, 'high': 0.6807762572939615, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 1.0
