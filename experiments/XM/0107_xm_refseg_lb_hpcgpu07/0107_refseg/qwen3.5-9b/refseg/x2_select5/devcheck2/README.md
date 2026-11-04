# X-2 · qwen3.5-9b · refseg · x2_select5 · devcheck2

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7439496512663997（95% CI {'low': 0.7168121078313061, 'high': 0.7728201834124354, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7574213424188548（95% CI {'low': 0.739628874774496, 'high': 0.7750839349002518, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
