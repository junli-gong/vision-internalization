# X-2 · qwen3.5-4b · refseg · x2_select5 · explore_val_refcocog_umd_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7129146056051208（95% CI {'low': 0.6675081728707081, 'high': 0.7567909914489513, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.697878863455881（95% CI {'low': 0.6655838356807883, 'high': 0.7294757462928432, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 3.0
