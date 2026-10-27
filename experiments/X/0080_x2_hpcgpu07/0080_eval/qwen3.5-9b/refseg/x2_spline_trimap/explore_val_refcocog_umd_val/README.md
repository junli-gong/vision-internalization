# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · explore_val_refcocog_umd_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7082815317510271（95% CI {'low': 0.6666896522594113, 'high': 0.7518961538778799, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.701832313480176（95% CI {'low': 0.6711723699538931, 'high': 0.7324725898216785, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
