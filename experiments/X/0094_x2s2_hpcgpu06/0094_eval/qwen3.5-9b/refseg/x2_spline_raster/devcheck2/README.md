# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7211011626694449（95% CI {'low': 0.6948632550824236, 'high': 0.7478534710542913, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7266891593164466（95% CI {'low': 0.7100016141719322, 'high': 0.7426232802619418, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
