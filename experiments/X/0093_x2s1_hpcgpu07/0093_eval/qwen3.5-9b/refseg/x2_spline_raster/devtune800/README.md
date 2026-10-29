# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7147613603208065（95% CI {'low': 0.6923963207192746, 'high': 0.7357245421069531, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7148319967522829（95% CI {'low': 0.6989088501561979, 'high': 0.7296646026034088, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
