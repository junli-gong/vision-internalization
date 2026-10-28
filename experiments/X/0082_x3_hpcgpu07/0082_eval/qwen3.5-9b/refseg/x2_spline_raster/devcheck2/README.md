# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7250616506716592（95% CI {'low': 0.6991564327548868, 'high': 0.7509140935332879, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.731176751290073（95% CI {'low': 0.7147847328350202, 'high': 0.7473508760756864, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
