# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7213968453078136（95% CI {'low': 0.6994822793135871, 'high': 0.7412549193150018, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7177239558515598（95% CI {'low': 0.7021461953373946, 'high': 0.731080638750729, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
