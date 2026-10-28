# X-2 · qwen3.5-9b · refseg · x2_spline_raster · explore_val_refcoco+_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6729427014422046（95% CI {'low': 0.6349919148249733, 'high': 0.7091071418305648, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6810203276409676（95% CI {'low': 0.6546060473114064, 'high': 0.7061547831415041, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
