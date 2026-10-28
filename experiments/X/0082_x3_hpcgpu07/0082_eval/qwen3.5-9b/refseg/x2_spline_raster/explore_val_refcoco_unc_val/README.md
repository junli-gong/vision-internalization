# X-2 · qwen3.5-9b · refseg · x2_spline_raster · explore_val_refcoco_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7337313635469616（95% CI {'low': 0.7078258817590491, 'high': 0.757385964277808, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7257112322971614（95% CI {'low': 0.704738430099373, 'high': 0.7451681715577599, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
