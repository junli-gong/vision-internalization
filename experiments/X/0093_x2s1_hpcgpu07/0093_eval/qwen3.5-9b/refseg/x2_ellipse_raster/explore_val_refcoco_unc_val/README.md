# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · explore_val_refcoco_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5739073530694586（95% CI {'low': 0.5501909110730235, 'high': 0.5976490853503376, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5831617741617258（95% CI {'low': 0.5652029100493643, 'high': 0.6011684680939982, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
