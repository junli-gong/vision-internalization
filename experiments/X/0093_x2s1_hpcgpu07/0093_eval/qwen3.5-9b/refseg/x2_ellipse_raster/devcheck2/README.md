# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · devcheck2

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5382874652553977（95% CI {'low': 0.5149316042772968, 'high': 0.5638974940317257, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5748232566072524（95% CI {'low': 0.5578090860765691, 'high': 0.592397277975167, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
