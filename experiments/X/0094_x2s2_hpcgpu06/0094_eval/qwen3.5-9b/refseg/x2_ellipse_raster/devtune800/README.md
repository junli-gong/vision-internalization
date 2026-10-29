# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · devtune800

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5419662628332881（95% CI {'low': 0.5245388503672028, 'high': 0.5598097877346333, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5619998614475528（95% CI {'low': 0.5479875497374566, 'high': 0.5761531195666403, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
