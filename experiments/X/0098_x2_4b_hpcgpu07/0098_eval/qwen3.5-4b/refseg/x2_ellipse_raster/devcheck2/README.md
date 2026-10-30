# X-2 · qwen3.5-4b · refseg · x2_ellipse_raster · devcheck2

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5347130873319816（95% CI {'low': 0.511348308189194, 'high': 0.5598876872319122, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5729155469008177（95% CI {'low': 0.5556647359763143, 'high': 0.5898262226883976, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
