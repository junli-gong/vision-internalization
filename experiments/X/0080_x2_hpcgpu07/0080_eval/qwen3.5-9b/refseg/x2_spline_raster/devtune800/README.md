# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7141909512426646（95% CI {'low': 0.6933972587225381, 'high': 0.7343175785189262, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7114735603787972（95% CI {'low': 0.6964800761146065, 'high': 0.7248631968813698, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
