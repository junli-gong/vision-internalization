# X-2 · qwen3.5-4b · refseg · x2_spline_raster · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7002047442868914（95% CI {'low': 0.6728244583488817, 'high': 0.728006961254871, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.712272602326929（95% CI {'low': 0.6944729242977836, 'high': 0.7300709668101355, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
