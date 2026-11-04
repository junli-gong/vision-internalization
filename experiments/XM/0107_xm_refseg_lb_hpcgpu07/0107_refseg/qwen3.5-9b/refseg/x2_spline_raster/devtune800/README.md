# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7262836002391739（95% CI {'low': 0.7054401619062376, 'high': 0.7448535492844733, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7238778106169221（95% CI {'low': 0.7087449494264614, 'high': 0.7378177411762084, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
