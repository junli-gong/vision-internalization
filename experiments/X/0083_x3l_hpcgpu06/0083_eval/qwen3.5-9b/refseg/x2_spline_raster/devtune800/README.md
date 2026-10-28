# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7124947465224971（95% CI {'low': 0.690717633148576, 'high': 0.732126439867319, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7100520803913238（95% CI {'low': 0.694923666179903, 'high': 0.7234075902727725, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
