# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7126191859588558（95% CI {'low': 0.684720514089434, 'high': 0.7407786475903912, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7201380125716293（95% CI {'low': 0.7014040476499916, 'high': 0.7384848025777173, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
