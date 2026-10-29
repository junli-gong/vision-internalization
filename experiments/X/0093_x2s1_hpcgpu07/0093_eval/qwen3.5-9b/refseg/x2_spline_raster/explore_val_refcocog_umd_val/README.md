# X-2 · qwen3.5-9b · refseg · x2_spline_raster · explore_val_refcocog_umd_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.702891062275459（95% CI {'low': 0.6560788136393517, 'high': 0.7499701806871855, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6904869123439461（95% CI {'low': 0.6612477025544676, 'high': 0.7208640820175821, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 199, 'parse_failure': 1}；每句模型调用 2.0
