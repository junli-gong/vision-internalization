# X-2 · qwen3.5-9b · refseg · x2_spline_raster · explore_val_refcoco+_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6746945965859281（95% CI {'low': 0.6344202100203745, 'high': 0.7099930486362603, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6837817513415824（95% CI {'low': 0.6573347018386734, 'high': 0.7065922689641504, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
