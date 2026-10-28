# X-2 · qwen3.5-9b · refseg · x2_spline_raster · explore_val_refcoco_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7437651976224683（95% CI {'low': 0.7197973937232224, 'high': 0.7659239586635068, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7341848358531848（95% CI {'low': 0.7165228612257771, 'high': 0.751258164731925, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
