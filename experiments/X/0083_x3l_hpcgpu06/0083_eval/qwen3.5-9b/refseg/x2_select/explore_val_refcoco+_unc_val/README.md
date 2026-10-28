# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcoco+_unc_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5948447667067567（95% CI {'low': 0.5582379057474373, 'high': 0.633528355316887, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6226407041400803（95% CI {'low': 0.5980183850491816, 'high': 0.6478628332275593, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
