# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · explore_val_refcoco+_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5194511476331161（95% CI {'low': 0.4889657724332582, 'high': 0.5488289974478634, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5433868193523814（95% CI {'low': 0.521713418707469, 'high': 0.5650588554646808, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
