# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · explore_val_refcoco_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5765237187185482（95% CI {'low': 0.5520611023755666, 'high': 0.5996194628429055, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5894876087264683（95% CI {'low': 0.5709752593913012, 'high': 0.6075621653041327, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
