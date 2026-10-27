# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · explore_val_refcoco+_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5130686739202134（95% CI {'low': 0.4851670277480216, 'high': 0.5411874818460486, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5337670975081332（95% CI {'low': 0.5119015097870909, 'high': 0.5548528187635109, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
