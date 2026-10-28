# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · explore_val_refcoco_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5637703618037991（95% CI {'low': 0.5398706516510549, 'high': 0.5883889418010881, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5781331133351858（95% CI {'low': 0.558832994102522, 'high': 0.597378583207573, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
