# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · devcheck2

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5389668482482888（95% CI {'low': 0.5156047679389394, 'high': 0.5643865350862165, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5753465833394957（95% CI {'low': 0.5579193337757599, 'high': 0.5931350124824688, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
