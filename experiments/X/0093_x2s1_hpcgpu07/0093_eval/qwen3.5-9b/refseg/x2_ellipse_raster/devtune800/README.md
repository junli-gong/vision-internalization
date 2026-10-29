# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · devtune800

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5478790253962356（95% CI {'low': 0.530686856577404, 'high': 0.5655684810121335, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5694829462999274（95% CI {'low': 0.5552540050799166, 'high': 0.5828029786063054, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
