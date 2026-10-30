# X-2 · qwen3.5-4b · refseg · x2_ellipse_raster · devtune800

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5399368515980494（95% CI {'low': 0.5230841427551078, 'high': 0.5573276370445059, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5609020925178997（95% CI {'low': 0.5471239360744495, 'high': 0.5745930994044072, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
