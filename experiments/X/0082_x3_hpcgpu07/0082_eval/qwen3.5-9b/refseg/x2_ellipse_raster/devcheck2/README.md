# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · devcheck2

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5401574698961655（95% CI {'low': 0.5187328199787844, 'high': 0.5642804777050479, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5773121920682307（95% CI {'low': 0.5615796473723143, 'high': 0.5948980574273697, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
