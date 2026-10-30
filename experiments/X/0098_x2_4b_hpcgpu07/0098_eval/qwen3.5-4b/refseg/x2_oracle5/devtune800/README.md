# X-2 · qwen3.5-4b · refseg · x2_oracle5 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7443502276979191（95% CI {'low': 0.7224128952065785, 'high': 0.765085578205252, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7481353374221358（95% CI {'low': 0.7328491334705547, 'high': 0.7622238232230529, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
