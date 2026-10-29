# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7640832779675765（95% CI {'low': 0.7379390973196723, 'high': 0.7887387867311397, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7672807733071666（95% CI {'low': 0.7500572864537903, 'high': 0.7840952634946255, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
