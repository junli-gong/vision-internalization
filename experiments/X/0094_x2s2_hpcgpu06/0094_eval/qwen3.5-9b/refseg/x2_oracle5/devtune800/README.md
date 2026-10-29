# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.751298979013087（95% CI {'low': 0.7288693032438831, 'high': 0.7711934944116693, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7523526971671299（95% CI {'low': 0.7378131824293623, 'high': 0.7665859302926313, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
