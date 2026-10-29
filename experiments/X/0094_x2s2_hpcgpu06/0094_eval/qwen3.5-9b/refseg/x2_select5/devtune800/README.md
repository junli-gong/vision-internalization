# X-2 · qwen3.5-9b · refseg · x2_select5 · devtune800

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7345454354064809（95% CI {'low': 0.7108565495557286, 'high': 0.7560446456792991, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7387943099232938（95% CI {'low': 0.7230197292390792, 'high': 0.7538354910136548, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
