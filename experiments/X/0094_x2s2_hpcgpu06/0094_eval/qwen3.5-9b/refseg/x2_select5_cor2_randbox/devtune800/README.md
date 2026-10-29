# X-2 · qwen3.5-9b · refseg · x2_select5_cor2_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6744350535068192（95% CI {'low': 0.6482906646684908, 'high': 0.6998672034752526, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6693721464468139（95% CI {'low': 0.6489685310808969, 'high': 0.6892375509488141, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 5.0
