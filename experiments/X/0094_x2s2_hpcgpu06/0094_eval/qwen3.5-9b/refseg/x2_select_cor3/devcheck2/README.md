# X-2 · qwen3.5-9b · refseg · x2_select_cor3 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6397308436931429（95% CI {'low': 0.6080239462928513, 'high': 0.6729804842580794, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6438949710417416（95% CI {'low': 0.6196472042324885, 'high': 0.6690368518476585, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
