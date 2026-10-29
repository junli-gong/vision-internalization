# X-2 · qwen3.5-9b · refseg · x2_select5_cor2 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6987321649642679（95% CI {'low': 0.6720993562768838, 'high': 0.7270792326486318, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6923125729780807（95% CI {'low': 0.6682946466806275, 'high': 0.7177426615376244, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
