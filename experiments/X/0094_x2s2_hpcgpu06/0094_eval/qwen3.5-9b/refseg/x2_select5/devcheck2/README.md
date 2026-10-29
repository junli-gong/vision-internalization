# X-2 · qwen3.5-9b · refseg · x2_select5 · devcheck2

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7466132685354449（95% CI {'low': 0.7200646564759141, 'high': 0.7730965004394589, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7535494865263984（95% CI {'low': 0.7359911046249888, 'high': 0.7712139936263804, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
