# X-2 · qwen3.5-9b · refseg · x2_select5_cor3 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6599653571036092（95% CI {'low': 0.6247114869077016, 'high': 0.6940467734653948, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6527430689249972（95% CI {'low': 0.6276700244039437, 'high': 0.6773131369478904, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 6.0
