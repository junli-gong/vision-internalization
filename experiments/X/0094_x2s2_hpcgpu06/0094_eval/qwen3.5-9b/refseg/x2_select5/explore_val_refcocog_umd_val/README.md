# X-2 · qwen3.5-9b · refseg · x2_select5 · explore_val_refcocog_umd_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7008853515317658（95% CI {'low': 0.6559800209143429, 'high': 0.7488465689274143, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7029582280968153（95% CI {'low': 0.6699779886897756, 'high': 0.7389295940910459, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 3.0
