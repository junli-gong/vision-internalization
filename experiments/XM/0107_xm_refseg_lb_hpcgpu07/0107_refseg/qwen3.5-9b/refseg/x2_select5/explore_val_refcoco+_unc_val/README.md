# X-2 · qwen3.5-9b · refseg · x2_select5 · explore_val_refcoco+_unc_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6975216120698864（95% CI {'low': 0.6559354127108962, 'high': 0.7365115381595879, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7114671274898562（95% CI {'low': 0.6849151463707687, 'high': 0.7350714729989807, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
