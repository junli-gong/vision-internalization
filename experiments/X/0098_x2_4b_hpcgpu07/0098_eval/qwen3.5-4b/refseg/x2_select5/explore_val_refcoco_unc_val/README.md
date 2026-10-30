# X-2 · qwen3.5-4b · refseg · x2_select5 · explore_val_refcoco_unc_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7536145137541631（95% CI {'low': 0.7269653512990678, 'high': 0.7795030300872782, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7468264343733643（95% CI {'low': 0.7275045298568424, 'high': 0.7663698578021311, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
