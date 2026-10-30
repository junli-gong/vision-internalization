# X-2 · qwen3.5-4b · refseg · x2_select5 · explore_val_refcoco+_unc_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6822884362197759（95% CI {'low': 0.6404488007905837, 'high': 0.7185159501125297, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.694525173337726（95% CI {'low': 0.6663376416694065, 'high': 0.7181717141609344, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
