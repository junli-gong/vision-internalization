# X-2 · qwen3.5-9b · refseg · x2_select5 · explore_val_refcoco_unc_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7604756788911932（95% CI {'low': 0.7351389640629852, 'high': 0.7849994171418571, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7556985453825902（95% CI {'low': 0.737918187575235, 'high': 0.774469190577603, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
