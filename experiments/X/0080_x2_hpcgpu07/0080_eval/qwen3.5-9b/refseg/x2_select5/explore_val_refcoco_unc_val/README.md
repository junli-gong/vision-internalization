# X-2 · qwen3.5-9b · refseg · x2_select5 · explore_val_refcoco_unc_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7724373055756453（95% CI {'low': 0.7478624656390299, 'high': 0.7967877580915461, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7578057816188667（95% CI {'low': 0.7398375185465599, 'high': 0.7762961034099903, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
