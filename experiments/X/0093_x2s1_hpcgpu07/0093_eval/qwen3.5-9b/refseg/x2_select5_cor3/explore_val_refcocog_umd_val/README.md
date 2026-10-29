# X-2 · qwen3.5-9b · refseg · x2_select5_cor3 · explore_val_refcocog_umd_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6329257059022793（95% CI {'low': 0.57689098824127, 'high': 0.6869921062400298, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5861992266690549（95% CI {'low': 0.5384928478431316, 'high': 0.6308861684265034, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 6.0
