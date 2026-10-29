# X-2 · qwen3.5-9b · refseg · x2_select_cor3 · explore_val_refcoco+_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5995637548860469（95% CI {'low': 0.5563160202706662, 'high': 0.6397851619664723, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5966190466458147（95% CI {'low': 0.5643572375808961, 'high': 0.6283024446635819, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 5.0
