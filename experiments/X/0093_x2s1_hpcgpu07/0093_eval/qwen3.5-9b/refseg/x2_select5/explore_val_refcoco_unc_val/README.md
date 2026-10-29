# X-2 · qwen3.5-9b · refseg · x2_select5 · explore_val_refcoco_unc_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7634681176437993（95% CI {'low': 0.7383834792395778, 'high': 0.7871203626426495, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7530301786639407（95% CI {'low': 0.7345602407388961, 'high': 0.7713422127294531, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
