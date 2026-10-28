# X-2 · qwen3.5-9b · refseg · x2_select5 · explore_val_refcoco+_unc_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6878932164907069（95% CI {'low': 0.6499320604784216, 'high': 0.7249936464467358, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.700296714399643（95% CI {'low': 0.673273104780126, 'high': 0.72645313777422, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
