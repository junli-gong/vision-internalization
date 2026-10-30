# X-2 · qwen3.5-4b · refseg · x2_select5_cor2 · explore_val_refcocog_umd_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6773393647553758（95% CI {'low': 0.6258328859325345, 'high': 0.7222600604646408, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6333071634904374（95% CI {'low': 0.5944379006150654, 'high': 0.6724851733535404, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 5.0
