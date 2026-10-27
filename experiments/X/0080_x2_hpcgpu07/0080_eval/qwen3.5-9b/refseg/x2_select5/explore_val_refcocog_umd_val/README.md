# X-2 · qwen3.5-9b · refseg · x2_select5 · explore_val_refcocog_umd_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7057120295517245（95% CI {'low': 0.6629157549611374, 'high': 0.7489036612329913, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7049733893621553（95% CI {'low': 0.6743841197021292, 'high': 0.7378065985498994, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 3.0
