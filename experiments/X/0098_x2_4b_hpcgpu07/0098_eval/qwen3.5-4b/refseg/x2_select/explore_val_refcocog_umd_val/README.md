# X-2 · qwen3.5-4b · refseg · x2_select · explore_val_refcocog_umd_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6309367854068726（95% CI {'low': 0.5743965211201291, 'high': 0.6805205739147132, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6270321138286955（95% CI {'low': 0.5946191352496859, 'high': 0.6597943604871553, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
