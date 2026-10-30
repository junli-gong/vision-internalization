# X-2 · qwen3.5-4b · refseg · x2_select · devcheck2

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6367583590431883（95% CI {'low': 0.6101388415423541, 'high': 0.6644535506355363, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.664060642323296（95% CI {'low': 0.6453189220813509, 'high': 0.6826566064355892, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
