# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcocog_umd_test

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcocog_umd_test；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6936760299971566（95% CI {'low': 0.6671878597210352, 'high': 0.7167592897890662, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6942682768312438（95% CI {'low': 0.6795579588589054, 'high': 0.7092582391396657, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
