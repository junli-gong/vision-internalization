# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcocog_umd_test

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcocog_umd_test；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7111720813965086（95% CI {'low': 0.6872315098908887, 'high': 0.732929092172165, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7062783031676074（95% CI {'low': 0.6912002910108839, 'high': 0.7202031709395164, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
