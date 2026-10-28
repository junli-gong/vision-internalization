# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devcheck3_refcocog_umd_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck3_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.674950383596972（95% CI {'low': 0.5845906681856359, 'high': 0.7626389320987708, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7062341176250919（95% CI {'low': 0.6629273536792245, 'high': 0.7450022615032886, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 100}；fallback {'ok': 100}；每句模型调用 2.0
