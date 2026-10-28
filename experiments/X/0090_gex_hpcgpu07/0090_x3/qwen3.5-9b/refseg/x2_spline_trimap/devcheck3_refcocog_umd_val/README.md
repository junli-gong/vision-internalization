# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devcheck3_refcocog_umd_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck3_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6797095310597995（95% CI {'low': 0.5922801712885555, 'high': 0.7658253651245717, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7062063775294756（95% CI {'low': 0.6614103696927016, 'high': 0.7455213976099324, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 100}；fallback {'ok': 100}；每句模型调用 2.0
