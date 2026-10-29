# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · explore_val_refcocog_umd_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7006431779665658（95% CI {'low': 0.6556186894223335, 'high': 0.7460409732831266, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6999312001254316（95% CI {'low': 0.6666604490929845, 'high': 0.7341549757236376, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
