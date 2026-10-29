# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · explore_val_refcocog_umd_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7202310841731057（95% CI {'low': 0.6720757934993631, 'high': 0.7685430904538347, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7101085760900819（95% CI {'low': 0.6808897843044932, 'high': 0.7402891466006575, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 199, 'parse_failure': 1}；每句模型调用 2.0
