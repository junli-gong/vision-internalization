# X-2 · qwen3.5-4b · refseg · x2_spline_trimap · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7213416048368492（95% CI {'low': 0.6934053346457618, 'high': 0.7504654376159526, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7335939892535026（95% CI {'low': 0.7150766213427097, 'high': 0.751243789386207, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
