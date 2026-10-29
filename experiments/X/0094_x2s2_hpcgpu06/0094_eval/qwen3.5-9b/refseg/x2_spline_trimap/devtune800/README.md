# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.733973286647073（95% CI {'low': 0.7113046777806772, 'high': 0.7546942448686245, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7330350579255347（95% CI {'low': 0.7175786292202869, 'high': 0.7474030302880987, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
