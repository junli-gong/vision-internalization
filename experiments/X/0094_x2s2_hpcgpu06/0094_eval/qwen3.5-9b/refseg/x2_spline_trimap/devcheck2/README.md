# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7431441975409605（95% CI {'low': 0.7158903366888387, 'high': 0.7704084077442299, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7477701204869209（95% CI {'low': 0.7301893599176666, 'high': 0.7644541211872926, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
