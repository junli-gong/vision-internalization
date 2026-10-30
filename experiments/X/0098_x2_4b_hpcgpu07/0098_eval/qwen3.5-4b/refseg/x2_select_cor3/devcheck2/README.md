# X-2 · qwen3.5-4b · refseg · x2_select_cor3 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6300693188411649（95% CI {'low': 0.5971164798581677, 'high': 0.6624109974823907, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.610762888290868（95% CI {'low': 0.5829473146742966, 'high': 0.6375164273084652, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
