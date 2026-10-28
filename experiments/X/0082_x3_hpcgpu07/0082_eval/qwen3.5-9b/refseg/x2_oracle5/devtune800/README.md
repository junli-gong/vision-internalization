# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7544285277166671（95% CI {'low': 0.7324467729998413, 'high': 0.7744616139840836, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7538767589720681（95% CI {'low': 0.738074509556784, 'high': 0.7671172020635093, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
