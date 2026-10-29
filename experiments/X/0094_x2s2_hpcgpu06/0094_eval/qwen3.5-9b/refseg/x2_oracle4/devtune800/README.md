# X-2 · qwen3.5-9b · refseg · x2_oracle4 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6562010122956566（95% CI {'low': 0.6345901187893843, 'high': 0.6767209066435139, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6627976466044615（95% CI {'low': 0.6463517357483141, 'high': 0.6782109014216073, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
