# X-2 · qwen3.5-4b · refseg · x2_oracle4 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6481193798259228（95% CI {'low': 0.6275416568287568, 'high': 0.6693310580273768, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6597490745424243（95% CI {'low': 0.6425199246724097, 'high': 0.6750063261639272, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
