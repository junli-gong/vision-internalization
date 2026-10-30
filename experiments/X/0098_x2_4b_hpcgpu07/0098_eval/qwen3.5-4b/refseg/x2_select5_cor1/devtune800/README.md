# X-2 · qwen3.5-4b · refseg · x2_select5_cor1 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.605498222743749（95% CI {'low': 0.5723465062469077, 'high': 0.6337844717133313, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5929560858264113（95% CI {'low': 0.5696093195666065, 'high': 0.6156848466418353, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 4.0
