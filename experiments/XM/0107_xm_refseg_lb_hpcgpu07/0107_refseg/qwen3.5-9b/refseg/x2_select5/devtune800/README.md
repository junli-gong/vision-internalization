# X-2 · qwen3.5-9b · refseg · x2_select5 · devtune800

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7423800945737561（95% CI {'low': 0.7209509728715122, 'high': 0.7626942573750742, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7467859516492114（95% CI {'low': 0.7308408098178311, 'high': 0.7614003432112315, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
