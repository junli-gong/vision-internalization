# X-2 · qwen3.5-4b · refseg · x2_select_cor3 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6131520335868691（95% CI {'low': 0.5847375412053791, 'high': 0.6413186863603505, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5993439294068917（95% CI {'low': 0.5765410066504227, 'high': 0.6193231985690743, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 5.0
