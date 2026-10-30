# X-2 · qwen3.5-4b · refseg · x2_spline_trimap · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7261993782568913（95% CI {'low': 0.7028904607901904, 'high': 0.7476402645855739, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7266836598617452（95% CI {'low': 0.71017062195586, 'high': 0.7411976218135805, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
