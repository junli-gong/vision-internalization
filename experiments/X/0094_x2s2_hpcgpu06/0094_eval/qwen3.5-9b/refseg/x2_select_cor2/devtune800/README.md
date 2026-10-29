# X-2 · qwen3.5-9b · refseg · x2_select_cor2 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6496598180811266（95% CI {'low': 0.6242641853627081, 'high': 0.6737706725023762, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.642724808405461（95% CI {'low': 0.6228729163127165, 'high': 0.6622943148833952, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 4.0
