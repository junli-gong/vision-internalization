# X-2 · qwen3.5-9b · refseg · x2_select_cor3 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6106325450973736（95% CI {'low': 0.577563630026549, 'high': 0.6397529902966831, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6066151185897969（95% CI {'low': 0.5846039082688057, 'high': 0.6282439031371633, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 5.0
