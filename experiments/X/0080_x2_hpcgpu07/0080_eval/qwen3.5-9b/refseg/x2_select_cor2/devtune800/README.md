# X-2 · qwen3.5-9b · refseg · x2_select_cor2 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.647303336182528（95% CI {'low': 0.6214620084593057, 'high': 0.6708588413814882, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6447513841108535（95% CI {'low': 0.6245525156731694, 'high': 0.6628098239596731, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 4.0
