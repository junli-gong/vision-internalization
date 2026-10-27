# X-2 · qwen3.5-9b · refseg · x2_select_cor2_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6111415299586526（95% CI {'low': 0.5869236651444856, 'high': 0.6331270635925464, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6139024210332517（95% CI {'low': 0.5955237341410725, 'high': 0.6320191731754626, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 4.0
