# X-2 · qwen3.5-4b · refseg · x2_select_cor2_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5798091308360859（95% CI {'low': 0.5517756890889337, 'high': 0.6059954141259875, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5943462989154917（95% CI {'low': 0.5752844660694378, 'high': 0.6127898008395738, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 4.0
