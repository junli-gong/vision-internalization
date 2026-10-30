# X-2 · qwen3.5-4b · refseg · x2_select_cor3_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5777247244193378（95% CI {'low': 0.5523366596239779, 'high': 0.6057603620101157, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5726661522625881（95% CI {'low': 0.5527279459932254, 'high': 0.592044828705137, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 5.0
