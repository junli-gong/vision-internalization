# X-2 · qwen3.5-9b · refseg · x2_select_cor2_randbox · devcheck2

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.618081974977243（95% CI {'low': 0.5932062473729317, 'high': 0.6462989335941088, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6297694088602281（95% CI {'low': 0.6087216667173051, 'high': 0.6520384936785439, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 4.0
