# X-2 · qwen3.5-9b · refseg · x2_select_cor1_randbox · devcheck2

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6033154587508014（95% CI {'low': 0.5759649199033694, 'high': 0.6328755754536136, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6095505415098703（95% CI {'low': 0.5855732764098104, 'high': 0.6340998633491397, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
