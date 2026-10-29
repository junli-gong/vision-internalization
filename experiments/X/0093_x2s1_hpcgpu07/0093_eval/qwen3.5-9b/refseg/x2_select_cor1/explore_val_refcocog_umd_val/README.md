# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · explore_val_refcocog_umd_val

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6175271779310609（95% CI {'low': 0.5640097831070545, 'high': 0.6744036280199532, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5616751132990774（95% CI {'low': 0.5176303366532944, 'high': 0.6041639905848938, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 3.0
