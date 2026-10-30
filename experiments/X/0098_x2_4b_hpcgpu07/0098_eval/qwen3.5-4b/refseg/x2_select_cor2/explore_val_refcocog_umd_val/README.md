# X-2 · qwen3.5-4b · refseg · x2_select_cor2 · explore_val_refcocog_umd_val

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6247691784970381（95% CI {'low': 0.5650024088660217, 'high': 0.6786480443201797, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6040759368142171（95% CI {'low': 0.5655192716002633, 'high': 0.643833394130158, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 4.0
