# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcocog_umd_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6481949894664198（95% CI {'low': 0.5941967449754584, 'high': 0.6976075539678005, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6468050584964697（95% CI {'low': 0.6138444732160354, 'high': 0.6790107178404905, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 1.0
