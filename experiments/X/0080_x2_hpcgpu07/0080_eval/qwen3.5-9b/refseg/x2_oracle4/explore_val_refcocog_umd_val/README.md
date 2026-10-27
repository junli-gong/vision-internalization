# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcocog_umd_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6477157451300812（95% CI {'low': 0.5927169469121655, 'high': 0.6971057633425642, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6449492617095828（95% CI {'low': 0.6119592441571761, 'high': 0.6780827954044261, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 1.0
