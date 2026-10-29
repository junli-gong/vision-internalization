# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcocog_umd_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6483264516237087（95% CI {'low': 0.6095140497313961, 'high': 0.6897055285179667, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6346870816068769（95% CI {'low': 0.6014895066673546, 'high': 0.6666362998220098, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
