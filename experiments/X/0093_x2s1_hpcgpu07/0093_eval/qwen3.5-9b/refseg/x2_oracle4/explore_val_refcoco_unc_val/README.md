# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6689017106544431（95% CI {'low': 0.6400670042011796, 'high': 0.6955312911929884, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6736901574581509（95% CI {'low': 0.6537577618786491, 'high': 0.6939914909412644, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
