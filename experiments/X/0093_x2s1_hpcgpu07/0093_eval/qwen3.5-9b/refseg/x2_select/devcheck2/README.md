# X-2 · qwen3.5-9b · refseg · x2_select · devcheck2

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6370088084006134（95% CI {'low': 0.6101826973805407, 'high': 0.6649779310010359, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6630136949916172（95% CI {'low': 0.643076486721961, 'high': 0.683441363574599, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
