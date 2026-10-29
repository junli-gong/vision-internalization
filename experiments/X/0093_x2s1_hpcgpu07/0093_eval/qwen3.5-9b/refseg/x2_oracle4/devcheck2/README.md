# X-2 · qwen3.5-9b · refseg · x2_oracle4 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6559515104423084（95% CI {'low': 0.629438643118892, 'high': 0.6830364891040046, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6768401688915688（95% CI {'low': 0.6580926798956981, 'high': 0.6965933735159954, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
