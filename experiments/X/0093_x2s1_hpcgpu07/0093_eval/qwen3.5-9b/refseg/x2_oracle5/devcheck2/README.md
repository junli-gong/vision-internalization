# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7581918189125966（95% CI {'low': 0.7297873632013764, 'high': 0.7853088534659419, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7635321083020674（95% CI {'low': 0.7451284679853276, 'high': 0.7809903208204889, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
