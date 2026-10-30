# X-2 · qwen3.5-4b · refseg · x2_oracle4 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6496553491140692（95% CI {'low': 0.6219310329598182, 'high': 0.6774808903089178, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6755954361837145（95% CI {'low': 0.6568081498260858, 'high': 0.6942741001466277, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
