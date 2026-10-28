# X-2 · qwen3.5-9b · refseg · x2_select5 · devcheck2

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7417136615527031（95% CI {'low': 0.7152238773120491, 'high': 0.7697662184560969, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7526920134544999（95% CI {'low': 0.733901602899619, 'high': 0.7710314314629821, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
