# X-2 · qwen3.5-9b · refseg · x2_oracle4 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6529279153024878（95% CI {'low': 0.6280178799111678, 'high': 0.6818387775750742, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6765721231845298（95% CI {'low': 0.657768077769899, 'high': 0.6966287921256802, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
