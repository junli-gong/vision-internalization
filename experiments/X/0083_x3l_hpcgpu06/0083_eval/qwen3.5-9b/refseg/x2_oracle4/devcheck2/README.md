# X-2 · qwen3.5-9b · refseg · x2_oracle4 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6588103665719316（95% CI {'low': 0.6329868931431449, 'high': 0.6852645350625357, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6808651075895534（95% CI {'low': 0.6627478543103598, 'high': 0.6993415869875019, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
