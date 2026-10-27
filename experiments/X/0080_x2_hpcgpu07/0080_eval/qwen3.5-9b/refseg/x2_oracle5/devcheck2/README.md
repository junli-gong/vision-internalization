# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.757670256384975（95% CI {'low': 0.7308076055329273, 'high': 0.7848752142865488, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7632650837547431（95% CI {'low': 0.7452939618903331, 'high': 0.7812666293809913, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
