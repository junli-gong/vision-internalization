# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7634630353835812（95% CI {'low': 0.7372009038685451, 'high': 0.7905240058902337, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7692215869921901（95% CI {'low': 0.752239653336341, 'high': 0.7861043597748332, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
