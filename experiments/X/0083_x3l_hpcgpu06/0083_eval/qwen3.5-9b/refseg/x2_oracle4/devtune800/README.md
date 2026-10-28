# X-2 · qwen3.5-9b · refseg · x2_oracle4 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6602120273545992（95% CI {'low': 0.6392392320253741, 'high': 0.6808955018133953, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6659654830108451（95% CI {'low': 0.649874450908632, 'high': 0.6810819486050049, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
