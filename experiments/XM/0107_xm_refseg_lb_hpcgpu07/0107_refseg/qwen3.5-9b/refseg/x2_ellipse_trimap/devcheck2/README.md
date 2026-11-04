# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · devcheck2

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5553615405414204（95% CI {'low': 0.5321436236722064, 'high': 0.5805745565718657, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5967648061109415（95% CI {'low': 0.580846363301801, 'high': 0.6148330347471396, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
