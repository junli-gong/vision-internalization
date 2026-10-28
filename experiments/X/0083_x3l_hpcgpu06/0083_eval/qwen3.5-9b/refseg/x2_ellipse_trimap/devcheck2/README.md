# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · devcheck2

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5591255358525755（95% CI {'low': 0.5357250684466917, 'high': 0.5846656364866196, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5983035338292384（95% CI {'low': 0.5813874624631444, 'high': 0.6163223507536417, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
