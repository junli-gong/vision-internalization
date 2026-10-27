# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · devcheck2

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.556566038236697（95% CI {'low': 0.5324748040241897, 'high': 0.5825597227613474, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5944316333255621（95% CI {'low': 0.5767338965748044, 'high': 0.6128497303224986, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
