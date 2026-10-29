# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · devcheck2

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5560205112401112（95% CI {'low': 0.5316368131857602, 'high': 0.5821602916782468, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5939939494344134（95% CI {'low': 0.5760757166625666, 'high': 0.6122561992720824, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
