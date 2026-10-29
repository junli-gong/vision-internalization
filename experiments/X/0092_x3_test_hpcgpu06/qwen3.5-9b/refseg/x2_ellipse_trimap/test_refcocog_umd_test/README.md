# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · test_refcocog_umd_test

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcocog_umd_test；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5464187168598973（95% CI {'low': 0.5284423403110065, 'high': 0.5631064154110862, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5621308066996464（95% CI {'low': 0.5475847970064279, 'high': 0.5761482793059178, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 1.0
