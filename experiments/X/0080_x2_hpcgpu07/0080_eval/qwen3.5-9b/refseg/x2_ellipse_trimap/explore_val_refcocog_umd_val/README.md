# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · explore_val_refcocog_umd_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5598203380245047（95% CI {'low': 0.5227202324729914, 'high': 0.6005583295518229, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5622292737296368（95% CI {'low': 0.5332427797273579, 'high': 0.5929392593983165, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 1.0
