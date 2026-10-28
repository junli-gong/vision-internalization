# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · explore_val_refcoco_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.597470153238432（95% CI {'low': 0.5718026487111575, 'high': 0.6213215226910895, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6108386440228962（95% CI {'low': 0.5918636639034457, 'high': 0.6297364772376722, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
