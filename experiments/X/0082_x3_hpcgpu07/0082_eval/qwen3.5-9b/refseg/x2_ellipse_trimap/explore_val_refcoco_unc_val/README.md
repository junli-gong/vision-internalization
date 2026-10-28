# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · explore_val_refcoco_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5833895263512414（95% CI {'low': 0.5583704579901897, 'high': 0.6090053584878423, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5983361705789432（95% CI {'low': 0.578452942931148, 'high': 0.61826748836991, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
