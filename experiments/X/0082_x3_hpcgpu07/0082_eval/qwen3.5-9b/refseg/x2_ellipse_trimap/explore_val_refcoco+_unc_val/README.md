# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · explore_val_refcoco+_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5297391464928591（95% CI {'low': 0.4987918853083349, 'high': 0.5626646669895025, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5521716463834507（95% CI {'low': 0.5270877504069872, 'high': 0.5758386952022014, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
