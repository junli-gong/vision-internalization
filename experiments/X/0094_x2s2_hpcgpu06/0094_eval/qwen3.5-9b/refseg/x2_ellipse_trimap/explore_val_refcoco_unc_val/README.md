# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · explore_val_refcoco_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5921507666910913（95% CI {'low': 0.5671001600991692, 'high': 0.6172293576827554, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6080818607938783（95% CI {'low': 0.5892335333292009, 'high': 0.6267525655101012, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
