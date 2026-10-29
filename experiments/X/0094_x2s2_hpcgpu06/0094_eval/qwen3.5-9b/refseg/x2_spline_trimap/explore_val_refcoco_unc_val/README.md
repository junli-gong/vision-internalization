# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · explore_val_refcoco_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7660156096225411（95% CI {'low': 0.7400248200629355, 'high': 0.7905217584394431, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7535755531651912（95% CI {'low': 0.7345298894690715, 'high': 0.7715321916625937, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
