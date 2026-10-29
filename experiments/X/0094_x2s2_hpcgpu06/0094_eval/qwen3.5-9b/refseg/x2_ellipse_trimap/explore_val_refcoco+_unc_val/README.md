# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · explore_val_refcoco+_unc_val

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5299301031321433（95% CI {'low': 0.4986656930808434, 'high': 0.559604101160768, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5582176701210559（95% CI {'low': 0.534372897499376, 'high': 0.5806451295682381, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
