# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · explore_val_refcoco_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7529245936302614（95% CI {'low': 0.7261565478506223, 'high': 0.77758693581911, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7456827028720288（95% CI {'low': 0.7249730542264207, 'high': 0.7656867675318723, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
