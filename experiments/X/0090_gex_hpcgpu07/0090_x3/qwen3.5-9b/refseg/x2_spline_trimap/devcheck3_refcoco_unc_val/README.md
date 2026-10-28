# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devcheck3_refcoco_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck3_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7180749408375866（95% CI {'low': 0.6774175793567885, 'high': 0.7578364566151067, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7267225524319179（95% CI {'low': 0.6982993355930347, 'high': 0.7559493825807475, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
