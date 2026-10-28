# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7471880059672438（95% CI {'low': 0.7212106600629152, 'high': 0.7743129128461284, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7537632372391222（95% CI {'low': 0.7364923905321279, 'high': 0.7710150384457911, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
