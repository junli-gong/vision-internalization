# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7392020871801039（95% CI {'low': 0.7169136333791897, 'high': 0.7590379243365601, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7370684499401008（95% CI {'low': 0.7209148848180372, 'high': 0.750924305290912, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
