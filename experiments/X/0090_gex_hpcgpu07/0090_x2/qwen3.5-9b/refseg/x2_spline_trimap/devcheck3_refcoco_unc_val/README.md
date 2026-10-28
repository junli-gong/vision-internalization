# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devcheck3_refcoco_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck3_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7165349011820756（95% CI {'low': 0.6727587555177319, 'high': 0.7565321054170978, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7218365950990606（95% CI {'low': 0.6904759382078602, 'high': 0.751069938235731, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
