# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcoco+_unc_testA

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco+_unc_testA；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7334806812383503（95% CI {'low': 0.7130781195989613, 'high': 0.7547762787716189, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7372467955625917（95% CI {'low': 0.7243977595324625, 'high': 0.7503711521567047, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
