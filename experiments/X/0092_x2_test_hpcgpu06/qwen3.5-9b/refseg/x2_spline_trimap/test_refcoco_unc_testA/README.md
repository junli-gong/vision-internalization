# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcoco_unc_testA

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco_unc_testA；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7558343874006179（95% CI {'low': 0.7382466864362602, 'high': 0.7717758396694157, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7516940151199579（95% CI {'low': 0.7405219479866619, 'high': 0.7624014305782362, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
