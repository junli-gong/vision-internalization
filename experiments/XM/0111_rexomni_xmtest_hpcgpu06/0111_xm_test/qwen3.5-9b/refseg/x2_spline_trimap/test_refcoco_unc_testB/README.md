# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcoco_unc_testB

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco_unc_testB；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7266562029809137（95% CI {'low': 0.7043521956352943, 'high': 0.7492516449051949, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7389246866842705（95% CI {'low': 0.7234132534957292, 'high': 0.7545877747952902, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
