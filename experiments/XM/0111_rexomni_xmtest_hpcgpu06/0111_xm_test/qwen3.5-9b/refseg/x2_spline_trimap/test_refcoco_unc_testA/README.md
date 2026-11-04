# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcoco_unc_testA

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco_unc_testA；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7522344597849411（95% CI {'low': 0.7342351655879767, 'high': 0.7688463288577122, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7486589880483219（95% CI {'low': 0.7372926213431351, 'high': 0.7598195119247109, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
