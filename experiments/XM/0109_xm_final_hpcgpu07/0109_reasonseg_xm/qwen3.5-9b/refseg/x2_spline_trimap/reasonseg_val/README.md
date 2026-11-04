# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · reasonseg_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/reasonseg_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.4623829128605986（95% CI {'low': 0.37490779291068, 'high': 0.5583764528633032, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5447895176256052（95% CI {'low': 0.5017097495593369, 'high': 0.5882983174437427, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 196}；fallback {'ok': 196}；每句模型调用 2.0
