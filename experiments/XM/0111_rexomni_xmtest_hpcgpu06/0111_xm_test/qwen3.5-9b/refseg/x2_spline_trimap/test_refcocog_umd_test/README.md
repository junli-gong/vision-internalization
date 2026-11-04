# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcocog_umd_test

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcocog_umd_test；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7012134248307028（95% CI {'low': 0.6799235815956172, 'high': 0.7235430414907565, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6960351698148627（95% CI {'low': 0.6812886827876449, 'high': 0.7107607979701572, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
