# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · test_refcocog_umd_test

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcocog_umd_test；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5402188568307249（95% CI {'low': 0.5218195333974793, 'high': 0.5575137171736279, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5574150191563745（95% CI {'low': 0.5440772847928904, 'high': 0.571328449891417, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 1.0
