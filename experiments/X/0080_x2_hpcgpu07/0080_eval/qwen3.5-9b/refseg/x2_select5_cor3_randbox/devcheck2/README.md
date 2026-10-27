# X-2 · qwen3.5-9b · refseg · x2_select5_cor3_randbox · devcheck2

- 定义：{"kind": "randbox", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6579272926084273（95% CI {'low': 0.6207628566174004, 'high': 0.6920019716303302, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6704198987866389（95% CI {'low': 0.6462384715303756, 'high': 0.6934929964132243, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 6.0
