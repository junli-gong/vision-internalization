# X-2 · qwen3.5-9b · refseg · x2_select5_cor2_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6832928940682597（95% CI {'low': 0.6575212696721607, 'high': 0.7077730828642644, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.685620490934067（95% CI {'low': 0.6659399500881643, 'high': 0.7048589866231109, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 5.0
