# X-2 · qwen3.5-9b · refseg · x2_select5_cor3_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6566551975422265（95% CI {'low': 0.6278322259873673, 'high': 0.6851093566414873, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6443291204851503（95% CI {'low': 0.6232529020628442, 'high': 0.6650546730645533, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 6.0
