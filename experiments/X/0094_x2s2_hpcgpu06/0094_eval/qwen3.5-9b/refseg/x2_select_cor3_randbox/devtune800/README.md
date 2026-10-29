# X-2 · qwen3.5-9b · refseg · x2_select_cor3_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5903150839463819（95% CI {'low': 0.5622305775166375, 'high': 0.6163106113024983, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5875170953138916（95% CI {'low': 0.5674337326300634, 'high': 0.6066966746074923, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 5.0
