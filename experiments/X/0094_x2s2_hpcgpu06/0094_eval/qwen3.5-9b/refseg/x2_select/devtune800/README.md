# X-2 · qwen3.5-9b · refseg · x2_select · devtune800

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6442241794163849（95% CI {'low': 0.6229958395244727, 'high': 0.664970622956102, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.652359015705803（95% CI {'low': 0.6353804783659857, 'high': 0.6679601743401524, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
