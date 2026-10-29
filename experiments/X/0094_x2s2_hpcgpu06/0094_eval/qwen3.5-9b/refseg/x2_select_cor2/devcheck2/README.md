# X-2 · qwen3.5-9b · refseg · x2_select_cor2 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6657648437999079（95% CI {'low': 0.6346061937353181, 'high': 0.6954493659208628, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6639628004151726（95% CI {'low': 0.6387019135083692, 'high': 0.6879233593541064, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 4.0
