# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5961117737676778（95% CI {'low': 0.5604862164049313, 'high': 0.6320763668289983, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6145005337872045（95% CI {'low': 0.5876328165043533, 'high': 0.6412834215696736, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
