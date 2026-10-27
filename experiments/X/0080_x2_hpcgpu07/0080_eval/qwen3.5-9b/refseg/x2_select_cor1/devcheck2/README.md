# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6210207540799005（95% CI {'low': 0.5885424092913069, 'high': 0.6519298899101409, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6252665804948865（95% CI {'low': 0.5989536806372826, 'high': 0.6497128022346782, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
