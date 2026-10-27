# X-2 · qwen3.5-9b · refseg · x2_select_cor3 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6357899822742495（95% CI {'low': 0.5996141280813255, 'high': 0.6670222495926397, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6429097326223739（95% CI {'low': 0.6173730605837174, 'high': 0.6674320909447226, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
