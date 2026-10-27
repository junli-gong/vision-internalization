# X-2 · qwen3.5-9b · refseg · x2_select5 · devcheck2

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7416315437760678（95% CI {'low': 0.7144772154966492, 'high': 0.768726888284202, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7489880791645599（95% CI {'low': 0.7301940917148846, 'high': 0.7672083234416213, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
