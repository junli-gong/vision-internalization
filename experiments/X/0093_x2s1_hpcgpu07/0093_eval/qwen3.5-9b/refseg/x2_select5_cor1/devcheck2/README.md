# X-2 · qwen3.5-9b · refseg · x2_select5_cor1 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6649870586694715（95% CI {'low': 0.6291128932612285, 'high': 0.6976172244588984, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6461622269598937（95% CI {'low': 0.61816243536193, 'high': 0.6726924274704504, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 4.0
