# X-2 · qwen3.5-9b · refseg · x2_select5_cor2_randbox · devcheck2

- 定义：{"kind": "randbox", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.678164493731991（95% CI {'low': 0.6483672195072713, 'high': 0.7101941613377576, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6817834328529644（95% CI {'low': 0.6590542168396522, 'high': 0.7054381338581898, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
