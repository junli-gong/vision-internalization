# X-2 · qwen3.5-4b · refseg · x2_select_cor1_randbox · explore_val_refcocog_umd_val

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5557693579171746（95% CI {'low': 0.49389306239384273, 'high': 0.6217946955858097, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5522335736855483（95% CI {'low': 0.5095577907028396, 'high': 0.5962929327657198, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 3.0
