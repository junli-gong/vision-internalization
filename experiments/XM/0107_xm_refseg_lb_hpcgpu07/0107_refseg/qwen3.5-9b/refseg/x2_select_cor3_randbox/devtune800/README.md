# X-2 · qwen3.5-9b · refseg · x2_select_cor3_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6079923807385205（95% CI {'low': 0.5845314101794463, 'high': 0.6342337822769548, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6031999586729434（95% CI {'low': 0.5852166829006404, 'high': 0.6214647931773458, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 5.0
