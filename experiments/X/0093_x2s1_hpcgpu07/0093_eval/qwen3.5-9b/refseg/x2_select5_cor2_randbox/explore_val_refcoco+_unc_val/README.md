# X-2 · qwen3.5-9b · refseg · x2_select5_cor2_randbox · explore_val_refcoco+_unc_val

- 定义：{"kind": "randbox", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6397650436852943（95% CI {'low': 0.5994881985269428, 'high': 0.6745119655125359, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6367550050176043（95% CI {'low': 0.6057306256831411, 'high': 0.6650025143594035, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 5.0
