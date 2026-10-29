# X-2 · qwen3.5-9b · refseg · x2_select_cor1_randbox · explore_val_refcoco+_unc_val

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.550466491065607（95% CI {'low': 0.5117897965661556, 'high': 0.5878138417118955, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5519590284415382（95% CI {'low': 0.5216300300744784, 'high': 0.5824957608975346, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
