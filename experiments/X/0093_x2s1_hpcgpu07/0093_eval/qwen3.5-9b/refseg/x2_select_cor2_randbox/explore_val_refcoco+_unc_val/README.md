# X-2 · qwen3.5-9b · refseg · x2_select_cor2_randbox · explore_val_refcoco+_unc_val

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5483794469249447（95% CI {'low': 0.5088415305953579, 'high': 0.5865848478177126, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5665924605502712（95% CI {'low': 0.5361501066965383, 'high': 0.5936585592953687, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 4.0
