# X-2 · qwen3.5-9b · refseg · x2_select5_cor2_randbox · explore_val_refcoco_unc_val

- 定义：{"kind": "randbox", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.685906155696998（95% CI {'low': 0.6492229679801496, 'high': 0.7201161167854865, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6834621936811496（95% CI {'low': 0.6558488770981454, 'high': 0.7087017645699473, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 5.0
