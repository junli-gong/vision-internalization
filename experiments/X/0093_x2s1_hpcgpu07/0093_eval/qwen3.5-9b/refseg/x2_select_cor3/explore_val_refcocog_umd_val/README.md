# X-2 · qwen3.5-9b · refseg · x2_select_cor3 · explore_val_refcocog_umd_val

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.626238840438743（95% CI {'low': 0.5746955190663987, 'high': 0.6784564023760664, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5829973538022197（95% CI {'low': 0.5420488190867679, 'high': 0.6247992400389836, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 5.0
