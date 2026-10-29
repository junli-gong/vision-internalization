# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcoco_unc_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6535318631004269（95% CI {'low': 0.6241467046908935, 'high': 0.6811171843367663, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6584358896309109（95% CI {'low': 0.6374832388213301, 'high': 0.6790322569313285, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
