# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcocog_umd_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6307288824919698（95% CI {'low': 0.5763021398481301, 'high': 0.6800816868067459, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6329789955114383（95% CI {'low': 0.5995729461395612, 'high': 0.667618089536448, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
