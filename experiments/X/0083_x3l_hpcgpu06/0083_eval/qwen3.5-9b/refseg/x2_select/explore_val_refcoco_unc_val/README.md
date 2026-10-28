# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcoco_unc_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6530421074410817（95% CI {'low': 0.6241608944185111, 'high': 0.6801861980359661, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6623286313727738（95% CI {'low': 0.6418909978493349, 'high': 0.6823052655334212, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
