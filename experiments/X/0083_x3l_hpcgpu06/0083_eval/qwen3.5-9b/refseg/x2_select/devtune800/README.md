# X-2 · qwen3.5-9b · refseg · x2_select · devtune800

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6504705587115308（95% CI {'low': 0.6294793818234431, 'high': 0.6702733023023589, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.655621313808785（95% CI {'low': 0.638564993615832, 'high': 0.6715764833967398, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
