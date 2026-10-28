# X-2 · qwen3.5-9b · refseg · x2_select · devcheck2

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6391773776809647（95% CI {'low': 0.6158549645104331, 'high': 0.6662672606879653, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6657279065911341（95% CI {'low': 0.6474643139973096, 'high': 0.6849581776824537, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
