# X-2 · qwen3.5-9b · refseg · x2_select · devcheck2

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6374455320781118（95% CI {'low': 0.612342897328936, 'high': 0.6668727557601275, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6640393023503816（95% CI {'low': 0.64481768928673, 'high': 0.6837798391327886, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
