# X-2 · qwen3.5-9b · refseg · x2_select5_cor2 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6917083963104722（95% CI {'low': 0.6590085767970661, 'high': 0.7245868478203625, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6931763200476688（95% CI {'low': 0.6697960569656793, 'high': 0.7165679661087216, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
