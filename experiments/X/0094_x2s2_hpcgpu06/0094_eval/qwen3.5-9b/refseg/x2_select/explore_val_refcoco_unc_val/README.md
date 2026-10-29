# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcoco_unc_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6533585280902146（95% CI {'low': 0.6234386442043014, 'high': 0.6827024670806234, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6640870994809487（95% CI {'low': 0.643315025349736, 'high': 0.6850862548634165, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
