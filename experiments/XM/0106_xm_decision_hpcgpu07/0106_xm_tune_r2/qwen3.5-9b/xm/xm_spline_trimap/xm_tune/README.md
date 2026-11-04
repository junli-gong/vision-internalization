# X-M · qwen3.5-9b · xm_spline_trimap · xm_tune

- 定义：主设置：X-M 的框 + X-M 的样条 → 三元图贴边（mask：spline_trimap；调用：['list', 'trace']）

## det（box IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.7388（95% CI [0.7149, 0.7796]）；P 0.7623、R 0.7167
- f1_rexomni_miou：0.5632（95% CI [0.5372, 0.5973]）；P 0.5840、R 0.5439
- f1_micro：0.6630（95% CI [0.6256, 0.6992]）；P 0.6658、R 0.6602
- matched_mean_iou：0.8324（95% CI [0.8233, 0.8423]）
- f1_rexomni_exact：0.7615；P 0.8007、R 0.7260
- f1_rexomni_exact_miou：0.5955；P 0.6479、R 0.5510

## seg（mask IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.6517（95% CI [0.6240, 0.6913]）；P 0.6727、R 0.6319
- f1_rexomni_miou：0.3877（95% CI [0.3649, 0.4158]）；P 0.4017、R 0.3746
- f1_micro：0.5617（95% CI [0.5253, 0.5999]）；P 0.5583、R 0.5652
- matched_mean_iou：0.7471（95% CI [0.7385, 0.7575]）
- f1_rexomni_exact：0.6852；P 0.7262、R 0.6485
- f1_rexomni_exact_miou：0.4456；P 0.5299、R 0.3845

- 失败与退回：{"list_failure": {"ok": 921, "truncated": 2}, "list_counts": {"n_raw": 2392, "n_capped": 0, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 25, "n_pred_boxes": {"n": 923, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 9.0, "max": 30, "mean": 2.5644637053087758, "n_distinct": 27, "top": [[1, 578], [2, 122], [3, 66], [4, 47], [5, 26], [6, 13], [7, 13], [8, 7], [9, 7], [11, 6]]}, "trace_failure": {"count_mismatch": 3, "ok": 918}, "spline_failure": {"count_mismatch": 81, "ok": 2286}, "fallback": {"box_grabcut": 81, "ok": 2286}, "tool_not_ok": 0}
