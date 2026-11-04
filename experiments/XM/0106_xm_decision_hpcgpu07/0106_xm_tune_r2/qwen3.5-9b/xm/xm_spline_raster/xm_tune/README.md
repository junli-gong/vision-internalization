# X-M · qwen3.5-9b · xm_spline_raster · xm_tune

- 定义：诊断：样条直接光栅化，不贴边（mask：spline_raster；调用：['list', 'trace']）

## det（box IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.7388（95% CI [0.7149, 0.7796]）；P 0.7623、R 0.7167
- f1_rexomni_miou：0.5632（95% CI [0.5372, 0.5973]）；P 0.5840、R 0.5439
- f1_micro：0.6630（95% CI [0.6256, 0.6992]）；P 0.6658、R 0.6602
- matched_mean_iou：0.8324（95% CI [0.8233, 0.8423]）
- f1_rexomni_exact：0.7615；P 0.8007、R 0.7260
- f1_rexomni_exact_miou：0.5955；P 0.6479、R 0.5510

## seg（mask IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.6421（95% CI [0.6160, 0.6816]）；P 0.6630、R 0.6225
- f1_rexomni_miou：0.3493（95% CI [0.3294, 0.3781]）；P 0.3618、R 0.3376
- f1_micro：0.5515（95% CI [0.5170, 0.5898]）；P 0.5480、R 0.5550
- matched_mean_iou：0.7287（95% CI [0.7201, 0.7383]）
- f1_rexomni_exact：0.6768；P 0.7194、R 0.6389
- f1_rexomni_exact_miou：0.4182；P 0.5274、R 0.3465

- 失败与退回：{"list_failure": {"ok": 921, "truncated": 2}, "list_counts": {"n_raw": 2392, "n_capped": 0, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 25, "n_pred_boxes": {"n": 923, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 9.0, "max": 30, "mean": 2.5644637053087758, "n_distinct": 27, "top": [[1, 578], [2, 122], [3, 66], [4, 47], [5, 26], [6, 13], [7, 13], [8, 7], [9, 7], [11, 6]]}, "trace_failure": {"count_mismatch": 3, "ok": 918}, "spline_failure": {"count_mismatch": 81, "ok": 2286}, "fallback": {"box_grabcut": 81, "ok": 2286}, "tool_not_ok": 0}
