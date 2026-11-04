# X-M · qwen3.5-9b · xm_spline_raster · xm_final

- 定义：诊断：样条直接光栅化，不贴边（mask：spline_raster；调用：['list', 'trace']）

## det（box IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.6953（95% CI [0.6823, 0.7254]）；P 0.7176、R 0.6744
- f1_rexomni_miou：0.5214（95% CI [0.5059, 0.5505]）；P 0.5388、R 0.5050
- f1_micro：0.6696（95% CI [0.6474, 0.6897]）；P 0.6883、R 0.6519
- matched_mean_iou：0.8339（95% CI [0.8286, 0.8400]）
- f1_rexomni_exact：0.7050；P 0.7385、R 0.6744
- f1_rexomni_exact_miou：0.5365；P 0.5722、R 0.5050

## seg（mask IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.5727（95% CI [0.5570, 0.6047]）；P 0.5853、R 0.5606
- f1_rexomni_miou：0.3055（95% CI [0.2949, 0.3274]）；P 0.3116、R 0.2996
- f1_micro：0.5480（95% CI [0.5247, 0.5700]）；P 0.5519、R 0.5442
- matched_mean_iou：0.7178（95% CI [0.7127, 0.7228]）
- f1_rexomni_exact：0.5982；P 0.6321、R 0.5677
- f1_rexomni_exact_miou：0.3401；P 0.3868、R 0.3035

- 失败与退回：{"list_failure": {"ok": 2933, "truncated": 7}, "list_counts": {"n_raw": 7729, "n_capped": 99, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 39, "n_pred_boxes": {"n": 2940, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 9.0, "max": 30, "mean": 2.581972789115646, "n_distinct": 30, "top": [[1, 1847], [2, 426], [3, 192], [4, 99], [5, 74], [6, 52], [7, 46], [8, 33], [10, 25], [9, 24]]}, "trace_failure": {"count_mismatch": 19, "ok": 2914}, "spline_failure": {"count_mismatch": 436, "ok": 7155}, "fallback": {"box_grabcut": 436, "ok": 7155}, "tool_not_ok": 0}
