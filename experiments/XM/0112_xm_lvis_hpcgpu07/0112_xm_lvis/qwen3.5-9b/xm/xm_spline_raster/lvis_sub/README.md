# X-M · qwen3.5-9b · xm_spline_raster · lvis_sub

- 定义：诊断：样条直接光栅化，不贴边（mask：spline_raster；调用：['list', 'trace']）

## det（box IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.7202（95% CI [0.7093, 0.7467]）；P 0.8047、R 0.6518
- f1_rexomni_miou：0.5317（95% CI [0.5222, 0.5602]）；P 0.5968、R 0.4795
- f1_micro：0.5807（95% CI [0.5380, 0.6235]）；P 0.7828、R 0.4615
- matched_mean_iou：0.8274（95% CI [0.8205, 0.8347]）
- f1_rexomni_exact：0.7843；P 0.8828、R 0.7056
- f1_rexomni_exact_miou：0.6221；P 0.7762、R 0.5190

## seg（mask IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.5619（95% CI [0.5453, 0.5898]）；P 0.6306、R 0.5066
- f1_rexomni_miou：0.2801（95% CI [0.2698, 0.2986]）；P 0.3156、R 0.2517
- f1_micro：0.4493（95% CI [0.4129, 0.4865]）；P 0.6174、R 0.3531
- matched_mean_iou：0.7120（95% CI [0.7050, 0.7185]）
- f1_rexomni_exact：0.6885；P 0.7863、R 0.6124
- f1_rexomni_exact_miou：0.4137；P 0.6460、R 0.3043

- 失败与退回：{"list_failure": {"ok": 2211, "truncated": 2}, "list_counts": {"n_raw": 5174, "n_capped": 65, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 23, "n_pred_boxes": {"n": 2213, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 7.0, "max": 30, "mean": 2.298237686398554, "n_distinct": 28, "top": [[1, 1441], [2, 365], [3, 116], [4, 87], [5, 47], [6, 30], [7, 19], [8, 18], [13, 13], [9, 8]]}, "trace_failure": {"count_mismatch": 5, "ok": 2206}, "spline_failure": {"count_mismatch": 109, "ok": 4977}, "fallback": {"box_grabcut": 109, "ok": 4977}, "tool_not_ok": 0}
