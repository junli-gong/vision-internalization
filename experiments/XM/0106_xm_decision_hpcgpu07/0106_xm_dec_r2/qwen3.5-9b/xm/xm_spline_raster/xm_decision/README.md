# X-M · qwen3.5-9b · xm_spline_raster · xm_decision

- 定义：诊断：样条直接光栅化，不贴边（mask：spline_raster；调用：['list', 'trace']）

## det（box IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.7266（95% CI [0.7107, 0.7615]）；P 0.7435、R 0.7106
- f1_rexomni_miou：0.5343（95% CI [0.5227, 0.5692]）；P 0.5467、R 0.5225
- f1_micro：0.6531（95% CI [0.6229, 0.6838]）；P 0.6757、R 0.6319
- matched_mean_iou：0.8303（95% CI [0.8232, 0.8375]）
- f1_rexomni_exact：0.7311；P 0.7528、R 0.7106
- f1_rexomni_exact_miou：0.5508；P 0.5825、R 0.5225

## seg（mask IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.5909（95% CI [0.5752, 0.6284]）；P 0.5995、R 0.5826
- f1_rexomni_miou：0.3203（95% CI [0.3093, 0.3450]）；P 0.3245、R 0.3163
- f1_micro：0.5151（95% CI [0.4868, 0.5450]）；P 0.5165、R 0.5138
- matched_mean_iou：0.7198（95% CI [0.7125, 0.7273]）
- f1_rexomni_exact：0.6058；P 0.6309、R 0.5826
- f1_rexomni_exact_miou：0.3675；P 0.4383、R 0.3163

- 失败与退回：{"list_failure": {"ok": 1436, "truncated": 1}, "list_counts": {"n_raw": 3901, "n_capped": 109, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 35, "n_pred_boxes": {"n": 1437, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 10.0, "max": 30, "mean": 2.614474599860821, "n_distinct": 26, "top": [[1, 891], [2, 219], [3, 94], [4, 60], [5, 33], [7, 22], [6, 19], [10, 15], [8, 14], [9, 8]]}, "trace_failure": {"count_mismatch": 14, "ok": 1422}, "spline_failure": {"count_mismatch": 366, "ok": 3391}, "fallback": {"box_grabcut": 366, "ok": 3391}, "tool_not_ok": 0}
