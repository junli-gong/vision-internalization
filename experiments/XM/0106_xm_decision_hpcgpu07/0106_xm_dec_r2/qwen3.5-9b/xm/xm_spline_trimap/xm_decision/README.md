# X-M · qwen3.5-9b · xm_spline_trimap · xm_decision

- 定义：主设置：X-M 的框 + X-M 的样条 → 三元图贴边（mask：spline_trimap；调用：['list', 'trace']）

## det（box IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.7266（95% CI [0.7107, 0.7615]）；P 0.7435、R 0.7106
- f1_rexomni_miou：0.5343（95% CI [0.5227, 0.5692]）；P 0.5467、R 0.5225
- f1_micro：0.6531（95% CI [0.6229, 0.6838]）；P 0.6757、R 0.6319
- matched_mean_iou：0.8303（95% CI [0.8232, 0.8375]）
- f1_rexomni_exact：0.7311；P 0.7528、R 0.7106
- f1_rexomni_exact_miou：0.5508；P 0.5825、R 0.5225

## seg（mask IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.6076（95% CI [0.5916, 0.6443]）；P 0.6170、R 0.5985
- f1_rexomni_miou：0.3541（95% CI [0.3418, 0.3806]）；P 0.3589、R 0.3494
- f1_micro：0.5297（95% CI [0.5005, 0.5610]）；P 0.5309、R 0.5285
- matched_mean_iou：0.7408（95% CI [0.7332, 0.7485]）
- f1_rexomni_exact：0.6208；P 0.6448、R 0.5985
- f1_rexomni_exact_miou：0.3943；P 0.4524、R 0.3494

- 失败与退回：{"list_failure": {"ok": 1436, "truncated": 1}, "list_counts": {"n_raw": 3901, "n_capped": 109, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 35, "n_pred_boxes": {"n": 1437, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 10.0, "max": 30, "mean": 2.614474599860821, "n_distinct": 26, "top": [[1, 891], [2, 219], [3, 94], [4, 60], [5, 33], [7, 22], [6, 19], [10, 15], [8, 14], [9, 8]]}, "trace_failure": {"count_mismatch": 14, "ok": 1422}, "spline_failure": {"count_mismatch": 366, "ok": 3391}, "fallback": {"box_grabcut": 366, "ok": 3391}, "tool_not_ok": 19}
