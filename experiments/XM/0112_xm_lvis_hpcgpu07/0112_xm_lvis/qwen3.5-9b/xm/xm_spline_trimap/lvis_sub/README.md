# X-M · qwen3.5-9b · xm_spline_trimap · lvis_sub

- 定义：主设置：X-M 的框 + X-M 的样条 → 三元图贴边（mask：spline_trimap；调用：['list', 'trace']）

## det（box IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.7202（95% CI [0.7093, 0.7467]）；P 0.8047、R 0.6518
- f1_rexomni_miou：0.5317（95% CI [0.5222, 0.5602]）；P 0.5968、R 0.4795
- f1_micro：0.5807（95% CI [0.5380, 0.6235]）；P 0.7828、R 0.4615
- matched_mean_iou：0.8274（95% CI [0.8205, 0.8347]）
- f1_rexomni_exact：0.7843；P 0.8828、R 0.7056
- f1_rexomni_exact_miou：0.6221；P 0.7762、R 0.5190

## seg（mask IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.5946（95% CI [0.5773, 0.6204]）；P 0.6666、R 0.5367
- f1_rexomni_miou：0.3283（95% CI [0.3177, 0.3491]）；P 0.3703、R 0.2949
- f1_micro：0.4778（95% CI [0.4396, 0.5154]）；P 0.6546、R 0.3761
- matched_mean_iou：0.7400（95% CI [0.7330, 0.7470]）
- f1_rexomni_exact：0.7082；P 0.8087、R 0.6299
- f1_rexomni_exact_miou：0.4559；P 0.6676、R 0.3461

- 失败与退回：{"list_failure": {"ok": 2211, "truncated": 2}, "list_counts": {"n_raw": 5174, "n_capped": 65, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 23, "n_pred_boxes": {"n": 2213, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 7.0, "max": 30, "mean": 2.298237686398554, "n_distinct": 28, "top": [[1, 1441], [2, 365], [3, 116], [4, 87], [5, 47], [6, 30], [7, 19], [8, 18], [13, 13], [9, 8]]}, "trace_failure": {"count_mismatch": 5, "ok": 2206}, "spline_failure": {"count_mismatch": 109, "ok": 4977}, "fallback": {"box_grabcut": 109, "ok": 4977}, "tool_not_ok": 16}
