# X-M · qwen3.5-9b · xm_ellipse_trimap · lvis_sub

- 定义：消融：X-M 的框的内切椭圆 → 三元图贴边（同 X-2）（mask：ellipse_trimap；调用：['list']）

## det（box IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.7202（95% CI [0.7093, 0.7467]）；P 0.8047、R 0.6518
- f1_rexomni_miou：0.5317（95% CI [0.5222, 0.5602]）；P 0.5968、R 0.4795
- f1_micro：0.5807（95% CI [0.5380, 0.6235]）；P 0.7828、R 0.4615
- matched_mean_iou：0.8274（95% CI [0.8205, 0.8347]）
- f1_rexomni_exact：0.7843；P 0.8828、R 0.7056
- f1_rexomni_exact_miou：0.6221；P 0.7762、R 0.5190

## seg（mask IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.5226（95% CI [0.5063, 0.5504]）；P 0.5864、R 0.4712
- f1_rexomni_miou：0.2560（95% CI [0.2465, 0.2722]）；P 0.2907、R 0.2287
- f1_micro：0.4331（95% CI [0.3980, 0.4688]）；P 0.5943、R 0.3406
- matched_mean_iou：0.7072（95% CI [0.7007, 0.7135]）
- f1_rexomni_exact：0.6686；P 0.7735、R 0.5887
- f1_rexomni_exact_miou：0.3912；P 0.6206、R 0.2857

- 失败与退回：{"list_failure": {"ok": 2211, "truncated": 2}, "list_counts": {"n_raw": 5174, "n_capped": 65, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 23, "n_pred_boxes": {"n": 2213, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 7.0, "max": 30, "mean": 2.298237686398554, "n_distinct": 28, "top": [[1, 1441], [2, 365], [3, 116], [4, 87], [5, 47], [6, 30], [7, 19], [8, 18], [13, 13], [9, 8]]}, "fallback": {"ok": 5086}, "tool_not_ok": 0}
