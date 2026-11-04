# X-M · qwen3.5-9b · xm_box_grabcut · lvis_sub

- 定义：对照（判定第 1 条）：X-M 的框 → W1-F 的缺省 GrabCut，不描样条（mask：box_grabcut；调用：['list']）

## det（box IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.7202（95% CI [0.7093, 0.7467]）；P 0.8047、R 0.6518
- f1_rexomni_miou：0.5317（95% CI [0.5222, 0.5602]）；P 0.5968、R 0.4795
- f1_micro：0.5807（95% CI [0.5380, 0.6235]）；P 0.7828、R 0.4615
- matched_mean_iou：0.8274（95% CI [0.8205, 0.8347]）
- f1_rexomni_exact：0.7843；P 0.8828、R 0.7056
- f1_rexomni_exact_miou：0.6221；P 0.7762、R 0.5190

## seg（mask IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.3959（95% CI [0.3824, 0.4274]）；P 0.4474、R 0.3551
- f1_rexomni_miou：0.2533（95% CI [0.2443, 0.2761]）；P 0.2890、R 0.2254
- f1_micro：0.3011（95% CI [0.2715, 0.3277]）；P 0.4259、R 0.2328
- matched_mean_iou：0.7850（95% CI [0.7774, 0.7917]）
- f1_rexomni_exact：0.5726；P 0.6881、R 0.4903
- f1_rexomni_exact_miou：0.4005；P 0.5612、R 0.3113

- 失败与退回：{"list_failure": {"ok": 2211, "truncated": 2}, "list_counts": {"n_raw": 5174, "n_capped": 65, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 23, "n_pred_boxes": {"n": 2213, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 7.0, "max": 30, "mean": 2.298237686398554, "n_distinct": 28, "top": [[1, 1441], [2, 365], [3, 116], [4, 87], [5, 47], [6, 30], [7, 19], [8, 18], [13, 13], [9, 8]]}, "fallback": {"ok": 5086}, "tool_not_ok": 4}
