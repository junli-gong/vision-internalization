# X-M · qwen3.5-9b · base_box_grabcut · xm_tune

- 定义：起点：基座零样本的框 → 缺省 GrabCut（mask：box_grabcut；调用：['list']）

## det（box IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.6955（95% CI [0.6704, 0.7461]）；P 0.7028、R 0.6884
- f1_rexomni_miou：0.5203（95% CI [0.4951, 0.5604]）；P 0.5279、R 0.5130
- f1_micro：0.6049（95% CI [0.5651, 0.6493]）；P 0.5967、R 0.6133
- matched_mean_iou：0.8313（95% CI [0.8230, 0.8411]）
- f1_rexomni_exact：0.7251；P 0.7551、R 0.6974
- f1_rexomni_exact_miou：0.5653；P 0.6197、R 0.5196

## seg（mask IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.4094（95% CI [0.3811, 0.4587]）；P 0.4172、R 0.4020
- f1_rexomni_miou：0.2578（95% CI [0.2322, 0.2915]）；P 0.2640、R 0.2518
- f1_micro：0.3121（95% CI [0.2783, 0.3515]）；P 0.3026、R 0.3221
- matched_mean_iou：0.7567（95% CI [0.7467, 0.7668]）
- f1_rexomni_exact：0.4807；P 0.5459、R 0.4295
- f1_rexomni_exact_miou：0.3352；P 0.4443、R 0.2691

- 失败与退回：{"list_failure": {"ok": 883, "out_of_range": 2, "truncated": 38}, "list_counts": {"n_raw": 2628, "n_capped": 97, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 110, "n_pred_boxes": {"n": 923, "min": 0, "p05": 0.0, "p50": 1.0, "p95": 9.0, "max": 30, "mean": 2.62296858071506, "n_distinct": 26, "top": [[1, 454], [2, 212], [3, 67], [0, 51], [4, 37], [5, 24], [7, 14], [6, 11], [11, 8], [30, 7]]}, "fallback": {"ok": 2421}, "tool_not_ok": 9}
