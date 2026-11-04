# X-M · qwen3.5-9b · xm_ellipse_trimap · xm_decision

- 定义：消融：X-M 的框的内切椭圆 → 三元图贴边（同 X-2）（mask：ellipse_trimap；调用：['list']）

## det（box IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.7266（95% CI [0.7107, 0.7615]）；P 0.7435、R 0.7106
- f1_rexomni_miou：0.5343（95% CI [0.5227, 0.5692]）；P 0.5467、R 0.5225
- f1_micro：0.6531（95% CI [0.6229, 0.6838]）；P 0.6757、R 0.6319
- matched_mean_iou：0.8303（95% CI [0.8232, 0.8375]）
- f1_rexomni_exact：0.7311；P 0.7528、R 0.7106
- f1_rexomni_exact_miou：0.5508；P 0.5825、R 0.5225

## seg（mask IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.5386（95% CI [0.5178, 0.5701]）；P 0.5461、R 0.5312
- f1_rexomni_miou：0.2671（95% CI [0.2533, 0.2840]）；P 0.2704、R 0.2639
- f1_micro：0.4693（95% CI [0.4446, 0.4958]）；P 0.4738、R 0.4648
- matched_mean_iou：0.6917（95% CI [0.6846, 0.6986]）
- f1_rexomni_exact：0.5596；P 0.5829、R 0.5380
- f1_rexomni_exact_miou：0.3110；P 0.3718、R 0.2673

- 失败与退回：{"list_failure": {"ok": 1436, "truncated": 1}, "list_counts": {"n_raw": 3901, "n_capped": 109, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 35, "n_pred_boxes": {"n": 1437, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 10.0, "max": 30, "mean": 2.614474599860821, "n_distinct": 26, "top": [[1, 891], [2, 219], [3, 94], [4, 60], [5, 33], [7, 22], [6, 19], [10, 15], [8, 14], [9, 8]]}, "fallback": {"ok": 3757}, "tool_not_ok": 17}
