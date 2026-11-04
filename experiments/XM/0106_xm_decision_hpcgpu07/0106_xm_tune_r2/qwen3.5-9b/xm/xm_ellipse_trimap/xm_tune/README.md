# X-M · qwen3.5-9b · xm_ellipse_trimap · xm_tune

- 定义：消融：X-M 的框的内切椭圆 → 三元图贴边（同 X-2）（mask：ellipse_trimap；调用：['list']）

## det（box IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.7388（95% CI [0.7149, 0.7796]）；P 0.7623、R 0.7167
- f1_rexomni_miou：0.5632（95% CI [0.5372, 0.5973]）；P 0.5840、R 0.5439
- f1_micro：0.6630（95% CI [0.6256, 0.6992]）；P 0.6658、R 0.6602
- matched_mean_iou：0.8324（95% CI [0.8233, 0.8423]）
- f1_rexomni_exact：0.7615；P 0.8007、R 0.7260
- f1_rexomni_exact_miou：0.5955；P 0.6479、R 0.5510

## seg（mask IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.5600（95% CI [0.5308, 0.6017]）；P 0.5804、R 0.5410
- f1_rexomni_miou：0.2879（95% CI [0.2643, 0.3090]）；P 0.3002、R 0.2766
- f1_micro：0.4960（95% CI [0.4635, 0.5307]）；P 0.4932、R 0.4989
- matched_mean_iou：0.7024（95% CI [0.6943, 0.7113]）
- f1_rexomni_exact：0.6161；P 0.6699、R 0.5702
- f1_rexomni_exact_miou：0.3609；P 0.4734、R 0.2916

- 失败与退回：{"list_failure": {"ok": 921, "truncated": 2}, "list_counts": {"n_raw": 2392, "n_capped": 0, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 25, "n_pred_boxes": {"n": 923, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 9.0, "max": 30, "mean": 2.5644637053087758, "n_distinct": 27, "top": [[1, 578], [2, 122], [3, 66], [4, 47], [5, 26], [6, 13], [7, 13], [8, 7], [9, 7], [11, 6]]}, "fallback": {"ok": 2367}, "tool_not_ok": 0}
