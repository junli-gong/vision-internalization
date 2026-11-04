# X-M · qwen3.5-9b · xm_box_grabcut · xm_tune

- 定义：对照（判定第 1 条）：X-M 的框 → W1-F 的缺省 GrabCut，不描样条（mask：box_grabcut；调用：['list']）

## det（box IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.7388（95% CI [0.7149, 0.7796]）；P 0.7623、R 0.7167
- f1_rexomni_miou：0.5632（95% CI [0.5372, 0.5973]）；P 0.5840、R 0.5439
- f1_micro：0.6630（95% CI [0.6256, 0.6992]）；P 0.6658、R 0.6602
- matched_mean_iou：0.8324（95% CI [0.8233, 0.8423]）
- f1_rexomni_exact：0.7615；P 0.8007、R 0.7260
- f1_rexomni_exact_miou：0.5955；P 0.6479、R 0.5510

## seg（mask IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.4243（95% CI [0.3910, 0.4710]）；P 0.4376、R 0.4118
- f1_rexomni_miou：0.2730（95% CI [0.2437, 0.3024]）；P 0.2843、R 0.2626
- f1_micro：0.3314（95% CI [0.3002, 0.3683]）；P 0.3262、R 0.3367
- matched_mean_iou：0.7593（95% CI [0.7504, 0.7693]）
- f1_rexomni_exact：0.4893；P 0.5510、R 0.4400
- f1_rexomni_exact_miou：0.3456；P 0.4498、R 0.2806

- 失败与退回：{"list_failure": {"ok": 921, "truncated": 2}, "list_counts": {"n_raw": 2392, "n_capped": 0, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 25, "n_pred_boxes": {"n": 923, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 9.0, "max": 30, "mean": 2.5644637053087758, "n_distinct": 27, "top": [[1, 578], [2, 122], [3, 66], [4, 47], [5, 26], [6, 13], [7, 13], [8, 7], [9, 7], [11, 6]]}, "fallback": {"ok": 2367}, "tool_not_ok": 3}
