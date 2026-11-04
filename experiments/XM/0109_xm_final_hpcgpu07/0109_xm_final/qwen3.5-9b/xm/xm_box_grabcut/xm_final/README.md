# X-M · qwen3.5-9b · xm_box_grabcut · xm_final

- 定义：对照（判定第 1 条）：X-M 的框 → W1-F 的缺省 GrabCut，不描样条（mask：box_grabcut；调用：['list']）

## det（box IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.6953（95% CI [0.6823, 0.7254]）；P 0.7176、R 0.6744
- f1_rexomni_miou：0.5214（95% CI [0.5059, 0.5505]）；P 0.5388、R 0.5050
- f1_micro：0.6696（95% CI [0.6474, 0.6897]）；P 0.6883、R 0.6519
- matched_mean_iou：0.8339（95% CI [0.8286, 0.8400]）
- f1_rexomni_exact：0.7050；P 0.7385、R 0.6744
- f1_rexomni_exact_miou：0.5365；P 0.5722、R 0.5050

## seg（mask IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.4120（95% CI [0.3949, 0.4386]）；P 0.4191、R 0.4052
- f1_rexomni_miou：0.2398（95% CI [0.2296, 0.2597]）；P 0.2434、R 0.2363
- f1_micro：0.3491（95% CI [0.3290, 0.3701]）；P 0.3470、R 0.3512
- matched_mean_iou：0.7467（95% CI [0.7407, 0.7526]）
- f1_rexomni_exact：0.4279；P 0.4407、R 0.4157
- f1_rexomni_exact_miou：0.2613；P 0.2832、R 0.2425

- 失败与退回：{"list_failure": {"ok": 2933, "truncated": 7}, "list_counts": {"n_raw": 7729, "n_capped": 99, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 39, "n_pred_boxes": {"n": 2940, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 9.0, "max": 30, "mean": 2.581972789115646, "n_distinct": 30, "top": [[1, 1847], [2, 426], [3, 192], [4, 99], [5, 74], [6, 52], [7, 46], [8, 33], [10, 25], [9, 24]]}, "fallback": {"ok": 7591}, "tool_not_ok": 6}
