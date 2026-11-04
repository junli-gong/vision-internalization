# X-M · qwen3.5-9b · xm_box_grabcut · xm_decision

- 定义：对照（判定第 1 条）：X-M 的框 → W1-F 的缺省 GrabCut，不描样条（mask：box_grabcut；调用：['list']）

## det（box IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.7266（95% CI [0.7107, 0.7615]）；P 0.7435、R 0.7106
- f1_rexomni_miou：0.5343（95% CI [0.5227, 0.5692]）；P 0.5467、R 0.5225
- f1_micro：0.6531（95% CI [0.6229, 0.6838]）；P 0.6757、R 0.6319
- matched_mean_iou：0.8303（95% CI [0.8232, 0.8375]）
- f1_rexomni_exact：0.7311；P 0.7528、R 0.7106
- f1_rexomni_exact_miou：0.5508；P 0.5825、R 0.5225

## seg（mask IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.4136（95% CI [0.3996, 0.4550]）；P 0.4191、R 0.4082
- f1_rexomni_miou：0.2489（95% CI [0.2384, 0.2761]）；P 0.2519、R 0.2459
- f1_micro：0.3158（95% CI [0.2903, 0.3452]）；P 0.3129、R 0.3189
- matched_mean_iou：0.7475（95% CI [0.7385, 0.7563]）
- f1_rexomni_exact：0.4462；P 0.4775、R 0.4188
- f1_rexomni_exact_miou：0.2918；P 0.3459、R 0.2523

- 失败与退回：{"list_failure": {"ok": 1436, "truncated": 1}, "list_counts": {"n_raw": 3901, "n_capped": 109, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 35, "n_pred_boxes": {"n": 1437, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 10.0, "max": 30, "mean": 2.614474599860821, "n_distinct": 26, "top": [[1, 891], [2, 219], [3, 94], [4, 60], [5, 33], [7, 22], [6, 19], [10, 15], [8, 14], [9, 8]]}, "fallback": {"ok": 3757}, "tool_not_ok": 21}
