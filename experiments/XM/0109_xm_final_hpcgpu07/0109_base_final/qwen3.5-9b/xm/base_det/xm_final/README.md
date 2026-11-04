# X-M · qwen3.5-9b · base_det · xm_final

- 定义：对照（判定第 2 条）：基座零样本，只出框（mask：None；调用：['list']）

## det（box IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.6425（95% CI [0.6290, 0.6768]）；P 0.6260、R 0.6598
- f1_rexomni_miou：0.4770（95% CI [0.4628, 0.5091]）；P 0.4675、R 0.4870
- f1_micro：0.5808（95% CI [0.5567, 0.6058]）；P 0.5379、R 0.6312
- matched_mean_iou：0.8275（95% CI [0.8219, 0.8336]）
- f1_rexomni_exact：0.6533；P 0.6469、R 0.6598
- f1_rexomni_exact_miou：0.4932；P 0.4995、R 0.4870

- 失败与退回：{"list_failure": {"inverted": 1, "no_box": 1, "ok": 2837, "out_of_range": 2, "truncated": 99}, "list_counts": {"n_raw": 10244, "n_capped": 865, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 290, "n_pred_boxes": {"n": 2940, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 12.0, "max": 30, "mean": 3.0914965986394556, "n_distinct": 30, "top": [[1, 1428], [2, 687], [3, 187], [0, 130], [4, 112], [5, 70], [30, 61], [6, 41], [7, 40], [8, 31]]}}
