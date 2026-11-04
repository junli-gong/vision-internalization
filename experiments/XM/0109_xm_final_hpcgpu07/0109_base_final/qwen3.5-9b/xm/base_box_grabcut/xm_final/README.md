# X-M · qwen3.5-9b · base_box_grabcut · xm_final

- 定义：起点：基座零样本的框 → 缺省 GrabCut（mask：box_grabcut；调用：['list']）

## det（box IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.6425（95% CI [0.6290, 0.6768]）；P 0.6260、R 0.6598
- f1_rexomni_miou：0.4770（95% CI [0.4628, 0.5091]）；P 0.4675、R 0.4870
- f1_micro：0.5808（95% CI [0.5567, 0.6058]）；P 0.5379、R 0.6312
- matched_mean_iou：0.8275（95% CI [0.8219, 0.8336]）
- f1_rexomni_exact：0.6533；P 0.6469、R 0.6598
- f1_rexomni_exact_miou：0.4932；P 0.4995、R 0.4870

## seg（mask IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.3929（95% CI [0.3758, 0.4214]）；P 0.3828、R 0.4036
- f1_rexomni_miou：0.2308（95% CI [0.2210, 0.2518]）；P 0.2248、R 0.2370
- f1_micro：0.3094（95% CI [0.2910, 0.3300]）；P 0.2812、R 0.3440
- matched_mean_iou：0.7468（95% CI [0.7409, 0.7528]）
- f1_rexomni_exact：0.4088；P 0.4038、R 0.4140
- f1_rexomni_exact_miou：0.2541；P 0.2660、R 0.2432

- 失败与退回：{"list_failure": {"inverted": 1, "no_box": 1, "ok": 2837, "out_of_range": 2, "truncated": 99}, "list_counts": {"n_raw": 10244, "n_capped": 865, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 290, "n_pred_boxes": {"n": 2940, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 12.0, "max": 30, "mean": 3.0914965986394556, "n_distinct": 30, "top": [[1, 1428], [2, 687], [3, 187], [0, 130], [4, 112], [5, 70], [30, 61], [6, 41], [7, 40], [8, 31]]}, "fallback": {"ok": 9089}, "tool_not_ok": 56}
