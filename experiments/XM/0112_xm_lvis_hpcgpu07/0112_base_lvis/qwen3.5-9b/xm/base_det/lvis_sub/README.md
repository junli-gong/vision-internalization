# X-M · qwen3.5-9b · base_det · lvis_sub

- 定义：对照（判定第 2 条）：基座零样本，只出框（mask：None；调用：['list']）

## det（box IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.6792（95% CI [0.6718, 0.7093]）；P 0.7210、R 0.6420
- f1_rexomni_miou：0.5044（95% CI [0.4972, 0.5322]）；P 0.5396、R 0.4735
- f1_micro：0.5346（95% CI [0.4932, 0.5767]）；P 0.6563、R 0.4510
- matched_mean_iou：0.8264（95% CI [0.8193, 0.8337]）
- f1_rexomni_exact：0.7551；P 0.8194、R 0.7001
- f1_rexomni_exact_miou：0.6013；P 0.7197、R 0.5164

- 失败与退回：{"list_failure": {"inverted": 1, "malformed": 3, "no_box": 1, "ok": 2153, "out_of_range": 2, "truncated": 53}, "list_counts": {"n_raw": 6159, "n_capped": 316, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 125, "n_pred_boxes": {"n": 2213, "min": 0, "p05": 1.0, "p50": 1.0, "p95": 8.0, "max": 30, "mean": 2.5838228648892905, "n_distinct": 31, "top": [[1, 1082], [2, 597], [3, 148], [4, 85], [0, 80], [5, 49], [6, 35], [30, 19], [7, 18], [9, 16]]}}
