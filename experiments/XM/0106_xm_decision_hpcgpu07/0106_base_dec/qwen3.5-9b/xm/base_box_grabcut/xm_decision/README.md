# X-M · qwen3.5-9b · base_box_grabcut · xm_decision

- 定义：起点：基座零样本的框 → 缺省 GrabCut（mask：box_grabcut；调用：['list']）

## det（box IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.6748（95% CI [0.6643, 0.7183]）；P 0.6546、R 0.6963
- f1_rexomni_miou：0.4855（95% CI [0.4774, 0.5264]）；P 0.4701、R 0.5020
- f1_micro：0.5931（95% CI [0.5613, 0.6255]）；P 0.5760、R 0.6113
- matched_mean_iou：0.8271（95% CI [0.8199, 0.8344]）
- f1_rexomni_exact：0.6836；P 0.6713、R 0.6963
- f1_rexomni_exact_miou：0.5076；P 0.5133、R 0.5020

## seg（mask IoU；1437 个对、500 张图、79 类）
- f1_rexomni：0.3899（95% CI [0.3788, 0.4359]）；P 0.3758、R 0.4051
- f1_rexomni_miou：0.2348（95% CI [0.2260, 0.2637]）；P 0.2262、R 0.2440
- f1_micro：0.2945（95% CI [0.2699, 0.3209]）；P 0.2772、R 0.3142
- matched_mean_iou：0.7501（95% CI [0.7411, 0.7593]）
- f1_rexomni_exact：0.4238；P 0.4324、R 0.4156
- f1_rexomni_exact_miou：0.2762；P 0.3079、R 0.2504

- 失败与退回：{"list_failure": {"malformed": 1, "ok": 1368, "out_of_range": 1, "truncated": 67}, "list_counts": {"n_raw": 4641, "n_capped": 359, "n_clipped": 0, "n_inverted": 0}, "nms_suppressed": 130, "n_pred_boxes": {"n": 1437, "min": 0, "p05": 0.0, "p50": 1.0, "p95": 11.200000000000045, "max": 30, "mean": 2.8893528183716075, "n_distinct": 30, "top": [[1, 703], [2, 318], [3, 94], [0, 89], [4, 54], [5, 34], [30, 21], [6, 19], [8, 17], [11, 11]]}, "fallback": {"ok": 4152}, "tool_not_ok": 10}
