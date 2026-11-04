# Rex-Omni（外部参照）· rexomni_det · lvis_sub

- 批准：D165 第四节（外部参照，仅检测）；D177-c；CLAUDE.md 硬约束第 1 条的单独运行的对照系统。学习型检测器，单独运行；输出不进入我们的任何数据、标签、奖励或挑选。
- 设置：诊断：与 X-M 同协议（每个（图, 类别）对一次，只问这一个类别）（query_mode per_category）。每个（图像, 类别）一次调用，提示里只有这一个类别。**不是**官方 COCO/LVIS 评测的每图一次多类别提示，读数不能当作 Rex-Omni 论文数字的复现；只作与 X-M 同协议的诊断。
- 官方代码 https://github.com/IDEA-Research/Rex-Omni @ 6508981c；权重 IDEA-Research/Rex-Omni @ 0e5693d2（IDEA License 1.0（限非商业研究；权重另受 Qwen Research License 约束））。

## det（框 IoU；2213 个对、1000 张图、446 类）
- f1_rexomni：0.6926（95% CI [0.6846, 0.7223]）；P 0.8012、R 0.6100
- f1_rexomni_miou：0.5034（95% CI [0.4961, 0.5305]）；P 0.5836、R 0.4426
- f1_micro：0.5896（95% CI [0.5527, 0.6272]）；P 0.8198、R 0.4603
- matched_mean_iou：0.8201（95% CI [0.8117, 0.8277]）
- f1_rexomni_exact：0.7731；P 0.8867、R 0.6852
- f1_rexomni_exact_miou：0.6015；P 0.7610、R 0.4973

- 调用 2213 次；解析失败 0（{"ok": 2213}）；截断 2；停止 {"eos": 2211, "max_new_tokens": 2}；未问类名的坐标串 0
- 解析计数：{"n_coord_lists": 6174, "n_other_category": 19, "n_not_box": 2, "n_boxes_parsed": 6153, "n_capped": 1300, "n_out_of_grid": 0, "n_inverted": 0}
- 每次调用生成 token：均值 20.1094、最大 3707.0000；每次调用墙钟均值 0.5481 s
