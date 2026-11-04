# Rex-Omni（外部参照）· rexomni_det · xm_final

- 批准：D165 第四节（外部参照，仅检测）；D177-c；CLAUDE.md 硬约束第 1 条的单独运行的对照系统。学习型检测器，单独运行；输出不进入我们的任何数据、标签、奖励或挑选。
- 设置：诊断：与 X-M 同协议（每个（图, 类别）对一次，只问这一个类别）（query_mode per_category）。每个（图像, 类别）一次调用，提示里只有这一个类别。**不是**官方 COCO/LVIS 评测的每图一次多类别提示，读数不能当作 Rex-Omni 论文数字的复现；只作与 X-M 同协议的诊断。
- 官方代码 https://github.com/IDEA-Research/Rex-Omni @ 6508981c；权重 IDEA-Research/Rex-Omni @ 0e5693d2（IDEA License 1.0（限非商业研究；权重另受 Qwen Research License 约束））。

## det（框 IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.7078（95% CI [0.6945, 0.7303]）；P 0.7673、R 0.6569
- f1_rexomni_miou：0.5188（95% CI [0.5043, 0.5429]）；P 0.5623、R 0.4814
- f1_micro：0.7016（95% CI [0.6855, 0.7174]）；P 0.7661、R 0.6471
- matched_mean_iou：0.8286（95% CI [0.8237, 0.8344]）
- f1_rexomni_exact：0.7118；P 0.7766、R 0.6569
- f1_rexomni_exact_miou：0.5292；P 0.5875、R 0.4814

- 调用 2940 次；解析失败 0（{"ok": 2940}）；截断 0；停止 {"eos": 2940}；未问类名的坐标串 0
- 解析计数：{"n_coord_lists": 6728, "n_other_category": 2, "n_not_box": 0, "n_boxes_parsed": 6726, "n_capped": 0, "n_out_of_grid": 0, "n_inverted": 0}
- 每次调用生成 token：均值 17.0585、最大 225.0000；每次调用墙钟均值 0.4683 s
