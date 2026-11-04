# Rex-Omni（外部参照）· rexomni_det · xm_tune

- 批准：D165 第四节（外部参照，仅检测）；D177-c；CLAUDE.md 硬约束第 1 条的单独运行的对照系统。学习型检测器，单独运行；输出不进入我们的任何数据、标签、奖励或挑选。
- 设置：诊断：与 X-M 同协议（每个（图, 类别）对一次，只问这一个类别）（query_mode per_category）。每个（图像, 类别）一次调用，提示里只有这一个类别。**不是**官方 COCO/LVIS 评测的每图一次多类别提示，读数不能当作 Rex-Omni 论文数字的复现；只作与 X-M 同协议的诊断。
- 官方代码 https://github.com/IDEA-Research/Rex-Omni @ 6508981c；权重 IDEA-Research/Rex-Omni @ 0e5693d2（IDEA License 1.0（限非商业研究；权重另受 Qwen Research License 约束））。

## det（框 IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.7571（95% CI [0.7347, 0.7944]）；P 0.8110、R 0.7099
- f1_rexomni_miou：0.5611（95% CI [0.5359, 0.5989]）；P 0.6003、R 0.5267
- f1_micro：0.6986（95% CI [0.6438, 0.7452]）；P 0.7211、R 0.6774
- matched_mean_iou：0.8291（95% CI [0.8207, 0.8395]）
- f1_rexomni_exact：0.7759；P 0.8423、R 0.7191
- f1_rexomni_exact_miou：0.5980；P 0.6802、R 0.5335

- 调用 923 次；解析失败 0（{"ok": 923}）；截断 1；停止 {"eos": 922, "max_new_tokens": 1}；未问类名的坐标串 0
- 解析计数：{"n_coord_lists": 3110, "n_other_category": 0, "n_not_box": 1, "n_boxes_parsed": 3109, "n_capped": 709, "n_out_of_grid": 0, "n_inverted": 0}
- 每次调用生成 token：均值 22.4680、最大 3661.0000；每次调用墙钟均值 0.6045 s
