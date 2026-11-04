# Rex-Omni（外部参照）· rexomni_det_joint · xm_final

- 批准：D165 第四节（外部参照，仅检测）；D177-c；CLAUDE.md 硬约束第 1 条的单独运行的对照系统。学习型检测器，单独运行；输出不进入我们的任何数据、标签、奖励或挑选。
- 设置：标题读数：Rex-Omni 自己的协议（每图一次，列出该图本划分里的全部正类别，category_id 升序）（query_mode joint）。每张图一次调用，提示里列出该图本划分里的全部正类别（category_id 升序；官方 COCO.jsonl 的顺序不是升序，见 config.yaml 的 protocol.deviations）。这是 Rex-Omni 自己的协议，作论文里外部参照的标题读数；划分是 X-M 的子集，不是论文的完整 val2017。
- 官方代码 https://github.com/IDEA-Research/Rex-Omni @ 6508981c；权重 IDEA-Research/Rex-Omni @ 0e5693d2（IDEA License 1.0（限非商业研究；权重另受 Qwen Research License 约束））。

## det（框 IoU；2940 个对、1000 张图、79 类）
- f1_rexomni：0.7109（95% CI [0.6976, 0.7340]）；P 0.7627、R 0.6656
- f1_rexomni_miou：0.5204（95% CI [0.5059, 0.5453]）；P 0.5589、R 0.4869
- f1_micro：0.6992（95% CI [0.6804, 0.7174]）；P 0.7534、R 0.6523
- matched_mean_iou：0.8283（95% CI [0.8233, 0.8341]）
- f1_rexomni_exact：0.7160；P 0.7747、R 0.6656
- f1_rexomni_exact_miou：0.5317；P 0.5857、R 0.4869

- 调用 1000 次；解析失败 0（{"ok": 1000}）；截断 1；停止 {"eos": 999, "max_new_tokens": 1}；未问类名的坐标串 0
- 解析计数：{"n_coord_lists": 7222, "n_other_category": 0, "n_not_box": 0, "n_boxes_parsed": 7222, "n_capped": 282, "n_out_of_grid": 0, "n_inverted": 0}
- 每次调用生成 token：均值 57.9920、最大 3647.0000；每次调用墙钟均值 1.4434 s
