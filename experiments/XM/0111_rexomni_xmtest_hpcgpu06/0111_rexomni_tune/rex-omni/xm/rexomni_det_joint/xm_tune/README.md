# Rex-Omni（外部参照）· rexomni_det_joint · xm_tune

- 批准：D165 第四节（外部参照，仅检测）；D177-c；CLAUDE.md 硬约束第 1 条的单独运行的对照系统。学习型检测器，单独运行；输出不进入我们的任何数据、标签、奖励或挑选。
- 设置：标题读数：Rex-Omni 自己的协议（每图一次，列出该图本划分里的全部正类别，category_id 升序）（query_mode joint）。每张图一次调用，提示里列出该图本划分里的全部正类别（category_id 升序；官方 COCO.jsonl 的顺序不是升序，见 config.yaml 的 protocol.deviations）。这是 Rex-Omni 自己的协议，作论文里外部参照的标题读数；划分是 X-M 的子集，不是论文的完整 val2017。
- 官方代码 https://github.com/IDEA-Research/Rex-Omni @ 6508981c；权重 IDEA-Research/Rex-Omni @ 0e5693d2（IDEA License 1.0（限非商业研究；权重另受 Qwen Research License 约束））。

## det（框 IoU；923 个对、300 张图、78 类）
- f1_rexomni：0.7414（95% CI [0.7177, 0.7812]）；P 0.7863、R 0.7015
- f1_rexomni_miou：0.5515（95% CI [0.5259, 0.5899]）；P 0.5835、R 0.5229
- f1_micro：0.6974（95% CI [0.6521, 0.7391]）；P 0.7246、R 0.6721
- matched_mean_iou：0.8291（95% CI [0.8207, 0.8390]）
- f1_rexomni_exact：0.7576；P 0.8114、R 0.7106
- f1_rexomni_exact_miou：0.5898；P 0.6653、R 0.5296

- 调用 300 次；解析失败 0（{"ok": 300}）；截断 1；停止 {"eos": 299, "max_new_tokens": 1}；未问类名的坐标串 0
- 解析计数：{"n_coord_lists": 2523, "n_other_category": 0, "n_not_box": 0, "n_boxes_parsed": 2523, "n_capped": 120, "n_out_of_grid": 0, "n_inverted": 7}
- 每次调用生成 token：均值 73.0933、最大 3645.0000；每次调用墙钟均值 1.8128 s
- 冒烟线：f1_rexomni_exact = 0.7576，门槛 0.6：通过
