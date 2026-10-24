# W2-c 训练数据 · qwen3.5-9b

- 格式：docs/w2_plan.md §5；w2c_data/ 下 16/16 片，16000/16000 句（complete = True）
- w2 配置 sha256 c4d083489783039d…；候选清单 sha256 c134cfab5d94adae…（K = 4：['gc_default', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts']）；manifest.json sha256 22fd7767342d6f4f…
- 出框失败 309（0.0193）：{'malformed': 1, 'multiple_boxes': 287, 'no_box': 1, 'truncated': 20}
- 最优候选的平均 IoU 0.6453；1 号（W1-F）的平均 IoU 0.5468（有候选的句子）
- 标签的分布：{'1': 2791, '2': 3582, '3': 3915, '4': 5403, 'null': 309}；并列的句子 1475
- 各编号作为最优的次数（并列都算）：{'1': 2791, '2': 4104, '3': 5378, '4': 5898}

## 文件

- w2c_data/manifest.json：总体、去重（E3 的剔除清单、W2 的评测图）、抽样各步的条数与 sha256，选中的 id（处理顺序）与它们的图；
- propose/：E1 refexp_box 的运行（E1 格式）；w2c_data/*.jsonl：数据；config.yaml：运行身份；metrics.json：由数据算出；logs/：每句候选生成的墙钟与每次调用的记录。
