# E5 · 训练后（base_warmup → E5）· qwen3.5-2b · n64000 · seed 0 · physics（d）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 0ea63f1c30af02dc…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest b0d3144d34e79cae…；起点 base_warmup（manifest 58ba382e9cd0b5bf…）；训练运行身份 6435020c591a36d1…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.6526（95% CI [0.6277, 0.6771]），有效率 1.0000，条件 0.6526（中位数 0.7048），n = 400
- mask IoU（全分辨率）：严格 0.5985（95% CI [0.5773, 0.6184]），有效率 1.0000，条件 0.5985（中位数 0.6600），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.6526 vs 最强非真值基线 copy 0.2906；差 0.3620，CI [0.3322, 0.3904] → pass=True
- 判定条件：not_applicable
- imagining：True

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
