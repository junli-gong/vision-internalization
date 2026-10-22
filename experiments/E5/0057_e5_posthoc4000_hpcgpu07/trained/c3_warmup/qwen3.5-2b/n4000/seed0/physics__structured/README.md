# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n4000 · seed 0 · physics（d）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 0ea63f1c30af02dc…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest 85f41063b47214a5…；起点 c3_warmup（manifest dca6a89d9a3e920c…）；训练运行身份 f208ccb9ae2c34ce…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.5126（95% CI [0.4866, 0.5375]），有效率 1.0000，条件 0.5126（中位数 0.5517），n = 400
- mask IoU（全分辨率）：严格 0.4829（95% CI [0.4602, 0.5043]），有效率 1.0000，条件 0.4829（中位数 0.5226），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.5126 vs 最强非真值基线 copy 0.2906；差 0.2220，CI [0.1934, 0.2527] → pass=True
- 判定条件：not_applicable
- imagining：True

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
