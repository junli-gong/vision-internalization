# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n4000 · seed 1 · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest c0019d300d4787c5…；起点 c3_warmup（manifest e547056affbd7f0b…）；训练运行身份 d682d02ed8bebd80…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.5309（95% CI [0.5095, 0.5523]），有效率 1.0000，条件 0.5309（中位数 0.5427），n = 300
- mask IoU（全分辨率）：严格 0.5234（95% CI [0.5025, 0.5435]），有效率 1.0000，条件 0.5234（中位数 0.5363），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.5309 vs 最强非真值基线 convex_hull 0.7081；差 -0.1772，CI [-0.1993, -0.1555] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
