# E5 · 训练后（e3b → E5）· qwen3.5-2b · n1000 · seed 0 · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest c9d40666daeec412…；起点 e3b（manifest acb7f58d1ae3aa0c…）；训练运行身份 3169f1982035ddfa…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.6095（95% CI [0.5874, 0.6306]），有效率 1.0000，条件 0.6095（中位数 0.6274），n = 300
- mask IoU（全分辨率）：严格 0.5942（95% CI [0.5730, 0.6151]），有效率 1.0000，条件 0.5942（中位数 0.6105），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.6095 vs 最强非真值基线 convex_hull 0.7081；差 -0.0987，CI [-0.1208, -0.0781] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
