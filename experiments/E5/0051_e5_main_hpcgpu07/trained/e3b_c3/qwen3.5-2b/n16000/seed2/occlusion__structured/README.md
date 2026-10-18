# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n16000 · seed 2 · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest ce9af8dca172b4ef…；起点 e3b_c3（manifest aaacbbab3e2f2ef3…）；训练运行身份 0f239a5c0d4a7255…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.5726（95% CI [0.5425, 0.6017]），有效率 1.0000，条件 0.5726（中位数 0.6203），n = 300
- mask IoU（全分辨率）：严格 0.5591（95% CI [0.5301, 0.5868]），有效率 1.0000，条件 0.5591（中位数 0.6101），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.5726 vs 最强非真值基线 convex_hull 0.7081；差 -0.1356，CI [-0.1622, -0.1107] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
