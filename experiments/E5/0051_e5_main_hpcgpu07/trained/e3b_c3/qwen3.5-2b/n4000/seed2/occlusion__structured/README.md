# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n4000 · seed 2 · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest fa8699f4866e5ebd…；起点 e3b_c3（manifest aaacbbab3e2f2ef3…）；训练运行身份 181e2256a0658d0a…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.5857（95% CI [0.5639, 0.6082]），有效率 1.0000，条件 0.5857（中位数 0.5964），n = 300
- mask IoU（全分辨率）：严格 0.5718（95% CI [0.5504, 0.5937]），有效率 1.0000，条件 0.5718（中位数 0.5783），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.5857 vs 最强非真值基线 convex_hull 0.7081；差 -0.1224，CI [-0.1456, -0.1005] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
