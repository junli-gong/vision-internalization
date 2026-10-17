# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n16000 · seed 0 · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest 3e78b64d04b4ad21…；起点 c3_warmup（manifest dca6a89d9a3e920c…）；训练运行身份 4b5a7b9d735677b5…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2164（95% CI [0.1881, 0.2449]），有效率 1.0000，条件 0.2164（中位数 0.0522），n = 300
- mask IoU（全分辨率）：严格 0.2140（95% CI [0.1858, 0.2425]），有效率 1.0000，条件 0.2140（中位数 0.0518），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2164 vs 最强非真值基线 convex_hull 0.7081；差 -0.4918，CI [-0.5218, -0.4630] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
