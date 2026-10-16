# E5 · 训练后（e3b → E5）· qwen3.5-2b · n4000 · seed 2 · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest ec3409b1ae6b94e6…；起点 e3b（manifest a4073791af6478a6…）；训练运行身份 bb513e754e5776c2…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.6096（95% CI [0.5868, 0.6316]），有效率 1.0000，条件 0.6096（中位数 0.6458），n = 300
- mask IoU（全分辨率）：严格 0.5947（95% CI [0.5721, 0.6157]），有效率 1.0000，条件 0.5947（中位数 0.6290），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.6096 vs 最强非真值基线 convex_hull 0.7081；差 -0.0985，CI [-0.1194, -0.0784] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
