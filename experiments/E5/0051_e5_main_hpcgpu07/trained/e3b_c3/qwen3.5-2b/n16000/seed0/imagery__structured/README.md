# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n16000 · seed 0 · imagery（e）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 f94cb7b097c8d72b…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest 1c2960ebad25dd2a…；起点 e3b_c3（manifest bf73188b8c88c760…）；训练运行身份 5d66deea556bcfc4…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2500（95% CI [0.2235, 0.2759]），有效率 1.0000，条件 0.2500（中位数 0.2034），n = 400
- mask IoU（全分辨率）：严格 0.2426（95% CI [0.2170, 0.2676]），有效率 1.0000，条件 0.2426（中位数 0.2011），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2500 vs 最强非真值基线 no_transform 0.2146；差 0.0354，CI [0.0115, 0.0610] → pass=True
- 判定条件：not_applicable
- imagining：True

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
