# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n4000 · seed 1 · imagery（e）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 f94cb7b097c8d72b…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest c0019d300d4787c5…；起点 c3_warmup（manifest e547056affbd7f0b…）；训练运行身份 d682d02ed8bebd80…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.3536（95% CI [0.3316, 0.3744]），有效率 1.0000，条件 0.3536（中位数 0.3647），n = 400
- mask IoU（全分辨率）：严格 0.3449（95% CI [0.3245, 0.3646]），有效率 1.0000，条件 0.3449（中位数 0.3472），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.3536 vs 最强非真值基线 no_transform 0.2146；差 0.1391，CI [0.1157, 0.1633] → pass=True
- 判定条件：not_applicable
- imagining：True

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
