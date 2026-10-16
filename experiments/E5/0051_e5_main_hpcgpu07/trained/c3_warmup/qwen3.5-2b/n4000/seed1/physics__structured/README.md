# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n4000 · seed 1 · physics（d）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 0ea63f1c30af02dc…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest 7681747aceca195a…；起点 c3_warmup（manifest e547056affbd7f0b…）；训练运行身份 0d94263309b2b116…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2837（95% CI [0.2588, 0.3086]），有效率 1.0000，条件 0.2837（中位数 0.2612），n = 400
- mask IoU（全分辨率）：严格 0.2664（95% CI [0.2435, 0.2889]），有效率 1.0000，条件 0.2664（中位数 0.2546），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2837 vs 最强非真值基线 copy 0.2906；差 -0.0068，CI [-0.0358, 0.0234] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
