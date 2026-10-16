# E5 · 训练后（e3b → E5）· qwen3.5-2b · n1000 · seed 2 · imagery（e）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 f94cb7b097c8d72b…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest 79befcd86761a437…；起点 e3b（manifest a4073791af6478a6…）；训练运行身份 987b75975c17cbd9…
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.0872（95% CI [0.0738, 0.1020]），有效率 1.0000，条件 0.0872（中位数 0.0000），n = 400
- mask IoU（全分辨率）：严格 0.0865（95% CI [0.0729, 0.1005]），有效率 1.0000，条件 0.0865（中位数 0.0000），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0872 vs 最强非真值基线 no_transform 0.2146；差 -0.1273，CI [-0.1490, -0.1052] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
