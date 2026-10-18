# E5 · 训练后（c3_warmup → E5，E5 LoRA 不合并）· qwen3.5-2b · n1000 · seed 0 · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…，与零样本相同的清单）；生成器任务版本 2
- 检查点：E5 manifest ef3341f513e53eb6…；起点 c3_warmup（manifest dca6a89d9a3e920c…）；训练运行身份 d75c1bd7c0eef63b…
- E5 LoRA：**不合并**（lora_merge = unmerged，LoRA 状态 injected）。D99-2 的敏感性分析：同一个检查点、同一条评测路径（run_task + decode_grids(decode=structured)），样本、提示、批大小、解析、评分、汇总与门槛都与正式评测相同；正式评测（合并后）的结果在同一模型目录的 occlusion__structured/
- 评测前的核对：起点 manifest 的 sha256 与权重指纹照常；跳过：e5_lora_merge（D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json）
- LoRA 审计：解码期间逐批核对每个被调用的 LoRA 层都走了 lora_B（LoraAudit，规则同 DEV-0046），结果在 logs/run_<k>.json 的 lora_audit；审计不过就报错，不写 metrics
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.3271（95% CI [0.3048, 0.3494]），有效率 1.0000，条件 0.3271（中位数 0.3277），n = 300
- mask IoU（全分辨率）：严格 0.3244（95% CI [0.3027, 0.3465]），有效率 1.0000，条件 0.3244（中位数 0.3259），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.3271 vs 最强非真值基线 convex_hull 0.7081；差 -0.3811，CI [-0.4045, -0.3570] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
