# E5 · 训练后（c3_warmup → E5，E5 LoRA 不合并）· qwen3.5-2b · n1000 · seed 2 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest 17a7e4ccdbafc9d0…；起点 c3_warmup（manifest 1ef34cdf543d80fc…）；训练运行身份 14753a441ea8f515…
- E5 LoRA：**不合并**（lora_merge = unmerged，LoRA 状态 injected）。D99-2 的敏感性分析：同一个检查点、同一条评测路径（run_task + decode_grids(decode=structured)），样本、提示、批大小、解析、评分、汇总与门槛都与正式评测相同；正式评测（合并后）的结果在同一模型目录的 rotation3d__structured/
- 评测前的核对：起点 manifest 的 sha256 与权重指纹照常；跳过：e5_lora_merge（D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json）
- LoRA 审计：解码期间逐批核对每个被调用的 LoRA 层都走了 lora_B（LoraAudit，规则同 DEV-0046），结果在 logs/run_<k>.json 的 lora_audit；审计不过就报错，不写 metrics
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.7310（95% CI [1.5977, 1.8754]），有效率 0.9900，条件 1.7283（中位数 1.4587），n = 300
- mask IoU（网格）：严格 0.3973（95% CI [0.3784, 0.4169]），有效率 1.0000，条件 0.3973（中位数 0.3898），n = 300
- mask IoU（全分辨率）：严格 0.3949（95% CI [0.3764, 0.4136]），有效率 1.0000，条件 0.3949（中位数 0.3911），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.3973 vs 最强非真值基线 copy 0.6878；差 -0.2905，CI [-0.3110, -0.2684] → pass=False
- 判定条件：evaluated，系统起伏误差 1.7310，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 0.7295，CI [0.6021, 0.8655] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.4950，CI [0.3575, 0.6227] → 胜过=False
  - vs copy_convex：基线 0.9363；差 0.7947，CI [0.6727, 0.9242] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
