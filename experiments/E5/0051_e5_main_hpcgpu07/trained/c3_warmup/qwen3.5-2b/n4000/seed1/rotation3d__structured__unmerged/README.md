# E5 · 训练后（c3_warmup → E5，E5 LoRA 不合并）· qwen3.5-2b · n4000 · seed 1 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest 7681747aceca195a…；起点 c3_warmup（manifest e547056affbd7f0b…）；训练运行身份 0d94263309b2b116…
- E5 LoRA：**不合并**（lora_merge = unmerged，LoRA 状态 injected）。D99-2 的敏感性分析：同一个检查点、同一条评测路径（run_task + decode_grids(decode=structured)），样本、提示、批大小、解析、评分、汇总与门槛都与正式评测相同；正式评测（合并后）的结果在同一模型目录的 rotation3d__structured/
- 评测前的核对：起点 manifest 的 sha256 与权重指纹照常；跳过：e5_lora_merge（D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json）
- LoRA 审计：解码期间逐批核对每个被调用的 LoRA 层都走了 lora_B（LoraAudit，规则同 DEV-0046），结果在 logs/run_<k>.json 的 lora_audit；审计不过就报错，不写 metrics
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.9665（95% CI [1.9460, 1.9847]），有效率 0.0567，条件 1.4091（中位数 1.6810），n = 300
- mask IoU（网格）：严格 0.0277（95% CI [0.0147, 0.0409]），有效率 1.0000，条件 0.0277（中位数 0.0000），n = 300
- mask IoU（全分辨率）：严格 0.0276（95% CI [0.0147, 0.0408]），有效率 1.0000，条件 0.0276（中位数 0.0000），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0277 vs 最强非真值基线 copy 0.6878；差 -0.6601，CI [-0.6809, -0.6405] → pass=False
- 判定条件：evaluated，系统起伏误差 1.9665，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 0.9650，CI [0.9421, 0.9874] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.7305，CI [0.5712, 0.8663] → 胜过=False
  - vs copy_convex：基线 0.9363；差 1.0302，CI [1.0037, 1.0540] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
