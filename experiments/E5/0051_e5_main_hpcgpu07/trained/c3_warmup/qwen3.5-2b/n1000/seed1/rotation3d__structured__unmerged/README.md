# E5 · 训练后（c3_warmup → E5，E5 LoRA 不合并）· qwen3.5-2b · n1000 · seed 1 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest a7a2eac74cfbd2c2…；起点 c3_warmup（manifest e547056affbd7f0b…）；训练运行身份 67c0d40b563d0d7b…
- E5 LoRA：**不合并**（lora_merge = unmerged，LoRA 状态 injected）。D99-2 的敏感性分析：同一个检查点、同一条评测路径（run_task + decode_grids(decode=structured)），样本、提示、批大小、解析、评分、汇总与门槛都与正式评测相同；正式评测（合并后）的结果在同一模型目录的 rotation3d__structured/
- 评测前的核对：起点 manifest 的 sha256 与权重指纹照常；跳过：e5_lora_merge（D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json）
- LoRA 审计：解码期间逐批核对每个被调用的 LoRA 层都走了 lora_B（LoraAudit，规则同 DEV-0046），结果在 logs/run_<k>.json 的 lora_audit；审计不过就报错，不写 metrics
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.8044（95% CI [1.6120, 2.0406]），有效率 0.9967，条件 1.8037（中位数 1.3109），n = 300
- mask IoU（网格）：严格 0.5000（95% CI [0.4826, 0.5163]），有效率 1.0000，条件 0.5000（中位数 0.4926），n = 300
- mask IoU（全分辨率）：严格 0.4948（95% CI [0.4779, 0.5107]），有效率 1.0000，条件 0.4948（中位数 0.4909），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.5000 vs 最强非真值基线 copy 0.6878；差 -0.1877，CI [-0.2089, -0.1676] → pass=False
- 判定条件：evaluated，系统起伏误差 1.8044，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 0.8029，CI [0.6096, 1.0310] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.5684，CI [0.4174, 0.7456] → 胜过=False
  - vs copy_convex：基线 0.9363；差 0.8680，CI [0.6801, 1.0964] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
