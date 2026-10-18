# E5 · 训练后（e3b_c3 → E5，E5 LoRA 不合并）· qwen3.5-2b · n1000 · seed 2 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest 18d0fdc3d8341fea…；起点 e3b_c3（manifest aaacbbab3e2f2ef3…）；训练运行身份 4dde32f7428a15cd…
- E5 LoRA：**不合并**（lora_merge = unmerged，LoRA 状态 injected）。D99-2 的敏感性分析：同一个检查点、同一条评测路径（run_task + decode_grids(decode=structured)），样本、提示、批大小、解析、评分、汇总与门槛都与正式评测相同；正式评测（合并后）的结果在同一模型目录的 viewpoint__structured/
- 评测前的核对：起点 manifest 的 sha256 与权重指纹照常；跳过：e5_lora_merge（D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json）
- LoRA 审计：解码期间逐批核对每个被调用的 LoRA 层都走了 lora_B（LoraAudit，规则同 DEV-0046），结果在 logs/run_<k>.json 的 lora_audit；审计不过就报错，不写 metrics
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.4035（95% CI [1.3469, 1.4603]），有效率 0.8762，条件 1.3192（中位数 1.2797），n = 315
- mask IoU（网格）：严格 0.2813（95% CI [0.2569, 0.3053]），有效率 1.0000，条件 0.2813（中位数 0.2451），n = 315
- mask IoU（全分辨率）：严格 0.2800（95% CI [0.2559, 0.3036]），有效率 1.0000，条件 0.2800（中位数 0.2411），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2813 vs 最强非真值基线 translate 0.2630；差 0.0183，CI [-0.0024, 0.0399] → pass=False
- 判定条件：evaluated，系统起伏误差 1.4035，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.2652，CI [0.2034, 0.3294] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.6330，CI [0.5626, 0.7023] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.2631，CI [0.2031, 0.3277] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
