# E5 · 训练后（e3b_c3 → E5，E5 LoRA 不合并）· qwen3.5-2b · n4000 · seed 1 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest 6e6aebc2a23df08a…；起点 e3b_c3（manifest 7c72a5007e461cca…）；训练运行身份 aea71d1028b8ccc2…
- E5 LoRA：**不合并**（lora_merge = unmerged，LoRA 状态 injected）。D99-2 的敏感性分析：同一个检查点、同一条评测路径（run_task + decode_grids(decode=structured)），样本、提示、批大小、解析、评分、汇总与门槛都与正式评测相同；正式评测（合并后）的结果在同一模型目录的 viewpoint__structured/
- 评测前的核对：起点 manifest 的 sha256 与权重指纹照常；跳过：e5_lora_merge（D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json）
- LoRA 审计：解码期间逐批核对每个被调用的 LoRA 层都走了 lora_B（LoraAudit，规则同 DEV-0046），结果在 logs/run_<k>.json 的 lora_audit；审计不过就报错，不写 metrics
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.9076（95% CI [1.8715, 1.9479]），有效率 0.1905，条件 1.5147（中位数 1.4678），n = 315
- mask IoU（网格）：严格 0.0665（95% CI [0.0479, 0.0839]），有效率 1.0000，条件 0.0665（中位数 0.0000），n = 315
- mask IoU（全分辨率）：严格 0.0662（95% CI [0.0477, 0.0832]），有效率 1.0000，条件 0.0662（中位数 0.0000），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0665 vs 最强非真值基线 translate 0.2630；差 -0.1965，CI [-0.2244, -0.1720] → pass=False
- 判定条件：evaluated，系统起伏误差 1.9076，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.7693，CI [0.7155, 0.8265] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 1.1371，CI [1.0779, 1.1985] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.7673，CI [0.7128, 0.8262] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
