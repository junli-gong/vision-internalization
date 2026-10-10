# E5 · 只评起点（e3b_c3，N = 0，不经 E5 训练）· qwen3.5-2b · n0 · seed 1 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 起点：e3b_c3（manifest 7c72a5007e461cca…；合并起点 LoRA 之后的权重指纹 e3958f54ee83b9b3…）；配种子 match（起点的种子 1，--allow-seed-mismatch False）
- 评测前的核对：起点检查点的合并核对（grid_eval：merge、restricted，判据 v1）与权重指纹（= factory.load_trained(起点, merge=True) 的）；跳过：e5_lora_merge（训练后评测的“E5 LoRA 合并前后”核对（stacked_merge_check）不适用：只评起点没有 E5 LoRA，起点 LoRA 已合并，评测的模型就是起点合并评测时的权重（由起点检查点的合并核对与权重指纹核对））
- 用途：N = 0 的读数，区分截距与斜率（E5 预注册 §8 第 2 项），不单独参与 H5 判定
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 2.9796（95% CI [2.7695, 3.2071]），有效率 0.9267，条件 3.0571（中位数 2.5311），n = 300
- mask IoU（网格）：严格 0.0859（95% CI [0.0792, 0.0925]），有效率 1.0000，条件 0.0859（中位数 0.0791），n = 300
- mask IoU（全分辨率）：严格 0.0860（95% CI [0.0793, 0.0927]），有效率 1.0000，条件 0.0860（中位数 0.0783），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0859 vs 最强非真值基线 copy 0.6878；差 -0.6019，CI [-0.6192, -0.5855] → pass=False
- 判定条件：evaluated，系统起伏误差 2.9796，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 1.9781，CI [1.7692, 2.2010] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 1.7436，CI [1.5769, 1.9122] → 胜过=False
  - vs copy_convex：基线 0.9363；差 2.0433，CI [1.8375, 2.2627] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
