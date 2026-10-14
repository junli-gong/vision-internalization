# E5 · 只评起点（e3b_c3，N = 0，不经 E5 训练）· qwen3.5-2b · n0 · seed 0 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 起点：e3b_c3（manifest bf73188b8c88c760…；合并起点 LoRA 之后的权重指纹 6d0d2e0210ffdd31…）；配种子 match（起点的种子 0，--allow-seed-mismatch False）
- 评测前的核对：起点检查点的合并核对（grid_eval：merge、restricted，判据 v1）与权重指纹（= factory.load_trained(起点, merge=True) 的）；跳过：e5_lora_merge（训练后评测的“E5 LoRA 合并前后”核对（stacked_merge_check）不适用：只评起点没有 E5 LoRA，起点 LoRA 已合并，评测的模型就是起点合并评测时的权重（由起点检查点的合并核对与权重指纹核对））
- 用途：N = 0 的读数，区分截距与斜率（E5 预注册 §8 第 2 项），不单独参与 H5 判定
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 3.5504（95% CI [3.2426, 3.8916]），有效率 1.0000，条件 3.5504（中位数 2.7411），n = 300
- mask IoU（网格）：严格 0.0939（95% CI [0.0883, 0.0994]），有效率 1.0000，条件 0.0939（中位数 0.0859），n = 300
- mask IoU（全分辨率）：严格 0.0940（95% CI [0.0885, 0.0995]），有效率 1.0000，条件 0.0940（中位数 0.0867），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0939 vs 最强非真值基线 copy 0.6878；差 -0.5939，CI [-0.6108, -0.5776] → pass=False
- 判定条件：evaluated，系统起伏误差 3.5504，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 2.5489，CI [2.2453, 2.8823] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 2.3144，CI [2.0975, 2.5554] → 胜过=False
  - vs copy_convex：基线 0.9363；差 2.6141，CI [2.3128, 2.9385] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
