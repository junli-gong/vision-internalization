# E5 · 只评起点（e3b_c3，N = 0，不经 E5 训练）· qwen3.5-2b · n0 · seed 1 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 起点：e3b_c3（manifest 7c72a5007e461cca…；合并起点 LoRA 之后的权重指纹 e3958f54ee83b9b3…）；配种子 match（起点的种子 1，--allow-seed-mismatch False）
- 评测前的核对：起点检查点的合并核对（grid_eval：merge、restricted，判据 v1）与权重指纹（= factory.load_trained(起点, merge=True) 的）；跳过：e5_lora_merge（训练后评测的“E5 LoRA 合并前后”核对（stacked_merge_check）不适用：只评起点没有 E5 LoRA，起点 LoRA 已合并，评测的模型就是起点合并评测时的权重（由起点检查点的合并核对与权重指纹核对））
- 用途：N = 0 的读数，区分截距与斜率（E5 预注册 §8 第 2 项），不单独参与 H5 判定
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 2.3875（95% CI [2.2973, 2.4770]），有效率 0.8444，条件 2.4589（中位数 2.3632），n = 315
- mask IoU（网格）：严格 0.1246（95% CI [0.1171, 0.1318]），有效率 1.0000，条件 0.1246（中位数 0.1250），n = 315
- mask IoU（全分辨率）：严格 0.1246（95% CI [0.1172, 0.1318]），有效率 1.0000，条件 0.1246（中位数 0.1245），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.1246 vs 最强非真值基线 translate 0.2630；差 -0.1384，CI [-0.1627, -0.1145] → pass=False
- 判定条件：evaluated，系统起伏误差 2.3875，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 1.2492，CI [1.1444, 1.3514] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 1.6171，CI [1.5208, 1.7090] → 胜过=False
  - vs copy_convex：基线 1.1403；差 1.2472，CI [1.1408, 1.3490] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
