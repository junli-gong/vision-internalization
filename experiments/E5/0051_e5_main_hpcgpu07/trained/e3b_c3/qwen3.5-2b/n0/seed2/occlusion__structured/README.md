# E5 · 只评起点（e3b_c3，N = 0，不经 E5 训练）· qwen3.5-2b · n0 · seed 2 · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…，与零样本相同的清单）；生成器任务版本 2
- 起点：e3b_c3（manifest aaacbbab3e2f2ef3…；合并起点 LoRA 之后的权重指纹 d8c53de5f6d30a4d…）；配种子 match（起点的种子 2，--allow-seed-mismatch False）
- 评测前的核对：起点检查点的合并核对（grid_eval：merge、restricted，判据 v1）与权重指纹（= factory.load_trained(起点, merge=True) 的）；跳过：e5_lora_merge（训练后评测的“E5 LoRA 合并前后”核对（stacked_merge_check）不适用：只评起点没有 E5 LoRA，起点 LoRA 已合并，评测的模型就是起点合并评测时的权重（由起点检查点的合并核对与权重指纹核对））
- 用途：N = 0 的读数，区分截距与斜率（E5 预注册 §8 第 2 项），不单独参与 H5 判定
- 解码：结构强制（D83-3，v3 的正式口径；binary，每个格子位置只允许 2 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.0012（95% CI [0.0000, 0.0036]），有效率 1.0000，条件 0.0012（中位数 0.0000），n = 300
- mask IoU（全分辨率）：严格 0.0011（95% CI [0.0000, 0.0034]），有效率 1.0000，条件 0.0011（中位数 0.0000），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0012 vs 最强非真值基线 convex_hull 0.7081；差 -0.7069，CI [-0.7250, -0.6881] → pass=False
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
