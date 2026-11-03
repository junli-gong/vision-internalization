# X-M 的外部参照与文献读数（D165；2026-11-03 核对，来源为论文、官方仓库与模型卡）

## Rex-Omni（arXiv 2510.12798，"Detect Anything via Next Point Prediction"；仓库称 CVPR 2026，会议录未核）

- **许可证**：代码与权重都是 IDEA License 1.0，限非商业研究；权重基于 Qwen2.5-VL-3B，另受 Qwen Research License 约束，同样限非商业研究。
  - 我们是非商业研究，**允许在同样的数据上跑**（D165 的条件满足）。
  - 再分发须附许可证与声明。我们不再分发。
- **权重**：`IDEA-Research/Rex-Omni`，BF16，8.14 GB，单张 40 GB A100 放得下。
- **评测协议**：
  - 提示 `Detect {类别}. Output the bounding box coordinates in [x0, y0, x1, y1] format.`；
  - 温度 0；
  - 只提示图中出现的类别（positive only），不出现的类别不提示、不罚。
- **F1 的定义**（代码 `evaluation/metrics/coco_lvis_metric.py` 与 COCOeval 移植）：
  - 按（图像, 类别）用 COCOeval 的贪心匹配，按输出顺序（所有预测的分数都是常数 0.2）。每个真值只能匹配一次，重复的框记 FP；crowd 区域上的预测忽略；maxDets 100。
  - 每个类别在全部图上汇总，得 P_c、R_c；再对类别取宏平均，F1 = 2PR / (P + R)。
    - 它的 safe_mean 会丢掉 ≤ 0 的类别；precision 有一处没按类别重置。
  - F1@mIoU：P、R 各在 IoU 0.5:0.05:0.95 上取平均后再求调和平均。
  - **Rex-Omni 没有"匹配上的平均 IoU"这个指标**，也不报 mask F1。
- **COCO val2017 检测**（表 2，完整 4,952 张）：

  | F1@0.5 | 值 |
  |---|---|
  | Qwen2.5-VL-3B | 64.7 |
  | Qwen2.5-VL-7B | 64.0 |
  | Grounding DINO-T（最优阈值） | 69.8 |
  | DINO-Swin-L（最优阈值） | 75.6 |
  | SEED1.5-VL | 71.3 |
  | Rex-Omni | 72.0 |

  - 除 Rex-Omni、SEED1.5-VL 外，MLLM 都是每次只问一个类别，与我们的协议相同。
  - 专用检测器取 0–1 扫出的最优分数阈值。
- **LVIS**：19,626 张，其中 14,874 张来自 COCO train2017。COCO 上训过的模型在它上面有图像重叠。我们的 LVIS 子集只取 val2017 的图（D165 的管理员补充）。

## SAM 3（arXiv 2511.16719）、LLMSeg（arXiv 2610.29409）

- SAM 3：
  - COCO val 检测 AP 56.4。没有 COCO 的 mask AP，也没有 COCO 的 cgF1。
  - 它的 cgF1 是匈牙利匹配、IoU 0.5:0.95 平均，并乘以图像级的 MCC，**与 F1@0.5 不可比**。
- LLMSeg：
  - 在 COCO val2017 的 1,500 个（图像, 类别）上报 cgF1 56.1，其中一半是负类，不是我们的协议；
  - ReasonSeg val gIoU / cIoU 69.6 / 71.5。
- 这两者只在正文里引用，不画在 F1 轴上。

## ReasonSeg val（LISA，arXiv 2308.00692）

- 200 条（图 + 推理型指令 + mask），4 条真值为空。官方只有一个 Google Drive 文件夹（val.zip 264 MB）。
- 代码 Apache-2.0，数据集没有单独写许可证（未核）；图来自 OpenImages 与 ScanNetv2。
- 读数（val，gIoU / cIoU）：

  | 方法 | gIoU / cIoU |
  |---|---|
  | LISA-7B | 44.4 / 46.0 |
  | LISA-7B (ft) | 52.9 / 54.0 |
  | LISA-13B-LLaVA1.5 (ft) | 65.0 / 72.9 |
  | Seg-Zero（Qwen2.5-VL-7B） | 62.6 / 62.0 |
  | SAM 3 Agent（Gemini 2.5 Pro） | 77.0 / 66.0 |
  | LLMSeg（Gemini） | 69.6 / 71.5 |

- LISA 的 v1 权重用 train+val 训过。

## Grounding DINO（arXiv 2303.05499）

- COCO val2017 box AP：
  - Swin-T 零样本 48.4，微调 57.2；
  - Swin-L 零样本 52.5，微调 62.6。
- AP 与 F1 不可比，只在正文里引用。
