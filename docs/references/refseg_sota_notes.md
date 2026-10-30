# 指代分割的文献读数：统一主表的灰色参考线（D154；预注册 U-3、U-3a）

2026-10-30 核对。数字逐个读自论文 PDF 或官方仓库，比较过 arXiv 各版本。

## 口径

- **oIoU 与 cIoU 是同一个量**：整个划分的 Σ交 / Σ并。
  - UNINEXT、LAVT、PolyFormer、UniLSeg、OneRef、DeRIS 叫 oIoU；
  - LISA、ReLA、EVF-SAM 叫 cIoU。
- **mIoU 即 LISA 所说的 gIoU**：逐句 IoU 的均值，与 cIoU 不同，不画在 cIoU 轴上。
- RefCOCOg 一律是 UMD 划分。
- **文献数字都是在完整 test 划分上算的**，我们的统一主表是每个划分 300 句的子集（预注册 U-1），所以只作参考，不参与配对。

## 选作灰线的 4 行（每个模型取一整行，不跨论文挑各划分的最高值）

顺序：RefCOCO testA / testB | RefCOCO+ testA / testB | RefCOCOg test(U)。

| 模型 | 类别 | 出处 | 量 | 5 个 test 划分 |
|---|---|---|---|---|
| LISA-7B (ft)，在 refCOCO(+/g) 上微调 | LLM + SAM | LISA，arXiv 2308.00692（CVPR'24），表 2；v1–v3 相同 | cIoU | 79.1 / 72.3 / 70.8 / 58.1 / 70.6 |
| GLaMM (FT)，Vicuna-7B + SAM-H，GranD 预训练后在 refCOCO/+/g 上微调 | LLM + SAM | GLaMM，arXiv 2311.03356 v2/v3（CVPR'24），表 4 | cIoU（论文没写名字，见注 2） | 83.2 / 76.9 / 78.7 / 64.6 / 74.9 |
| UNINEXT-H，ViT-H；Objects365 → COCO + RefCOCO/+/g → 视频数据 | 专用模型，引用最多的 SOTA | arXiv 2303.06674（CVPR'23），表 4 | oIoU | 83.44 / 81.33 / 76.42 / 66.22 / 76.37 |
| DeRIS-L，Swin-B（Mask2Former 初始化）+ BEiT-3-L；RefCOCO/+/g 混合训练 | 专用模型，已发表的最强 | arXiv 2507.01738（ICCV'25）；oIoU 取官方仓库 README（论文只给 mIoU） | oIoU | 86.49 / 82.87 / 82.34 / 74.41 / 80.12 |

## 备查（不画）

- LISA-7B 混合训练、未微调（同表 2）：76.5 / 71.1 / 67.4 / 56.5 / 68.5。
- **我们跑的 `xinlai/LISA-7B-v1` 没有公开读数**，论文、README 与 HF 仓库里都没有。它的读数只能来自我们自己的运行（RUN 待发）。
- GLaMM (ZS)：58.1 / 52.2 / 47.1 / 39.5 / 55.6。
- PolyFormer-L（CVPR'23，明确去掉了 val/test 的图）：oIoU 78.29 / 73.25 / 74.56 / 61.87 / 70.19。
- EVF-SAM 多任务（arXiv 2406.20076 v4 起，期刊版待核）：cIoU 84.2 / 80.2 / 80.0 / 71.9 / 78.3。
- 只有 mIoU 的不进 cIoU 轴：InstanceVG-L、DeRIS 论文表、OneRef 表 3。

## 注

1. LISA：
   - 表 2 只有 cIoU。
   - "(ft)" 从 v2 起才解释为"在 refCOCO(+/g) 上微调"；论文模型以 LLaVA-7B-v1-1 为底。
   - README 说 v1 模型"用 train+val 训练"，所以论文读数不描述 v1 权重。
2. GLaMM：
   - v1 的 FT 一行（81.5 / 74.4 / …）在 v2/v3 被替换，引 v2/v3 或 CVPR 版。
   - 量没有命名。按以下三点当作 cIoU：官方评测脚本返回 class_iou（Σ交 / Σ并）；同表的 LISA 行就是 LISA 的 cIoU；DeRIS 表 1 标 GLaMM 为 cIoU。
   - 它表里 LISA-7B test(U) 抄成 68.4（LISA 自己是 68.5），引 LISA 自己的表。
3. UNINEXT 与 COCO 联合训练，没有看到它说去掉了 RefCOCO val/test 的图；PolyFormer 明确去掉了。
4. DeRIS 的 oIoU 只在官方 README 里，是官方来源，但不在论文里，表注写明。

来源：
- LISA：https://arxiv.org/abs/2308.00692 、https://github.com/dvlab-research/LISA
- GLaMM：https://arxiv.org/abs/2311.03356 、https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/evaluation.md
- UNINEXT：https://arxiv.org/abs/2303.06674
- DeRIS：https://arxiv.org/abs/2507.01738 、https://github.com/Dmmm1997/DeRIS
- PolyFormer：https://arxiv.org/abs/2302.07387
- EVF-SAM：https://arxiv.org/abs/2406.20076
