# W2 外部对照 LISA-7B · lisa-7b-v1 · lisa_7b · test_refcocog_umd_test

- 硬约束第 1 条：LISA 把 SAM 接在 LLM 旁边（外接）。它只作对照单独运行，输出只用来计分与出表，绝不进入我们的训练数据、标签、奖励或任何挑选（D154；简报 §8.8；预注册 U-2）。
- 官方代码：https://github.com/dvlab-research/LISA@3cb2d4301f1a（third_party/lisa，未改动；PROVENANCE sha256 2ab8d4ec28e25250…）；论文 Lai et al., LISA: Reasoning Segmentation via Large Language Model（CVPR 2024；arXiv 2308.00692）
- 权重：xinlai/LISA-7B-v1@43c754eef758；视觉塔 openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41
- 协议：官方 refer_seg 评测（论文表格的口径）：train_ds.py --eval_only 的 validate()，数据走 utils/dataset.py 的 ValDataset（refer_seg 分支）与 collate_fn；回答由评测直接给出 "[SEG]."（teacher forcing），LLM 一次前向，取 [SEG] 的隐藏状态经 SAM 解码；精度 bf16
- 句子：300 句（configs/w2_llmseg_subsets.json，sha256 8c8a1d134ab324e0…），与 LLMSeg 相同；批准 D154

## 读数

- cIoU：0.6599，95% CI [0.6108, 0.7051]，n = 300
- gIoU：0.6364，95% CI [0.5959, 0.6760]，n = 300

- 失败 {'ok': 300}（按空 mask 计）；mask 全空 0；读图尺寸与标注不符 0；带 EXIF 方向 0
- 每句前向 {'llm': 1.0, 'sam_image_encoder': 1.0, 'sam_mask_decoder': 1.0}；complete = True（300 / 300）

## 文件

- predictions.jsonl（逐句：mask 的 RLE、score、failure、LISA 的 [SEG] 与前向次数）；metrics.json；compute.json；config.yaml；meta.json；logs/timing.jsonl。原始输出在 ../../../raw/。
