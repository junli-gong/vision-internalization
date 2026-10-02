#### 加载（lisa_predictions/logs/run_*.json 的 runner_info；入口检查不通过时脚本会报错退出，本次通过）

| 项 | 值 |
|---|---|
| fixes_applied | addedtoken_str_patch, vision_tower_local_snapshot |
| tokenizers 是否可用 | False |
| 入口检查：[SEG] | 单 token，id 32003 |
| 入口检查：分词器长度 / embed_tokens 行数 | 32004 / 32004 |
| 入口检查：bos / eos / unk / pad 的 id | 1 / 2 / 0 / 0 |
| 入口检查：生成配置 | {"do_sample": false, "num_beams": 1} |
| 加载用时 | 5.568 s |
| 显存峰值（torch max_memory_allocated）：加载后 / 推理 | 15444875776 B / 17203474944 B |
| 参数量 | 7,704,747,312 |
| 版本 | {"python": "3.12.13", "torch": "2.13.0+cu130", "transformers": "4.31.0", "numpy": "1.26.4", "cv2": "4.8.0", "cuda": "13.0", "gpu": "NVIDIA B200", "capability": [10, 0]} |

#### 逐任务（lisa_predictions/<任务>.jsonl）

| 任务 | 条数 | 成功 | 失败（按原因） | [SEG] 个数分布 | 平均生成 token | 每条墙钟 s：均值 / 中位 / 最大 | 掩码像素：中位 / 最小 / 最大 |
|---|---|---|---|---|---|---|---|
| rotation3d | 16 | 16 | 无 | {"1": 16} | 10.0 | 0.270 / 0.253 / 0.521 | 27286.0 / 5287 / 51544 |
| viewpoint | 16 | 16 | 无 | {"1": 16} | 9.3 | 0.240 / 0.253 / 0.261 | 25412.5 / 7493 / 58672 |
| occlusion | 12 | 12 | 无 | {"1": 12} | 9.2 | 0.243 / 0.251 / 0.318 | 24247.0 / 5639 / 201526 |
| physics | 16 | 16 | 无 | {"1": 16} | 8.9 | 0.233 / 0.253 / 0.272 | 7339.0 / 567 / 14625 |

#### 评分（external/lisa-7b-v1；严格 = 失败计 0，条件 = 只算有效条；n 很小，只是核对）

| 任务 | 范围 | n | 有效 | mask_iou_grid 严格 | 条件 | mask_iou_full 严格 | 起伏误差 |
|---|---|---|---|---|---|---|---|
| rotation3d | 全部档 | 16 | 16 | 0.709 | 0.709 | 0.702 | — |
| rotation3d | 档 0 | 4 | 4 | 0.806 | 0.806 | 0.803 | — |
| rotation3d | 档 1 | 4 | 4 | 0.653 | 0.653 | 0.643 | — |
| rotation3d | 档 2 | 4 | 4 | 0.677 | 0.677 | 0.653 | — |
| rotation3d | 档 3 | 4 | 4 | 0.700 | 0.700 | 0.708 | — |
| rotation3d | 标题范围：档 [1, 2, 3]（下限 ≥ 30°） | 12 | 12 | 0.677 | 0.677 | 0.668 | — |
| viewpoint | 全部档 | 16 | 16 | 0.352 | 0.352 | 0.352 | — |
| viewpoint | 档 0 | 4 | 4 | 0.416 | 0.416 | 0.419 | — |
| viewpoint | 档 1 | 4 | 4 | 0.547 | 0.547 | 0.546 | — |
| viewpoint | 档 2 | 4 | 4 | 0.169 | 0.169 | 0.168 | — |
| viewpoint | 档 3 | 4 | 4 | 0.276 | 0.276 | 0.274 | — |
| viewpoint | 标题范围：环绕移动（全部档） | 9 | 9 | 0.311 | 0.311 | 0.312 | — |
| occlusion | 全部档 | 12 | 12 | 0.541 | 0.541 | 0.514 | — |
| occlusion | 档 0 | 4 | 4 | 0.768 | 0.768 | 0.732 | — |
| occlusion | 档 1 | 4 | 4 | 0.340 | 0.340 | 0.338 | — |
| occlusion | 档 2 | 4 | 4 | 0.515 | 0.515 | 0.473 | — |
| occlusion | 标题范围：全部样本 | 12 | 12 | 0.541 | 0.541 | 0.514 | — |
| physics | 全部档 | 16 | 16 | 0.244 | 0.244 | 0.239 | — |
| physics | 档 0 | 4 | 4 | 0.427 | 0.427 | 0.412 | — |
| physics | 档 1 | 4 | 4 | 0.285 | 0.285 | 0.267 | — |
| physics | 档 2 | 4 | 4 | 0.179 | 0.179 | 0.181 | — |
| physics | 档 3 | 4 | 4 | 0.085 | 0.085 | 0.096 | — |
| physics | 标题范围：全部样本 | 16 | 16 | 0.244 | 0.244 | 0.239 | — |

#### 同一批样本上的平凡基线（baseline/；全部档 / 标题范围的 mask_iou_grid 严格）

| 任务 | 基线 | 全部档 | 标题范围 | 起伏误差 严格：全部档 / 标题范围 |
|---|---|---|---|---|
| rotation3d | copy | 0.758 | 0.711 | 1.119 / 1.338 |
| rotation3d | copy_flat | 0.758 | 0.711 | 0.973 / 0.986 |
| rotation3d | gt_affine_upper | 0.949 | 0.941 | 1.095 / 1.201 |
| rotation3d | instr_affine | 0.553 | 0.447 | 1.214 / 1.457 |
| rotation3d | **lisa-7b-v1** | 0.709 | 0.677 | — |
| viewpoint | copy | 0.466 | 0.297 | — |
| viewpoint | translate | 0.523 | 0.298 | — |
| viewpoint | **lisa-7b-v1** | 0.352 | 0.311 | — |
| occlusion | convex_hull | 0.714 | 0.714 | — |
| occlusion | visible | 0.586 | 0.586 | — |
| occlusion | **lisa-7b-v1** | 0.541 | 0.541 | — |
| physics | const_velocity | 0.290 | 0.290 | — |
| physics | copy | 0.381 | 0.381 | — |
| physics | **lisa-7b-v1** | 0.244 | 0.244 | — |

#### 文本举例（每个任务前 3 条的 text，原样）

- rotation3d #0: `Sure, the segmentation result is [SEG]  .</s>`
- rotation3d #1: `Sure, the segmentation result is [SEG]  .</s>`
- rotation3d #2: `Sure, the segmentation result is [SEG]  .</s>`
- viewpoint #0: `Sure, the segmentation result is [SEG]  .</s>`
- viewpoint #1: `Sure, the segmentation result is [SEG]  .</s>`
- viewpoint #2: `Sure, the segmentation result is [SEG]  .</s>`
- occlusion #0: `Sure, it is [SEG]  .</s>`
- occlusion #1: `Sure, the segmentation result is [SEG]  .</s>`
- occlusion #2: `Sure, the segmentation result is [SEG]  .</s>`
- physics #0: `Sure, the segmentation result is [SEG]  .</s>`
- physics #1: `Sure, the segmentation result is [SEG]  .</s>`
- physics #2: `Sure, it is [SEG]  .</s>`
