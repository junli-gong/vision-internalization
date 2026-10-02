# RUN-0019 — LISA-7B-v1 在 E5 开发集上的首次运行（外部对照，隔离；val 划分，每档 4 条，a–d）

服务器 CC，b200，2026-10-02 08:50–09:00 UTC，值守中接单。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- **彩排（`--limit 5`）：** 入口检查通过，`fixes_applied` = `addedtoken_str_patch`、`vision_tower_local_snapshot`，与 RUN-0015 的两处处理相同。4 个任务各 5 条，20/20 成功。
- **全量：** 60/60 成功（rotation3d 16、viewpoint 16、occlusion 12、physics 16）。
  - 没有 `no_seg`、`empty_mask` 或异常；每条恰好 1 个 `[SEG]`。
  - 平均生成 8.9–10.0 个 token，每条墙钟均值 0.23–0.27 s；全量推理 26.7 s（含加载）。
- **加载：** 用时 5.57 s。torch `max_memory_allocated`：加载后 15.44 GB，推理 17.20 GB。
- **评分（mask_iou_grid 严格；60 条都有效，所以条件读数等于严格读数）：**
  - 全部档：rotation3d 0.709、viewpoint 0.352、occlusion 0.541、physics 0.244。
  - 标题范围：rotation3d 0.677、viewpoint 0.311。occlusion 与 physics 的标题范围是全部样本，与上一行相同。
  - a 类的起伏误差：`judgment.status = not_applicable`，原文是"这个系统不给起伏（只给 mask）"。
- **文本：** 60 条的 text 只有三种："Sure, the segmentation result is [SEG]  .</s>" 48 条，"Sure, it is [SEG]  .</s>" 11 条，"Sure, [SEG]  .</s>" 1 条，举例见 §3。
- **隔离：** LISA 的输出只在本运行目录的两个外部预测目录里，没有写 `$PROJ/data` 或任何训练、数据目录。导出目录用完已删除，删除前 372,418 B、64 个文件。
- **用时与磁盘：** 任务合计约 10 min。根盘 38.186 → 38.185 GB，`$PROJ` 4.554 → 4.555 GB，内存盘峰值 21.042 GB。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `60fba2b`。`24c4c22`（DEV-0019 合入）与本任务书所在的 `85f7e09` 都在 HEAD 的历史里。
- **LISA：** `$PROJ/ext/lisa/LISA` 的 HEAD 是 `3cb2d4301f1af4691bd4f3938335ef06e76f155a`，`git status --short` 为空。venv 沿用 RUN-0015 的，没有重装。
- **权重：** LISA-7B-v1 @ `43c754e` 10/10、16.122 GB；CLIP @ `32bd642` 8/8、1.714 GB。逐文件的哈希与 RUN-0015 的 `weights_download.json` 完全相同。
  - 内存盘上原有的其他权重只有 Qwen3.5-0.8B，按任务书保留。
- **环境：**
  - LISA 那几步 export 了 `HF_HOME=$ROOT/.hf`、`TMPDIR=$PROJ/tmp`、`PIP_NO_CACHE_DIR=1`。
  - 另加了 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`：权重已在内存盘，路径都是本地快照。
- **运行用的代码：** 全部在固定于 `60fba2b` 的 worktree 里运行，跑完已删除。
- **改动范围：**
  - 没有修改我们仓库的 `src/`、`scripts/`、`tests/`、`configs/`，也没有改 LISA 的源码，没有安装任何包。
  - 新写的文件只在 `probe_scripts/` 下：`run_lisa_19.sh`、`p23_overlays.py`、`p24_lisa_summary.py`。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。
- **划分：** 只用了 val 划分，没有运行 test 划分。

## 2. 执行记录

| 步骤 | 墙钟 | 结果 |
|---|---|---|
| 权重下载与核对 | 55.4 s（LISA 35.8 s、CLIP 9.8 s，其余是核对） | 18/18 一致 |
| 导出 | 1.8 s | 4 个任务，60 条 |
| 彩排 `--limit 5` | 13.9 s | 20/20 ok；加载 5.555 s；推理峰值 17,202,832,896 B |
| 全量 | 26.7 s | 60/60 ok |
| 评分 external | 4.0 s | rc=0 |
| 评分 baseline | 8.4 s | rc=0 |
| 删除权重 | <1 s | 释放 17,836,162,608 B；hf-hub 只剩 Qwen3.5-0.8B |

- **GPU：** NVIDIA B200，bf16。每 5 s 一次的 nvidia-smi 采样在两次 LISA 运行里各抓到 3 个与 6 个样本，按进程峰值分别是 15,832 MiB 和 15,960 MiB。运行很短，采样可能漏掉峰值；torch 的读数见 §3。

## 3. 结果表（原样）

由 `probe_scripts/p24_lisa_summary.py` 生成，与 `lisa_summary.md` 相同。

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

#### a 类起伏误差的状态（`external/lisa-7b-v1/rotation3d__dev/metrics.json` 的 gate，原样摘录）

- `judgment`：`{"status": "not_applicable", "note": "这个系统不给起伏（只给 mask）", "pass": null}`
- `prerequisite`（标题范围，n=12）：系统 0.6766，最强的平凡基线 copy 0.7111，diff −0.0345，95% CI [−0.0738, +0.0016]，`pass=False`。只是照录，n 很小，不作结论。

#### 目检记录（`overlays/`，彩排输出里每个任务前 2 条成功的样本；只记事实）

| 图 | 指令里点名的物体 | 叠加图上 mask 覆盖的区域 |
|---|---|---|
| `rotation3d_val_0.png` | 黄色椭球（绕水平轴转 24°） | 输入图里那个黄色椭球的整个可见区域 |
| `rotation3d_val_1.png` | 灰色椭球（绕竖直轴转 51°） | 输入图里那个灰色椭球的可见区域；顶边与底边有几处小的未覆盖点 |
| `viewpoint_val_0.png` | 粉色三棱柱（相机上移 0.10 m） | 三棱柱粉色的顶面，以及靠近圆环一侧侧面的上部；深色侧面的大部分没有覆盖 |
| `viewpoint_val_1.png` | 黄色 L 形块（相机下移 0.33 m） | 输入图里那个黄色 L 形块；旁边紫色圆锥的边上有少量红色点 |
| `occlusion_val_0.png` | 棕色月牙（要求含被遮挡部分） | 棕色月牙的可见部分，以及与它重叠的橙黑格子长方形的一部分 |
| `occlusion_val_1.png` | 棕黑格子三角形（要求完整 amodal mask） | 紫黑格子的十字形；棕黑格子三角形基本没有覆盖 |
| `physics_val_0.png` | 灰黑格子圆（被推向右侧） | 中间那个棕黑格子正方形；灰黑格子圆上有零星的小块 |
| `physics_val_1.png` | 蓝色长方形与灰黑格子圆（相向运动） | 灰黑格子圆（左数第三个物体），以及它右边灰色条纹圆上的零星小块 |

## 4. 与任务书假设不符的发现

- **两个离线变量：** 任务书只要求设 `HF_HOME`、`TMPDIR`、`PIP_NO_CACHE_DIR`。我另设了 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。权重路径都是本地快照，设这两个变量是为了确保 LISA 那一侧不访问 Hub。
- **用时：** 任务书预计 15–30 min，每条超过 10 s 时有停止规则。实际每条 0.23–0.27 s（均值），全量 26.7 s，没有触到停止规则。
- **采样：** nvidia-smi 每 5 s 一次，LISA 两次运行只有 13.9 s 和 26.7 s，各只抓到 3 个和 6 个样本。显存峰值以 torch 的读数为准。

## 5. 生成的文件

`experiments/E5/0019_lisa_e5_dev/`（1.8 MB，141 个文件）：

- 外部预测：`lisa_predictions/`、`lisa_predictions_limit5/`；
- 评分：`external/`、`baseline/`；
- `overlays/`（8 张图与 `index.json`）；
- `lisa_summary.md/json`、`weights_download.json`、`README.md`、`disk_ledger.tsv`；
- `logs/`、`probe_scripts/`。

一次推回。

## 6. 磁盘与内存盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 08:52:47 | 0019 start | 38.186 | 3.204 | 4.554 |
| 08:53:43 | after download LISA+CLIP | 38.186 | 21.041 | 4.554 |
| 08:55:34 | before full LISA run | 38.185 | 21.042 | 4.555 |
| 08:56:01 | after full LISA run | 38.185 | 21.042 | 4.555 |
| 08:56:34 | after delete LISA+CLIP | 38.183 | 3.205 | 4.556 |
| 08:56:34 | 0019 end | 38.185 | 3.076 | 4.555 |

- **根盘：** 可用空间从 38.186 GB 到 38.185 GB，全程远高于 3 GB 红线。
- **`$PROJ`：** 从 4.554 GB 到 4.555 GB，在 15 GB 份额以内。导出目录（372,418 B）用完已删除。
- **内存盘：** 峰值 21.042 GB，其中权重 19.6 GB（LISA 16.122 + CLIP 1.714 + Qwen3.5-0.8B 1.770），低于 25 GB 上限。结束时 `/dev/shm/visint` 是 3.076 GB，hf-hub 只剩 Qwen3.5-0.8B；worktree 已删除。
