# 论文用图表（docs/figures/）

<!-- 由 scripts/figures/make_paper_figures.py 生成；不要手改。数据补齐后重跑：python scripts/figures/make_paper_figures.py -->

- 生成时间：2026-10-23 09:31 UTC
- 代码 commit：725e3d32b315db8e9ad17fd5f026969d027add80 (working tree has uncommitted changes)
- 结果树：`仓库根`
- matplotlib 3.11.2；PDF 是矢量（TrueType 字体，无 Type 3），PNG 300 dpi；宽度：单栏 3.3 in，双栏 6.9 in。
- 配色：dataviz 参考调色板。Qwen3.5 四个尺寸用一个有序的蓝色阶（浅→深 = 0.8B→9B），InternVL3.5 用第二个有序色阶（橙）；两类对比用分类槽 1（蓝）与槽 2（橙）。都在白底上过了 validate_palette.js（有序：单色相、亮度单调、相邻 ΔL ≥ 0.06；分类两色全对 CVD ΔE 24.7）。
- 每个数字都从结果文件读；逐点来源见 `data/*.json`（每个点带 source 路径）。

## 缺的数据（missing）

- 干扰 9B: P_seg（RUN-0056 还没有两臂配对的读数）
- 干扰 9B: P_depth（RUN-0056 还没有两臂配对的读数）

E3 表里缺的格子（按种子）：

- E3 2B e3b seed 1: refseg structured grid（rejected by merge check）
- E3 2B e3b seed 1: depth structured grid（rejected by merge check）
- E3 4B e3b seed 2: refseg structured grid（rejected by merge check）
- E3 4B e3b seed 2: depth structured grid（rejected by merge check）
- E3 9B e3b seed 0: refseg structured grid（no result file）
- E3 9B e3b seed 0: depth structured grid（no result file）

## 1. E1 差距图：`fig_e1_gap.pdf` / `fig_e1_gap.png`

- 数据：
  - 定位 Acc@0.5、多边形 cIoU、grid16 gIoU：`experiments/E1/0027_e1_main/<模型>/{refexp_box,refexp_poly,grid16}/metrics.json` 的 headline（严格读数，scope = all，bootstrap 95% CI）。
  - 自己的框 cIoU、结构强制多边形 cIoU 与配对差：`experiments/E1/0033_e1_posthoc2/ownbox.json`（refexp_poly 的 own_box、poly_structured、poly_structured_min8 三行，strict.cIoU 与 paired_diff.cIoU）。
  - 核对：ownbox.json 的结构强制读数与 `experiments/E1/0033_e1_posthoc2/<模型>/refexp_poly__poly_structured/metrics.json` 逐个相同。
- 口径（RESULTS.md G1 第 1 条、G1 §2、D81-2）：
  - "知道在哪"：定位 Acc@0.5（RefCOCO/+/g 全部评测划分，G1 §2.3 的规模斜率口径）；自己的框按 E1 多边形同一个光栅化成 mask 后的 cIoU（思考子集的那批句子，n 见图注）。
  - "画轮廓"：主评测的多边形 cIoU、16×16 网格 gIoU、结构强制多边形 cIoU（≥3 点，与自己的框同一批句子）；各自的 n 见图注（从 metrics.json 读）。
  - (b) 的配对差：结构强制多边形 − 自己的框，同一批 id 的配对 bootstrap 95% CI；两种结构强制（≥3 点、≥8 点）× 四个尺寸。
- 其他家族：InternVL3.5-2B/4B 作参照画进 (a)（都能作答，定位有效回答率 InternVL3.5-2B 100%, InternVL3.5-4B 100%；没有跑结构强制，标 n/a）。Gemma 4 E2B/E4B 不画：定位以拒答为主（有效回答率 Gemma-4-E2B 20%, Gemma-4-E4B 25%，G1 §2.5），两个"知道在哪"的读数量到的是拒答而不是定位，画出来可能出现"多边形高于定位"的假象；数字照样写进 `data/fig_e1_gap.json`。

**图注（英文）**

Knowing where vs. drawing the outline (E1, zero-shot, T-text channel). (a) Qwen3.5 0.8B–9B (blue) and InternVL3.5 2B/4B (orange, cross-family reference). Knows where: grounding Acc@0.5 on all RefCOCO/+/g evaluation splits (n=56,857) and the model's own predicted box rasterised as a mask (cIoU, n=200 expressions). Draws the outline: free-form polygon cIoU (n=8,000), 16×16 grid gIoU (n=500) and structure-forced polygon cIoU (≥3 vertices; n=200, the same expressions as the own box; run for Qwen3.5 only, n/a for InternVL3.5). Strict readings: invalid answers score 0. Error bars: 95% bootstrap CI (1,000 resamples). (b) Paired difference structure-forced polygon − own box on the same expressions (≥3 and ≥8 vertices), 95% paired bootstrap CI; 8 of 8 CIs exclude 0. Gemma-4-E2B, Gemma-4-E4B are omitted: their grounding answers are mostly refusals (valid answers: Gemma-4-E2B 20%, Gemma-4-E4B 25%), so both knows-where readings measure refusal rather than localisation.

核对：

- 一致：qwen3.5-0.8b 结构强制多边形 cIoU（ownbox.json = experiments/E1/0033_e1_posthoc2/qwen3.5-0.8b/refexp_poly__poly_structured/metrics.json）
- 一致：qwen3.5-2b 结构强制多边形 cIoU（ownbox.json = experiments/E1/0033_e1_posthoc2/qwen3.5-2b/refexp_poly__poly_structured/metrics.json）
- 一致：qwen3.5-4b 结构强制多边形 cIoU（ownbox.json = experiments/E1/0033_e1_posthoc2/qwen3.5-4b/refexp_poly__poly_structured/metrics.json）
- 一致：qwen3.5-9b 结构强制多边形 cIoU（ownbox.json = experiments/E1/0033_e1_posthoc2/qwen3.5-9b/refexp_poly__poly_structured/metrics.json）

## 2. E2 逐层曲线：`fig_e2_layers.pdf` / `fig_e2_layers.png`

- 数据：`experiments/E2/0029_e2_qwen/results_0029/0029_e2_qwen/<模型>/<任务>/metrics.json` 的 `probes["L<i>/linear"].eval`（标题指标与它的 `_ci`）。标题指标按 `scripts/mgr/e2_summary.py` 的 headline_of 取（metrics.json 的第一个 headline）。
- 核对：逐层读数与 `experiments/E2/0029_e2_qwen/e2_summary.json`（管理员汇总，G2 用的那份）的 linear 曲线逐层相同。
- 口径（G2 §2.1）：线性探针（1×1 卷积）；深度 AbsRel↓（NYU）、法向平均角误差°↓（NYU）、语义分割 mIoU↑（ADE20K val）、指代分割 cIoU↑（RefCOCO val，指令在前）；n 见图注（metrics.json 的 n_eval）。横轴 = 层号 ÷ 最后一层的层号（9B 隔层取）；空心圆 = 第 0 层，实心方块 = 最后一层；阴影 = 评测样本的 bootstrap 95% CI。

**图注（英文）**

Linear probes on frozen Qwen3.5 hidden states (E2). x: relative depth = layer index / number of layers. Tasks: depth (NYUv2, n=654; AbsRel ↓); surface normals (NYUv2, n=654; Mean angular error (°) ↓); semantic segmentation (ADE20K val, n=2,000; mIoU ↑); referring segmentation (RefCOCO val, instruction first, n=10,634; cIoU ↑). Shaded: 95% bootstrap CI over evaluation samples. Open circle: layer 0 (the image tokens entering the LLM); filled square: last layer. 9B probed at every other layer.

核对：

- 一致：qwen3.5-0.8b/depth 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-0.8b/normals 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-0.8b/semseg 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-0.8b/refseg 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-2b/depth 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-2b/normals 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-2b/semseg 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-2b/refseg 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-4b/depth 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-4b/normals 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-4b/semseg 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-4b/refseg 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-9b/depth 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-9b/normals 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-9b/semseg 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json
- 一致：qwen3.5-9b/refseg 逐层读数对 experiments/E2/0029_e2_qwen/e2_summary.json

## 3. E3 对照表：`tab_e3_comparison.tex` / `tab_e3_comparison.md`

- 数据（每个尺寸的目录，先到先得，与 interference.py 同一套相对路径）：
  - 0.8B（hpcgpu07；RUN-0050）：臂 `experiments/E3/0050_hpc_eval_08b_hpcgpu07`；基座 `experiments/E3/0050_hpc_eval_08b_hpcgpu07/qwen3.5-0.8b`
  - 2B（B200；RUN-0038, RUN-0039, RUN-0046; base: RUN-0027, RUN-0030, RUN-0033）：臂 `experiments/E3/0038_e3_eval_2b_s0, experiments/E3/0039_e3b_diag_2b_s0, experiments/E3/0046_e3_eval_2b_h1`；基座 `experiments/E1/0027_e1_main/qwen3.5-2b, experiments/E1/0030_e1_posthoc/qwen3.5-2b, experiments/E1/0033_e1_posthoc2/qwen3.5-2b`
  - 4B（hpcgpu06；RUN-0052）：臂 `experiments/E3/0052_hpc_eval_4b_hpcgpu06`；基座 `experiments/E3/0052_hpc_eval_4b_hpcgpu06/qwen3.5-4b`
  - 9B（hpcgpu06；RUN-0056）：臂 `experiments/E3/0056_hpc_9b_eval_hpcgpu06`；基座 `experiments/E3/0056_hpc_9b_eval_hpcgpu06/qwen3.5-9b`
  - 臂的 T 文本：`<目录>/t_text/<变体>/seed<k>/<模型>/<任务>/metrics.json`；网格：`<目录>/eval/<变体>/<模型>/seed<k>/<refseg|depth>__structured/metrics.json`；基座：`<基座目录>/<任务>/metrics.json`；合并核对：同一层目录的 `merge_check.json`。
- 口径：列是 §13 的 8 个 T 文本读数（`scripts/mgr/interference.py` 的 L_ITEMS，标题读数 = metrics.json 第一个 headline 的严格值，×100），加结构强制网格的指代分割 cIoU（×100）与深度 AbsRel（单轮、guided，D83-3）。种子平均 ± 样本标准差（statistics.stdev，同 interference.summary）。
- 行：基座、C3、E3b+C3、E3b（基座放在每组第一行作参照）。任务书写的是"9B 有了就加一列组"；列已经是 10 个读数，所以尺寸做成行组，9B 有了就多一个行组。

**表注（英文）**

- Seed mean ± sample standard deviation over seeds 0–2 (0.8B, 2B, 4B); seed 0 only (9B) (E3 preregistration §3); n=k marks cells with fewer seeds than planned. Base rows are single deterministic evaluations.
- T-text columns are the 8 readouts of the interference analysis (§13), ×100: strict grounding Acc@0.5, polygon cIoU, counting accuracy and COCO detection F1@0.5; NYU depth pairs with letter swap; position-debiased BLINK, CV-Bench and EmbSpatial. RefSeg cIoU (×100) and depth AbsRel use structure-forced decoding of the pixel-token grid, single turn, guided upsampling.
- Machines (all effects computed within one machine, D93-1; each base row was evaluated on the same machine as its arms): 0.8B: hpcgpu07 (NVIDIA A100-PCIE-40GB); 2B: B200 (NVIDIA B200); 4B: hpcgpu06 (NVIDIA A100-PCIE-40GB); 9B: hpcgpu06 (NVIDIA A100-PCIE-40GB).
- Seeds without readings because the pre-evaluation merge check (criteria v1) rejected them (N37, kept as v1 per D102-2): 2B E3b seed 1 (grid, B200): restricted decoding vs. generate_batch exact-row agreement 0.945 < 0.95; 4B E3b seed 2 (grid, hpcgpu06): restricted decoding vs. generate_batch exact-row agreement 0.930 < 0.95.
- E3b T-text readouts exist only for 2B seed 0 (grid-only LoRA; T-text is not part of the E3b protocol elsewhere); other E3b T-text cells are --.
- n/a: Base and C3 have no pixel-token channel.

被合并核对拒绝的种子（N37；判据 v1 的 failures 原文）：

- 2B e3b seed 1（grid，B200）：{"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9450 < 0.95"], "merge": []}；`experiments/E3/0046_e3_eval_2b_h1/eval/e3b/qwen3.5-2b/seed1/merge_check.json`
- 4B e3b seed 2（grid，hpcgpu06）：{"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9300 < 0.95"], "merge": []}；`experiments/E3/0052_hpc_eval_4b_hpcgpu06/eval/e3b/qwen3.5-4b/seed2/merge_check.json`

## 4. 干扰随尺寸的曲线：`fig_interference_size.pdf` / `fig_interference_size.png`

- 数据：`docs/analysis/e3_interference_by_size.json`（`scripts/mgr/interference.py` 的 JSON 输出）。JSON 里没有的尺寸（9B）在评测目录出现后，用同一个脚本的 size_block 按上表的目录现算；秩 128 的 0.8B 同样用 size_block，C3 与 E3b+C3 只取 RUN-0054 的目录，E3b 取 RUN-0050 的 r32（§13a、D99-1）。
- 口径（E3 预注册 §13、D96-4）：L = C3 −（E3b+C3）的 8 项平均（点）；P_seg = E3b − (E3b+C3) 的 refseg cIoU；P_depth = (E3b+C3) − E3b 的 depth AbsRel；R_seg = P_seg ÷ E3b 的 cIoU（事后、描述性）。逐种子配对；空心 = 各种子，实心 = 种子平均。横轴 = 名义参数量（尺寸标签，对数轴）。
- §13 的趋势判定（JSON 原文，只引用不重判）：{"L": "随尺寸缩小", "P_seg": "不缩小", "P_depth": "不缩小"}；相对干扰（描述）：{"R_seg": "明显缩小", "R_depth": "不缩小"}。
- 各点的来源：
  - 0.8B：docs/analysis/e3_interference_by_size.json
  - 0.8B r128：computed with scripts/mgr/interference.py size_block: C3/E3b+C3 from experiments/E3/0054_r128_eval_hpcgpu07; E3b from experiments/E3/0050_hpc_eval_08b_hpcgpu07
  - 2B：docs/analysis/e3_interference_by_size.json
  - 4B：docs/analysis/e3_interference_by_size.json
  - 9B：computed with scripts/mgr/interference.py size_block over experiments/E3/0056_hpc_9b_eval_hpcgpu06

**图注（英文）**

Interference vs. model size (E3, exploratory analysis §13). L = C3 − (E3b+C3), mean difference over the 8 T-text readouts of Table E3 (points). P_seg = E3b − (E3b+C3) referring-segmentation cIoU and P_depth = (E3b+C3) − E3b depth AbsRel, both with structure-forced grid decoding (positive = mixing in text hurts the pixel channel). R_seg = P_seg / cIoU(E3b), post hoc and descriptive (D96-4). Hollow markers: individual seeds (paired within seed); filled: seed mean. x: nominal parameter count, log scale. Effects are computed within one machine (D93-1): 0.8B: hpcgpu07; 2B: B200; 4B: hpcgpu06; 9B: hpcgpu06. Seeds with paired readings, where fewer than planned (see Table E3 for seeds rejected by the merge check): 2B P_seg: 2 of 3 seeds; 2B P_depth: 2 of 3 seeds; 4B P_seg: 2 of 3 seeds; 4B P_depth: 2 of 3 seeds; 9B P_seg: 0 of 1 seeds; 9B P_depth: 0 of 1 seeds. Star: 0.8B with LoRA rank 128 for both C3 and E3b+C3 (post hoc, §13a/D99-1; P uses the rank-32 E3b), drawn slightly right of 0.8B. Missing: Qwen3.5-9B (RUN-0056): P_seg not yet available; Qwen3.5-9B (RUN-0056): P_depth not yet available.

核对：

- 一致：0.8B 用 experiments/E3/0050_hpc_eval_08b_hpcgpu07 重算的 L/P/R 对 docs/analysis/e3_interference_by_size.json
- 一致：2B 用 experiments/E3/0038_e3_eval_2b_s0, experiments/E3/0039_e3b_diag_2b_s0, experiments/E3/0046_e3_eval_2b_h1 重算的 L/P/R 对 docs/analysis/e3_interference_by_size.json
- 一致：4B 用 experiments/E3/0052_hpc_eval_4b_hpcgpu06 重算的 L/P/R 对 docs/analysis/e3_interference_by_size.json

## 文件

- `fig_e1_gap.pdf`
- `fig_e1_gap.png`
- `fig_e1_gap_caption.tex`
- `data/fig_e1_gap.json`
- `fig_e2_layers.pdf`
- `fig_e2_layers.png`
- `fig_e2_layers_caption.tex`
- `data/fig_e2_layers.json`
- `tab_e3_comparison.tex`
- `tab_e3_comparison.md`
- `data/tab_e3_comparison.json`
- `fig_interference_size.pdf`
- `fig_interference_size.png`
- `fig_interference_size_caption.tex`
- `data/fig_interference_size.json`
