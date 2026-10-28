# 论文用图表（docs/figures/）

<!-- 由 scripts/figures/make_paper_figures.py 生成；不要手改。数据补齐后重跑：python scripts/figures/make_paper_figures.py -->

- 生成时间：2026-10-28 09:52 UTC
- 代码 commit：5ea73714c0142fb341ea8810329ba80c13de2a93 (working tree has uncommitted changes)
- 结果树：`仓库根`
- matplotlib 3.11.2；PDF 是矢量（TrueType 字体，无 Type 3），PNG 300 dpi；宽度：单栏 3.3 in，双栏 6.9 in。
- 配色：dataviz 参考调色板。Qwen3.5 四个尺寸用一个有序的蓝色阶（浅→深 = 0.8B→9B），InternVL3.5 用第二个有序色阶（橙）；两类对比用分类槽 1（蓝）与槽 2（橙）。都在白底上过了 validate_palette.js（有序：单色相、亮度单调、相邻 ΔL ≥ 0.06；分类两色全对 CVD ΔE 24.7）。
- W2 的图（DEV-0071）：同一个系统在各张图里同一个颜色——W2-c 槽 1（蓝）、E3b+C3 槽 2（橙）、W2-c + 回放 槽 3（青）；基座与 C3 （对照）用中性灰（浅、深）；W2-a 只在 fig_w2_size 出现，用槽 2（与 E3b+C3 不同图）。槽 1–3 三色全对过 validate_palette.js（CVD ΔE ≥ 9.2）；--with-explore 的 X-2、X-3、X-3L 用槽 4–6，按画出的相邻顺序过了检查。青色对白底不足 3:1，图上带数值标签或有表可查（lang_cost_9b.md、count_errors_9b.md）。
- 每个数字都从结果文件读；逐点来源见 `data/*.json`（每个点带 source 路径）。

## 缺的数据（missing）

- 无

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
  - 2B（B200；RUN-0038, RUN-0039, RUN-0046; base: RUN-0027, RUN-0030, RUN-0033）：臂 `experiments/E3/0071_e3_reeval_e5v2_hpcgpu07, experiments/E3/0038_e3_eval_2b_s0, experiments/E3/0039_e3b_diag_2b_s0, experiments/E3/0046_e3_eval_2b_h1`；基座 `experiments/E1/0027_e1_main/qwen3.5-2b, experiments/E1/0030_e1_posthoc/qwen3.5-2b, experiments/E1/0033_e1_posthoc2/qwen3.5-2b`；补评目录（--e3-prepend，排在最前、先到先得）`experiments/E3/0071_e3_reeval_e5v2_hpcgpu07`
  - 4B（hpcgpu06；RUN-0052）：臂 `experiments/E3/0071_e3_reeval_e5v2_hpcgpu07, experiments/E3/0052_hpc_eval_4b_hpcgpu06`；基座 `experiments/E3/0052_hpc_eval_4b_hpcgpu06/qwen3.5-4b`；补评目录（--e3-prepend，排在最前、先到先得）`experiments/E3/0071_e3_reeval_e5v2_hpcgpu07`
  - 9B（hpcgpu06；RUN-0056）：臂 `experiments/E3/0056_hpc_9b_eval_hpcgpu06`；基座 `experiments/E3/0056_hpc_9b_eval_hpcgpu06/qwen3.5-9b`
  - 臂的 T 文本：`<目录>/t_text/<变体>/seed<k>/<模型>/<任务>/metrics.json`；网格：`<目录>/eval/<变体>/<模型>/seed<k>/<refseg|depth>__structured/metrics.json`；基座：`<基座目录>/<任务>/metrics.json`；合并核对：同一层目录的 `merge_check.json`。
- 口径：列是 §13 的 8 个 T 文本读数（`scripts/mgr/interference.py` 的 L_ITEMS，标题读数 = metrics.json 第一个 headline 的严格值，×100），加结构强制网格的指代分割 cIoU（×100）与深度 AbsRel（单轮、guided，D83-3）。种子平均 ± 样本标准差（statistics.stdev，同 interference.summary）。
- 行：基座、C3、E3b+C3、E3b（基座放在每组第一行作参照）。任务书写的是"9B 有了就加一列组"；列已经是 10 个读数，所以尺寸做成行组，9B 有了就多一个行组。

**表注（英文）**

- Seed mean ± sample standard deviation over seeds 0–2 (0.8B, 2B, 4B); seed 0 only (9B) (E3 preregistration §3); n=k marks cells with fewer seeds than planned. Base rows are single deterministic evaluations.
- T-text columns are the 8 readouts of the interference analysis (§13), ×100: strict grounding Acc@0.5, polygon cIoU, counting accuracy and COCO detection F1@0.5; NYU depth pairs with letter swap; position-debiased BLINK, CV-Bench and EmbSpatial. RefSeg cIoU (×100) and depth AbsRel use structure-forced decoding of the pixel-token grid, single turn, guided upsampling.
- Machines (all effects computed within one machine, D93-1; each base row was evaluated on the same machine as its arms): 0.8B: hpcgpu07 (NVIDIA A100-PCIE-40GB); 2B: B200 (NVIDIA A100-PCIE-40GB, NVIDIA B200); 4B: hpcgpu06 (NVIDIA A100-PCIE-40GB); 9B: hpcgpu06 (NVIDIA A100-PCIE-40GB).
- Pre-evaluation merge check: criteria v1 (D50-3) for all readings except these structure-forced grid readings, re-evaluated under criteria e5-v2 (the E5 criteria, which tolerate tie flips between the two decoding implementations; unified per N34, decided before the rejected seeds' readings were seen; D102-2): 2B E3b+C3 seed 1 (refseg/depth; criteria e5-v2; NVIDIA A100-PCIE-40GB); 2B E3b seed 1 (refseg/depth; criteria e5-v2; NVIDIA A100-PCIE-40GB); 4B E3b+C3 seed 2 (refseg/depth; criteria e5-v2; NVIDIA A100-PCIE-40GB); 4B E3b seed 2 (refseg/depth; criteria e5-v2; NVIDIA A100-PCIE-40GB).
- E3b T-text readouts exist only for 2B seed 0 (grid-only LoRA; T-text is not part of the E3b protocol elsewhere); other E3b T-text cells are --.
- n/a: Base and C3 have no pixel-token channel.

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
- (d) R_seg 的离群读数（DEV-0071，任务书第 5 条，二选一选了"箭头 + 数值"，没有用断轴）：秩 128 的读数（种子或平均）落在 r32 全部 R_seg 读数（各种子与平均）的范围 [lo, hi] 向外再扩 1 倍跨度的窗口之外时，不画进坐标轴，改在轴边画一个指向它的箭头并写数值；纵轴只容纳窗口内的点（含秩 128 的其余点与平均值）。其余三个面板不变。本次：0.8B r128 种子 2 = -3.278（窗口 [-0.838, 1.772]，纵轴 [-1.422, 1.117]）。

**图注（英文）**

Interference vs. model size (E3, exploratory analysis §13). L = C3 − (E3b+C3), mean difference over the 8 T-text readouts of Table E3 (points). P_seg = E3b − (E3b+C3) referring-segmentation cIoU and P_depth = (E3b+C3) − E3b depth AbsRel, both with structure-forced grid decoding (positive = mixing in text hurts the pixel channel). R_seg = P_seg / cIoU(E3b), post hoc and descriptive (D96-4). Hollow markers: individual seeds (paired within seed); filled: seed mean. x: nominal parameter count, log scale. Effects are computed within one machine (D93-1): 0.8B: hpcgpu07; 2B: B200; 4B: hpcgpu06; 9B: hpcgpu06. Pixel pairs re-evaluated under merge-check criteria e5-v2, both arms (see Table E3): 2B seed 1; 4B seed 2. Star: 0.8B with LoRA rank 128 for both C3 and E3b+C3 (post hoc, §13a/D99-1; P uses the rank-32 E3b), drawn slightly right of 0.8B. In (d), a rank-128 reading lies far outside the range of the rank-32 readings and is marked by an arrow at the axis edge with its value (0.8B rank 128 seed 2: R_seg = −3.28); the axis covers all other points.

核对：

- 一致：0.8B 用 experiments/E3/0050_hpc_eval_08b_hpcgpu07 重算的 L/P/R 对 docs/analysis/e3_interference_by_size.json
- 一致：2B 用 experiments/E3/0071_e3_reeval_e5v2_hpcgpu07, experiments/E3/0038_e3_eval_2b_s0, experiments/E3/0039_e3b_diag_2b_s0, experiments/E3/0046_e3_eval_2b_h1 重算的 L/P/R 对 docs/analysis/e3_interference_by_size.json
- 一致：4B 用 experiments/E3/0071_e3_reeval_e5v2_hpcgpu07, experiments/E3/0052_hpc_eval_4b_hpcgpu06 重算的 L/P/R 对 docs/analysis/e3_interference_by_size.json

## 5. W2 test 主表：`tab_w2_test.tex` / `tab_w2_test.md`（gIoU 附录表 `tab_w2_test_giou.*`）

- 数据（都是已有的读数，不重算 bootstrap）：
  - 9B 各设置（自己的框、W1-F、随机挑选、最优候选、W2-a、W2-c、E1 零样本多边形）的读数与 95% CI：`docs/analysis/w2_test_table.json` 的 models.qwen3.5-9b.per_split（RUN-0072；CI 是各运行 metrics.json 里运行时算的，1,000 次重抽）。核对：与 `experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/<设置>/<划分>/metrics.json` 逐格相同。
  - W2-c + 回放：`experiments/W2/0076_w2c_replay_test_hpcgpu06/qwen3.5-9b/refseg/w2c/<划分>/metrics.json` 的 headline（RUN-0076 补充 2）。
  - 配对差 W2-c − W1-F、W2-a − W1-F：同一份 JSON 的 pairs 与 pairs_pooled（配对 bootstrap，ratio-of-sums，10,000 次，种子 20260929；W2-a 由 RUN-0081 在 HPC 上补齐）。
  - 回放 − W2-c：**没有 JSON**，只在 `docs/analysis/w2c_replay.md` §1 的表里（3 位小数；重抽次数文档里没写），从那张表解析；核对：表里的 W2-c、回放读数与上两项的值四舍五入相同。
  - E1 零样本多边形的解析失败：`experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/baseline_e1/<划分>/metrics.json` 的 parse。
  - 外部基线 LLMSeg（RUN-0078）：修正前（seg）、修正后（rec，主配置）的读数与 CI 取 `experiments/W2/0078_llmseg_hpcgpu06/qwen3.5-9b/refseg/llmseg/<划分>/metrics.json` 的 readouts.<seg|rec>.headline；合并读数、同样 300 句上的 W2-c 读数（RUN-0072）与 LLMSeg − W2-c 的配对差取 `experiments/W2/0078_llmseg_hpcgpu06/compare_w2c/compare.json`（D139，CPU 补跑）；每图调用次数取 readouts.<seg|rec>.compute_per_image.n_model_calls。
- 版式：系统为行、划分（加合并）为列——论文里指代分割表的惯例；任务书"列：自己的框、W1-F…"按"列出这些系统"理解。三块：9B 各系统；配对差（W2-c − W1-F、W2-a − W1-F、回放 − W2-c）；外部基线 LLMSeg（修正前、修正后两行，同句 W2-c 并列，下接两行配对差）。数值 ×100、一位小数，CI 用 [low, high]。
- 合并列：分析里没有逐系统的合并读数（w2_test_table.json 只有合并的配对差；W2-a 的逐句数据超过同步上限，不在仓库），所以 9B 各系统一块的合并列是"—"，合并只出现在配对差与 LLMSeg 两块（LLMSeg 的合并读数没有 CI）。
- gIoU 附录表：同一版式；W2 的配对差一块没有（分析只算了 cIoU 的配对差），LLMSeg − W2-c 的 gIoU 差取 compare.json。

**表注（英文，cIoU）**

- Qwen3.5-9B on the five W2 test splits (1,000 expressions each, the E1 polygon subset); cIoU ×100 with 95% bootstrap CI over expressions (1,000 resamples, as computed by each run). cIoU = summed intersection / summed union.
- Own box: the model's own box (E1 protocol) as the mask; W1-F: that box plus the default GrabCut candidate (a fixed pipeline, no LLM decision); random / oracle pick: a random / the ground-truth-best of the four tool candidates; W2-a: the frozen model picks a candidate (zero-shot; candidate 1 = W1-F, also submitted when the pick fails to parse); W2-c: LoRA text supervision to propose the box and pick; W2-c + replay: W2-c trained with self-distillation replay (post hoc variant, descriptive).
- E1 zero-shot polygon: all 1,000 outputs per split fail to parse (strict scoring gives 0).
- Paired differences (points ×100): per-expression paired bootstrap of the ratio of sums, 95%, 10,000 resamples; pooled = the five splits together (5,000 expressions). Per-system pooled cIoU is not part of the analysis, so the pooled column has differences only.
- External baseline LLMSeg (official code, commit f2191e6) driven by the same Qwen3.5-9B, zero-shot, on 300 hash-selected expressions per split (pooled 1,500), with W2-c on the same expressions alongside. Before refinement = after its first two stages; after refinement = its main configuration. Model calls per image (split means): 13.5–14.0 after vs. 2.0 before refinement; W2-c 2. LLMSeg − W2-c: paired bootstrap, 95%, 1,000 resamples; the pooled LLMSeg and W2-c readings have no CI in the analysis.
- Data: RUN-0072, RUN-0076, RUN-0078 (LLMSeg comparison computed on CPU, D139); W2-a paired differences completed in RUN-0081.

核对：

- 一致：docs/analysis/w2_test_table.json 的 9B 各设置读数与 experiments/W2/0072_w2_test_hpcgpu06 各叶目录的 metrics.json 逐格相同
- 一致：docs/analysis/w2c_replay.md §1 的 W2-c 与回放读数（3 位）对 docs/analysis/w2_test_table.json 与 experiments/W2/0076_w2c_replay_test_hpcgpu06 的 metrics.json
- 一致：experiments/W2/0078_llmseg_hpcgpu06/compare_w2c/compare.json 的 LLMSeg 读数与各叶目录 metrics.json（rec、seg）逐格相同；W2-c 的 rec、seg 两份读数相同

## 6. W2-a 随尺寸：`fig_w2_size.pdf` / `fig_w2_size.png`

- 数据：`docs/analysis/w2_test_table.json` 的 models.<模型>.pairs_pooled（w2a-w1f；9B 另有 w2c-w1f），合并 5,000 句；RUN-0072 (9B), RUN-0073 (2B, 4B), RUN-0081 (W2-a paired differences)。
- 挑选的解析失败率（横轴刻度下）：W2-a 叶目录 metrics.json 的 pick.parse_failure，5 个划分合计 Σn ÷ Σof（分母 = 有候选的句子）：2B：`experiments/W2/0073_w2_test_hpcgpu07/qwen3.5-2b/refseg/w2a/<split>/metrics.json :: pick.parse_failure`; 4B：`experiments/W2/0073_w2_test_hpcgpu07/qwen3.5-4b/refseg/w2a/<split>/metrics.json :: pick.parse_failure`; 9B：`experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2a/<split>/metrics.json :: pick.parse_failure`。9B 的数在 RUN-0072 的结果目录里有（w2_test.md 第 2 节写的"见 RUN-0072 报告"），直接取。
- 口径：纵轴 = 配对差 ×100；误差棒 = 95% 配对 bootstrap CI；横轴 = 名义参数量（对数轴）。W2-c 只在 9B 跑过（菱形）。

**图注（英文）**

W2-a − W1-F across model size (Qwen3.5 2B, 4B, 9B; W2 test, five splits pooled, 5,000 expressions; cIoU ×100; 95% paired bootstrap CI, 10,000 resamples). Diamond: W2-c − W1-F, run at 9B only. Below each size: the share of W2-a's selection answers that fail to parse (2B 48% (2,359/4,938); 4B 21% (1,027/4,831); 9B 2.9% (144/4,897)); a selection that fails to parse falls back to W1-F, which pulls W2-a toward W1-F at small sizes. W2-c selection parse failures: 9B 0.0%. Data: RUN-0072 (9B), RUN-0073 (2B, 4B), RUN-0081 (W2-a paired differences).

## 7. 语言代价：`fig_lang_cost.pdf` / `fig_lang_cost.png`

- 数据：`docs/analysis/lang_cost_9b.json`（`scripts/mgr/lang_cost_table.py`，D125-3）的 battery（大题量 MMLU-Pro 2,000 题、IFEval 541 条）与 ttext（定位、计数）四个读数，每格是相对同节点基座的差（百分点）与 95% CI；健康检查的标红（health.*.flag）只写进图注，不进图。各系统的来源路径（JSON 原文）：
  - E3b+C3：health `experiments/W2/0074_lang_battery_hpcgpu06/health/health_compare_e3b_c3.json`；battery `experiments/W2/0074_lang_battery_hpcgpu06/lb_compare_e3b_c3.json`；ttext `experiments/E3/0056_hpc_9b_eval_hpcgpu06/t_text/e3b_c3/seed0/qwen3.5-9b`；ttext_base `experiments/E3/0056_hpc_9b_eval_hpcgpu06/qwen3.5-9b`
  - W2-c：health `experiments/W2/0069_w2c_hpcgpu06/health/health_compare.json`；battery `experiments/W2/0074_lang_battery_hpcgpu06/lb_compare_w2c.json`；ttext `experiments/W2/0069_w2c_hpcgpu06/0069_ttext/t_text/w2c/seed0/qwen3.5-9b`；ttext_base `experiments/W2/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b`
  - W2-c+回放：health `experiments/W2/0076_w2c_replay_eval_hpcgpu06/health/health_compare.json`；battery `experiments/W2/0076_w2c_replay_eval_hpcgpu06/lb_compare_w2c_replay.json`；ttext `experiments/W2/0076_w2c_replay_ttext_hpcgpu06/t_text/w2c_replay/seed0/qwen3.5-9b`；ttext_base `experiments/W2/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b`
- 探索的候选（X-2、X-3、X-3L）：判定还没做，缺省不画；`--with-explore` 打开后，大题量取 `experiments/X/0084_lb_explore_hpcgpu07/lb_compare_x2.json`、`experiments/X/0084_lb_explore_hpcgpu07/lb_compare_x3.json`、`experiments/X/0084_lb_explore_hpcgpu06/lb_compare_x3l.json`，健康检查的标红取 `experiments/X/0080_x2_hpcgpu07/0080_eval/health/health_compare.json`、`experiments/X/0082_x3_hpcgpu07/0082_eval/health/health_compare.json`、`experiments/X/0083_x3l_hpcgpu06/0083_eval/health/health_compare.json`（RUN-0080, RUN-0082, RUN-0083 (health), RUN-0084 (battery)）；它们没有 T 文本读数，图注写明。（本次生成时 --with-explore 关着，缺省）
- 版式：读数自上而下（底色 = T 文本通道），每个读数里三个系统各一行，点 = 差，横线 = 95% CI，右侧写数值。

**图注（英文）**

Language cost on Qwen3.5-9B: difference from the base model evaluated on the same node (percentage points; 95% paired bootstrap CI). MMLU-Pro (letter answers) and IFEval (prompt-level strict) from the large language battery; grounding Acc@0.5 and counting accuracy (CountBench) on the T-text channel (shaded; base 87.1% / 89.4%). Systems: E3b+C3 (E3), W2-c, and W2-c + replay (post hoc self-distillation replay). Not drawn: the small health check (brief §8.11: MMLU-Pro 100, GSM8K 50, IFEval 50 items) flags E3b+C3: MMLU-Pro −8.0 (n=100), IFEval −20.0 (n=50); W2-c: MMLU-Pro −6.0 (n=100), IFEval −6.0 (n=50); W2-c + replay: IFEval −6.0 (n=50). Data: lang_cost_9b.json (RUN-0056, RUN-0069, RUN-0074, RUN-0076).

## 8. 计数分档：`fig_count_bins.pdf` / `fig_count_bins.png`

- 数据：`docs/analysis/count_errors_9b.json`（`scripts/mgr/count_errors.py`）的三个大档 2–4, 5–7, 8–10：(a) 各系统的准确率（bins.<档>.acc）；(b) 与配对基座的差及配对 bootstrap 95% CI（bins.<档>.vs_base）。预测文件（JSON 原文）：
  - base@0069：`experiments/W2/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b/count/predictions.jsonl`
  - W2-c：`experiments/W2/0069_w2c_hpcgpu06/0069_ttext/t_text/w2c/seed0/qwen3.5-9b/count/predictions.jsonl`
  - W2-c+回放：`experiments/W2/0076_w2c_replay_ttext_hpcgpu06/t_text/w2c_replay/seed0/qwen3.5-9b/count/predictions.jsonl`
  - base@0056：`experiments/E3/0056_hpc_9b_eval_hpcgpu06/qwen3.5-9b/count/predictions.jsonl`
  - E3b+C3：`experiments/E3/0056_hpc_9b_eval_hpcgpu06/t_text/e3b_c3/seed0/qwen3.5-9b/count/predictions.jsonl`
  - C3：`experiments/E3/0056_hpc_9b_eval_hpcgpu06/t_text/c3/seed0/qwen3.5-9b/count/predictions.jsonl`
- 为什么分两个面板：分析 JSON 里只有"与基座之差"的 CI，没有各档准确率本身的 CI（不重算），所以 CI 画在 (b) 上，(a) 只画准确率。
- 基座：W2-c、W2-c + 回放配 base@0069，C3、E3b+C3 配 base@0056；两次基座读数逐条相同（1.0）时只画一根（base@0069）。

**图注（英文）**

Counting accuracy by ground-truth count on CountBench (442 items; bins 2–4 (n=149), 5–7 (n=140), 8–10 (n=153)), Qwen3.5-9B, T-text channel. (a) Accuracy. (b) Difference from the base, 95% paired bootstrap CI (10,000 resamples). W2-c, W2-c + replay paired with the base run of RUN-0069; C3, E3b+C3 paired with the base run of RUN-0056. The two base runs give identical answers (agreement 1.000), so one base bar is drawn. Data: count_errors_9b.json (RUN-0056, RUN-0069, RUN-0076).

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
- `tab_w2_test.tex`
- `tab_w2_test.md`
- `data/tab_w2_test.json`
- `tab_w2_test_giou.tex`
- `tab_w2_test_giou.md`
- `fig_w2_size.pdf`
- `fig_w2_size.png`
- `fig_w2_size_caption.tex`
- `data/fig_w2_size.json`
- `fig_lang_cost.pdf`
- `fig_lang_cost.png`
- `fig_lang_cost_caption.tex`
- `data/fig_lang_cost.json`
- `fig_count_bins.pdf`
- `fig_count_bins.png`
- `fig_count_bins_caption.tex`
- `data/fig_count_bins.json`
