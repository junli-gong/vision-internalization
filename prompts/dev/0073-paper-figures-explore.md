[DEV-TASK 0073] 论文图表：加入通过 G-EX 的主方法（C1 = X-2）与消融，补主表的合并列，新增语言代价四方表与计算表

## 背景

- 探索的两个候选都通过了 G-EX（D149），并在 W2 的 test 上跑完（RUN-0092）。负责人定的报告方式（D147）：
  - C1（X-2）是论文的主方法，C2（X-3）作为"去掉修正流的简化版"消融；
  - "样条换成框内椭圆"的能力归属消融进论文主表；
  - LLMSeg 外部基线 cIoU 与 gIoU 都报；
  - **语言代价四方对照表**（9B，同一套检查）：E3b+C3、W2-c、W2-c+回放、X-2，是论文第三段的核心证据。
- 管理员的分析，都已在仓库里：
  - `docs/analysis/x_test_table.{md,json}`（`scripts/mgr/x_test_table.py`）：
    - C1、C2 在 test 上每个划分与合并的 cIoU、gIoU 与 CI；
    - 配对差：− W2-c、− W1-F、− 椭圆消融、− LLMSeg（同样 1,500 句），cIoU 与 gIoU；
    - `compute` 与 `compute_reference`：每句的调用次数、token、时间，含 C1、C2、椭圆消融、W2-c、W1-F、LLMSeg。
  - `docs/analysis/w2_test_pooled.{md,json}`（`scripts/mgr/w2_test_pooled.py`，D152）：9B 各系统合并 5,000 句的 cIoU、gIoU 与 CI。
    - W2-a 一行要等 RUN-0095 推回 `experiments/W2/0095_w2a_scores/` 后重跑脚本才有。
    - 你开始时可能还是"缺"。缺就照现有的 missing 机制写进 README，输出"—"，不要自己补。
  - `docs/analysis/lang_cost_9b.json`：systems 有 `E3b+C3`、`W2-c`、`W2-c+回放`、`X-2（C1）`、`X-3（C2，消融）`。每个都有 health、battery、ttext 三块与来源路径。
  - `docs/analysis/count_errors_9b.json`：systems 新增了 `X-2（C1）`、`X-3（C2）`。
  - 门报告 `reports/gates/G-EX.md` §1（方法）、§6（test 读数），只用来写 caption，不从中取数。
- 出图脚本 `scripts/figures/make_paper_figures.py`（DEV-0052、0063、0071）已有 `--with-explore` 开关，缺省关。那时判定还没做，现在做完了。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `scripts/figures/make_paper_figures.py` 及其测试 `tests/test_paper_figures.py`、`docs/figures/README.md`（风格、配色、caption、missing 机制）
- 上面列的分析 JSON 与 `reports/gates/G-EX.md`
- `PREREGISTRATION.md` 的 X-5a、X-6（主方法与消融的叫法）

## 任务

都从仓库里已有的 JSON 与结果目录读。不跑模型，不重算 bootstrap，CI 一律取分析 JSON 或 metrics.json 里已有的。

1. **`tab_w2_test`（全表，进附录）与 `tab_w2_test_giou`**：
   - 9B 各系统一块，新增三行：
     - "C1 = X-2（ours）"；
     - "C1，样条换成框内椭圆"；
     - "C2 = X-3（去掉修正流）"。
   - 每个划分的读数取各叶目录 `metrics.json` 的 headline，与现有行同一口径。叶目录在 `experiments/X/0092_x{2,3}_test_hpcgpu06/qwen3.5-9b/refseg/<x2_spline_trimap|x2_ellipse_trimap>/<划分>/`。
   - **合并列**：所有系统都取 `w2_test_pooled.json`。缺的写"—"，并进 missing。表注改成说明合并读数的来源与重抽次数（10,000），删掉"Per-system pooled cIoU is not part of the analysis"。
   - 配对差一块新增：C1 − W2-c、C1 − W1-F、C1 − 椭圆消融、C2 − W2-c。取 `x_test_table.json`，每个划分加合并；gIoU 表取对应的 `g_` 键。
   - LLMSeg 一块新增：
     - "C1，同样的句子"一行；
     - C1 − LLMSeg（修正后）、C1 − LLMSeg（修正前）两行配对差，取 `x_test_table.json` 的 per_split.*.llmseg 与 pooled.llmseg。
2. **新增 `tab_main`（.tex 与 .md，论文正文主表，紧凑）**：
   - 行：自己的框、W1-F、最优候选（4 个工具候选里的真值最优）、W2-a、W2-c、**C1 = X-2（ours）**、C1 的椭圆消融、C2 = X-3。
   - 列：5 个划分与合并，cIoU ×100，一位小数。CI 只在合并列给。
   - 末列"每句调用 / 生成 token"，取 `x_test_table.json` 的 compute。没有计算记录的行写"—"，并在表注说明。
   - 下接一小块外部基线：同样 1,500 句上 LLMSeg（修正后，主配置）与 C1 的合并读数，以及配对差与 CI。
   - 表注写清：
     - C1 是什么：一个 LoRA；9B 先提框、再描 16 个控制点的 B 样条；样条作为 GrabCut 的 trimap 种子；GrabCut 不含学习；
     - 消融各改了什么；
     - 配对 bootstrap 的口径；
     - 数据来源的 RUN 编号。
   - 版式参照现有表；`\best` 一类的加粗只加在合并列的最高值上。
3. **新增 `tab_lang_cost`（.tex 与 .md）：语言代价四方对照表**，取 `lang_cost_9b.json`：
   - 列：E3b+C3、W2-c、W2-c+回放、X-2（C1）。
   - 行：
     - 健康检查三项：MMLU-Pro 100、GSM8K 50、IFEval 50，差与是否标红；
     - 大题量 MMLU-Pro 2,000 与 IFEval 541：差 [95% CI]；
     - T 文本的定位 Acc@0.5 与计数准确率：差 [95% CI]。
   - 每格是相对同节点基座的百分点。表头下给基座的读数；同一读数在不同节点的基座若不同，就分别注明。
   - 附录版 `tab_lang_cost_full` 再加一列 X-3（C2，消融）。
4. **新增 `tab_compute`（.tex 与 .md）**：
   - 每句的模型调用、生成 token、输入 token、模型调用时间、墙钟，取 `x_test_table.json`。
   - 行：W1-F、W2-c、C1、C1 的椭圆消融、C2、LLMSeg。
   - LLMSeg 的墙钟写"—"，原因照 `x_test_table.md` 的说明写进表注。
5. **`fig_lang_cost`**：
   - 缺省画四个系统：E3b+C3、W2-c、W2-c+回放、X-2（C1）。X-2 的 T 文本现在有了，取 `lang_cost_9b.json`。
   - `--with-explore` 改为再加 X-3（C2）与 X-3L。
   - 健康检查的标红仍只写在 caption：X-2 没有标红。
   - 配色：给 X-2 一个固定的分类槽，并在 C1 出现的所有图里都用同一个颜色。过 `validate_palette.js`，写进 README。
6. **`fig_count_bins`**：缺省加 X-2（C1），`--with-explore` 时再加 X-3（C2）。都取 `count_errors_9b.json`。
7. caption（.tex）与 `docs/figures/README.md`：照现有写法，每张图表写数据来源（文件与 RUN 编号）。README 开头的"缺的数据"如实列出。
8. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_paper_figures.py -q`）：
   - 用仓库里的真数据出一遍，新文件都生成；
   - 新表的数值与分析 JSON 逐项一致，至少抽查 C1 合并、C1 − W2-c 合并、C1 − 椭圆消融合并、四方表里 X-2 的大题量 MMLU-Pro 与计数；
   - `w2_test_pooled.json` 里 W2-a 缺与不缺两种情况都能出表。不缺的情况用临时目录里的合成 JSON。
   - E1、E2、E3 与干扰图的输出逐字节不变。W2 已有的图表只按上面的条目改。

## 不做

- 不改任何分析 JSON、分析脚本与结果目录。不改 `docs/analysis/` 里的文档。
- 种子波动（RUN-0093、0094 还在跑）不进图表。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 新图表的清单与数据来源；
- `tab_main` 的最终版式（贴 .md 全文）；
- 配色的检查结果；
- 测试结果原文；
- 缺的数据（大概率只有 W2-a 的合并读数，取决于 RUN-0095 是否已推回）。

尽量 3 小时内完成。
