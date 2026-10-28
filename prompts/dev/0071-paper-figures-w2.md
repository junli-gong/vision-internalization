[DEV-TASK 0071] 论文图表：W2 一节（test 主表、尺寸、语言代价、计数分档），外加干扰图 R_seg 面板的离群值

## 背景

- W2 的 test、外部基线 LLMSeg、语言代价、回放变体都已有终稿数据，管理员的分析在 `docs/analysis/`：
  - `w2_test.md` 与 `w2_test_table.json`：§11 的全表；W2-a 的配对差由 RUN-0081 在 HPC 上补齐；
  - `llmseg_baseline.md`，数据在 `experiments/W2/0078_llmseg_hpcgpu06/compare_w2c/compare.json`；
  - `lang_cost_9b.{md,json}`：D125-3 的并排；
  - `count_errors_9b.{md,json}`：计数分档；
  - `w2c_replay.md`：回放变体，test 读数在 `experiments/W2/0076_w2c_replay_test_hpcgpu06/`。
- 现有的出图脚本 `scripts/figures/make_paper_figures.py`（DEV-0052、0063）只有 E1、E2、E3 与干扰随尺寸。
- 遗留：干扰图里 R_seg 面板 (d) 的秩 128 一点是离群值，把坐标轴撑得太大，其余点挤在一起。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `scripts/figures/make_paper_figures.py` 及其测试、`docs/figures/README.md`（风格、配色、caption 的写法、README 的生成方式）
- 上面列的分析文档与 JSON

## 任务

在同一个脚本里加 W2 的部分。新增的图表都从仓库里已有的 JSON 与结果目录读，不跑模型，不重算 bootstrap：CI 取分析 JSON 里已有的。

1. **`tab_w2_test`**（.tex 与 .md）：9B 的 5 个划分加合并，每格 cIoU [95% CI]。
   - 列：自己的框、W1-F、随机挑选、最优候选、W2-a、W2-c、W2-c+回放、E1 零样本多边形。
   - LLMSeg 另起一块：它只跑了每个划分 300 句；同样 300 句上的 W2-c 并列；修正前与修正后两行。
   - 表下是配对差 W2-c − W1-F、W2-a − W1-F、回放 − W2-c 及 CI。
   - gIoU 放附录表 `tab_w2_test_giou`。
2. **`fig_w2_size`**：2B、4B、9B 的 W2-a − W1-F（合并 5,000 句，CI 误差棒），9B 另画 W2-c − W1-F。
   - 每个尺寸标上挑选的解析失败率。数从 RUN-0073 的报告与结果目录取，取不到就在 caption 里写"见正文"。
3. **`fig_lang_cost`**：9B，相对基座的差（百分点）与 95% CI。
   - 读数：大题量 MMLU-Pro、大题量 IFEval prompt 严格、T 文本定位、T 文本计数。
   - 系统：E3b+C3、W2-c、W2-c+回放。
   - 健康检查的标红在 caption 里用文字说明，不进图。
   - 探索的方法暂不进图：判定还没做，留一个开关 `--with-explore`，缺省关。
4. **`fig_count_bins`**：CountBench 按真值个数分 2–4、5–7、8–10 三档的准确率，系统为基座、W2-c、W2-c+回放、C3、E3b+C3，带 CI。
5. **干扰图 R_seg 面板**：把秩 128 一点画在坐标轴之外，用箭头加数值标注，或用断轴，二选一，在 README 里写明，使其余点的差别看得清。其余面板不变。
6. caption（.tex）与 `docs/figures/README.md`：照现有的写法，每张图表都写数据来源（文件与 RUN 编号）。
7. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_paper_figures.py -q`）：
   - 用仓库里的真数据出一遍，新文件都生成；
   - 数值与分析 JSON 逐项一致，抽查若干格；
   - 原有的 E1、E2、E3 输出逐字节不变，只有干扰图那一张按第 5 条改。

## 不做

- 不改任何分析 JSON 与结果目录；不改 `docs/analysis/` 里的文档。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 新图表的清单与数据来源；
- R_seg 面板的处理方式；
- 测试结果原文；
- 数据里取不到、只能写"见正文"的地方。

尽量 3 小时内完成。
