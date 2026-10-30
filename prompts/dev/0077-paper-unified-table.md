[DEV-TASK 0077] 论文图表：统一主表（5 × 300 句）、分档图、种子波动附录表

## 背景

- 负责人要求论文主表改为**统一主表**：所有系统在同样的 5 个 test 划分 × 300 句上，跑完全部照实报告，不挑（D154）。专家 SOTA 用文献数字作灰色参考线。
  - 读数已全部算好，在 `docs/analysis/unified_table.{md,json}`（`scripts/mgr/unified_table.py`；预注册 U）。
  - 共 12 个系统，分四组（本文的方法、各种"手"、外部方法、附带）：每个划分与合并的 cIoU、gIoU 及 CI，以及 X-2 − 该系统的配对差。
  - 专家 SOTA 的文献数字在 `docs/references/refseg_sota.json`，4 行，每行 5 个 test 划分；说明在 `refseg_sota_notes.md`，选法在预注册 U-3a。
- 按划分、类别、形状复杂度的分档：`docs/analysis/x_bins.{md,json}`（`scripts/mgr/x_bins.py`；预注册 U-6）。
- 主方法 C1 三个种子的波动：`docs/analysis/seed_variance.{md,json}`（`scripts/mgr/seed_variance.py`）。
- 现有出图脚本 `scripts/figures/make_paper_figures.py`（DEV-0052、0063、0071、0073）有 tab_main（全 5,000 句）、tab_w2_test、tab_lang_cost 等。
  - tab_main 保留，改为附录的"全 test 5,000 句"表；论文正文的主表换成这次新增的 tab_unified。
- 定性图在 HPC 上出（RUN-0099），不在本任务里。语言代价一段等负责人答 N46，本任务不动 tab_lang_cost。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `PREREGISTRATION.md` 的 U 节（U-1、U-2、U-3a、U-6）
- `scripts/figures/make_paper_figures.py` 及其测试、`docs/figures/README.md`：风格、配色（C1 = X-2 固定用槽 7 的紫）、caption 与 README 的写法、missing 机制
- 上面列的分析 JSON 与 `docs/references/refseg_sota.json`

## 任务

都从仓库里已有的 JSON 读。不跑模型，不重算 bootstrap，CI 一律取 JSON 里已有的。

1. **`tab_unified`（.tex 与 .md，论文正文主表）**：
   - 行：照 `unified_table.json` 的 systems 顺序，按 group 分块，系统一个都不能少。
   - 列：5 个划分的 cIoU，合并的 cIoU [95% CI]、合并的 gIoU，"X-2 − 它"的 cIoU 配对差 [CI]。×100，一位小数。
   - 灰色参考块（`refseg_sota.json` 的 4 行）放在表的最后：
     - 只填 5 个划分，合并一列与配对差一列写"—"；
     - .tex 里用灰色字；
     - 表注写明：这是完整 test 划分上的文献数字，不是这 300 句的子集；不参与配对；各自用了学习型视觉模型；各自的出处。
   - 表注还要写：
     - 子集的取法：hash_rank，每个划分 300 句，与 LLMSeg 相同；
     - 照实报告、不挑；
     - LISA 有两处：实线是我们跑的发布权重 v1，灰线是论文的微调版，两者不是同一个权重；
     - 各系统一句话的定义，可参照现有 tab_main 的写法；
     - 数据来源的 RUN 编号：RUN-0072、0078、0092、0095、0097、0100。
   - 不加粗。
2. **`fig_unified`（PDF + PNG）**：
   - 5 个划分各一个小面板，横条是各系统的 cIoU，误差棒取各叶目录 metrics.json 的 CI（`unified_table.json` 里有 per_split），按组分色；
   - 每个面板里，4 条 SOTA 画灰色竖虚线，并标名字；
   - 再加一个"合并"面板，只画我们的系统，误差棒是 10,000 次的 CI，不画 SOTA，因为文献没有合并读数；
   - X-2 用槽 7 的紫，其他颜色照现有约定，过 `validate_palette.js`。
3. **`fig_bins`（PDF + PNG）**，取 `x_bins.json`：
   - (a) 人与非人；(b) 形状复杂度四分位档；
   - 两组横条：X-2 − W2-c 与 X-2 − 椭圆，带 CI；
   - 每档的句数标在刻度下；
   - caption 写分档规则（U-6）与每个划分里人的占比。
4. **`tab_seeds`（.tex 与 .md，附录）**，取 `seed_variance.json`：
   - 三个种子的 `x2_spline_trimap` 合并 cIoU 与 CI、− W2-c 参照、− seed 0；椭圆消融；
   - 语言读数：大题量 MMLU-Pro、IFEval，定位，计数，相对基座的差 [CI]；
   - 健康检查的标红用 † 标出；
   - 表注写清这些都只在开发数据与语言检查上，不是 test，以及各自的 RUN 编号。
5. tab_main 的 caption 改为说明它是附录的全 5,000 句版、正文用 tab_unified。数值不变。
6. README 与 caption 照现有写法，每张图表写数据来源。
7. **测试**（`HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/test_paper_figures.py -q`）：
   - 用仓库里的真数据出一遍，新文件都生成；
   - 抽查数值与 JSON 逐项一致：X-2 合并、LISA 合并、E3b 合并、一个 SOTA 格、bins 的人与非人、seeds 的计数；
   - tab_unified 的系统数与 JSON 相同；
   - 已有图表除 tab_main 的 caption 外逐字节不变。

## 不做

- 不改任何分析 JSON、分析脚本、结果目录、`docs/references/`。
- 不动 tab_lang_cost、fig_lang_cost：等 N46。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 新图表清单与数据来源；
- tab_unified 的 .md 全文；
- 配色检查；
- 测试结果原文；
- 待裁的地方。

尽量 2.5 小时内完成。
