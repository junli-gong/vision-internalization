[DEV-TASK 0078] 论文图表：语言代价四方表加"3 个种子的范围"、表注与图注按 D162 改，新增 4B 补充实验表

## 背景

- 负责人答复 N46（D162，原文 `docs/lead_answers/2026-10-30-n46.md`）。
  - 四方表的 X-2 列保留 seed 0，因为它与 test 主表是同一个检查点；加一行"3 个种子的范围"；表注写明其余各列都是单种子。
  - 以大题量为准，健康检查的标红只作并列说明。
- 要点与数字整理在 `docs/analysis/lang_cost_writing.md`，§1–§3、§5、§6。数据：
  - `docs/analysis/seed_variance.json`：C1 三个种子的 `language`，含大题量、定位、计数，相对同节点基座的差与 CI；
  - `docs/analysis/lang_cost_9b.json`：四方表现有的数据；
  - `docs/analysis/x2_4b.json`：4B 补充实验（预注册 X-7、RUN-0098、0101），分割、健康检查、大题量（`battery`）；
  - `docs/analysis/count_errors_c1_seeds.json`：三个种子的计数分档，含数多、数少。
- 出图脚本 `scripts/figures/make_paper_figures.py` 里已有 tab_lang_cost（及 _full）、fig_lang_cost、tab_seeds（DEV-0073、0077）。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/lead_answers/2026-10-30-n46.md`、`docs/analysis/lang_cost_writing.md`
- `scripts/figures/make_paper_figures.py` 及其测试、`docs/figures/README.md`
- 上面列的 JSON

## 任务

只从仓库里已有的 JSON 读，不重算 bootstrap。

1. **tab_lang_cost 与 tab_lang_cost_full**：
   - X-2 列照旧用 seed 0；
   - 在大题量与 T 文本几行下面，各加一行或一列"X-2，3 个种子的范围"：MMLU-Pro −0.2 到 −2.3，IFEval 0.0 到 −3.5，定位 +3.4 到 +3.5，计数 −8.1 到 −16.1（均值 −10.9）。版式由你选，在总结里说明；
   - 表注改为：
     - 其余各列都是单种子；
     - 以大题量为准，健康检查的标红只作并列说明；
     - 50 条 IFEval 上 seed 1、2 标红，以大题量为准三个种子都没有超过 3.5 点。
2. **fig_lang_cost**：
   - X-2 的点仍是 seed 0，另画 3 个种子的范围，例如一条细的横线或空心点；
   - 图注同上改。
3. **新增 tab_x2_4b（.tex 与 .md，附录）**：
   - 4B 与 9B（三个种子的范围）并列：
     - 分割，开发数据 2,300 句：样条、椭圆、只挑候选，样条 − 椭圆；
     - 大题量：MMLU-Pro、IFEval，相对同节点基座的差 [CI]；
     - 健康检查：标红用 †。
   - 表注写明：补充实验，预注册 X-7；只在开发数据上，不是 test；4B 只有一个种子；数据来源 RUN-0098、0101。
4. **tab_seeds**：表注加一句"三个种子的分割几乎相同，差异全在语言的副作用上"，并把计数的三个值列出来；如果已有，就核对。
5. README 与 caption 照现有写法，写清数据来源。
6. **测试**（`HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/test_paper_figures.py -q`，另跑 `tests/test_constants.py`）：
   - 新表、新行的数值与 JSON 逐项一致：至少抽查 3 个种子的范围、4B 的 IFEval 差与 CI、4B 样条；
   - 其余图表除这里点名改动的以外逐字节不变。
   - 重出图表照 README 的命令，带两条 `--e3-prepend`。

## 不做

- 不改任何分析 JSON、分析脚本、结果目录。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 新版 tab_lang_cost.md 与 tab_x2_4b.md 的全文；
- 测试结果原文；
- 待裁的地方。

尽量 2 小时内完成。
