[DEV-TASK 0084] 论文用图表加 X-M 一节：主表（检测、实例分割与消融）、LVIS、指代分割、语言代价、判定的附录表（D182）

## 你的 worktree（管理员事先建好的，不是 Agent 自动建的那种）

- 路径 `/Users/tim/Research/CVPR2027/.claude/worktrees/dev0084`，分支 `dev0084`，建在 origin/main 128eba5d2。
- 稀疏签出：`experiments/` 只签出了 `experiments/XM/`。开发机磁盘只剩约 8 GB，不要签出其余的 experiments。
- 所有命令都先 `cd` 到这个路径；不要碰主仓库 `/Users/tim/Research/CVPR2027`。不 commit、不 push。
- 测试：`cd <worktree> && HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/test_paper_figures.py -q`，加上你新增的测试。**不要跑全量**。跑之前看一眼 `df -h /`，低于 3 GB 就停下告诉管理员。
- 不在 worktree 里重新生成整个 `docs/figures/`：其余各节要读这里没签出的 experiments。合入后由管理员在 main 上重新生成。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D165、D170、D177–D183。**D182 是负责人的写作要求，逐条照做。**
- `reports/gates/G-XM.md`
- `scripts/figures/make_paper_figures.py` 的模块说明，特别是 W2 一节、X-2（DEV-0073）、统一主表（DEV-0077）、语言代价的种子范围（DEV-0078）的写法；`tests/test_paper_figures.py`
- 数据：
  - `docs/analysis/xm_results.json`：RUN-0113 的输出，检测与实例分割的点估计和配对差、RefCOCO test、ReasonSeg；
  - `docs/analysis/xm_lang.json`：`scripts/mgr/xm_lang_table.py` 生成。计数与定位等 RUN-0114 跑完才有；现在可以先在 worktree 里运行它生成一份，缺的格子是"待 RUN-0114"；
  - `experiments/XM/0108_gate_xm_hpcgpu07/0108_gate/gate_xm.json`：判定；
  - `docs/references/xm_refs_notes.md`：文献数字。

## 任务

照现有的约定：只读结果文件，脚本里不写任何读数；每张表出 LaTeX（booktabs）与 Markdown；`docs/figures/data/*.json` 记来源；README 里写来源与缺的数据。

1. **`tab_xm_main`（正文主表）**：最终报告集 xm_final，1,000 张、2,940 对。
   - 行：
     - 基座零样本；
     - 基座的框交给 GrabCut；
     - **X-M（主设置：样条 → GrabCut）**；
     - 消融块，各行附相对主设置的配对差与 CI：**样条直接光栅化、不贴边**、框内椭圆、X-M 的框交给 GrabCut；
     - Rex-Omni 一块：joint，即它自己的协议。逐类的那一行放附录版。
   - 列：检测 F1@0.5、P/R、实例分割 F1@0.5，各带 CI；另加检测 F1@mIoU。
   - D182 的写法：
     - **与 Rex-Omni 写成"相当"（comparable），不写"超过"**。差是 −0.016 [−0.033, +0.005]，区间跨 0，表注写明。
     - "样条直接光栅化、不贴边"只比主设置低 0.012，表注点明：这是"不需要专用解码器"最直接的证据，即形状完全来自 LLM，GrabCut 只再加约 0.01。
   - 附录版 `tab_xm_main_full`：再加判定集一块，以及逐类 Rex-Omni。
2. **`tab_xm_lvis`**：LVIS 子集，1,000 张，每图最多 3 类，2,213 对，描述性。同主表的列。
   - 表注：只作描述；Rex-Omni 的训练数据可能与 LVIS 重叠；每图类别上限见 XM-2b。高于 Rex-Omni 的 +0.027 不写成结论。
3. **`tab_xm_refseg`**：
   - RefCOCO 的 W2 test：5 × 1,000 句，X-M 对 X-2，同在 06；各划分与合并的 cIoU、gIoU，以及配对差。
   - ReasonSeg val：196 条，X-M 对 X-2，同在 07。**如实写出 cIoU 比 X-2 低 0.052、区间跨 0**；gIoU 持平。
4. **`tab_xm_lang`**：X-M 相对同节点基座的语言代价，取自 `docs/analysis/xm_lang.json`。
   - 行：大题量 MMLU-Pro、大题量 IFEval prompt 严格、指令级严格、**计数 CountBench**、定位 acc@0.5，健康检查三项放附录版。
   - 每行给差与 CI，并列 X-2 三个种子的范围。
   - **不以阈值作论据**，表里不出现 −5.7、−8.1。表注写明 MMLU-Pro、IFEval 落在 X-2 三个种子的范围之内（MMLU-Pro −0.2 到 −2.3，IFEval 0.0 到 −3.5）；IFEval 约 −3 点的代价写进局限。
   - 计数照实给出，与 X-2 的 −8.1 到 −16.1 并列。缺数据时写 missing，不留空。
5. **`tab_xm_gate`（附录）**：G-XM 的四条，各给读数、差、CI、阈值、结果。
   - 第 4 条三种写法并列：D170（判定用）、D165 原文（诊断，MMLU-Pro −1.15 [−2.40, +0.10] 不满足）、N47 的 (b)（IFEval −6.10 < −5.5 不满足）。数字从 gate_xm.json 读。
   - 表注：D170 写于任何 X-M 训练之前；它比 (b) 宽，理由是 (b) 连 X-2 seed 0 自己都过不了。
6. **文献的灰行**：新建 `docs/references/xm_sota.json`，数字照 `xm_refs_notes.md`，例如 Rex-Omni 在完整 COCO val2017 上 72.0、Qwen2.5-VL-7B 64.0 等，每条注明出处。在主表最后作灰行，标"不是同一批图"。
7. **测试**：在 `tests/test_paper_figures.py` 的假结果树里加 X-M 的输入，核对以下几点：
   - 每张表都生成；
   - 数字等于输入 JSON 里的值；
   - 缺 xm_lang 的计数时写 missing；
   - 主表注里有 comparable、没有 outperform；
   - 语言表里没有阈值。
8. 不改其余各节的输出：已有测试照过。

## 不做

- 不改任何结果文件与 `scripts/mgr/` 下的分析脚本。缺的口径，报给管理员。
- 不画由图像生成的定性图（D28-9）。
- 不写论文正文。

## 完成总结要写

- 改了哪些文件；
- 每张新表在 Markdown 里的样子，用现有数据生成，计数可以缺；
- 测试结果原文；
- 待裁的地方。

尽量 3 小时内完成。
