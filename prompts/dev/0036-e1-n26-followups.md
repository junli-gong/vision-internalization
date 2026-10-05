[DEV-TASK 0036] E1 事后读数的后续（N26 → D72-2/3）：NYU 点对"对调标记"去偏、BLINK 标题改自由作答、结构强制的多边形（含"至少 8 个点"变体与近矩形占比）、"模型自己的框"基线；另加 C1′ 的 T 文本解码屏蔽网格 token

## 背景

- RUN-0030 发现：NYU 点对与 BLINK 的选项文字指向图上的 A、B 标记，循环移位去偏会误罚"按标记作答"的模型（N26）。
- 负责人 D72-2：NYU 改为对调图上的标记，BLINK 的标题改用自由作答。
- D72-3：多边形补一个结构强制读数，加"至少 8 个点"的变体，报告近矩形占比；最要紧的是加"模型自己的框"基线：同一句话、同一个模型在 E1 给出的框，光栅化后算 cIoU（多边形）与 gIoU（16×16 网格）。
- D72-4：C1′ 的 T 文本评测在解码时屏蔽网格 token。
- 这些都是事后的，标为事后，不改 G1 的结论。E1 主评测的输出与 E1 配置 sha256（`f3353de0…`）不能变。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `PREREGISTRATION.md` 的 E1 §10；`DECISIONS.md` 的 D66、D68、D70、D72；`reports/0030-e1-posthoc.md`
- `src/e1/`（debiased、forced、posthoc、posthoc_runner、tasks）、`scripts/run_e1.py`、`scripts/summarize_e1_posthoc.py`、`configs/e1_posthoc.json`
- NYU 点对的构造与渲染：找出图上 A、B 标记是在哪里画的，`src/data/` 或 `src/e1/`

## 任务

### 1. NYU 点对的"对调标记"去偏（新读数，例如 `letter_swap`）

- 每题渲染两张图：原图，以及两个点的 A、B 标记互换的图。选项文字不变，仍是 "(A) A is closer / (B) B is closer"。
- 每张图读一次受限字母读数，两种写法取大，同 D36-4。把字母概率映射到**物理点**：原图里 A 是点 1，换标记的图里 A 是点 2。两张图按物理点取平均，最大者为答案。
- **核对**：
  - 原图那一遍与 RUN-0027 的字母读数逐条一致，同 D68 的 s = 0 核对；
  - 换标记的图除标记文字外逐像素相同，测试钉住。
- 全量 3,270 条。输出与报告格式同 debiased，另记两张图各自的准确率与"最常见字母占比"。
- BLINK 的标记画在原图里，不做。

### 2. 标题口径的汇总改动

- `summarize_e1_posthoc.py` 的 validity 汇总按 D72-2 出新的标题表：
  - NYU 点对用对调标记的去偏读数；
  - BLINK 用自由作答（RUN-0027 的 free，112 条全量）；
  - CV-Bench、EmbSpatial 仍用循环移位的去偏读数。
- 有效性检查（§10.1 的两条规则）照旧逐格给出，NYU 改用新读数。
- 顺手修：汇总脚本把 `probe_scripts` 等非模型目录当成模型（RUN-0030 报告里多出"缺"的行）。

### 3. 结构强制的多边形（Qwen 四个尺寸，200 句，与 refexp_poly 思考子集同一批 id）

- **语法**：用逐位置的 logits 处理器，与网格的结构强制同理（`src/e1/forced.py`）。
  - 输出必须是 `[[x, y], [x, y], …]`：每个点恰好两个 0–1000 的整数；点与点之间是 `], [`，结尾是 `]]`，再接停止 token。
  - 点数由模型决定，上限写明，例如 64。
  - 数字的 token 化按本家族分词器处理：Qwen 的数字是逐位的，写明做法。
- **两个变体**：
  - `poly_structured`：至少 3 个点；
  - `poly_structured_min8`：至少 8 个点。第 8 个点之前不允许结束符。
- **报告**：严格读数（cIoU、gIoU）、有效率、条件读数；**近矩形占比**：恰好 4 个点，且多边形面积与它外接框面积之比 ≥ 0.95。阈值写进配置。
- 解析与光栅化同 E1。

### 4. "模型自己的框"基线（不需要新的模型运行）

- 对 refexp_poly 与 grid16 的同一批 id（思考子集的 id），找到同一句话、同一个模型在 E1 主评测（RUN-0027 的 refexp_box eval）里给出的框。
  - 句子的 id 要能对上，核对方式写明；对不上的条数照实报。
  - 框光栅化成 mask，规则同 E1：
    - 多边形对照：算 cIoU 与 gIoU；
    - 网格对照：把框填进 16×16 网格，算 gIoU 与 cIoU。
- 再做一份开思考的版本：RUN-0028 refexp_box 思考子集里同一句话的框。只在 id 有交集时做，交集的条数照报。
- **汇总表**：对每个 Qwen 尺寸，把下面这些放在同一张表里，同一批 id，配对差与 bootstrap CI：
  - 多边形：主评测、思考、预填、结构强制、结构强制 ≥ 8 点；
  - 网格：主评测、思考、受限、结构强制；
  - "自己的框"基线。

  其余 4 个模型也给出"自己的框"基线，它们没有格式强制。

### 5. C1′ 的 T 文本评测：解码时屏蔽网格 token（D72-4）

- `run_e1.py --checkpoint … --mask-grid-tokens`：生成时把检查点登记的网格 token 的 logits 置为 −inf，其余不变。
  - 适用于自由生成，受限字母读数本来就只在字母上取；
  - 运行身份记下这个选项。
- **测试**：带网格 token 的小检查点上，加了这个选项的输出里没有网格 token；不带网格 token 的检查点与 base 运行，输出逐字节不变。

### 6. 顺手

- `src/e1/forced.py` 的说明改为"被 check_training_interfaces import"（D70-5）。
- `run_e1 --task all --subset free` 自动展开为四个字母读数任务（D70-5）。

### 7. 运行估算

总结里给出 B200 用时：NYU 对调标记 8 个模型，加多边形两个变体 × Qwen 四个尺寸。按 RUN-0030 的实测推算。

## 允许改动的文件

- `src/e1/`、`scripts/run_e1.py`、`scripts/summarize_e1_posthoc.py`、`configs/e1_posthoc.json`（只能新增条目，已有条目不动；sha256 会变，写进总结）
- NYU 点对的渲染代码：只为加"互换标记"的选项，默认行为不变，测试钉住
- 对应的测试与 fixture

**不许碰**：`configs/e1_prompts.json`、`src/adapters/`、`src/e2/`、`src/e3/`、`src/e5/`、`src/readout/`、`src/synth/`。另有 DEV-0037 在并行改 `src/e5/`，不要动。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 开发机磁盘紧张，剩约 10 GB：临时输出放 pytest 的 tmp 或 worktree 里，用完即删。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 新读数的 `--tiny`
  - E1 主评测 `--tiny` 的输出逐字节不变
- 总结里写明：
  - 互换标记的渲染与核对；
  - 多边形语法的实现与 token 化；
  - 近矩形的定义；
  - "自己的框"如何对上 id 与对不上的条数；
  - 新配置的 sha256；
  - B200 用时；
  - 所有偏离。
