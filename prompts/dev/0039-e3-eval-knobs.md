[DEV-TASK 0039] E3 评测的两处补充：E3b 评测只跑单轮的开关；冻结对照的 T 文本抽查比对工具

## 背景

- D72-4 (c)：多轮与匹配计算量的评测，每个尺寸只做 1 个种子；单轮的主读数保留全部种子。
  - 现在 `scripts/eval_e3b.py` 每个检查点都按配置的 `eval.multi_turn.k_values`（1、2、4）跑多轮，再跑匹配计算量的单轮，没有只跑单轮的办法。改配置会改变 E3b 配置的 sha256，不行。
- D72-4：冻结 LLM 的对照（C1、C2，以及屏蔽网格 token 之后的 C1′）不重跑 T 文本评测，直接引用基座的读数，抽查几百条确认相同。
  - 抽查的运行用现有的 `run_e1 --checkpoint … --limit N`（C1′ 加 `--mask-grid-tokens`）。
  - 缺一个工具：把抽查的输出与基座的主评测（RUN-0027，`experiments/E1/0027_e1_main/<模型>/<任务>/`）逐条对上并给出判定。
- 先读：
  - `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
  - `DECISIONS.md` 的 D36-7、D43、D65、D72、D77
  - `PREREGISTRATION.md` 的 E3（已批准 v1，§4 评测）
  - `scripts/eval_e3b.py`、`src/e3/evaluate.py`（`settings` 的构造与 multi_turn、matched_compute 两段）、`configs/e3b_train.json` 的 eval 节
  - `scripts/run_e1.py`（`--checkpoint`、`--limit`、`--mask-grid-tokens`）、`src/e1/`；RUN-0027 的输出目录结构（`experiments/E1/0027_e1_main/qwen3.5-2b/`）

## 任务

### 1. `eval_e3b.py --k-values`（`scripts/eval_e3b.py`、`src/e3/evaluate.py`）

- 新选项 `--k-values K [K …]`：覆盖配置的 `eval.multi_turn.k_values`。
  - 必须是配置那一列的前缀，从 1 开始，例如 `1` 或 `1 2`；否则报用法错误。
  - 匹配计算量的单轮随之只做这些 K。`--k-values 1` 时不做任何采样。
- 不给时行为不变，输出逐字节相同。
- 运行身份（`config.yaml`）记下 `k_values` 与"是否覆盖"。`metrics.json` 的结构不变，只是 by_K 里的 K 变少。
- `--resume` 时，已有的输出若是用另一组 K 跑的，拒绝混写，与现有的运行身份核对规则相同。
- **测试**：
  - tiny 检查点上，`--k-values 1` 的单轮输出（predictions 与单轮指标）与不给时的单轮部分逐字节相同；
  - 不是前缀的取值被拒；
  - 不给时与改动前逐字节相同，用现有的测试或 fixture；
  - `--resume` 换了一组 K 时被拒。

### 2. 冻结对照的 T 文本抽查比对（`scripts/compare_ttext_spotcheck.py`，新）

- 输入：
  - 基座的主评测目录：`<RUN-0027 输出>/<模型>`；
  - 抽查运行的目录：`run_e1 --checkpoint … --checkpoint-label … --limit N` 的输出，`<out>/<标签>/<模型>`；
  - 任务列表，默认两边都有的任务。
- 先核对：两边的模型、提示配置 sha256、图像口径、批配置相同；抽查的条目是基座评测清单的前 N 条，id 逐条相同。不同就报错退出。
- 逐任务输出：
  - 比对条数；
  - 原始输出文本逐条相同的比例、解析结果相同的比例、逐条得分相同的比例；
  - 两边在这 N 条上的任务指标，以及配对差。
- 判定（写进脚本与说明）：
  - "相同"：所有任务的原始输出逐条相同；
  - 否则"不完全相同"：列出不同的条目 id（每个任务最多 20 条），再给解析结果的一致率。管理员据此决定能不能引用基座的读数。
  - 说明里写清楚为什么 N 取批大小（64）的整数倍：批按样本顺序连续组成，前 N 条的批组成才与全量运行相同（D36-7、D43）。
- 输出写成 json 加 md，路径由 `--out` 给出。
- **测试**：
  - 同一份输出与自己比，判"相同"；
  - 改一条原始输出，判"不完全相同"并列出那条 id；
  - id 对不上时报错；
  - 提示配置的 sha256 不同时报错。
  - 用仓库里已有的 E1 fixture 或小的合成目录，不读 `experiments/` 的真实大文件。

### 3. 彩排

在开发机上用 tiny 检查点，把下面几条各空跑一遍，写进完成总结：

- `train_e3a.py --tiny --variant c1` 之后 `run_e1 --tiny --checkpoint <c1 的 final/model> --limit 4`；
- `train_e3b.py --tiny --supervision grid --variant c1prime` 之后 `run_e1 --tiny --checkpoint … --mask-grid-tokens --limit 4`。

这两条确认冻结对照的检查点能直接给 `run_e1 --checkpoint`，合并核对对没有 LoRA 的检查点怎么处理。不能处理就在总结里写清楚，不要改 `src/e3/merge_check`，那是禁区。

## 允许改动的文件

- `scripts/eval_e3b.py`、`src/e3/evaluate.py`：只加选项，不改已有行为
- `scripts/compare_ttext_spotcheck.py`（新）
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/e1/`、`src/e3/` 除 `evaluate.py` 以外的文件、`src/readout/`、`src/e5/`、`src/synth/`、所有 `configs/*.json`、`scripts/run_e1.py`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。临时输出放系统临时目录，用完即删。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 第 3 节的两条彩排
- 总结里写明：
  - `--k-values` 的做法与逐字节核对的结果；
  - 比对工具的判定规则；
  - 冻结对照的检查点给 `run_e1 --checkpoint` 时的实际行为（合并核对、加载）；
  - 所有偏离。
