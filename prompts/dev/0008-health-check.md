[DEV-TASK 0008] 健康检查框架（简报 §8.11）：固定 200 题 + 三个评分器 + 运行脚本

## 背景

简报 §8.11 要求：每个门都在合并后的模型上跑一个约 200 题的固定集合（MMLU-Pro 100、GSM8K 50、IFEval 50），同时报告基座在同一集合上的分数；任何一项下降超过 5 个点，在门报告里标红。

本任务把这个集合固定下来，并写评分与运行脚本。基座分数由之后的一个 RUN 在服务器上测。

题目挑选与协议写进 D 条目，在看到任何分数之前固定。

先读：
- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 简报 §4 第 5、9 条，§8.11
- `DECISIONS.md` 的 D10、D19、D22（解析层）
- 已合入的 `src/parsing/`、`src/adapters/base.py`、`src/accounting.py`、`src/runlog.py`、`scripts/smoke_adapter.py`（参考它的 `--tiny` 与家族分派写法）

## 任务

### 1. 固定题目（`configs/health_check_items.json`，入库）

- **来源与许可**：
  - MMLU-Pro：`TIGER-Lab/MMLU-Pro` 的 test 划分，MIT；
  - GSM8K：`openai/gsm8k` 的 main/test，MIT；
  - IFEval：`google/IFEval` 的 train（共 541 条），Apache-2.0。

  每个来源都记下 HF 数据集的 commit hash、划分和许可。
- **抽样**，种子 20260930：
  - MMLU-Pro 按 14 个类别分层，每类 7 题，共 98 题；再从剩余题目里均匀抽 2 题，凑满 100。
  - GSM8K 均匀抽 50 题。
  - IFEval 均匀抽 50 题。
- JSON 里存每题的原始 id 与全部评分所需字段（题干、选项、答案、IFEval 的 instruction_id_list 与 kwargs），**服务器不必再下载数据集**。
- 挑选脚本入库：`scripts/build_health_items.py`，可以复现同一份 JSON。
- 数据用 `huggingface_hub` 下载到你的 scratchpad，不要用 `datasets`，不入库原始文件。

### 2. 协议（写进模块 docstring；我会据此写 D 条目）

- **输入**：纯文本，走适配器 `build_inputs([], prompt, "instruction_first")`。
  - 先确认三家族都支持零张图；不支持就写进偏离，不要改适配器。
- **生成**：`enable_thinking=False`，greedy，每次显式设置。
- **MMLU-Pro**：
  - 提示：题干，然后每行一个选项（"A. …"），最后一句 "Answer with the letter of the correct option only."
  - `max_new_tokens=32`。
  - 解析用 `src.parsing.parse_choice`，letters 取该题的实际选项字母（最多到 J）。
- **GSM8K**：
  - 提示：题干，加一句 "Solve the problem step by step. On the last line, write the final answer as '#### <number>'."
  - `max_new_tokens=512`。
  - 解析分两种口径：
    - strict：取最后一个 "####" 之后的数；
    - flexible：取全文最后一个数。
  - 两种口径都去掉千分位逗号，数值与答案相等（整数比较）即对。
  - 标题口径用 flexible，两种都报告。
  - 这两个解析函数放在 `src/health/`，不改 `src/parsing/`。
- **IFEval**：
  - 提示原样。`max_new_tokens=1024`。
  - 评分用官方 checker：google-research `instruction_following_eval` 的 `instructions.py`、`instructions_registry.py`、`instructions_util.py`、`evaluation_lib.py` 的评分部分。
    - 固定 commit，原样放进 `src/health/ifeval/`，文件头保留 Apache-2.0 声明与出处；
    - 只允许做 import 路径上的最小改动，并在文件头注明改了什么。
  - 标题口径：prompt-level strict accuracy。另报 instruction-level strict 与 loose。
- **计分**：三项都以百分数报告，保留一位小数。附 `ParseTally`（MMLU-Pro 与 GSM8K）和逐题记录。
- **比较**：`compare(base_json, merged_json)` 输出每项的差值（百分点），下降超过 5 个点的项标出，供门报告使用。

### 3. 脚本 `scripts/health_check.py`

- 参数：`--model <注册键> --device --dtype --out <json>`，另有 `--tiny`（CPU 小模型彩排，数值无意义，只验证全流程）和 `--limit N`（只跑前 N 题，调试用）。
- 输出 JSON 包含：
  - 运行记录（`src/runlog.py`：config、种子、git commit、题目文件的 sha256）；
  - 每题的原始输出与解析结果；
  - 三项分数、ParseTally、计算记账（`src/accounting.py`）。

### 4. 依赖

- 只允许新增 IFEval checker 需要的包：`langdetect`、`nltk`、`immutabledict`，以及官方代码 import 到的 `absl-py`。尽量避开，避不开就写进偏离。
- 写进 `requirements.txt` 并注明版本下限。
- nltk 的 punkt 数据：
  - 文档里写明用 `NLTK_DATA` 指到 `$PROJ` 下的目录；
  - 测试里没有数据时要么跳过并给出原因，要么用仓库里的小 fixture，不要去写全局的 `~/nltk_data`。

### 5. 测试（`tests/test_health*.py`）

- 题目文件：数量、分层、字段完整；用 JSON 重新挑一次，结果逐字相同（不联网：比较入库的 JSON 与脚本对 fixture 的输出；或者只检查 JSON 的结构与哈希。择一，写进偏离）。
- GSM8K 两种解析：逗号、负数、多个 "####"、没有数等情况。
- MMLU-Pro 用 parse_choice，字母上限按选项数截断。
- IFEval 评分：用官方仓库自带的样例输入输出，对照官方期望值至少 10 条。
- `compare`：标红规则。
- `--tiny` 端到端：三家族各跑 `--limit 3`，退出码 0。

## 允许改动的文件

- 新建：`src/health/`、`configs/health_check_items.json`、`scripts/build_health_items.py`、`scripts/health_check.py`、`tests/test_health*.py`，以及必要的小 fixture（放在 `tests/fixtures/health/`）
- `requirements.txt`：只加上面列出的包

**不许碰**：`src/adapters/`、`src/parsing/`、`src/metrics/`、`src/synth/`（另一个 agent 在改）、`scripts/smoke_adapter.py`。

## 环境与验收

- conda 环境 `visint`，可以 pip 安装上面列出的包。不需要 GPU，不下载权重。
- 验收：
  - `conda run -n visint pytest tests/ -q` 全绿，报告新增测试数与耗时；
  - 三条 `--tiny --limit 3` 冒烟的退出码与输出摘要。
- 总结里逐条列出偏离，至少包括：
  - IFEval 官方代码的 commit 与改动；
  - 三家族零张图输入是否都支持；
  - 题目文件的复现方式；
  - 你对协议的意见。协议固定后不再改，现在是唯一的机会。
