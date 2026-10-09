[DEV-TASK 0044] E3a+C3 混合训练（2B 用；读出损失 + C3 文本的语言损失，文本样本流与 C3 逐槽位相同）

## 背景（D83、D84、D85）

- E3a 只用读出损失训练 LoRA，冲掉了 T 文本：自由生成退化成 " a a a…"。
- 负责人要 **E3a+C3**，只在 2B 上做，3 个种子，用于 G2 时写下的机制比较：E3b 的跨通道迁移是否大于 E3a（E3 预注册 §10a、§12）。
- 做法与 E3b+C3 对称（DEV-0042 已实现 E3b+C3，见 D85 与它的代码）：
  - 每步 8 条稠密样本 = E3a full 同种子的样本流第 8t … 8t + 7 个槽位，四个任务照 E3a 的混合；
  - 每步 8 条文本样本 = C3 同种子的样本流第 8t … 8t + 7 个槽位，与 DEV-0042 的文本流同一个函数；
  - 8,000 步；
  - 损失 = 读出损失（定义同 E3a，各任务的权重同 E3a）+ 文本样本的逐 token 平均语言损失（同 C3），1:1，事先固定；
  - 学习率日程按 8,000 步排，写明是 E3a、C3 的 2 倍长。
- E3a 不看指令（D84：代码上不是 bug，是学习问题）。**本任务不改 E3a 的读出结构**。给 refseg 头加文本侧属于架构改动，没有批准。
- 先读：
  - `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
  - `DECISIONS.md` 的 D71、D83、D84、D85
  - `PREREGISTRATION.md` 的 E3 §2、§12
  - `src/readout/`（config、data 的 Schedule、train）、`scripts/train_e3a.py`、`configs/e3a_train.json`
  - DEV-0042 的实现：`src/e3/data.py` 的 `MixedSchedule`、`stream_of` 与 `ExampleBuilder` 的 grid+text，`src/e3/train.py` 混合训练的微批与损失，`tests/test_e3_mixed.py`

## 任务

### 1. E3a+C3（`scripts/train_e3a.py --variant full --with-text`，或你认为更清楚的开关；输出变体名 `e3a_c3`）

- **样本流**：照背景里的定义。文本样本的构造直接调用 `src/e3` 的 C3 构造（只读复用），不另写。
  - 稠密样本照 E3a 的规则：语义分割的样本单独成微批，类别名区间只在语义分割的行上取。
  - 同一步里同一张图可能以两种监督各出现一次，说明里写明。
- **微批**：每个微批只含一种监督。组成进运行身份。
- **损失**：读出损失照 E3a 原样，加 C3 的语言损失。两者各自的归一化与单独训练时相同，相加 1:1。训练日志分别记两者。
- **LoRA 与查询**：
  - LoRA 照 E3a；查询、逐层混合、解码器照 E3a 原样训练。
  - 文本样本不插查询 token，与 C3 相同。
  - 文本样本只经过语言损失，不经过读出。
- **配置**：`configs/e3a_train.json` 新增一节，记两种样本的条数、损失权重、步数的来历。版本号与 sha256 随之改变。
  - **已有的 E3a full、C1、C2 的运行身份，除配置的 sha256 与版本号外逐项不变**，测试钉住，做法同 D85。
- **检查点**：与 E3a full 同一种（`final/model` 加 `final/readout`），能直接给 `eval_e3a`，以及 `run_e1 --checkpoint`（给 `final/model`）。
- **测试**：
  - **逐样本相同**：前若干步里，文本样本与 C3 同种子对应槽位的逐条相同；稠密样本与 E3a full 同种子的逐条相同。
  - 损失是两者之和、1:1，用合成的小例子核对数值。
  - 续跑逐字节相同。
  - 已有变体的运行身份不变，例外只有配置的 sha256 与版本号。
  - 新旧代码对比：已有变体在 tiny 下的训练日志与最终检查点逐字节相同，做法同 D85。

## 允许改动的文件

- `src/readout/`（config、data、train，以及确有需要的其余文件）、`scripts/train_e3a.py`、`configs/e3a_train.json`
- 对应的测试与 fixture

**不许碰**：`src/e3/`（只读复用）、`src/adapters/`、`src/e1/`、`src/e5/`、`src/synth/`、`src/grid/`、`configs/e1_*.json`、`configs/e3b_train.json`、`configs/e5_*.json`、所有评测脚本。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 两套全量测试并行时不要同时用两个 `conda run`。直接用 `/opt/miniconda3/envs/visint/bin/python -m pytest …` 与 `PATH=/opt/miniconda3/envs/visint/bin:$PATH PYTHON=/opt/miniconda3/envs/visint/bin/python bash scripts/pytest_server_parity.sh …`，各写各的日志。计时类测试（`tests/test_synth_*::test_speed_*`）在 CPU 被占满时可能失败；只有这类失败时，单独重跑确认，并写进总结。
- 验收：
  - 两套全量测试，在最后一次改动之后各一遍；
  - tiny 彩排：`train_e3a --tiny` 的 E3a+C3，然后 `eval_e3a --tiny` 与 `run_e1 --tiny --checkpoint <final/model> --limit 4`。
- 总结里写明：
  - 样本流与逐样本核对的结果；
  - 微批的组成；
  - 损失与学习率日程；
  - 新配置的 sha256；
  - 所有偏离。
