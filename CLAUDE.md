# 项目说明（给Claude Code）· v0.2

完整简报在 docs/RESEARCH_BRIEF.md。
- 开始任何任务前先读它。如果你已按v0.1开始工作，先读简报开头的"v0.2变更摘要"。
- 进入新阶段前，重读简报第2、3.1、4、8、10节。

## 方向

- **北极星是"心智"：** 让LLM能在内在的视觉-空间模型里预演世界。
- **第一个里程碑：** 把视觉内化进LLM本身。
- **CV任务的角色：** 训练场和度量尺，不是排行榜。我们不和接了专家模型的系统（如LISA、GLaMM）比CV分数。

## 最重要的规则（完整版见简报第4节和第10节）

- **能力必须长在LLM权重里。** 我们的方法中不使用任何学习型CV专家模型：工具、教师、伪标签、奖励模型都不行。唯一例外是经确认的LISA式对照系统，它单独运行，其输出绝不进入我们的数据或奖励。
- **"LLM能力"以T通道为准。** T通道包括文本/代码和像素token网格，结论都在合并后的单一权重上评测。
- **E3a的稠密解码器是训练仪器。** 它要够薄（默认不超过50M参数），跨任务共享，只读LLM隐藏状态，看不到像素。
- **E3b的像素token通道只允许新增至多300个token。** 不加额外解码器。任何架构改动都要先批准。
- **多轮流程必须做计算记账。** 叠加反馈、分块、边界细化这些流程，要在看到测试结果之前固定下来；单轮结果是主要数字，多轮结果附计算量，并与匹配计算量的单轮对比。
- **E5必须报告平凡基线。** 每类任务都要报告对应的平凡基线（原样复制、二维变换等）。
- **代码与模型无关。** 模型相关的逻辑放在 src/adapters/，不硬编码层号。

## 流程

- 每个决策门都要提交门报告。未经项目负责人确认，不进入下一阶段。
- 需要决策时写入 DECISIONS_NEEDED.md，然后继续做不受影响的工作。
- 主实验前写 PREREGISTRATION.md；每次工作都更新 RESEARCH_LOG.md。
- 预计超过12小时的任务，先估算，再请求批准。
- 不确定是否违反约束时，按违反处理，先问。

---

以下由管理员追加（2026-09-29，见 DECISIONS.md D2、D3）。以上是项目负责人给的 v0.2 原文，未改动。

## 三角色流程（必须遵守）

本仓库由三个 Claude Code 实例协作，完整流程见 skill `exp-triad-flow`（全局 `~/.claude/skills/exp-triad-flow/`；仓库内副本 `.claude/skills/exp-triad-flow/`，服务器没有全局 skill，用副本）。任何开发、运行、git 操作之前先读它。

角色由本对话第一条用户消息判定：

- 以 `[DEV-TASK` 开头 → 你是**开发 CC**：只做任务书里的事，不 commit、不 push。
- 以 `[RUN-TASK` 开头 → 你是**服务器 CC**：校验 hash、运行、结果推回，绝不改实验逻辑代码。
- 其他 → 你是**管理员 CC**：规划、写任务书、审查、git、解读，不写功能代码。

简报里"写入 DECISIONS_NEEDED.md"的动作由管理员执行：开发 CC 和服务器 CC 把问题写进完成总结或 report，由管理员汇总。用户答复后，在 DECISIONS.md 末尾追加 D 编号。优先级：用户当场指示 > DECISIONS.md > 简报正文。

## 文件

| 文件 | 作用 | 谁写 |
|---|---|---|
| `docs/RESEARCH_BRIEF.md` | 规格 v0.2，唯一来源 | 项目负责人 |
| `DECISIONS_NEEDED.md` | 待项目负责人决定的问题（简报 11.3 格式） | 管理员 |
| `DECISIONS.md` | 编号裁定 D1、D2…，只追加不插入 | 管理员 |
| `RESEARCH_LOG.md` | 日志兼台账（DEV/RUN 编号） | 管理员 |
| `PREREGISTRATION.md` / `RESULTS.md` | 事前预测 / 过门的结论 | 管理员 |
| `prompts/dev/`、`prompts/server/` | 任务书存档 | 管理员 |
| `reports/NNNN-*.md` | 服务器运行报告，只贴数字不解读 | 服务器 CC |
| `reports/gates/G*.md` | 门报告（简报 11.4 模板） | 管理员 |
| `experiments/<phase>/<NNNN_slug>/` | 运行产物，目录名必须带运行编号 | 服务器 CC |
| `docs/env_report.md`、`docs/model_notes.md`、`docs/data_sources.md` | Phase 0 核实记录 | 服务器 CC 写，管理员审 |

## 机器与路径

- 开发机（Mac，Apple Silicon，无 CUDA）：conda env `visint`（DEV-0001 建）；测试 `conda run -n visint pytest tests/ -q`。不往用户其他 conda 环境装东西。
- 算力机：`ssh b200`，root 多人共用。**不改任何全局文件**（`~/.ssh/config`、`~/.gitconfig`、`/root/.cache` 都是别人的）。
  - `ROOT=/home/ubuntu/workspace/wzg/gjl`，`PROJ=$ROOT/vision-internalization`，`REPO=$PROJ/repo`；日志 `$PROJ/logs`，探针脚本 `$PROJ/scripts`，临时文件 `$PROJ/tmp`。
  - venv `$PROJ/venv`（`--system-site-packages` 继承 `$ROOT/env`：py3.12、torch 2.13.0+cu130，不重装）。
  - `HF_HOME=$ROOT/.hf`（含 token）；模型权重暂放内存盘 `HF_HUB_CACHE=/dev/shm/visint/hf-hub`；pip 一律不留缓存。
  - 仓库级 git：`core.sshCommand` 用 `$PROJ/.ssh/id_ed25519_github`（deploy key，读写），身份 `b200-server-cc`。
  - **磁盘红线（D6）：** 整机只有一块 500 GB 盘，2026-09-29 只剩 6.1 GB。任何时刻 `df /` 剩余低于 3 GB 立即停止。`/dev/shm` 是 91 GB 内存盘，重启即丢。
- 启动脚本一律自己 export `HF_HOME`、`HF_HUB_CACHE`、`TMPDIR`、`PIP_NO_CACHE_DIR=1`，不依赖交互式 shell。
