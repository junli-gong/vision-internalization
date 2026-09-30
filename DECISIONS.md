# DECISIONS.md — 设计裁定

这里记录规格（`docs/RESEARCH_BRIEF.md` v0.2）留白或模糊处的编号裁定。效力低于用户当场指示，高于规格正文。
只追加，不插入；要修订旧条，就在末尾新增一条并注明修订哪条。待决问题在 `DECISIONS_NEEDED.md`。

## D1 — 规格版本，以及 v0.2 未声明的删减（2026-09-29，用户："按你的规划来"）

1. 规格以 `docs/RESEARCH_BRIEF.md`（v0.2，2026-09-29）为准。旧版归档为 `docs/archive/RESEARCH_BRIEF_v0.1_2026-09-27.md` 和 `docs/archive/CLAUDE_v0.1.md`。旧简报页眉误标"v0.2（2026-09-27）"，这里按 v0.2 变更摘要的叫法记为 v0.1。
2. 有些条目 v0.1 有、v0.2 删了，变更摘要却没列。按"只收紧、不放宽"处理：
   - **保留**：base 与 trained 的生成设置逐项相同，包括思考开关、采样参数、max tokens、图像分辨率和视觉 token 数（来自 v0.1 §6.3）。
   - **保留**：冒烟测试要包含"指令在前时，图像 token 的隐藏状态随指令改变"（来自 v0.1 §13 第4条）。另加一个对照："图在前、两条指令 token 长度相同时，图像 token 的隐藏状态逐位相同"，这是精确为零的仪器。
   - **保留**：Phase 0 的记录文件 `docs/env_report.md`、`docs/model_notes.md`（来自 v0.1 §13），另加 `docs/data_sources.md`。
   - **保留**：Phase 0 实测显存（v0.1 §5.4 的意图），结果写进 `env_report.md`。
   - **不做**：复现 Qwen3.5 官方"空间智能"一栏、文本通道的 16×16 粗网格深度。视为有意删除，如需恢复另开 D 条。
   - 附录B 的文献只作背景，见归档件。

**理由**：这些条目只让实验更严或记录更全，不改变任何假说、判据或通道定义。

## D2 — 三角色流程与文件对应（2026-09-29，用户确认）

1. 本项目按 `exp-triad-flow` 三角色流程执行：管理员 CC 在 Mac 上，开发 CC 与管理员共享 Mac 工作区，服务器 CC 在 b200 上。简报里"在服务器上执行实验的 Claude Code"拆成两个角色：开发 CC 写代码、服务器 CC 运行。
2. 文件以简报的命名为准，对应关系见 `CLAUDE.md` 的"文件"表。skill 的 `EXPERIMENTS.md` 并入 `RESEARCH_LOG.md`，`FINDINGS.md` 由 `RESULTS.md` 取代。"提问"与"裁定"分开记：问题写 `DECISIONS_NEEDED.md`，答复后在本文件追加。
3. `CLAUDE.md` 不用 `@` 自动导入简报，沿用 v0.2 的写法。每份任务书写明要先读哪些小节。
4. skill 副本放在 `.claude/skills/exp-triad-flow/`，因为服务器上没有全局 skill。

## D3 — 算力机布局（2026-09-29）

路径、环境、git 身份见 `CLAUDE.md` 的"机器与路径"。补充几点：

- 项目目录 `$PROJ` 和 deploy key 由管理员代建（省一轮往返），仓库级 `core.sshCommand` 与 mmvis 的做法相同。
- 共享的 `$ROOT/env`、`$ROOT/.hf`、`$ROOT/datasets` 只加不删。不属于本项目的文件一律不动。
- `$ROOT/datasets/nyuv2/splits.mat`（2.6 KB，2026-09-06 放入，来源不详）可以读，不能改。

## D4 — 代码布局与约定的单一来源（2026-09-29）

1. 目录照简报 §11.1：代码在 `src/` 下按子包组织。`src` 本身是包，import 写成 `from src.executors import raster`。
2. 跨模块约定只在 `src/constants.py` 定义，其余代码和测试一律 import，禁止重述数字。约定包括：坐标格式、网格尺寸、叠加渲染参数、深度分档、Eigen crop、Boundary IoU 膨胀比例、bootstrap 设置。
3. 测试不下载权重和数据。需要模型时，用本地配置构造小型随机模型。合成 fixture 故意 H≠W，内容和批内样本也互不相同。

## D5 — git 与远端（2026-09-29）

- 仓库 `junli-gong/vision-internalization`（私有）。本机推送走 HTTPS + gh 凭证助手；本地仓库身份 `Tim Gong <zhiyuanp2002@gmail.com>`。
- 提交前缀：`[mgr]`、`[dev] NNNN`、`[results] NNNN`。管理员逐文件 `git add`，禁止 `add -A` 和 `add .`。
- 服务器用 deploy key（读写），git 身份 `b200-server-cc`。

## D6 — 磁盘红线（过渡期，DECISIONS_NEEDED N1 答复前有效，2026-09-29）

- 根盘任何时刻剩余不得低于 3 GB。整机共用这一块盘，写满会拖垮其他租户。每次下载前先探体积。
- 本项目在根盘上的新增占用上限 3 GB。模型权重放 `/dev/shm/visint/`（内存盘，易失），上限 25 GB。
- 不删除、不移动任何非本项目文件。mmvis 等旧项目的占用是否回收，等 N1 的答复。

## D7 — Phase 0 的顺序（2026-09-29）

1. RUN-0001 和 DEV-0001 并行：
   - RUN-0001：环境、模型事实探针、数据源的体积与 schema，不落地大数据。
   - DEV-0001：与模型无关、E1/E2 要用的基础件，包括接口抽象、光栅化与几何、网格编解码、全部指标、统计、计算记账、运行记录。
2. 后续任务：
   - DEV-0002：E3 用的执行器（叠加渲染、分块拼接、无学习上采样、不确定度选区）和 mask 加噪过程。
   - DEV-0003 起：各家族适配器，等 RUN-0001 报回真实 schema 再写，不对假设的格式编码。
   - E5 合成数据生成器：单独任务，可与 Phase 0 并行。
3. G0 之前不跑任何训练。显存实测里的单步前向加反向不算训练。

## D8 — 改用模式 B：派发 + 值守（2026-09-29，用户确认）

1. **开发**：
   - DEV-0002 起，管理员用 Agent 工具派发后台开发 agent，每个任务一个 git worktree（在 `.claude/worktrees/`，已 gitignore）。审查、返工（最多两轮）、合入按 skill 的 manager.md §7。
   - 决策门判据会用到的仪器代码，另派独立审查 agent。
   - DEV-0001 由原开发 CC 会话（cvpr2027-48）完成，完成总结经跨会话消息直接交给管理员。该会话此后退役。
2. **服务器**：
   - RUN-0001 完成后，服务器 CC 按 `prompts/server/STANDBY.md` 进入值守，从 origin/main 自动接 `prompts/server/NNNN-*.md`。
   - 心跳写 `$PROJ/logs/standby.status`，完成清单记在 `$PROJ/logs/server_tasks_done.txt`。
   - 暂停：在仓库根提交 `HOLD`。
3. **管理员**：
   - 发出 run 任务书后，在后台挂 `wait_for_results.sh`：每 3 分钟 fetch 一次，6 小时重挂，同时只读查看服务器心跳。
   - 开发机用 `caffeinate -ims` 保持不休眠；管理员会话保持打开。
4. **自主范围**按 manager.md §7.4。以下情况一律停下，写 DECISIONS_NEEDED 并推送通知：
   - 简报 §10 的请示事项，以及决策门；
   - 删除非本项目文件；
   - 两轮返工仍不过；
   - 同一运行因代码 bug 失败两次；
   - 服务器值守失联。

**理由**：管理员与开发同机，人工转交是瓶颈；用户要求"不在电脑前实验也不断"。独立审查仍然成立：写代码的 agent 和审查的管理员是两个上下文，仪器代码还多一道独立审查。
