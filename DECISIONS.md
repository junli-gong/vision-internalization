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

## D9 — 模型变体、revision 固定与新增 token 的约定（2026-09-30，依据 RUN-0001 的 model_notes）

1. **变体**：
   - Qwen3.5 用后训练版 `Qwen/Qwen3.5-{0.8B,2B,4B,9B}`。
   - Gemma 4 用指令版 `google/gemma-4-{E2B,E4B}-it`，不用预训练版和 `-it-assistant`（MTP 草稿头）。
   - InternVL3.5 用 `OpenGVLab/InternVL3_5-{2B,4B}-HF`：最终阶段权重（CPT+SFT+CascadeRL），transformers 原生格式，不需要 `trust_remote_code`。8B 暂不用。
   - **理由**：三家族都取"最终后训练版"，口径一致；HF 原生格式让适配器不依赖远程代码。
2. **revision 固定**为 `docs/model_notes.md` §2 登记的 sha。下载与加载一律带 `revision=`；换 revision 要新开 D 条。
3. **Qwen3.5 新增 token**：
   - 复用嵌入里超出 tokenizer 的 243 个预留行（id 248077–248319，`add_tokens` 自然落在这里）。
   - **禁止调用 `resize_token_embeddings(len(tokenizer))`**：它会把 248320 行缩成 len(tokenizer)，已实测。
   - 按简报 §5.2，新 token 的嵌入行初始化为基础词表嵌入的均值。4B/9B 的 lm_head 与嵌入不共享，也用 lm_head 基础行的均值初始化对应行。预留行原有的值不用。
4. **思考开关**：Qwen3.5 的 0.8B/2B 模板默认关闭思考，4B/9B 默认开启。简报说"默认开启"与此不完全一致。按 D1，每次调用都显式传 `enable_thinking`，并直接作为 `apply_chat_template` 的关键字参数传入；`chat_template_kwargs` 在 transformers 里不生效，已实测。
5. **内存盘上的权重可以随时删**。9B 重新下载实测约 34 秒，D6 的 25 GB 上限不变：放不下时，先删眼下不用的模型。

## D10 — DEV-0001 合入、偏离裁定与独立审查的处理（2026-09-30）

1. **合入范围**：DEV-0001 按现状合入 main，便于 DEV-0002/0003 从它开 worktree。独立审查（manager.md §7.2）查出的问题由 `prompts/dev/0001-addendum-1.md` 修复，交还原开发 CC，算第 1 轮返工：
   - 会改变门判定的 S1–S3；
   - 中等问题 M1–M4；
   - 低风险的若干条。

   **addendum-1 合入前，任何实验不得使用** `stats.is_significant`、`stats.paired_bootstrap_ci`、`metrics.detection.detection_map`、`grid.codec.parse_grid_text`、`metrics.misc.normalize_choice`。
2. **对开发列出的 9 条偏离逐条裁定**：
   - (1) NYU 有效深度上端：改为开区间 gt < 10，与 BTS/AdaBins 的实现一致。对齐官方评测优先于任务书当初写的闭区间。见 addendum-1 第 12 条。
   - (2) 框光栅化取 (lo, hi]：接受。独立审查用 22,400 个随机框与 pycocotools 对照，0 处不一致。
   - (3) 两个 mask 都空时 Boundary IoU 记 1.0：接受，与 gIoU 的空并集约定一致（官方是 0/0）。
   - (4) 解析失败记错或 IoU 记 0，检测预测为空时 AP 记 0：接受。分割预测为 None 时同样记 0（addendum-1 第 6 条）。
   - (5) 无分数检测统一记 score=1.0：取代。改为 K 次随机打破平局取平均，另报单一工作点 P/R/F1（addendum-1 第 3 条）。标题指标见 N9。
   - (6) 元数据写 meta.json：接受。
   - (7) 任务书外的补充：接受 `ratio_of_sums`、`check_grid_token_names`，以及 visint 里只供测试对照的 opencv。
   - (8) 适配器接口的几处解释：在 DEV-0003 定。层号约定、`vision_features`、token 预算跨调用累计、`raw_text`，由 DEV-0003 补进 base.py。
   - (9) `is_significant` 的逐样本取法：取代，按 addendum-1 第 2 条重写（同一统计量、分方向、三条规则）。这仍是对简报 §8.4 措辞的实现解释，G0 报告里请项目负责人确认。
3. **审查的其他发现**：
   - 预测为 None 时，分割记 0；GT 为空时仍按空并集约定处理。
   - 多个候选块里恰好只有一个是 N×N 时，仍判失败（不猜）。
   - 思考段由适配器剥离，解析器只看最终答案（DEV-0003）。

