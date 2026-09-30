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

## D11 — 数据划分、去重与评测集的约定（2026-09-30，依据 RUN-0001 §6 与 report §4）

1. **RefCOCO 家族的训练图像** = RefCOCO(unc)、RefCOCO+(unc)、RefCOCOg(umd) 三者 train 划分的图像并集，减去三者任一评测划分里出现过的全部图像（按 COCO image_id）。评测划分指 unc 的 val/testA/testB 和 umd 的 val/test。
   - 实测交叉：umd train ∩ unc 评测 = 1,860 张，unc train ∩ umd 评测 = 1,891 张。
   - 评测一律用各自的官方评测划分。google 划分不用，它内部的 train/val/test 有图像重叠。
2. **NYUv2**：
   - 评测用官方 654（`splits.mat` 的 testNdxs）。Eigen crop 按 BTS 的写法：0 起、半开区间 [45:471, 41:601]，即 426×560，与 DEV-0001 的实现一致；MATLAB toolbox 的 427×561 写法不用。
   - 深度训练集用 BTS 清单（24,231 对），按场景过滤：基础场景名（去掉末尾字母后缀）属于官方 test 场景集合的帧全部去掉，至少包括 bookstore_0001d–j 的 1,383 帧。过滤前后的帧数写进去重报告。
   - 法向 GT 的来源在数据落地任务里选定，要求训练与测试同源、覆盖 654 测试集。选定之前不做法向训练。
3. **BLINK 相对深度**：只用 val 的 124 条（test 答案隐藏，官方评测渠道已于 2026-01-31 关闭）。只用于评测，不用于任何调参，报告注明 n=124。
4. **CountBench**：用 HF `nielsr/countbench` 的打包版（491 张有图），不抓原始 URL（22.8% 已失效）。报告注明 491/540。
5. **COCO val2017 开放词汇检测子集**：固定种子抽 500 张，排除 RUN-0001 探针用过的 000000039769 与 000000000139。
6. **只有测试划分的基准**（CV-Bench、EmbSpatial-Bench、BLINK val、CountBench）：各自用固定种子切出一个小开发子集（每个 50 条或 10%，取较小者），只用于解析器与输出格式的开发。这些样本从报告的数字里剔除，报告写明剩余的 n。
7. **去重**：数据落地后，按 image_id 与感知哈希在所有训练集与所有评测集之间去重（硬约束 7），并出去重报告。第 1、2 条先行处理。已知 CV-Bench 含 ADE20K val 633 张、COCO val2017 805 张、NYUv2 官方 test 36 张；这些都不在我们的训练集里，报告中照样核对。
8. **修订 D9-3**（新事实：Qwen3.5 只有 243 个预留行，而硬约束 4 允许最多 300 个新 token）：
   - 累计新增不超过 243 个时，复用预留行，不 resize；
   - 超过 243 个时，允许把嵌入（lm_head 不共享时也一起）扩到恰好需要的行数，新行用精确均值初始化，已有行逐位不变；
   - 任何时候不许缩表。已补发给 DEV-0003。

## D12 — DEV-0001 addendum-1 合入、偏离裁定与复核发现的处理（2026-09-30）

1. **合入**：addendum-1 合入。独立审查复核（manager.md §7.2）确认，原来的 S1–S3、M1–M4 全部修好：
   - 核心数值对照在改动后仍然逐位一致或只有浮点级误差；
   - 原先存活的 8 个变异全部被抓住。

   **解禁** `stats.is_significant`、`stats.paired_bootstrap_ci`、`metrics.detection.detection_map`。
2. **复核发现修复新引入的问题**，交给 addendum-2（第 2 轮返工，也是最后一轮）：
   - N1：`intersect_and_union` 的默认值使旧写法 reduce 两次、两类用法忘传参就算错；
   - N2：多选题的保守规则单向压低准确率；
   - N3：检测 bootstrap 静默丢弃预测；
   - N4：网格解析会被说明行误判；
   - N7：`is_significant` 接受转置的输入；
   - N8：检测缺配对显著性路径。

   **addendum-2 合入前，下列函数继续禁用**：`intersect_and_union` 与 `mean_iou`、`normalize_choice` 与 `multiple_choice_accuracy`、`parse_grid_text`、`detection_bootstrap_statistic`。
3. **对 addendum-1 的 9 条偏离的裁定**：
   - (1) 无分数 AP 的平局打破改为成对（antithetic）抽取：接受。复核证实它无偏，K=20 时标准差 0.0007，独立置换是 0.0023。调用方要按规范顺序传入预测，见 addendum-2 第 7 条。
   - (2) `intersect_and_union` 默认 `reduce_zero=True`：推翻，改为必填（N1）。
   - (3) `is_significant` 的输入约定：接受，另加防转置守卫（N7）。
   - (4) 检测工作点用最大基数匹配、crowd 按 COCOeval 口径忽略：接受。复核在 300 张随机图上（含 crowd 与重复框）得到的 TP/FP/nGT，与 COCOeval 在 IoU 0.5 下的计数完全一致。只有一个框同时覆盖多个同类 GT 时才与贪心不同，这时给出的是贪心的上界。
   - (5) 只含污染行的块判 ILLEGAL_CHAR：修改为"多数字符规则 + 候选块规则"（N4）。
   - (6) 多选题保守规则：推翻，收窄为括号形式、列表式 `X)`、以及 "answer is X"（N2）。报告要分系统列出 None 率。
   - (7) `as_binary_mask` 放在 `grid/codec.py`：接受。
   - (8) 记账的两种时间口径：接受。
   - (9) 法向的 `valid` 必须是 bool：接受。
4. **只改文档的两条**：
   - N5：检测 bootstrap 的点估计用 `n_tiebreaks`（默认 1），标题数字取 `detection_map`（K=20）。
   - N6：AP 忽略全集里没有 GT 的类别，F1 把这类框算作 FP。这一条补进 N9 供决策。
   - N9（变异 X3 存活）是冗余检查，不处理。

## D13 — Gemma 4 与 InternVL3.5 的适配约定（2026-09-30，依据 RUN-0002）

1. **新增 token**：
   - **InternVL3.5**：有 257 个预留行（id 151679–151935），规则同 D9-3 和 D11-8：累计不超过 257 个就复用预留行，超过才扩表，只增不缩，禁止缩表。它的 `resize_token_embeddings(len(tok))` 同样会缩表，已实测。lm_head 不共享（checkpoint 里两者的值不同，transformers 没有绑定），嵌入行和 lm_head 行都用各自基础行的精确均值初始化。
   - **Gemma 4**：没有预留行。复用 6227 个 `<unusedN>`：先用 `add_tokens` 注册，注册后仍是原 id、仍是单个 token，不 resize。要初始化两处，都用基础词表（不含 `<unusedN>` 与特殊 token）的精确均值：主嵌入行（与 lm_head 共享）和 PLE 的 `embed_tokens_per_layer` 对应行。
2. **思考开关**：
   - Gemma：把 `enable_thinking` 直接传给 `apply_chat_template`，默认关闭。
   - InternVL：模板没有开关。开启 = 加 model card 给出的 R1 system prompt，原文逐字存在适配器里并注明出处；关闭 = 不加 system prompt。
3. **生成**：三家族都显式用 greedy（Gemma 的 generation_config 默认是采样）。停止 token 一律从 tokenizer 查，不写死：
   - Qwen：`<|im_end|>` 与 `<|endoftext|>`；
   - Gemma：`<turn|>` 与 `<eos>`；
   - InternVL：`<|im_end|>`。
4. **坐标**：各家原生框格式由各自的适配器换算成内部的绝对像素 xyxy，轴序与量程写进单元测试：
   - Qwen：`bbox_2d`，0–1000，x 在前；
   - Gemma：`box_2d`，0–1000，**y 在前**；
   - InternVL：`<box>[[x1,y1,x2,y2]]</box>`，0–1000，x 在前。
5. **视觉 token 数**：每个家族一套固定策略，参数全部显式传入，**禁止依赖处理器默认值**（InternVL 的处理器默认 `crop_to_patches=True`，与 preprocessor_config 相反）。可调参数：
   - Qwen：`min_pixels`/`max_pixels`；
   - Gemma：`max_soft_tokens`，只能取 70/140/280/560/1120；
   - InternVL：`crop_to_patches` 与 `max_patches`。

   E1 的具体取值写进 E1 预注册，并在 G0 报告里列出。
6. **其他**：
   - Gemma 的 processor 做批处理时，图像要按样本嵌套传入；
   - Gemma 的按层属性不能全局读取，例如 `head_dim` 要从 `per_layer_config[i]` 取；
   - HF hub 1.x 的缓存 blob 全局共享，删模型用 `scan_cache_dir(...).delete_revisions(...)`，只删 `models--X` 目录腾不出空间（RUN-0002 实测，已写进 skill lessons）。

## D14 — DEV-0002 合入与偏离裁定（2026-09-30）

1. **合入**：DEV-0002 由 worktree 开发 agent 完成，合入 main（b545838）。
   - 管理员通读五个模块和 constants 的追加部分，在 worktree 里复跑 204 passed。
   - 合入后在干净的临时 worktree 里跑全量测试，发现 1 条集成失败：彩排按旧接口把 `mask_inter_union` 解包成两个值，而 addendum-1 起它返回三字段的 `MaskStats`。管理员改了一行测试（80c60e8）后，243 passed。
2. **任务书里两句数值上不成立的说法，按真实数学关系实现并用测试钉住，接受**：
   - A1：常数引导图时，引导滤波等于盒式均值连做两次，不是一次；
   - A2：三通道相同时，彩色版(eps) 等于灰度版(eps/3)，不是灰度版(eps)。
3. **设计选择**：
   - 以下暂定接受，**E3b 预注册前定稿**：B1 轮廓宽度 2；B10 熵为 0 的窗口照样可选，每张图的细化开销固定，便于计算记账；B12 噪声日程里 10% 的干净对。
   - 其余全部接受：B2–B9、B11、B13、B14。调色板的出处凭记忆写，但真正要紧的性质（两两色差 ΔE ≥ 25）有测试钉住。
4. **常量**：引导滤波（半径 8、eps 1e-3）、JBU（σ_s 1、σ_r 0.2、半径 2）、选区（k=4、边长 64）、分块（256/64）、加噪系数，都只在合成场景上做过合理性检查。按硬约束 11，要在真实开发子集上定稿，并且在看到任何 E3 测试结果之前固定。
5. **遗留**：
   - E3a 若要在训练中对 64 维嵌入场做全分辨率引导上采样，需要 torch/GPU 版本，届时与本实现对照；
   - 小物体的加噪幅度，要不要按网格分辨率另设下限，在 E3b 预注册前定。

## D15 — DEV-0003 合入、偏离裁定，以及对 D9-3、D11-8 的更正（2026-09-30）

1. **合入**：DEV-0003 由 worktree 开发 agent 完成，合入 main（a4a60e8）。
   - 管理员通读了 `qwen35.py`、`base.py` 的改动和 registry。在干净的 worktree 里跑全量测试，312 passed；`smoke --tiny` 退出码 0。
   - 冒烟脚本的各项检查都拿外部参照对比：官方 processor、`get_rope_index`、HF 原生前向路径。它们充当 G0 的独立检验。
   - 隐藏状态路径的独立代码审查安排在 E2 之前做。
2. **偏离裁定**：
   - (2) 层号采用 HF 惯例：最后一个层号取 final norm 之后的值，其余层号是各 decoder 层的原始输出。接受，三个家族一律照此执行，base.py 的 docstring 已写明。E2 的逐层曲线在最后一个点上是 post-norm，报告时注明。
   - (1)(3) `hidden_states` 的 `query_embeds` 不在子类检查里强制；`vision_features` 暂时不是抽象方法。暂时接受，DEV-0004 更新 base 的测试替身之后改为抽象方法。
   - (4)(5) 预算检查收紧；`GenerationResult` 加 `raw_text`（默认 None 只为兼容）和 `thinking_unclosed`。接受。
   - (6) 查询占位符用专用 token `<|visint_query|>`，占一个预留行，不计入 300 的预算。接受。之后 grid token 在扩表前只剩 242 个预留行。
   - (7a)(7c) 手写只增不缩的扩表，不调用 `resize_token_embeddings`；扩表后同步 `vocab_size`。接受。
   - (9) 适配器自己完成图像占位符的展开。在服务器上，像素值来自官方 processor 的 image_processor。冒烟检查 1 与官方 processor 逐项比对，是这条的前提；RUN-0003 核实。
   - (10)–(20) 接受。
3. **更正 D9-3**：lm_head 是否与嵌入共享，以运行时和 config 为准，适配器已按此实现。据 RUN-0001 存下的 4B config，0.8B/2B/4B 共享，只有 9B 不共享。D9-3 里"4B/9B 不共享"一句作废，RUN-0003 用 4B 实测核对。
4. **更正 D11-8 的措辞**：原句"已有的 248320 行逐位不变"与 D9-3"预留行原有的值不用"字面冲突。按本意改为：
   - 已有 token 的行（id < 原 len(tokenizer)）逐位不变；
   - 新 token 的行不论来自预留行还是扩出来的行，一律写成基础词表行的精确均值。

## D16 — DEV-0001 addendum-2 合入，DEV-0001 结案（2026-09-30）

1. **合入**：addendum-2 合入，是第 2 轮、也是最后一轮返工。
   - 管理员读完 diff，全仓库 331 passed。
   - 开发在新代码上重跑了独立审查的复核脚本；61 个变异全部被抓住，其中本轮新增 15 个，前两轮的 46 个仍被抓住。
   - 已到返工上限，不再加独立审查轮次。以后在这些函数上发现问题，一律开新任务。
2. **7 条偏离全部接受**：
   - (1) "answer is X" 中 X 的认定规则：偏向接受前缀答案；
   - (2) `multiple_choice_accuracy` 改为返回 `MultipleChoiceResult(accuracy, none_rate, n)`，报告要分系统列出 None 率；
   - (3) `f1_statistic` 放在 stats，`per_image_counts` 放在 detection；
   - (4) `per_image` 覆盖 GT 里的每一张图，两个系统能按行对齐；
   - (5) `ADE_REDUCE_ZERO_LABEL` 由调用方显式传入；
   - (6) 数字不足 ⌈N/2⌉ 的带注释短行仍算文本行，已知局限，有测试钉住；
   - (7) 允许 n_seeds == n_samples。
3. **解除 D12 的禁用**：`intersect_and_union`/`mean_iou`、`normalize_choice`/`multiple_choice_accuracy`、`parse_grid_text`、`detection_bootstrap_statistic` 可以使用。
4. **DEV-0001 结案**：原开发 CC 会话（cvpr2027-48）退役。以后的开发任务一律按模式 B 派发 worktree agent。

## D17 — DEV-0004（Gemma 4 适配器）合入与偏离裁定（2026-09-30）

1. **合入**：DEV-0004 由 worktree 开发 agent 完成，合入 main（2463c02）。
   - 管理员读了模块说明和 Gemma 特有的核心代码：思考 system 轮的拼接、思考段剥离、`<unusedN>` 注册、主嵌入与 PLE 的初始化、输入嵌入与 per-layer 输入、前向。
   - 合入后在干净的 worktree 里 424 passed；Gemma 与 Qwen 的 `--tiny` 冒烟都通过。
2. **偏离裁定**：
   - (1) 越界改了 `tests/test_adapter_qwen35.py` 两行：接受。`vision_features` 改成抽象方法后，那个测试替身必然失败，是我授权时漏列了它。
   - (2) 一维位置的家族遇到 grid2d 直接报错，不静默当成 seq1d：接受。E3a/E6 的配置按家族选位置模式；以后可在 base 加一个"支持哪些位置模式"的查询接口。
   - (3) `vision_features` 与第 0 层图像位置逐位相同，PLE 的精确作用方式写进了模块说明：接受。
   - (4) 复用 `<unusedN>`：grid token 从最小 id 取，查询占位符取最大 id，主嵌入和 PLE 行都用基础行的精确均值初始化：接受。
   - (5) 思考段格式 `<|channel>thought\n…<channel|>` 取自固定 revision 的 tokenizer_config：暂定接受，要在真实权重的冒烟里看到思考段闭合才算核实。
   - (6) 注册键用 `gemma-4-e2b-it`：接受。我任务书里写成了 `gemma4-e2b-it`，DEV-0005 同类。
   - (7)(8)(10)(12) 冒烟脚本的小改动；不预先缩放图像（按档位保持宽高比，D13-5）；小模型的层结构；其他细节：都接受。
   - (9) 从 qwen35 导入共用件：暂时接受，以后挪到公共模块。
   - (11) 查询位置的 PLE 身份用占位符自己的行：暂定接受，到 E6 用 Gemma 做 E3a 时再定。

## D18 — DEV-0003-a1 合入：CPU 上 GDN 走 torch 实现，开发机能按服务器条件跑测试（2026-09-30）

1. **合入**（4ed6bc3）：
   - Qwen3.5 适配器在 CPU 设备上用 `torch_gdn()` 上下文管理器，把 transformers 在 import 时绑到 fla 的两个 GDN 函数换成 torch 原函数；GPU 上保持 fla。
   - 冒烟检查 0 记录实际走的实现；检查 5 的 HF 原生参照也套同一个作用域。
   - 合入后在干净的 worktree 里：普通 pytest 与 `scripts/pytest_server_parity.sh` 都是 433 passed；Qwen 与 Gemma 的 `--tiny` 冒烟都通过。
   - 去掉这个修复，parity 脚本会复现 RUN-0003 服务器上的全部 16 个失败。
2. **偏离全部接受**：
   - 公开 `gdn_scope`、`torch_gdn`、`kernel_implementation` 等辅助件；
   - 检查 0 的实现名要先穿过分派包装再读；检查 5 在函数内 import；
   - 用 `VISINT_FAKE_FLA` 声明当前是服务器条件；
   - 保留两份模拟 fla：进程内那份用来计数，打包的那份模拟 import 时的绑定，两者由一条测试保持一致。
3. **新规矩**：凡是改动适配器或冒烟脚本的合入，除普通 pytest 外，还要跑一遍 `scripts/pytest_server_parity.sh`。它让开发机按服务器的可选内核绑定条件跑测试，挡住"开发机绿、服务器红"。新增一条绊线测试：假如服务器以后装了 causal-conv1d，这条测试会红，提醒 CPU 路径要同样处理。

## D19 — DEV-0005（InternVL3.5 适配器）合入；思考段判定规则统一（2026-09-30）

1. **合入**（00419da）：
   - 管理员读了模块说明和核心代码：思考 system 轮、剥离规则、单 tile 守卫、输入嵌入、扩表与均值初始化。
   - 冒烟脚本的家族分派冲突按 manager.md §7.2 的协议，交回开发 agent 在它的 worktree 里解决（第 1 轮）。
   - 合入后在干净 worktree 里：普通条件与 server-parity 条件都是 559 passed；三个家族的 `--tiny` 冒烟都通过。
2. **偏离裁定**：
   - grid2d 对一维位置的家族报错，与 D17 一致；
   - 多 tile 时 grid_hw = (16n, 16) 是 tile 纵向堆叠、不是整图网格，探针与查询相关的方法只接受单 tile；
   - R1 system prompt 逐字取自固定 revision 的 model card，并有测试比对；
   - `min_patches`、`size` 也一律显式传入；
   - lm_head 是否共享以运行时为准：2B 的 config 顶层写共享、text_config 写不共享，运行时实际不共享；
   - 框解析"不猜"：任何一个框坏了，整条判失败；
   - `ctx["checks"]` 与 `base_context`；
   - 以上全部接受。
3. **思考段判定统一**：
   - Gemma 与 InternVL 看输出本身：最后一个思考段开头之后没有结束标记，才算 unclosed；没写思考段就直接作答，取全文。
   - Qwen 的思考段开头由提示预填，所以开启思考时输出里没有 `</think>`，即为 unclosed。
   - base.py 的 docstring 已按此改写（b4b4810，只改文档）。
4. **管理员自己的失误，记录在案**：验证时用 zsh 循环 `for m in "" "--model x"; do ... $m ...` 拼参数，zsh 不对未加引号的变量做分词，两条冒烟被误判为失败（退出码 2），直接运行后都通过。已写进 skill lessons。

## D20 — P0 探针用过的图像从评测子集中排除；发出 RUN-0005（2026-09-30）

1. **探针用图的排除规则**：
   - P0 的探针与冒烟只用过两张 COCO val2017 图：`000000039769.jpg`、`000000000139.jpg`（RUN-0001、0002、0004、0005）。
   - 这两张图从 E1 的 COCO val2017 检测子集中排除；凡以 COCO val2017 为来源的评测集，同样排除。先剔除再抽样，报告写明。
   - 今后任何探针用到评测来源的图像，一律登记进 `docs/data_sources.md` 的"探针用图"一节，同样排除。
2. **RUN-0005（思考行为事实探针）**：
   - 起因：RUN-0004 里，InternVL 加了 R1 system prompt、greedy 解码，直接作答。E1 要在固定子集上对比开/关思考（简报 §6.3），需要先知道各家族在 greedy 与推荐采样下是否真的思考、思考多长、会不会在预算内写不完。
   - 性质：Phase 0 的事实探针，不评分，不调提示词，不是 E1 基线。
   - 结果只用于起草 E1 思考子集的协议（解码方式与思考预算），协议本身写进 E1 预注册，由项目负责人在 G0 与 E1 预注册时确认。

## D21 — G0 独立审查的处理（2026-09-30）

审查范围：适配器接口与冒烟结果。审查者是只读 agent，用 CPU 小模型核实。
- 判定：简报 §8.1 的接口全部满足。
- §12 的陷阱里：
  - "坐标约定"部分满足，Qwen 的框解析在 DEV-0006 里做；
  - "受限 logits 在训练和推理中一致"部分满足；
  - "LoRA 合并"还没有代码。

发现按下面四类处理。

1. **E1 开跑前修**（G0 通过后的 E1 准备开发任务）：
   - **批量生成**：现在只有 batch=1，E1 单是 RefCOCO 定位就要 70–115 GPU 小时。要做的：
     - 左填充、逐样本的思考前缀、Gemma 按样本嵌套传图、InternVL 可变 tile、逐行拆答与记账；
     - 在开发子集上报告与 batch-1 greedy 的一致率，不要求逐位相同。
   - **图像几何**：按 G0 §7.2 第 3 项的裁决实现。
   - **停止原因**：`GenerationResult` 加一个字段；解析失败率里把"被 max_new_tokens 截断"单列，以免截断的多边形或网格被当成更短的合法结果。
   - **采样参数**：凡用采样，全部参数显式给出，并记录实际生效的配置。不依赖仓库的 generation_config 或 HF 的全局默认：Gemma 仓库默认 top_k=64，HF 全局默认 top_k=50，而 Qwen3.5-0.8B 的固定 revision 没有 generation_config.json（RUN-0005）。
   - **计时**：首次调用含预热，记账前先预热。
2. **E3 开工前修**（单独的开发任务，E2 期间做）：
   - **训练侧的受限 logits**：
     - base 加 `training_forward(inputs, target_ids)`，与 generate 共用前缀和助手头。
     - 加唯一的 `restricted_logits()`，训练损失和受限解码都调它：只对允许的行做 lm_head 矩阵乘，并含 Gemma 的 final_logit_softcapping。
     - 配一条一致性测试。
   - **GPU 上的一致性判据**：写成一致率。Qwen 预填走 fla chunk、解码走 fla recurrent，bf16 下不逐位相同；"LoRA 合并前后一致"同理。
   - **新增 token 的保存与重载**：
     - 检查点随附 token 表（名字 → id、行哈希）；
     - 加一条"只核对、不初始化"的恢复路径；
     - 支持从本地路径加载合并后的权重；
     - Qwen 与 InternVL 的查询占位符固定取预留区末行，使 grid id 与调用顺序无关。

     现状下，训练后的模型重新挂适配器再加 grid token，会把训练过的行静默重置为均值。
   - **其他**：
     - 输入里有查询 token 时，强制显式传 `query_embeds`（现在不传会静默用占位符的行）；
     - 给出只含 LLM 线性层的 LoRA 目标模块表（视觉塔默认冻结）；
     - 写明 LoRA 要在 attach 之后注入；
     - 训练中途做验证前先切到 eval；
     - 为 E3a 的开放词汇头提供逐个类别名的 token 区间。
3. **写进后续预注册的设计事项**：
   - **E2**：vision_features 与第 0 层逐位相同，所以"视觉编码器参考线"就是第 0 层的探针，G2 解读时不能算作两条证据。如果项目负责人说的"视觉编码器输出"指投影或合并之前的视觉塔输出，那是另一条参考线，在 E2 预注册里请项目负责人定。
   - **E6**：Gemma E2B 的 35 层里有 28 层是 512 滑窗，靠后的查询 token 或网格 token 在这些层里看不到图像。这是模型属性，要写明。
   - **E1**：Gemma 每个视觉 token 对应 48 px，token 数相近时像素量约为 Qwen 的两倍，报告里写明。
4. **审查核实无误的**：
   - 多图顺序；
   - 三家族的视觉 token 都是行主序；
   - Qwen grid2d 的查询位置；
   - Gemma 查询位置的 PLE 与 HF 公式一致。

## D22 — DEV-0006（输出解析器）合入与偏离裁定（2026-09-30）

1. **合入**（f352489，审查 1 轮）：
   - 新的 `src/parsing/`：
     - 失败原因枚举（16 项，与网格解析的枚举不重名）；
     - `ParseTally` 与 `parse_generation`（先判思考段未闭合，再判空回答）；
     - `extract_json`；
     - 计数、多选、具名二选一、多边形的解析。
   - 适配器接口加抽象方法 `parse_boxes(text, *, width, height)`，`width`、`height` 必须是无默认值的关键字参数；基类提供 `parse_one_box`。
   - 三家族的原生框解析，用 model_notes 里的 8 条真实输出做测试。轴序读反时，与 COCO GT 的 IoU ≤0.22，所以写反一定会被测试抓住。
   - 合入后在干净 worktree 里验证：普通与 server-parity 条件都是 785 passed；三家族 `--tiny` 冒烟都通过。
2. **偏离裁定**：
   - **代码块**：多个代码块判 ambiguous，即使其中只有一个能解析；恰好一个代码块才解析。接受。
   - **小数坐标**：Qwen 与 Gemma 的 JSON 坐标接受小数；InternVL 保持整数语法（D19）。接受。
     - 风险：0–1 归一化坐标或像素坐标会被当成 0–1000，记为定位错误，而不是解析失败。
     - 对策：E1 按模型报告"全部坐标 ≤1 且含小数"的条数，作为诊断量，由 E1 框架实现。
   - **负数**：JSON 里的负数判 out_of_range；InternVL 语法里的负号仍判 malformed。接受。
   - **空列表**：Qwen/Gemma 的 `[]` 在 `parse_boxes` 里算解析成功、零个框（检测任务里"没有"是合法回答）；`parse_one_box` 判 no_box。InternVL 的 `<box>[]</box>` 仍判 malformed。接受。
   - **`parse_choice` 改为严格版**（审查第 1 轮）：
     - 回答以前缀式字母开头，又明确陈述了另一个字母，判 multiple_choices，与 D12 N2 一致；
     - 字母写法与 `metrics.misc` 共用一处定义。misc 只做导出，`normalize_choice` 的行为不变。
   - **其他**：
     - 新增 `no_json`，只由 `extract_json` 给出，框和多边形解析会把它换成 no_box / no_polygon；
     - JSON 对象里有重复键，判 ambiguous；
     - InternVL 的 label 取紧挨 `<box>` 的 `<ref>…</ref>`，两者之间只允许空白。

     以上全部接受。
   - **已知局限**：
     - "The answer is: D"（is 后面带冒号）判 no_choice：misc 的 "answer is X" 不含这种写法，改它会改变 `normalize_choice`（D12）。E1 先在开发子集上（D11-6）看解析失败的实际分布，需要时再开 D 条。
     - 计数里的代词 one、连字符数词、"no cats"、小数的处理，见 docstring。
3. **并入 E1 准备任务**：
   - 解析失败率用 `ParseTally.to_dict()` 按模型 × 任务单列，与准确率分开报；
   - 思考段未闭合和被 max_new_tokens 截断各自单列，截断的判定依赖 D21 的停止原因字段。

