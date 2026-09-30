# RESEARCH_LOG.md — 工作日志兼台账

管理员维护。台账表倒序（最新在上），日志按时间顺序。脚本编辑台账必须带命中断言。

## 台账

### 运行（RUN）

| RUN | 发出 | 校验 commit | 内容 | 状态 | 一句话结论 | Report |
|---|---|---|---|---|---|---|
| 0007 | 2026-09-30 | 647b007 | 环境更新 + 全量测试 + E5 v2 摘要（填入 d 的规范值）+ 8 个基座模型的健康检查（IFEval 只生成，N12） | 已发（值守自动接单） | — | — |
| 0006 | 2026-09-30 | 5b59be7 | 服务器装 pymunk + 全量测试 + E5 生成器跨机器复现核对（钉住摘要）+ 生成速度 | 完成（2b516f4） | pymunk 装好；849 passed + 1 skipped，2 条失败都是钉住摘要；10 个摘要里 8 个跨机器相同。差异来自光栅化和一个没量化的元数据浮点，不是物理位姿（D24） | [0006](reports/0006-env-synth.md) |
| 0005 | 2026-09-30 | 0277617 | 思考行为事实探针：8 个 E1 模型 × 12 条提示 × 3 种设置（nothink_greedy / think_greedy / think_sampled），D20 | 完成（9f198a6、e288c1c；8 个模型、240 条，约 55 分钟） | InternVL 两个尺寸开思考后约一半写不完 4096（card 推荐的采样也不解决）；Qwen 四个尺寸都有 1–3 条写满预算，且仓库都没有 generation_config；Gemma 两个尺寸开了的全部闭合 | [0005](reports/0005-thinking-probe.md) |
| 0004 | 2026-09-30 | b4b4810 | G0 冒烟（三家族）：服务器全量测试 + Qwen3.5-0.8B / Gemma 4 E2B-it / InternVL3.5-2B-HF 真实权重冒烟 + Qwen3.5-4B（取代 RUN-0003 §3–4） | 完成（806f4c0、0277617；值守自动接单） | 服务器 558 passed + 1 skipped；四份真实权重冒烟 all_ok=True；D1 仪器在三家族上重现；4B 的 lm_head 共享；InternVL 开思考时直接作答，没走到闭合分支 | [0004](reports/0004-g0-smoke-all.md) |
| 0003 | 2026-09-30 | a4a60e8 | G0 冒烟（Qwen3.5）：服务器全量测试（312）+ 0.8B 与 4B 真实权重冒烟 | **停在 §3**（37a29bd）：服务器上 16 failed，开发机绿、服务器红；冒烟未跑 | 服务器装了 fla，transformers 在 import 时把 GDN 绑到 Triton，CPU 小模型走进 GPU 内核；屏蔽 fla 后 311+1 与开发机一致 | [0003](reports/0003-g0-smoke-qwen.md) |
| 0002 | 2026-09-30 | 见任务书 | Phase 0：Gemma 4 E2B-it 与 InternVL3.5-2B-HF 事实探针（结构、模板、图像 token、位置、坐标、任务调制仪器、显存） | 完成（aa87b10、dc11314；服务器值守自动接单） | 三家族 D1 仪器全部成立（图在前逐位相同）；坐标轴序不同（Gemma y 在前）；Gemma 无预留行（复用 unusedN）；InternVL 处理器默认值与配置矛盾 | [0002](reports/0002-family-probe.md) |
| 0001 | 2026-09-29 | 见任务书 | Phase 0：环境、Qwen3.5 事实探针、模型登记、数据源体积与 schema（磁盘受限版） | 完成（45ce09b、27b00a5、8c010e4） | 仪器（图在前逐位相同）在 0.8B/9B 上成立；原生坐标 0–1000；数据源全部定位，许可与去重问题上报（N10、D11） | [0001](reports/0001-phase0-env.md) |

### 开发（DEV）

| DEV | 发出 | 合入 commit | 内容 | 状态 |
|---|---|---|---|---|
| 0009 | 2026-09-30 | f9f90d0 | E5 生成器跨机器逐位复现修复：整数光栅化、确定的三角函数与聚合、元数据只用量化值；版本 2，摘要扩大到 c/e 120 个、d 60 个（D24；模式 B，worktree agent） | 已合入（无返工；877 passed + 1 skipped；D25） |
| 0008 | 2026-09-30 | 647b007 | 健康检查框架（简报 §8.11）：固定 200 题（MMLU-Pro 100 分层、GSM8K 50、IFEval 50）+ 三个评分器 + 运行脚本（模式 B，worktree agent） | 已合入（审查 1 轮：先生成后评分、比较时给配对 CI 与符号检验；main 1030 passed + 11 skipped；D26） |
| 0007 | 2026-09-30 | 5b59be7 | E5 合成数据生成器第一部分：公共框架（场景规格、身份与种子划分、难度分档、按需生成）+ c 遮挡补全、d 短时物理（pymunk）、e 文本到意象 + 平凡基线（模式 B，worktree agent） | 已合入（无返工；main 852 passed，普通与服务器条件都通过；D23） |
| 0006 | 2026-09-30 | f352489 | 输出解析器：通用解析层（失败原因枚举、失败率记账、JSON/计数/多选/二选一/多边形）+ 三家族原生框格式（Phase 0 组件；模式 B，worktree agent） | 已合入（审查 1 轮；main 785 passed，普通与服务器条件都通过；D22） |
| 0003-a1 | 2026-09-30 | 4ed6bc3 | Qwen3.5 适配器在 CPU 上显式走 GDN 的 torch 实现（修 RUN-0003 的跨机器差异；模式 B，worktree agent） | 已合入（main 433 passed，普通与服务器条件都通过） |
| 0005 | 2026-09-30 | 00419da | InternVL3.5 适配器（照 Qwen35Adapter；模式 B，worktree agent） | 已合入（第 1 轮解冲突与思考规则；main 559 passed，普通/服务器条件都通过；三家族 --tiny 冒烟都通过） |
| 0004 | 2026-09-30 | 2463c02 | Gemma 4 适配器（照 Qwen35Adapter；另把 vision_features 改为抽象方法；模式 B，worktree agent） | 已合入（93 新测试；main 共 424 passed；两个家族的 --tiny 冒烟都通过） |
| 0003 | 2026-09-30 | a4a60e8 | Qwen3.5 适配器 + base.py 接口补充 + G0 冒烟脚本（模式 B，worktree agent） | 已合入（69 新测试；main 共 312 passed；--tiny 冒烟通过） |
| 0002 | 2026-09-30 | b545838（+80c60e8 集成修正） | E3 执行器（叠加、分块、引导滤波/JBU、熵选区）+ mask 加噪（模式 B，worktree agent） | 已合入（91 新测试；main 共 243 passed） |
| 0001-a2 | 2026-09-30 | 见 [dev] 0001-a2 提交 | 复核发现的新问题：N1 reduce_zero 必填、N2 选项标记收窄、N3 bootstrap 检查 id、N4 网格解析规则、N7 防转置、N8 检测逐图计数与 F1 配对（第 2 轮，最后一轮） | 已合入（main 331 passed）；DEV-0001 结案 |
| 0001-a1 | 2026-09-30 | 见下一条 [dev] 提交 | 独立审查修复：S1 NaN、S2 显著性重写、S3 无分数 AP、M1–M4、记账拆分等 12 条（原开发 CC 会话，第 1 轮返工） | 已合入（152 tests，46 个变异全部被抓住；复核通过，另有新问题交 a2） |
| 0001 | 2026-09-29 | 65a0c9d | 与模型无关的基础件：接口抽象、光栅化/几何、网格编解码、指标、统计、计算记账、运行记录（模式 A，开发 CC 会话 cvpr2027-48） | 已合入（带已知问题，见 D10；addendum-1 修复中） |

### 门

| 门 | 状态 | 报告 |
|---|---|---|
| G0 | **已提交，待项目负责人确认**（2026-09-30；N11） | [G0](reports/gates/G0.md) |

## 日志

### 2026-09-29（管理员）

- **做了什么**
  - 读完 v0.1 和 v0.2 四份文件。用户确认：走三角色流程；v0.2 未声明的删减按"只收紧"处理（D1）；立即开工。
  - 建仓库骨架与 D1–D7，DECISIONS_NEEDED 写 N1–N8；私有远端 `junli-gong/vision-internalization`。
- **服务器现状**（只读检查）
  - GPU 空闲（0/183 GB），30 核，180 GB 内存。
  - 整机只有一块 500 GB 盘，根盘剩 6.1 GB（99%）；`/dev/shm` 91 GB 空着。
  - 本项目可回收的旧占用约 36 GB，mmvis 已归档到本机。
- **异常**：磁盘是开工的硬阻塞，见 N1。在答复前按 D6 执行：根盘新增 ≤3 GB，权重放内存盘。
- **服务器准备**（管理员代做，省一轮往返）：建 `$PROJ` 目录，生成 deploy key 并加到 GitHub（读写），clone 到 `$PROJ/repo`。
- **下一步**
  - RUN-0001 与 DEV-0001 并行。
  - N1 答复后决定数据落地规模；RUN-0001 回来后写 DEV-0003（Qwen3.5 适配器）。
- **RUN-0001 的管理员预测**（发出前写；结果回来后逐条标注兑现与否）
  - (a) `get_device_capability()` = (10, 0)，现有 torch 2.13.0+cu130 可用。
  - (b) transformers 最新 release 已支持 Qwen3.5，不必装 main。
  - (c) Gated DeltaNet：fla 与 torch 实现的差距，fp32 最大绝对误差 ≤ 1e-4，bf16 相对误差在 1e-2 量级。
  - (d) 5.7：图在前时各层逐位相同；指令在前时第 0 层相同、之后各层不同。
  - (e) Qwen3.5 的原生定位坐标是 0–1000 归一化，与 Qwen3-VL 相同。
  - (f) 嵌入行数大于 tokenizer 词表，有预留行，新增网格 token 可以直接复用。
  - (g) vLLM nightly 的 dry-run 会另装一个不同于 2.13.0 的 torch，额外占 2 GB 以上。
  - (h) 9B 显存：bf16 推理 batch 1 的峰值约 20–25 GB；LoRA r=32、batch 8 的峰值约 40–60 GB（简报估算）。

### 2026-09-29（管理员，续：改用模式 B）

- 用户确认改为模式 B（D8），目标是"不在电脑前实验也不断"。
- 全局 skill `exp-triad-flow` 增补模式 B，仓库副本已同步：
  - SKILL.md 的两种模式；manager.md §7（派发、审查合入、结果监听、自主范围、保持循环）；dev.md 的子 agent 规则；server.md 的值守规则；
  - 两个脚本 `server_standby_wait.sh`、`wait_for_results.sh`，在本地假 origin 上测过 14 项：接单、完成清单、HOLD、addendum 排序、结果检测、心跳过期、blocked；
  - 两份模板（开发 agent 派发提示、服务器值守说明）；bootstrap 与 init 脚本加模式选择。
- 跨会话直连验证可用：DEV-0001 的完成总结由开发 CC 通过 SendMessage 直接发给管理员，未经用户转交。
- 仍需用户转交最后一次：让服务器 CC 在 RUN-0001 完成后按 `prompts/server/STANDBY.md` 进入值守。
- 下一步：审查 DEV-0001 → 合入 → 挂结果监听 → RUN-0001 回来后写 DEV-0003（Qwen3.5 适配器），DEV-0002（E3 执行器与加噪）可先派发。

### 2026-09-30（管理员）

- **RUN-0001 第 1、2 批回来**（`docs/env_report.md`、`docs/model_notes.md` §1–5）。对照发出前的预测：
  - (a) 兑现：(10, 0)。
  - (b) 兑现：transformers 5.17.0 release 已含 qwen3_5。
  - (c) 半兑现：fla chunk 内核 fp32 相对 fp64 的最大误差 1.1e-4，略超 1e-4；fla/transformers 的另外几种实现彼此只差约 1e-7。bf16 下相对误差 3e-3，比预测的小。机制猜测（未验证）：Triton 的 chunk 内核在 fp32 下用了 TF32 矩阵乘。整模型 bf16 logits 的最大差 1.28，argmax 309/321 一致；fp32 下 321/321 一致。
  - (d) 兑现：图在前 25/25（0.8B）和 33/33（9B）层逐位相同；指令在前只有第 0 层相同。D1 仪器成立。
  - (e) 兑现（实测）：零样本输出与 GT×1000/(W,H) 吻合。Qwen3.5 文档里没找到对坐标格式的说明。
  - (f) 兑现，但有陷阱：243 个预留行可直接用；`resize_token_embeddings(len(tok))` 会把嵌入缩行，已写进 D9。
  - (g) 未兑现：vLLM nightly 声明 torch==2.13.0，不会另装 torch，下载 1.16 GB，但会把 numpy 从 2.5.2 降到 2.3.5；SGLang 会把 transformers 降到 5.12.1。
  - (h) 半兑现：9B 的 LoRA r=32、batch 8 峰值 50.2 GiB，在预测区间内；9B 推理 batch 1 只有 17.8 GiB，低于预测的 20–25。
- **我自己的任务书写错一处**：RUN-0001 §3 把 `export ROOT=... PROJ=$ROOT/...` 写在同一行，`$ROOT` 在赋值之前就被展开，PROJ 成了 `/vision-internalization`。服务器 CC 发现后改成每行一个 export。STANDBY.md 已修（fe27b82），规则写进 skill 的 lessons。已发出的 RUN-0001 不改写。
- **D9**：三家族都用最终后训练版（Gemma `-it`、InternVL `-HF`），revision 固定；Qwen 新 token 复用预留行、禁止 resize、用均值初始化；思考开关直接作为 apply_chat_template 的关键字参数传。
- **发出 RUN-0002**：Gemma 4 E2B-it 与 InternVL3.5-2B-HF 的事实探针，为三家族适配器和 G0 冒烟做准备。会先删掉内存盘上的 9B，占用保持在 25 GB 以内。
- **DEV-0001 审查中**：自己复跑 113 passed、import 与彩排一致、范围干净；独立审查 agent 在跑（仪器代码，按 manager.md §7.2）。
- **DEV-0001 审查完成并合入**（65a0c9d）。独立审查 agent 的结论：
  - 核心数值与官方实现逐位一致：Boundary IoU 504/504，mIoU 最大差 ≤ 9.7e-9，BTS 深度差 0.0，另有 LISA 的 cIoU/gIoU、法向、框。
  - 有 3 个会改变门判定的缺陷：S1 一个 NaN 就让 CI 失效并判为显著；S2 显著性判定不分方向，比值指标没有配对路径；S3 无分数 AP 依赖图像编号。
  - 另有 M1–M4 与若干低风险问题。
  - 处理：D10 逐条裁定 9 条偏离；addendum-1 交回原开发 CC；接口补充并入 DEV-0003；无分数检测的标题指标上报 N9。
- **同时派发** DEV-0002、DEV-0003，各在自己的 worktree（`.claude/worktrees/`，都从 7d5665a 开出）。agent 的内部标识记在不入 git 的 `.claude/dispatch.local.tsv`；skill 已同步这条规则。
- **RUN-0001 第 3 批回来**（8c010e4）：`docs/data_sources.md` 与 `reports/0001-phase0-env.md`，全程没碰红线：根盘新增 0.47 GB，最低剩 5.9 GB；内存盘峰值 21.1 GB。据此：
  - **D11**：RefCOCO 家族训练图像去掉全家族的评测图像；NYUv2 的 BTS 训练帧按场景过滤掉官方 test 场景；BLINK 只用 val（124 条）；CountBench 用打包的 491 张；只有测试划分的基准切出开发子集并从报告数里剔除；Qwen 新 token 超过 243 个时只增不缩地扩表。
  - **N10**：数据许可待确认，阻塞 E1 的数据落地。
- 服务器 CC 在 report 里如实记下它派出的三个子代理的越界：一次只读 git status、C 组网络用量超预算、一个临时文件写出了允许路径。都没有造成改动。已记录，不追究。
- 已补发给 DEV-0003：超过 243 个新 token 时的扩表规则。
- **服务器进入值守**，06:38 起从 git 自动接走 RUN-0002，没经用户转交；模式 B 端到端验证通过。第 1 批（Gemma 4 E2B-it）已回（aa87b10）：
  - 图在前 36/36 层逐位相同，D1 仪器成立；
  - 坐标 `box_2d` 是 0–1000，**y 在前**；
  - 视觉 token 数只有 70/140/280/560/1120 五档；
  - 没有预留行，新 token 会扩表，或者复用注册后的 `<unusedN>`；
  - 默认开启采样，必须显式设 greedy。
- **DEV-0001 addendum-1 审完合入**：自己复跑 152 passed；独立审查复核确认 S1–S3、M1–M4 修好，同时查出 N1–N9。D12 逐条裁定，addendum-2 已发回原开发 CC（第 2 轮，最后一轮）。
- **RUN-0002 完成**（dc11314）：三家族的事实都齐了，据此写 D13（Gemma/InternVL 的适配约定）。三家族的"图在前逐位相同"仪器全部成立：Gemma 36/36 层，InternVL 29/29 层。服务器 CC 另发现 HF 缓存的 blob 全局共享，任务书里"删 9B 目录"的写法腾不出空间，它改用库函数删 revision。这一条已写进 skill lessons。
- **GPU 现在空闲**：G0 冒烟要等 DEV-0003 合入；数据落地要等 N1 磁盘腾出（服务器仍只剩 5.6 GB）和 N10 许可答复。Gemma、InternVL 适配器（DEV-0004、DEV-0005）要等 DEV-0003 合入后再派发，避免 base.py 冲突。
- **DEV-0002 合入**（b545838）：模式 B 下第一个 worktree agent 任务走完"派发 → 完成通知 → 审查 → 合入"全流程。合入后在干净的临时 worktree 里跑全量测试，抓到一条集成失败：DEV-0002 的 worktree 是在 addendum-1 之前开的，它的测试还按旧接口使用 `mask_inter_union`。管理员改一行测试后，243 passed（80c60e8）。这一条已写进 skill lessons。D14 为偏离裁定。
- **DEV-0003 合入**（a4a60e8）：Qwen3.5 适配器、base 的接口补充、G0 冒烟脚本。D15 为偏离裁定，同时更正了我自己在 D9-3 写错的一句（4B 实际共享 lm_head）和 D11-8 的措辞。
- **发出 RUN-0003**，服务器值守会自动接单。**管理员预测**（发出前写，结果回来后标注兑现与否）：
  - (a) 服务器全量测试 312 passed，与开发机相同；
  - (b) 两个尺寸都 all_ok=True；
  - (c) 检查 1 与官方 processor 逐项相等，包括像素；
  - (d) 检查 2：图在前全部层逐位相同（0.8B 25/25，4B 33/33），指令在前只有第 0 层相同；
  - (e) 0.8B 指令在前的逐层 max|Δ| 与 model_notes §5.7 的表逐项相同。依据：前向路径与 RUN-0001 的探针数学上相同。风险：use_cache=False 与 fla 内核的组合；
  - (f) 检查 3 的 grid token id 为 248077、248078；
  - (g) 4B 的 tied=True；
  - (h) 检查 6：vision_features 与第 0 层图像位置逐位相同。
- **DEV-0001 结案**：addendum-2 合入（331 passed），D16 解除了全部禁用。原开发 CC 会话退役，以后全部按模式 B 派发。DEV-0001 从交付到结案共两轮返工，独立审查查出的 3 个严重问题（会直接改变决策门判定）和 4 + 8 个中低问题都已修复，并有测试钉住。
- **RUN-0003 停在 §3**（37a29bd），服务器 CC 按纪律停下，只做诊断、不改代码：
  - 服务器 16 failed / 295 passed / 1 skipped，开发机 312 passed，是典型的"开发机绿、服务器红"。
  - 原因：服务器的 venv 装了 fla，transformers 5.17 在 import 时把 Qwen3.5 的 GDN 函数全局绑到 fla 的 Triton 内核，CPU 上的小型随机模型走进了 GPU 内核。
  - 屏蔽 fla 后 311 passed + 1 skipped（服务器没装 cv2），与开发机的 312 一致。
  - **预测 (a) 未兑现**；(b)–(h) 因冒烟没跑，暂不可判。
  - 另外，我的任务书把冒烟命令的参数写成 `--dtype bfloat16`，脚本只接受 `bf16`。服务器 CC 发现了，没有照抄执行。又是一次任务书里的命令没有对照脚本核对。
- **处理**：派发 DEV-0003-a1，让适配器在 CPU 上用上下文管理器显式换成 torch 实现（GPU 上照旧走 fla），并在开发机上用"模拟装了 fla"的测试钉住。它合入后，发 RUN-0003 的补充任务重跑 §3 和 §4。两条教训已写进 skill lessons。
- **DEV-0004 合入**（2463c02）：Gemma 4 适配器，偏离裁定见 D17。派发模板加了一条：临时文件放在 scratchpad 下以任务编号命名的子目录里。起因是并行的两个 agent 共用 scratchpad，发生过一次互相覆盖。
- **DEV-0003-a1 合入**（4ed6bc3，D18）：修好 RUN-0003 暴露的跨机器差异。新增的 parity 脚本在开发机上模拟服务器的 fla 绑定，去掉修复就能复现服务器上全部 16 个失败。今后改动适配器或冒烟脚本的合入都要多跑这一遍。
- **DEV-0005 第 1 轮**：已把最新的 main 合进它的分支。它在自己的 worktree 里解冒烟脚本的家族分派冲突（Gemma 接进它的 family_suite），并把思考段规则改成和 Gemma 一致：没写 `<think>` 就直接作答时取全文。
- **三家族适配器全部合入**：Qwen3.5、Gemma 4、InternVL3.5，main 共 559 passed。D19 为 DEV-0005 的裁定，并统一了思考段判定规则。
- **发出 RUN-0004**，取代 RUN-0003 的 §3–4。命令已对照最终版冒烟脚本的 argparse 核对过。**管理员预测**：
  - (a) 服务器 558 passed + 1 skipped（cv2）；
  - (b) 四份冒烟 all_ok=True；
  - (c) 检查 1 与官方 processor 逐项相等；
  - (d) 检查 2 图在前各层逐位相同（0.8B 25/25、Gemma 36/36、InternVL 29/29、4B 33/33），指令在前只有第 0 层相同；
  - (e) Qwen-0.8B 指令在前的逐层 max|Δ| 与 model_notes §5.7 的表逐项相同；
  - (f) grid token 的 id：Qwen 248077/248078，Gemma 6/7（`<unused0>`、`<unused1>`），InternVL 151679/151680；
  - (g) 4B 的 lm_head 共享；
  - (h) 各家族 vision_features 与第 0 层逐位相同；
  - (i) Qwen 在 GPU 上检查 0 记录的 GDN 实现是 fla，switched_to_torch=False；
  - (j) InternVL 在 2048 个思考 token 内出现闭合的思考段；Gemma 的思考段在真实权重上闭合，格式与 tokenizer_config 一致。

### 2026-09-30（管理员，续）

- **RUN-0004 结果**（806f4c0 批次 1，0277617 批次 2，[report](reports/0004-g0-smoke-all.md)）。预测逐条对照：
  - (a) 兑现：558 passed + 1 skipped（cv2 未装）。
  - (b) 兑现：四份冒烟 all_ok=True，退出码都是 0。
  - (c) 兑现：16 行（4 个模型 × 2 种顺序 × 思考开关）与官方 processor 逐项相等。
  - (d) 兑现：图在前 25/25、36/36、29/29、33/33 层逐位相同；指令在前都只有第 0 层相同。
  - (e) 兑现：0.8B 指令在前 25 层的 max|Δ| 与 model_notes §5.7 的 bf16 表逐项相同。那张表出自 RUN-0001 的探针脚本，这次走的是适配器 API：两套代码、两次运行，结果一致。
  - (f) 兑现：Qwen 248077/248078，Gemma 6/7，InternVL 151679/151680。
  - (g) 兑现：4B 的 lm_head 共享（D15 的更正成立）。
  - (h) 兑现：四个模型的 vision_features 与第 0 层图像位置 `torch.equal`。
  - (i) 兑现：两个 Qwen 的 chunk 与 recurrent GDN 都绑到 fla，switched_to_torch=False。
  - (j) **只兑现一半**：
    - Gemma 在真实权重上开了思考段并闭合（202 token，以 `<|channel>thought\n` 开头，与 D13 一致）；两个 Qwen 也都闭合（144、147 token）。
    - **InternVL 未兑现**：加了 R1 system prompt、greedy 解码时，直接回答 "Pink"（2 个 token，没有 `<think>`）。闭合分支没有在真实权重上走到，目前只有 fake 单测覆盖。
- **没预测到的事实**：
  - InternVL 多 tile（检查 9）：640×480、max_patches=6，得到 7 个 tile、1792 个 token；两种思考开关下都与官方 processor 逐项相等；探针类方法按设计拒绝多 tile。
  - 检查 1 里开思考的 8 行 text 都是空串、thinking_unclosed=True：检查 1 的生成预算只有 8 个 token，思考段写不完。符合规则，不是问题。
  - 开思考后提示长度的变化：Gemma 多 7 个 token（system 轮），InternVL 多 120 个（R1 system prompt），Qwen 少 2 个（关思考的模板里有一个空思考块）。
- **解读**：G0 要求的"每个家族最小尺寸跑通冒烟"已满足。唯一的缺口是 InternVL 的思考闭合分支。InternVL 的 model card 建议思考模式用 `do_sample=True, temperature=0.6`，我们一律 greedy，所以这是 E1 思考子集的协议问题，不是适配器 bug。下面发 RUN-0005，先测事实。
- **读代码时的新发现（G0 要报告）**：Qwen 适配器把每张图都强制缩放到固定的 H×W（默认 480×640），不保持长宽比。竖图（例如 480×640）会被拉成横图，长宽比失真约 1.78 倍。这对 E2/E3 的固定网格方便，但会压低 E1 零样本基线。Gemma 的处理器保持长宽比；InternVL 单 tile 时压成 448×448 的正方形（这是官方在 max_num=1 时的做法）。E1 的分辨率策略写进 G0 报告，请项目负责人定。
- **发出 RUN-0005**（思考行为事实探针，D20）。同时定了探针用图的排除规则：P0 用过的两张 COCO val2017 图，从所有评测子集中排除。**管理员预测**：
  - (a) InternVL-2B 的 think_greedy 开思考不到一半（≤6/12）；开了的都在 4096 内闭合。
  - (b) InternVL 两个尺寸的 think_sampled（T=0.6）开思考的条数都多于 think_greedy。
  - (c) Qwen 四个尺寸的 think_sampled 都是 12/12 闭合；0.8B 或 2B 的 think_greedy 至少有一条写满预算（重复循环）。
  - (d) Gemma E2B/E4B 的 think_greedy 都是 12/12 开思考并闭合。
  - (e) nothink_greedy 下没有任何模型输出思考标记。
  - (f) think_sampled 的思考长度中位数：Qwen 4B/9B 大于 0.8B/2B；所有模型的中位数都在 1500 token 以内。
  - (g) batch 1 的解码速度：0.8B ≥ 40 tokens/s，9B ≥ 20 tokens/s。
  - (h) 总墙钟 ≤ 3 小时。
- **派发 DEV-0006（输出解析器）**：简报 §8.1 把解析器列为 Phase 0 组件，现在只有 InternVL 有框解析，Qwen 与 Gemma 没有，自由文本里的多选字母、计数、二选一、多边形也都没有。等 G0 期间先补上：它不依赖 G0 要定的事，E1 一开始就要用。RUN-0005 的 A4/B4 定位输出回来后，追加为回归样例。
- **G0 门报告草稿**：`reports/gates/G0.md`。等独立审查与 RUN-0005 第一批结果补进去后，再交项目负责人。独立审查已派出（只读 agent）。
- **派发 DEV-0007（E5 合成数据生成器，第一部分）**：
  - 依据：简报 §8"E5 的合成数据生成可以从 Phase 0 起并行开发"；N5、N7 写的"答复前继续做的事"是照常开发，只用程序生成的几何体。
  - 第一部分只做 2D 能完成的三类任务：c 遮挡补全、d 短时物理、e 文本到意象；a、b 需要 3D 渲染，放到第二部分。
  - 生成器按需即时生成，不落盘（N1）。形状族、身份划分、难度档位这些默认值，最终以 E5 预注册为准，由项目负责人确认。
- **G0 独立审查回来**（只读 agent，用 CPU 小模型核实）。处理见 D21，要点：
  - **E1 开跑前必须修两件事**：
    - 批量生成：batch 1 下，E1 单是定位就要 70–115 GPU 小时；
    - 图像几何口径：Qwen 拉伸；InternVL 单 tile 时压成正方形。
  - **E3 之前要补两块**：
    - 训练侧的受限 logits 与统一的训练前向。漏掉 Gemma 的 softcapping 会静默不一致；
    - 新增 token 的保存与重载。现状下重新挂适配器会把训练过的行静默重置；Qwen 与 InternVL 的 grid id 还依赖调用顺序。
  - **备注**：vision_features 与第 0 层逐位相同，所以 E2 的"视觉编码器参考线"就是第 0 层探针本身，G2 解读时不能算两条证据。
  - 这些都不影响 G0 判定（接口满足 §8.1，冒烟通过），已排进 E1 准备与 E3 准备两个开发任务。
- **RUN-0005 第 1 批**（9f198a6：InternVL-2B、Qwen-0.8B、Gemma E2B）：
  - **InternVL-2B**：
    - greedy：开思考 8/12，只闭合 3 条，5 条写满 4096；
    - 采样（T=0.6）：开 9/12，闭合 5，写满 4；
    - 写满的多数是反复 "Wait…" 不收尾，少数是真循环。
  - **Qwen-0.8B**：greedy 12/12 开（提示预填），2 条写满（重复循环）；它的固定 revision 没有 generation_config.json，采样一项按任务书跳过。
  - **Gemma E2B**：两种解码都是 11/12 开且全部闭合；定位题 B4 不开思考。
  - 关思考时，三个模型都没有输出思考标记。
  - 预测对照：
    - (a) 未兑现；
    - (b) 2B 兑现；
    - (c) 后半兑现，前半因跳过不可判；
    - (d) E2B 未兑现（11/12）；
    - (e) 兑现；
    - (g) 0.8B 兑现；
    - 其余待第 2 批。
  - **解读**：E1 思考子集的协议必须把"写不完"当作一等公民处理：
    - 预算：4096 还是 8192；
    - 解码：greedy，还是 card 推荐的采样（参数全部显式给出）；
    - 写不完的样本单独计数，不能混进解析失败。
  
  这些写进 E1 预注册，交项目负责人批。
- **G0 门报告提交**：`reports/gates/G0.md`，待确认事项写进 N11。确认之前不写 E1 代码、不落地 E1 数据。
- **账户会话用量上限**：DEV-0006、DEV-0007 两个 agent 在 03:2x PT 被 "session limit" 打断，3:30 重置。两个 worktree 的半成品都保留着。先恢复快完成的 DEV-0006；DEV-0007 等它合入后再恢复。以后同时在跑的 agent 不超过两个，免得再撞上限、连管理员自己也停摆。
- **RUN-0005 完成**（e288c1c，[report](reports/0005-thinking-probe.md)；8 个模型、240 条，约 55 分钟）：

  | 模型 | greedy：开 / 闭合 / 写满 4096 | 采样：开 / 闭合 / 写满 | greedy 思考中位数 |
  |---|---|---|---|
  | InternVL-2B | 8 / 3 / 5 | 9 / 5 / 4 | 4095 |
  | InternVL-4B | 11 / 5 / 6 | 11 / 7 / 5 | 4095 |
  | Qwen-0.8B | 12 / 10 / 2 | 跳过 | 227 |
  | Qwen-2B | 12 / 9 / 3 | 跳过 | 312 |
  | Qwen-4B | 12 / 10 / 2 | 跳过 | 233 |
  | Qwen-9B | 12 / 11 / 1 | 跳过 | 297 |
  | Gemma E2B | 11 / 11 / 0 | 11 / 11 / 0 | 303 |
  | Gemma E4B | 11 / 11 / 0 | 10 / 10 / 0 | 312 |

  - 四个 Qwen 的固定 revision 都没有 generation_config.json，采样一项全部按任务书跳过。
  - 关思考时，8 个模型都没有输出思考标记。
  - batch 1 的解码速度在 34–83 tok/s 之间。
  - 根盘 09:44 掉了 0.30 GB，服务器 CC 查明是别的用户的 VSCode 扩展写入，不是本项目的文件。
  - 服务器 CC 还发现 Qwen 适配器的 `token_id()` 只认 attach 时缓存的 token，不认 `<think>`。探针改为从 tokenizer 查，并做单 token 校验。这条不影响已有结果，记入 E1 准备任务。
- **RUN-0005 的预测对照**：
  - (a) 未兑现：2B 开了 8/12，只闭合 3 条。
  - (b) 一半兑现：2B 采样比 greedy 多开 1 条；4B 两者都是 11 条，持平。
  - (c) 前半不可判（Qwen 全部跳过采样）；后半兑现：0.8B 写满 2 条，2B 写满 3 条。
  - (d) 未兑现：Gemma 两个尺寸都是 11/12 开思考，开了的全部闭合。
  - (e) 兑现。
  - (f) 前半不可判（没有采样）；greedy 下 Qwen 的思考长度与尺寸无单调关系。后半未兑现：InternVL 的中位数是 4095。
  - (g) 兑现：0.8B 62–83 tok/s，9B 45–61 tok/s。
  - (h) 兑现：约 55 分钟。

  我系统性地低估了 InternVL 写不完的问题。这类事实，比凭 card 的描述去猜可靠得多。
- **对 E1 思考子集协议的含义**（写进 E1 预注册，交项目负责人批）：
  - 思考模式下"写不完"是一等结果：单独计数，不混进解析失败。
  - 解码用 greedy，与主协议一致。原因：InternVL 采样也不解决问题；Qwen 仓库没有采样配置，采样就得我们自己定参数。
  - 预算定 4096。另在小子集上用 8192 做一次敏感性检查，看 InternVL 写不完是不是预算不够。
- **G0 报告发布**：私有网页 https://claude.ai/artifact/HP3jU72ci5typUjrJ1VVG7，内容与 `reports/gates/G0.md` 相同，方便项目负责人离开电脑时用手机看。推送通知只到了桌面：Remote Control 没开，手机收不到。
- **DEV-0006 合入**（f352489，D22）：
  - 审查第 1 轮改了三处：
    - `parse_choice` 改成与 D12 N2 一致的严格版，定义与 misc 共用；
    - `parse_boxes` 的 `width`/`height` 钉成必填关键字参数；
    - 两个测试替身补全。
  - 合入前在干净 worktree 里验证，普通与 server-parity 都是 785 passed。
  - 管理员自己的失误：又一次 `cd` 进了 agent 的 worktree，主工作目录被切走。已切回。以后对 worktree 一律用 `git -C` 和绝对路径，跑测试用子 shell 里的 cd。
- **派发 DEV-0008（健康检查框架）**：简报 §8.11 要求每个门都报告基座与合并后模型在固定 200 题上的分数，基座分数要在 G1 之前测好。它不依赖 G0 要定的事，只用刚合入的解析层。题目与协议在看到任何分数之前固定，之后写 D 条目。同时在跑的 agent 仍是两个（DEV-0007 与 DEV-0008）。
- **DEV-0007 合入**（5b59be7，D23）：E5 生成器第一部分，无返工。
  - 与解析层合并后在干净 worktree 里验证，普通与 server-parity 都是 852 passed；目检了遮挡任务的小样本。
  - 生成速度（开发机单核）：c 约 105、e 约 300、d 约 200 样本/秒。
- **发出 RUN-0006**：服务器装 pymunk、跑全量测试，并用钉住的摘要核对 E5 生成器能否跨机器逐位复现。**管理员预测**：
  - (a) 服务器全量测试 851 passed + 1 skipped，或者只有 d 的钉住摘要那一条失败；
  - (b) c、e 的钉住摘要在服务器上相同；
  - (c) d 的钉住摘要在服务器上**不同**：开发机 arm64 上的 Chipmunk 会做 FMA 融合，x86_64 默认没有 FMA；差异应只在浮点末位，位姿最大差远小于 1 px；
  - (d) 服务器单核生成速度：c ≥ 50 样本/秒。
- **RUN-0006 结果**（2b516f4，[report](reports/0006-env-synth.md)）：
  - pymunk 7.3.0 装好，只新增 3 个包。
  - 全量测试 849 passed + 1 skipped；2 条失败，都是钉住摘要。
  - 10 个摘要里 8 个跨机器相同。不同的两个是 `occlusion/test/2` 与 `physics/test/2`，恰好共用一个场景种子。
  - 管理员在开发机上逐分量对照：
    - 两者的场景规格都相同；
    - 遮挡那个是另一个物体的光栅化结果不同；
    - 物理那个的位姿与 mask 都相同，只有一个没量化的元数据浮点量不同。
  - 生成速度（服务器单核）：c 96.8、d 195、e 304 样本/秒。
- **RUN-0006 预测对照**：
  - (a) 未兑现：失败的是 c 和 d 各一条，不只是 d。
  - (b) 一半兑现：e 全部相同，c 有一条不同。
  - (c) 机制没有得到支持：物理位姿逐位相同，差异在光栅化和元数据；我猜的 Chipmunk FMA 不是这一例的原因。
  - (d) 兑现。
- **D24**：c、e 必须跨平台逐位相同；d 以服务器为规范平台，但生成器里我们自己写的部分必须确定。派发 DEV-0009 修复，生成器版本改为 2。DEV-0009 合入前，服务器上这两条失败是已知项。
- **DEV-0009 合入**（f9f90d0，D25）：E5 生成器版本 2。
  - c、e 的数据路径上已经没有依赖平台的计算。
  - 查明 RUN-0006 那条物理差异来自没量化的 `max_penetration_px`：两台机器只差浮点末位。
- **DEV-0008 合入**（647b007，D26）：健康检查框架。
  - 审查 1 轮：先生成后评分；比较时给出配对 bootstrap CI 与符号检验。
  - 与 main 合并时 `requirements.txt` 有追加冲突，保留两边；管理员同时按 D24 把 pymunk 钉成 ==7.3.0。
  - 验证：1030 passed + 11 skipped。
- **N12**：IFEval 评分器用的 punkt_tab 许可不明，已上报；在此之前 IFEval 只生成、不评分。
- **Mac 的 caffeinate 退出过一次**（退出码 144），已重挂并确认断言生效。
- **发出 RUN-0007**：环境更新、全量测试、填入 E5 d 的规范摘要、8 个基座模型的健康检查。**管理员预测**：
  - (a) pip 只新装 langdetect、nltk、immutabledict 及其依赖，不改任何已装包；
  - (b) 全量测试除 11 个 skip 外全部通过；c、e 的 120 个摘要，以及三角函数表、单位形状、光栅化的钉住摘要，在服务器上全部相同；
  - (c) d 的 60 个摘要与开发机也全部相同：量化挡住了 Chipmunk 的末位差异（把握约七成）；
  - (d) MMLU-Pro 随 Qwen 尺寸单调上升，9B ≥ 50%；GSM8K flexible 上 Qwen-4B、9B ≥ 80%；Gemma E4B 高于 E2B；每个模型的 MMLU-Pro 解析失败率 ≤ 10%；
  - (e) 每个模型 ≤ 45 分钟，总计 ≤ 6 小时。
- **RUN-0007 第 1 批**（d067563）：
  - 服务器全量测试 1030 passed + 11 skipped，全绿。
  - E5 v2 的 180 个摘要跨机器全部相同：c 60/60、d 60/60、e 60/60。
  - 预测对照：(b) 兑现；(c) 兑现，d 的 60 个也全部相同，说明量化挡住了 Chipmunk 的末位差异。
  - 管理员把服务器写出的规范清单复制回 `tests/data/synth_digests_v2.json`（常量级修改）：d 的摘要与占位值逐个相同，状态改为 canonical，并记下规范平台 x86_64 Linux、glibc 2.35、pymunk 7.3.0。开发机上复跑 `test_synth_repro.py`：13 passed + 1 skipped（非规范平台）。
- **RUN-0007 第 2 批**（1c0bd98，前 4 个基座；IFEval 只生成）：

  | 模型 | MMLU-Pro | GSM8K flexible / strict | MMLU-Pro 写满 32 token | 墙钟 |
  |---|---|---|---|---|
  | Qwen-0.8B | 14% | 58 / 56 | 40/100 | 469 s |
  | Qwen-2B | 27% | 76 / 72 | 11/100 | 409 s |
  | Gemma E2B | 25% | 80 / 78 | 54/100 | 647 s |
  | InternVL-2B | 34% | 80 / 76 | 13/100 | 356 s |

  - 预测 (d) 的最后一条"MMLU-Pro 解析失败率 ≤10%"对 4 个模型都没有兑现：Gemma E2B 53%，Qwen-0.8B 39%。这两个模型多数先写解释，32 token 内没给出字母。
  - 其余几条要等第 3 批。
  - 预测 (e) 目前兑现：每个模型 6–11 分钟，远低于 20–45 分钟的估计。
- **N13**：MMLU-Pro 的 32 token 限制让这一项被格式截断主导。已经看过结果，所以协议不自行修改，上报项目负责人。建议保留标题口径，另加一列 256 token 的诊断。

