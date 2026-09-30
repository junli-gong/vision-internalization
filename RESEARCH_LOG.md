# RESEARCH_LOG.md — 工作日志兼台账

管理员维护。台账表倒序（最新在上），日志按时间顺序。脚本编辑台账必须带命中断言。

## 台账

### 运行（RUN）

| RUN | 发出 | 校验 commit | 内容 | 状态 | 一句话结论 | Report |
|---|---|---|---|---|---|---|
| 0003 | 2026-09-30 | a4a60e8 | G0 冒烟（Qwen3.5）：服务器全量测试（312）+ 0.8B 与 4B 真实权重冒烟 | 已发（值守自动接单） | — | — |
| 0002 | 2026-09-30 | 见任务书 | Phase 0：Gemma 4 E2B-it 与 InternVL3.5-2B-HF 事实探针（结构、模板、图像 token、位置、坐标、任务调制仪器、显存） | 完成（aa87b10、dc11314；服务器值守自动接单） | 三家族 D1 仪器全部成立（图在前逐位相同）；坐标轴序不同（Gemma y 在前）；Gemma 无预留行（复用 unusedN）；InternVL 处理器默认值与配置矛盾 | [0002](reports/0002-family-probe.md) |
| 0001 | 2026-09-29 | 见任务书 | Phase 0：环境、Qwen3.5 事实探针、模型登记、数据源体积与 schema（磁盘受限版） | 完成（45ce09b、27b00a5、8c010e4） | 仪器（图在前逐位相同）在 0.8B/9B 上成立；原生坐标 0–1000；数据源全部定位，许可与去重问题上报（N10、D11） | [0001](reports/0001-phase0-env.md) |

### 开发（DEV）

| DEV | 发出 | 合入 commit | 内容 | 状态 |
|---|---|---|---|---|
| 0005 | 2026-09-30 | — | InternVL3.5 适配器（照 Qwen35Adapter；模式 B，worktree agent） | 已派发 |
| 0004 | 2026-09-30 | — | Gemma 4 适配器（照 Qwen35Adapter；另把 vision_features 改为抽象方法；模式 B，worktree agent） | 已派发 |
| 0003 | 2026-09-30 | a4a60e8 | Qwen3.5 适配器 + base.py 接口补充 + G0 冒烟脚本（模式 B，worktree agent） | 已合入（69 新测试；main 共 312 passed；--tiny 冒烟通过） |
| 0002 | 2026-09-30 | b545838（+80c60e8 集成修正） | E3 执行器（叠加、分块、引导滤波/JBU、熵选区）+ mask 加噪（模式 B，worktree agent） | 已合入（91 新测试；main 共 243 passed） |
| 0001-a2 | 2026-09-30 | 见 [dev] 0001-a2 提交 | 复核发现的新问题：N1 reduce_zero 必填、N2 选项标记收窄、N3 bootstrap 检查 id、N4 网格解析规则、N7 防转置、N8 检测逐图计数与 F1 配对（第 2 轮，最后一轮） | 已合入（main 331 passed）；DEV-0001 结案 |
| 0001-a1 | 2026-09-30 | 见下一条 [dev] 提交 | 独立审查修复：S1 NaN、S2 显著性重写、S3 无分数 AP、M1–M4、记账拆分等 12 条（原开发 CC 会话，第 1 轮返工） | 已合入（152 tests，46 个变异全部被抓住；复核通过，另有新问题交 a2） |
| 0001 | 2026-09-29 | 65a0c9d | 与模型无关的基础件：接口抽象、光栅化/几何、网格编解码、指标、统计、计算记账、运行记录（模式 A，开发 CC 会话 cvpr2027-48） | 已合入（带已知问题，见 D10；addendum-1 修复中） |

### 门

| 门 | 状态 | 报告 |
|---|---|---|
| G0 | 未提交 | — |

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

