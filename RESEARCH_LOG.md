# RESEARCH_LOG.md — 工作日志兼台账

管理员维护。台账表倒序（最新在上），日志按时间顺序。脚本编辑台账必须带命中断言。

## 台账

### 运行（RUN）

| RUN | 发出 | 校验 commit | 内容 | 状态 | 一句话结论 | Report |
|---|---|---|---|---|---|---|
| 0002 | 2026-09-30 | 见任务书 | Phase 0：Gemma 4 E2B-it 与 InternVL3.5-2B-HF 事实探针（结构、模板、图像 token、位置、坐标、任务调制仪器、显存） | 已发（等服务器进入值守后自动接单） | — | — |
| 0001 | 2026-09-29 | 见任务书 | Phase 0：环境、Qwen3.5 事实探针、模型登记、数据源体积与 schema（磁盘受限版） | 第 1、2 批已回（45ce09b、27b00a5），第 3 批（数据源 + report）未回 | 仪器（图在前逐位相同）在 0.8B/9B 上成立；原生坐标 0–1000；transformers 5.17.0 可用 | — |

### 开发（DEV）

| DEV | 发出 | 合入 commit | 内容 | 状态 |
|---|---|---|---|---|
| 0001 | 2026-09-29 | — | 与模型无关的基础件：接口抽象、光栅化/几何、网格编解码、指标、统计、计算记账、运行记录（模式 A，开发 CC 会话 cvpr2027-48） | 完成待审（113 tests，9 条偏离） |

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
