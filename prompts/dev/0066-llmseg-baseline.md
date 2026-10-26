[DEV-TASK 0066] 外部基线 LLMSeg：官方代码接本地 Qwen3.5-9B（零样本），在 W2 的 test 子集上评测（D125-4）

## 背景

- 负责人要求的外部基线（D125-4）：**LLMSeg**。
  - 代码：`https://github.com/ajhamdi/llmseg-code`，MIT 许可证，最新 commit `f2191e6605e40a5be5d932187fc055708ae098cf`；论文 arXiv 2610.29409。
  - 方法：现成的多模态 LLM 只输出坐标（框与 B 样条控制点，JSON），解析、限框、光栅化、mask 组合、停止都由确定性代码完成；不训练，没有分割头，也不用学习型 CV 模型。主配置默认 8 轮修正。
- 要求：
  - 用它的**官方代码**，配 **Qwen3.5-9B 零样本**，跑与 W2-c 相同的 test 子集，5 个划分各 300 句；4B 如果便宜也跑；
  - 计算量照常记账；
  - 跑 test 之前写进预注册，由管理员写；本任务的设计取值写进完成总结。
- **不用任何闭源 API**：官方代码只支持 Gemini、Anthropic、OpenAI、OpenRouter，要加一个本地提供方。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- LLMSeg 的 README、EVALUATION.md、`pipeline.py`、`llm.py`、`prompts.py`、`geometry.py`，以及它的 tests。看 `llm.py` 怎样按模型家族处理坐标：Qwen-VL 走 0–1000。用 `git clone` 到 scratchpad 或用 `gh api` 读，不要把整个仓库放进我们的仓库根。
- 我们这边：
  - `src/adapters/qwen35.py`（`chat_turn_batch`、`generate` 与图像处理）；
  - `src/w2/data.py`（test 划分怎样取样本）；
  - `src/w2/loop.py`、`runner.py` 的计分与记账；
  - `scripts/run_w2.py`。

## 任务

1. **引入官方代码**：放在 `third_party/llmseg/`，固定在 commit `f2191e66`。
   - 保留 LICENSE，写一份 `third_party/llmseg/PROVENANCE.md`：来源、commit、每个文件的 sha256、我们改了什么。
   - 只放运行管线必需的文件，不放 toy 图与评测数据。**方法本身的文件不改**。
   - 本地提供方尽量通过它的接口注册，例如在我们这边写一个提供方类再注入；实在要改它的文件，改动最小，并在 PROVENANCE 里逐行列出。
2. **本地提供方**：
   - 用我们的适配器加载 Qwen3.5-9B（4B 同理），按它对 Qwen-VL 的约定作答：坐标是 0–1000 的网格，由它的代码换算。
   - 思考：关思考，与 E1 的定位协议一致；写进总结。如果它的提示明显需要推理，另报开思考的小样本对照，但主配置关思考。
   - **批量**：它的管线是逐图顺序调用的，逐条跑 9B 会很慢。请实现一个批量的本地提供方：多个管线实例并发，请求在一个短时间窗内合批，调用 `chat_turn_batch` 或 `generate_batch`。合批不改变每条请求的内容与解码，写测试钉住，即合批与逐条的结果逐 token 相同（贪心）。
3. **评测**：`scripts/run_llmseg.py`。
   - 划分：W2 的 5 个 test 划分，每个取 1,000 句里按 `hash_rank(salt='w2_llmseg_v1')` 的前 300 句。清单写进 `configs/w2_llmseg_subsets.json`，含 id 与 sha256。
   - 参数：官方主配置，`--iterations` 默认 8，其余开关都关，即它论文的主配置。
   - 输出与 W2 相同的格式：`predictions.jsonl`（每句的最终 mask 的 RLE、各阶段的调用数、token 数、墙钟）、`metrics.json`（cIoU、gIoU 与 bootstrap 95% CI）、`compute.json`、`config.yaml`（运行身份：官方 commit、PROVENANCE 的 sha256、模型、迭代数）。
   - 计分与 W2 完全相同，同一份 GT 与同一个 `mask_inter_union`。
   - 再写一个比较脚本：同样 300 句上，W2-c 与 LLMSeg 的配对 cIoU 差与 95% CI。W2-c 的读数取 RUN-0072 的结果目录。
4. **用时估计**：
   - tiny 与开发机上测单句调用次数；
   - 按 9B 每次调用的实测，估 5 × 300 句的 GPU 小时，4B 另估；
   - 写进总结。负责人的新增预算合计约 25 GPU 小时，要单独看 LLMSeg 占多少。
5. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_llmseg*.py -q`）：
   - 用它自己的离线测试，即 scripted pipeline，确认引入后照常通过；
   - 本地提供方：合批与逐条逐 token 相同；
   - 坐标换算：我们的提供方返回的 0–1000 被它正确换回像素；
   - tiny 端到端：小模型 → `run_llmseg.py` → 计分；
   - 不会调用任何外部 API：没有 key 时不尝试联网，测试钉住。

## 不做

- 不用任何闭源 API，不下载它的评测数据集（我们用自己的 W2 test 子集）。
- 不改 W2 的代码与配置。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；PROVENANCE；
- 本地提供方的设计（批量、思考开关）；
- 子集的规则与 sha256；
- 用时估计；
- 测试结果原文；
- 与任务书的任何出入。
