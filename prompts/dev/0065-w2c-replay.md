[DEV-TASK 0065] W2-c 的缓解变体：训练里混入约 25% 的自蒸馏回放数据（D125-2 (b)）

## 背景

- W2-c（9B，LoRA，2,000 步，每步 8 条提出加 8 条挑选）通过了去留判定，但语言有代价：计数 −13 点，健康检查两项标红（G-W2 §4）。
- 负责人要求（D125）：做一个事后的缓解变体。
  - 在 W2-c 的训练里混入约 25% 的**自蒸馏回放数据**：基座模型自己对一组通用指令与计数提示的回答。其余设置不变，1 个种子。
  - 理由：回放的是基座自己的行为，不引入任何外部模型或新的监督。
  - 不用 devcheck2 判定，只在 test 上作描述，与 W2-c 并列，不替换它。
- 训练前要先写预注册，由管理员写；本任务的设计取值写进完成总结，管理员据此写预注册。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `src/w2/w2c_train.py`、`scripts/train_w2c.py`、`configs/w2c_train.json`：两条流、各自按监督 token 数平均、1 : 1
- `src/w2/w2c_data.py`：训练池与按图像去重
- E1 的计数提示：`configs/e1_prompts.json` 的 count；健康检查的题与 IFEval、MMLU-Pro、GSM8K 的来源（`src/health/`）
- `src/adapters/qwen35.py` 的 generate 与对话模板

## 任务

1. **回放数据** `scripts/build_w2c_replay.py`（在 HPC 上跑，GPU）：
   - **通用指令**：取一个公开的指令数据集的提示，例如 databricks-dolly-15k，或你找到的更合适的、许可允许研究用的。
     - 按 `hash_rank(salt='w2c_replay_v1')` 取 N_gen 条，默认 2,000；
     - 与全部评测（健康检查、`lang_battery`、IFEval、MMLU-Pro、GSM8K）的题目不重叠，写核对；
     - 选哪个、许可、理由，写进总结。
   - **计数**：从 W2-c 训练池同一批 train 图（按图像与 W2 全部评测图去重，同 DEV-0060），对图里有标注的 COCO 类别，用 E1 计数提示的原文问"有几个"。
     - 取 N_cnt 条，默认 2,000；与 CountBench 的图不重叠。
   - **回答**由 base 9B 自己生成：贪心，思考开关与 W2-c 的提出、挑选相同，即关思考；作答上限够长，通用指令取 512 个 token，计数取 16 个。
     - 截断的照实记，训练时不用。
   - 输出 jsonl：提示、图（计数）、回答、生成长度、是否截断。可续跑，除耗时外逐字节可复现。
2. **训练**：
   - `train_w2c.py` 加一条可选的回放流，配置 `configs/w2c_replay_train.json`：除回放一节外与 `configs/w2c_train.json` 逐项相同。
   - 回放一节：数据路径、每步条数（默认 5，即每步 8 + 8 + 5，回放约 24%）、损失权重（默认三条流各自按本流的监督 token 数平均，1 : 1 : 1）。
   - 回放样本的目标就是 base 的回答，用 base 的对话模板。
   - 默认配置（没有回放一节）时，`train_w2c.py` 的行为逐字节不变，测试钉住。
   - 检查点与 W2-c 同格式，`run_w2.py --setting w2c --checkpoint` 能直接评测。健康检查、`run_e1`、`lang_battery` 也能直接评测。
3. **服务器命令与用时估计**：回放数据生成、训练、test 的 5 个划分、健康检查、T 文本读数；写进脚本的文档字符串。
4. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_w2c_*.py -q`，加新测试）：
   - 回放数据的 tiny 端到端、去重与不重叠；
   - 三条流的日程与损失；
   - 默认配置逐字节不变；
   - tiny 训练能续跑；
   - 检查点能被 `run_w2.py --setting w2c` 加载。

## 不做

- 不改 W2-c 已有的训练与评测行为。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 回放数据的来源、许可、规模、去重；
- 混入比例与损失权重的默认值；
- 服务器命令与用时估计；
- 测试结果原文。
