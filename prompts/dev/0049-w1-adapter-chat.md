[DEV-TASK 0049] W1：适配器的多轮对话接口 `chat_turn_batch`（按轮批量）与 `chat_turn`。支持多轮、后续轮次的新图、Qwen3.5 的工具调用与每轮的思考预算；只实现 Qwen3.5 家族（D103；接口约定见 `docs/w1_plan.md` §1）

## 背景

- W1（D103）：冻结的 LLM 在多轮回路里调用不含学习的工具。
- 现在的适配器每次只能拼一条 user 消息（`build_inputs` → `_render`），没有 system、assistant 历史与 tool 消息，也不解析工具调用。
- Qwen3.5 的对话模板本身支持这些，原文见 `experiments/P0/0001_env_probe/raw/p4/Qwen__Qwen3.5-{2B,4B,9B}/chat_template.jinja`：
  - `tools` 渲染成 `# Tools` 的 system 块；
  - 调用格式是 `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`；
  - `role: "tool"` 渲染为 user 内的 `<tool_response>…</tool_response>`，content 可以含图像；
  - 最后一个真实 user 之前的 assistant 思考段会被去掉（`last_query_index`）。
  - token id 见 `docs/model_notes.md`：`<tool_call>` 248058/248059，`<tool_response>` 248066/248067。
- 思考预算与强制收尾已经有了：`src/adapters/budget.py`（`budgeted_greedy_loop`、`ThinkFormat`、`thinking_end` 四类，D59-1、D59-5）。

## 先读

- `CLAUDE.md`；`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w1_plan.md`（接口约定，§0 与 §1）
- `DECISIONS.md` 的 D103；`docs/lead_answers/2026-10-20-w1-directive.md`
- `src/adapters/base.py`、`src/adapters/qwen35.py`（`build_inputs`、`_content`、`_render`、`_check_instruction`、M-RoPE 的位置检查）、`src/adapters/budget.py`、`src/adapters/image_policy.py`
- `tests/test_adapter_budget.py`：用前向 hook 操纵 tiny 模型 logits 的做法

## 任务

1. **`src/adapters/chat.py`**（新）：`ToolSpec`、`ToolCall`、`ChatTurnResult`，签名照 `docs/w1_plan.md` §1。另加：
   - `parse_tool_calls(text) -> (list[ToolCall], list[str])`：从作答段解析 Qwen3.5 的 `<tool_call>` 块；格式不对的记进错误列表，不抛异常。
   - 消息的校验：role 只能是 system、user、assistant、tool；content 的元素类型只能是 text 与 image。
     - 文本里不得含特殊 token，沿用 `_check_instruction` 的规则；
     - 违反就报错。
2. **`ModelAdapter.chat_turn_batch(conversations, *, tools, thinking_budget, answer_budget) -> list[ChatTurnResult]`**，以及 `chat_turn`（批大小 1 的便捷形式）：
   - base.py 里声明，默认抛 `NotImplementedError`；在 `src/adapters/qwen35.py` 实现。
   - **批量是硬要求**（吞吐，见 `docs/w1_plan.md` §1）：
     - 多条对话左填充成一批，一次锁步生成，每行各自强制收尾；
     - 每条对话的 tools 可以不同（消融时某些组被关掉）；
     - 批内结果与逐条跑逐 token 相同，要有测试证明。
   - 用模板渲染完整对话（`apply_chat_template(..., tools=..., add_generation_prompt=True, enable_thinking=True)`）。
   - 所有图像都按适配器的图像策略处理，包括 tool 消息里的图。
   - 位置编码要对：多张图分布在多条消息里时，M-RoPE 位置照模板顺序正确，沿用现有检查。
   - 生成：单条、贪心、开思考，思考段到 `thinking_budget` 就强制收尾，作答段至多 `answer_budget`。规则与 `generate_batch_budgeted` 完全相同，尽量直接复用 `budgeted_greedy_loop`。
   - 返回字段：`ChatTurnResult` 的每个字段都要填。
     - token 计数口径与 `GenerationResult` 一致；
     - 补入的 token 不算生成。
   - **不改**现有的 `build_inputs`、`generate*`、`option_logits` 的行为与输出，用现有测试证明。
3. **测试**（`conda run -n visint pytest tests/ -q` 全部通过；新测试放 `tests/test_adapter_chat.py`）：
   - tiny Qwen3.5 上，用 hook 操纵 logits，造出以下情况，各自检查 `ChatTurnResult` 的字段：
     - 自然收尾；
     - 强制收尾（`n_inserted_tokens` 与 `thinking_end = forced`）；
     - 作答段截断（`stopped = False`）；
     - 作答段里写出一个工具调用（`tool_calls` 解析对）。
   - 渲染：给定 system、user（图）、assistant（带 tool_calls）、tool（文字加图）、user 的对话，渲染出的 token 序列与直接调模板的结果逐 token 相同。图像数与视觉 token 数也对。
   - `parse_tool_calls`：
     - 正常、多个调用、参数含换行与 JSON、缺 `</tool_call>`、函数名为空都要覆盖；
     - 后两种进错误列表。
   - **批一致性**：3 条长度、图数都不同的对话（其中一条有工具调用历史）成批生成，与逐条生成的结果逐 token 相同，各字段相同（`wall_time_s` 除外）。
   - 非 Qwen 家族调用 `chat_turn` 时抛 `NotImplementedError`。
   - 现有适配器测试照旧通过，作为不改旧行为的证据。
4. **估时**：在完成总结里粗估 9B 一轮的耗时（A100，提示约 3,000 token、2 张图、思考 2,048 加作答 256），分批大小 1 与 16 两种，参照 RUN-0055 的解码速度。另估批大小 16 时的显存（KV 与视觉 token），说明 40 GB 卡上建议的批大小。

## 不做

- 不写工具与回路（DEV-0050、0051）；不改 configs/。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；新接口的行为；
- 测试列表与结果原文；
- 估时；
- 对 `docs/w1_plan.md` §1 约定的任何偏离或建议，由管理员定。
