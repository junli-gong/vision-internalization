[DEV-TASK 0055] W1 回路：每轮的状态提示与"最后一轮"提示可以作为单独的 user 消息发出（开关，默认保持现状；W1 预注册 §10b；RUN-0061 臂 T1 的诊断）

## 背景

- RUN-0061 的臂 T1（9B，开发-调参集 96 句）：最后一轮只给提交工具，并在上一轮最后一条 tool 消息末尾追加 "This is your last turn. Only the submit tools are available now: submit your answer."
  - 只有 48/96 句提交。另有 41 句在最后一轮仍然调 `show`、`grabcut`、`select_superpixels` 等不在清单里的工具，被拒（`final_round`）。
  - 这些句子最后一轮的思考完全没提到"最后一轮"，都在接着上一步的结果往下做。
- 推测：追加在 `<tool_response>` 里的文字，模型当作工具输出的数据，不当作指令。
- 本任务加一个与任务无关的开关：这两种提示改为在这一轮的工具结果之后，单独追加一条 user 消息。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `src/w1/loop.py`（DEV-0053 的"协议开关"一节，`add_status_notice`、`add_last_round_notice`）、`src/w1/config.py`、`configs/w1.json`
- `src/adapters/qwen35.py` 的 `chat_inputs`、`chat_turn_batch` 与消息校验（`check_messages`，D104）；Qwen3.5 的对话模板怎样渲染 tool 消息与 `reasoning_content`
- DECISIONS.md 的 D104、D108、D109

## 任务

1. **两个开关**：`configs/w1.json` 的 `prompts` 下新增 `status_role` 与 `last_round_role`。
   - 取值都是 `"tool"`（默认，现状：追加在这一轮最后一条 tool 消息的末尾）或 `"user"`。
   - 取 `user` 时，这一轮的工具结果全部追加之后，再追加**一条** user 消息，内容只有文字：
     - 这一轮要发的提示按 status 在前、last_round 在后的顺序拼接，中间空一行。
     - 两种提示一个走 tool、一个走 user 时，各走各的。
   - 同一轮只追加一条 user 消息。
   - 默认配置下，tiny 彩排的 `predictions.jsonl` 与改动前逐字节相同，比对方法同 DEV-0053。
2. **记录**：
   - 轮记录里的 `status_notice`、`last_round_notice` 照旧。
   - 非默认时，再加 `"notice_message": "user"`。
   - 两个开关进运行身份。
3. **对话模板与校验**：
   - 确认 `chat_inputs`、`chat_turn_batch` 与 `check_messages` 接受"assistant → tool … → user → 生成"这样的序列。不接受的话，在适配器里放行这种序列，只放行这一种：tool 之后紧跟 user。
   - 用真的 Qwen3.5 对话模板（tiny 的 tokenizer）写测试：
     - user 消息渲染在工具结果之后，生成提示之前；
     - **前几轮的 `reasoning_content` 是否还在渲染结果里**。Qwen 的模板会丢掉"最后一条真正的 user 消息"之前的思考段，插入 user 消息之后前几轮的思考可能被丢掉。
   - 把实测结果写进完成总结：哪几轮的思考还在；提示的 token 数与默认时相比怎样变化。
   - 不要为了保留思考段去改模板。
4. **测试**（`conda run -n visint pytest tests/ -q` 只跑 W1 相关的测试与 `tests/test_adapter_chat.py` 即可，设 `HF_HUB_OFFLINE=1`）：
   - 每种取值组合；
   - 与 `final_round_tools`、`capped_tools` 一起开；
   - 默认配置逐字节不变；
   - 开了全部开关的 tiny 彩排 rc=0。

## 不做

- 不改提示的正文，不改上限。
- 不改 `src/w1/workbench.py`。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 两个开关的语义；
- 对话模板的实测：渲染出的片段原文，以及思考段的保留情况；
- 测试结果原文；
- 与本任务书的任何出入。
