[DEV-TASK 0053] W1 回路的调参开关：最后一轮只给提交工具、每轮报剩余预算、工作台选项透传、未冻结不许跑开发-检查集与 test（D103 §二.4；RUN-0060 冒烟的诊断）

## 背景

- RUN-0060（9B，开发-调参集 32 句，`experiments/W1/0060_w1_smoke_hpcgpu07/`）：
  - 指代分割 32 句里 30 句到第 6 轮还没提交（`no_answer`），cIoU 0.159。同样的句子上，"自己的框" 0.487，W1-F 0.518。
  - 模型每轮只发 **1 个** 工具调用（192 轮里没有一轮发两个），6 轮正好用完。
  - 第 6 轮它仍在调 `show`、`select_superpixels`、`refine` 等（30 句里 11 句是 `show`）。只有 3/32 的思考提到"最后一轮"的提示。
  - 提交了的 2 句，IoU 分别是 0.856、0.870。
- D103 把上限定死：每条样本最多 6 轮、8 次工具调用，不能加轮数。
- 本任务给回路加几个**与任务无关**的协议开关，默认值保持现在的行为。哪一组开关最好，由管理员在开发-调参集上比较后选定，然后冻结。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w1_plan.md`（§0 硬约束、§3 回路）、`configs/w1.json`
- `src/w1/loop.py`、`src/w1/config.py`、`src/w1/runner.py`、`scripts/run_w1.py`，以及它们的测试
- DECISIONS.md 的 D103、D104、D107

## 任务

### 1. 协议开关（`configs/w1.json` 的 `protocol` 与 `prompts.messages`；`src/w1/config.py` 校验；`src/w1/loop.py` 实现）

**默认值必须保持现有行为。** 新键写进 `configs/w1.json`，取默认值。现有测试原样通过；默认配置下，tiny 彩排的 `predictions.jsonl` 与改动前逐字节相同（运行身份里的配置 sha256 会变，这个可以）。

1. **`protocol.final_round_tools`**：取 `"all"`（默认）或 `"submit_only"`。
   - 取 `submit_only` 时，第 `max_rounds` 轮（`max_rounds > 1`）交给 `chat_inputs` 与 `chat_turn_batch` 的工具清单，只含这条样本 `specs()` 里属于 `submit_tools` 的那几个，顺序照 `specs()`。
   - 这一轮里调用非提交工具：不执行、不计数，`refused = "final_round"`，返回一条 tool 消息 `prompts.messages.final_round_refused`（取 `submit_only` 时必填，没有占位符）。
   - 同一段里，排在被拒调用后面的提交照常执行。
   - 文字形式的最终答案（定位、计数）照旧有效。
2. **`protocol.capped_tools`**：取 `"all"`（默认）或 `"submit_only"`。
   - 计数的调用已满 `max_tool_calls` 之后的各轮，工具清单只给提交工具。
   - 被拒的调用照旧记 `tool_cap`。
3. **`prompts.messages.status`**：字符串，或 `null`（默认，不追加）。
   - 每一轮执行完、样本还没结束时，作为一个文字项追加在这一轮最后一条 tool 消息的末尾（就是 `last_round` 提示现在放的位置）。
   - 占位符：`{round}`（刚结束的轮）、`{max_rounds}`、`{rounds_left}`、`{calls_used}`（计数的调用）、`{max_tool_calls}`、`{calls_left}`（不小于 0）。
   - 两者都有时，`status` 在前，`last_round` 在后。
   - 这一轮最后一条消息不是 tool 消息时报错（不会发生，样本没结束就一定有 tool 消息）。
4. **记录**：只在非默认时加字段，所以默认配置下逐字节不变。
   - 那一轮给的不是全部工具时，在轮记录里加 `"tools_offered": "submit_only"`；
   - 追加了 `status` 时，在轮记录里加 `"status_notice": true`。
   - `Protocol.as_dict()` 带上两个新开关，它们进运行身份。
5. **工作台选项透传**：`configs/w1.json` 新增一节 `"workbench": {"view_frames": false}`。
   - `W1Config.workbench_options()` 返回**与默认值不同**的项，默认时是空字典。
   - `run_episodes` 把它作为关键字参数传给 `workbench_factory(image, disabled_groups=…, **options)`。
   - 这一节整体进运行身份（config.yaml）。
   - `view_frames` 由并行的 DEV-0054 在 `Workbench.__init__` 里实现（关键字参数 `view_frames: bool = False`）。本任务只做透传，测试用接受 `**kwargs` 的假工作台。
   - 默认时什么都不传，所以 DEV-0054 合入前后都能跑。
6. `src/w1/config.py` 校验新键：枚举值合法；`status` 的占位符只能是上面六个；`final_round_tools = submit_only` 时 `final_round_refused` 必须有；`workbench` 只认 `view_frames`（bool）。

### 2. 冻结保护（`scripts/run_w1.py`）

- 划分是 `devcheck` 或以 `test` 开头时，`configs/w1.json`（或 `--w1-config` 给的文件）必须是 `"frozen": true`。
  - 否则退出码 2，报错写明"协议未冻结，不能碰开发-检查集与 test（D103 §二.4、§三）"。
  - 对全部设置都适用，基线也一样。
- `--tiny` 不受此限。
- `--w1-config` 可以指向别的文件（调参的变体放在 `configs/w1_tune/`，由管理员写）。运行身份里记这个文件的路径与 sha256（现在已记 sha256 的话，补上路径）。

### 3. 测试（`conda run -n visint pytest tests/ -q` 全部通过）

- 每个开关一组测试，用假适配器与假工作台：
  - `final_round_tools`：最后一轮给的工具清单；被拒的非提交调用；同一段里的提交照常执行；`max_rounds = 1` 时不生效。
  - `capped_tools`：满额之后的工具清单。
  - `status`：文字与占位符的值、位置，以及与 `last_round` 的先后。
  - 工作台选项透传：默认不传；非默认时传。
  - 冻结保护：未冻结时 `devcheck`、`test_*` 退出码 2；`devtune` 照常；`--tiny` 照常。
- 默认配置下 tiny 彩排的 `predictions.jsonl` 与改动前逐字节相同。做法：在改动前的 commit 上跑一次存下来，再比对。
- 写一个开了全部开关（`final_round_tools`、`capped_tools`、`status` 都开，`view_frames` 不开）的临时配置，跑一遍 tiny 彩排，rc=0。

## 不做

- 不改 `src/w1/workbench.py`、`src/w1/sandbox.py`、`docs/w1_tools.md`：这些由 DEV-0054 做。
- 不改提示词的正文，不改 `max_rounds`、`max_tool_calls` 等上限。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 每个开关的语义，以及与本任务书的任何出入；
- 测试结果原文；
- 逐字节核对的做法与结果。
