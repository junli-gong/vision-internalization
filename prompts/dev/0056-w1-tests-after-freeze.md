[DEV-TASK 0056] W1 测试：冻结之后，把"默认、未冻结配置"的测试改用一份冻结前的草稿配置作夹具（D111；不改被测代码）

## 背景

- D111 把 `configs/w1.json` 冻结为调参臂 U2：
  - `frozen: true`；
  - 最后一轮只给提交工具；
  - "最后一轮"提示的正文改写，并作为 user 消息发出。
- 有 17 项测试把仓库里的 `configs/w1.json` 当作"默认、未冻结"的配置来用，冻结之后都失败了：
  - `tests/test_w1_knobs.py`：
    - `test_status_notice_values_and_position`
    - `test_status_with_partial_placeholders_and_with_final_round_switch`
    - `test_default_config_keeps_previous_behaviour`
    - `test_unfrozen_config_refuses_devcheck_and_test`（10 个参数化实例）
    - `test_frozen_guard_exit_code_from_the_command_line`
  - `tests/test_w1_loop.py::test_history_format_and_last_round_notice`
  - `tests/test_w1_notice_role.py`：
    - `test_default_roles_are_tool_and_records_unchanged`
    - `test_config_defaults_and_roles`
    - `test_default_identity_records_tool_roles`
- 冻结前的草稿：`git show 4c9d92bb:configs/w1.json`，sha256 `1cd4524f89a44c4446923745ad91763ef701d32a3451414bb29f26b7770fb681`（DEV-0055 合入之后、D111 冻结之前的那一份）。

## 任务

1. 把这份草稿原样存为夹具 `tests/fixtures/w1/w1_draft_unfrozen.json`。测试里钉住它的 sha256。
2. 上面这些测试改为读夹具，不读 `configs/w1.json`。
   - 测试的意图不变：默认行为逐字节不变，未冻结时拒跑开发-检查集与 test，提示的默认走法。
   - 能用参数化或共用 fixture 的就共用，不要复制粘贴。
3. 另加 3 项测试，钉住冻结后的仓库配置：
   - `configs/w1.json` 的 `frozen` 为 true；
   - 它与 `configs/w1_tune/U2.json` 除 name、status、frozen 外完全相同；
   - 用冻结配置跑 `scripts/run_w1.py` 的参数检查时，`devcheck` 不再被冻结保护拒绝。只验证参数检查这一步，可以用 `--tiny`，或者直接调 `needs_frozen` 与读配置的那段逻辑。
4. 跑 W1 的全部测试与 `tests/test_adapter_chat.py`（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_w1_*.py tests/test_adapter_chat.py -q`），全部通过。

## 不做

- 不改 `src/`、`scripts/`、`configs/`。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些测试；
- 测试结果原文。
