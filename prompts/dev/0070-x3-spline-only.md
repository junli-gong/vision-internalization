[DEV-TASK 0070] 探索 X-3：只训描样条（去掉修正流），样条加量；运行器支持没有修正的检查点

## 背景

- X-2（RUN-0080，DEV-0069）在 explore_val 的前两层上，各 400 句：

  | 设置 | refcoco 层 | refcoco+ 层 |
  |---|---|---|
  | `x2_select`（W2-c 式） | 0.655 | 0.594 |
  | `x2_spline_trimap` | 0.767 | 0.696 |
  | `x2_select5` | 0.772 | 0.693 |
  | 椭圆消融 | 0.592 | 0.530 |

- 带符号的局部修正无用：不稳定，与随机框相近。
- 样条的训练损失到 2,000 步时仍在下降（0.96 → 0.87），每步 6 条，约 0.86 个 epoch。
- X-3 的设计：
  - 去掉修正流，减少与语言争抢的任务；
  - 样条每步从 6 条加到 10 条；
  - 其余同 X-2：提出、挑选（K = 4，W2-c 数据）、回放（通用 3、计数 2）。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `src/w2/w2c_train.py` 的 `x2_spec` 与 X-2 流的机制（DEV-0069）、`configs/x2_train.json`
- `src/explore/x2_runner.py`、`scripts/run_x2.py`、`configs/x2_run.json`
- `docs/exploration/log.md` 的 X-2 一节

## 任务

1. **`x2_spec` 允许只有 spline 一节**。
   - 没有 `correct` 一节时只开样条流。
   - 数据仍用同一份 X-2 数据（RUN-0080 的 `0080_x2_data`），只取样条行；`--x2-data` 的核对相应放宽，只核对样条部分，数据参数只比 spline 一节。
   - 两节都在时，行为与 DEV-0069 逐字节相同：X-2 的检查点、运行身份不变，用现有测试与快照钉住。
   - 没有 spline 一节、只有 correct 一节，仍然报错。
2. **`configs/x3_train.json`**：等于 `configs/x2_train.json` 去掉 `correct` 一节，`spline.per_step` 改为 10，`tiny_per_step` 2。`version`、`status` 与说明照实改，写明依据是探索日志 X-2。
   - 每步 8 + 8 + 10 + 3 + 2 = 31 条，权重 1 : 1 : 1 : 0.5 : 0.5，2,000 步。
   - 文档字符串里给用时估计，按 RUN-0080 的每步 17.5 s 折算。
3. **运行器**：
   - `run_x2.py` 能加载只训了样条的检查点；
   - 涉及修正的设置（`*_cor*`、`*_randbox`）在这种检查点上跳过，并在 `x2_summary` 里写明"检查点没有修正流"；
   - 其余设置照常；
   - `x2_select`、`x2_spline_*`、`x2_select5`、最优候选、椭圆消融的输出格式不变。
4. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_x2_*.py tests/test_w2c_*.py -q`）：
   - 只有 spline 一节时：tiny 训练、续跑；
   - 两节都在时逐字节不变；
   - `run_x2` 在只训了样条的 tiny 检查点上端到端，修正类设置被跳过并记录；
   - x3 配置除 correct 与 spline 的条数、version、status 外与 x2 配置相同。

## 不做

- 不改 X-2 的行为、配置与检查点格式。
- 不改 W2 冻结的配置。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- `configs/x3_train.json` 的 sha256 与用时估计；
- 服务器命令，同 RUN-0080 的形式：`--x2-data` 指向 RUN-0080 的 `0080_x2_data`；
- 测试结果原文；
- 与任务书的任何出入。

尽量 2 小时内完成。
