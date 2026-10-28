[DEV-TASK 0072] `run_x2.py` 在 test 划分上运行：只经通过 G-EX 的候选与负责人的批准编号放行（D147）

## 背景

- 负责人批准（D147）：通过 G-EX 判定的候选在与 W2-c 相同的 test 子集上运行，即 W2 的 5 个 test 划分，每个 1,000 句，进论文。
  - 同一次运行还要导出"样条换成框内椭圆"的消融（`x2_ellipse_trimap`），进论文主表。
- 现在 `scripts/run_x2.py` 对 test 一律拒绝（第 112、155 行；预注册 X §1）。
- 判定 RUN-0090 正在跑，结论由 `gate_explore.py` 写进 `gate_explore.json`。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `scripts/run_x2.py`、`src/explore/x2_runner.py`、`src/data/explore_splits.py`（密封守卫的写法）、`scripts/gate_explore.py`（`gate_explore.json` 的键）
- `scripts/run_w2.py`（test 划分的取样与冻结配置的要求）、预注册 X-5、X-5a、DECISIONS.md D147

## 任务

1. **`run_x2.py` 加 `--test-gate <gate_explore.json 的路径>` 与 `--test-approval <批准编号>`**：test 划分只有两者都给、且都合格时才放行。
   - `--test-approval` 必须恰是 `D147`，写成常量，注明依据。
   - `gate_explore.json` 必须满足：
     - 结论是"通过"，且不是"彩排，不作数"；
     - 判定点核对全部为真；
     - 其中候选的检查点 manifest sha256 等于本次 `--checkpoint` 的；
     - 其中候选的设置包含本次要导出的主设置。
   - 任何一条不满足就拒绝，退出码 2，什么都不写。
   - 放行时把批准编号、`gate_explore.json` 的路径与 sha256 写进运行身份（config.yaml），并追加一行到 `<out>/<模型>/test_unlock.jsonl`。
   - W2 配置必须冻结，同 devcheck2、devcheck3。
   - 不给这两个参数时，对 test 的拒绝与现在逐字相同。
2. **test 划分的取样**：与 `run_w2.py` 的 test 划分同一份清单（`configs/w2_splits.json` 的 test_*），同一个取样函数，句子集合逐 id 相同，便于与 RUN-0072 的 W2-c 逐句配对。
3. **其余行为不变**：explore_val、devtune800、devcheck2、devcheck3 的路径与输出逐字节不变，快照或测试钉住。
4. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_x2_*.py tests/test_gate_explore.py -q`）：
   - tiny 上用一份合成的 `gate_explore.json`，覆盖：通过时放行；不通过、彩排、manifest 不符、设置不符、批准编号不对、缺任一参数时都拒绝；
   - 放行后运行身份与 `test_unlock.jsonl` 有记录；
   - test 句子集合与 `run_w2` 的 test 划分逐 id 相同；
   - 其余划分输出不变。

## 不做

- 不改 `gate_explore.py` 与判定规则；不改 W2 冻结的配置。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 放行的条件与报错；
- 服务器命令：例子用 X-2 的检查点、`--settings x2_spline_trimap x2_ellipse_trimap`、5 个 test 划分；
- 测试结果原文。

尽量 1.5 小时内完成。
