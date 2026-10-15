[DEV-TASK 0046] E5 训练后评测前的合并核对改用判据 e5-v2（D96 第 1 项），加一项指标层面的合并前后比较（只报告，D96 第 2 项），加只做核对的模式

## 背景

- RUN-0051（hpcgpu07）的 E5 主比较到 10-15 23:00 UTC 做了 15 次训练后核对。照判据 v1（D50-3），只有 2 次通过，13 个运行的 test 评测被拒。
- 没过的只有两项：
  - restricted 的 `decode_vs_generate_batch.exact_rows`（< 0.95）；
  - merge 的全词表教师强制 argmax 一致率（< 0.98）。
- 只看允许集合时，合并前后一致率都 ≥ 0.999；所有分歧都在平局附近（gap 最大约 1.2）。
- E5 的正式评测走结构强制解码（D83-3），只在允许集合里选，上面两条路径都不走。
- 用户批准 (b′)（N34 → D96）：对 E5 训练后的核对，判定改看允许集合与受限教师强制，上面两项改为只报告；另加一项在 E5 正式指标上直接比较合并前后的检查，只报告。
- E1、E3 的核对（`scripts/check_training_interfaces.py` 的判据 v1）**一律不变**。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D50、D83、**D96**；`DECISIONS_NEEDED.md` 的 N34（含逐个运行的读数表）
- `src/e5/trained_eval.py`（`stacked_merge_check`、`_reusable`、`record`、`run_task`）
- `scripts/eval_e5_trained.py`（第 180–245 行：合并核对的调用与 key）
- `scripts/check_training_interfaces.py`：`criteria_block`、`grid_criteria_block`、`judge` 与它调用的各节判定函数，`GenAgg`、`TfAgg` 的 summary 结构（`n_divergence_without_gap`、`gap` 的 n / median / max）
- `src/e5/token_grid.py`（`decode_grids`、`STRUCTURED`）、`src/e5/scoring.py`（`score`、`GATE_IOU_KEY = "iou_grid"`、起伏的 `relief.error`）
- 真实的核对文件：`experiments/E5/0051_e5_main_hpcgpu07/trained/*/qwen3.5-2b/n*/seed*/merge_check.json`

## 任务

### 1. 判据 e5-v2（只用于 `stacked_merge_check`）

1. 新的判据版本 `e5-v2`。写成一个纯函数，输入 merge_check.json 的 `checks`，输出各节的失败原因，可以直接拿已有的文件测试。阈值放在一处常量里。
2. **判定项**（GPU / bf16，非精确模式），照 D96 第 1 项原文：
   - `restricted.across_merge` 的教师强制 `argmax_agreement` ≥ 0.98；
   - `restricted.by_thinking.<模式>.teacher_forcing.agreement` ≥ 0.999；
   - **所有比较**里分歧处的 gap：最大值 ≤ 2.0，中位数 ≤ 0.5。所有比较包括：
     - merge 的 generate 与 teacher_forcing；
     - restricted 的 decode_vs_generate_batch 与 teacher_forcing；
     - across_merge 的 teacher_forcing。
     - gap 的 n = 0（没有分歧）时这一条自然通过。只看 `gap`，`top2_gap` 只报告。
   - 任何比较里有分歧而算不出 gap 的（`n_divergence_without_gap` > 0），判不过；
   - 核对出错（`error` 不空），判不过。
3. **只报告**：
   - merge 的全词表 `teacher_forcing.argmax_agreement`；
   - `decode_vs_generate_batch.exact_rows`、`token_agreement`，以及 merge 的 `generate.*`。
4. **精确模式**（CPU fp32，`exact=True`，只在 tiny 测试里出现）：沿用 v1 的精确规则（处处逐个相同），不放宽。
5. **v1 的判定照算，只报告**：`doc["v1_report_only"] = {"criteria_version": "v1", "failures": …, "pass": …}`，用 `cti.judge` 与现有的 `criteria_block` / `grid_criteria_block`。
6. `doc["criteria_version"] = "e5-v2"`，`doc["criteria"]` 写清：判定项与阈值、只报告的项、精确模式的规则，依据写 "D96-1（N34 的 (b′)）"。`pass` 只由 e5-v2 决定。
7. **复用与旧文件**：
   - `_reusable` 改按 `e5-v2` 判。旧的 v1 文件（不论过没过）都不复用，重新核对。
   - 写新文件之前，如果已有的 merge_check.json 的 `criteria_version` 与当前不同，先改名留存为 `merge_check.<旧版本>.json`。重名就加 `.1`、`.2` 等后缀。证据不删。
   - 同一版本的重试照旧覆盖。

### 2. 指标层面的合并前后比较（只报告，D96 第 2 项）

1. 放在同一次 `stacked_merge_check` 里，用同一批 40 条 val 样本、同样的成批方式：
   - 合并之前（A，E5 LoRA 未合并）对每批跑一次 `decode_grids(..., decode=STRUCTURED)`；
   - 合并之后（B）再跑一次；
   - 两边都用 E5 的正式解析与评分，与 `record()` 走同一套 `e5tok.pred` 与 `SC.score`，relief_expected 同正式评测。
2. 写进 `doc["metric_level"]`，`report_only: true`，**不影响 `pass`**。内容：
   - **逐行**：id、任务、A / B 的 `iou_grid`、`iou_full`、起伏误差（a、b 两类）、两边的预测网格（与起伏）是否逐格相同。
   - **逐类与总体**：
     - A、B 的 `iou_grid` 平均、差（B − A）、逐行差绝对值的最大值；
     - 起伏误差（只取两边都有效的行）的平均、差、最大逐行差；
     - 预测完全相同的行数；无效预测数。
     - 另给只算标题口径行（`in_headline`）的同一组读数。
   - **阈值与标记**：
     - 总体 `iou_grid` 平均差的绝对值 > 0.01 → `flags.iou = true`；
     - 总体起伏误差平均差的绝对值 > 0.02 → `flags.relief = true`。
     - 0.02 是 D96-2 里管理员定的噪声底；阈值写进文件，依据写 D96-2。
   - 耗时（A、B 各自的秒数）。
3. 这一项出错时，把错误写进 `metric_level.error`，核对的 `pass` 不受影响；评测脚本在 stderr 打一行警告。不要静默吞掉。
4. 要确认 `decode_grids` 在 LoRA 挂着、未合并时走的是与正式评测相同的结构强制路径，没有偷偷合并或绕开 LoRA。精确模式下 A、B 的读数必须逐个相同、差恰好为 0，用测试钉住。
5. 正式评测本身（`run_task`）不变。

### 3. 只做核对的模式

- `scripts/eval_e5_trained.py` 加 `--check-only`：
  - 加载、做 `stacked_merge_check`（e5-v2 加指标层面），写 merge_check.json，打印摘要；
  - 不评任何任务，不建任何任务目录。
  - 通过退出 0，不过退出 1；退出码与 `--task` 无关。
- 用途：
  - 已经评过 test 的两个运行（判据 v1 下通过的），补出 e5-v2 与指标层面的读数，不重评 test；
  - 已有的 test 输出一字节都不能动。
- 正常评测（不加 `--check-only`）时，在 stderr 打一行指标层面的摘要（总体差与两个标记）。

### 4. 测试（`conda run -n visint pytest tests/ -q` 全部通过）

1. **真实文件的回归**：
   - 把仓库里现有的 `experiments/E5/0051_e5_main_hpcgpu07/trained/*/qwen3.5-2b/n*/seed*/merge_check.json` 的 `checks` 一节，连同路径与文件里记录的 pass / failures，抽成一个测试夹具（`tests/fixtures/e5/`）。测试不依赖 experiments 目录。
   - 断言：
     - v1 的 judge 复现每个文件里记录的判定；
     - e5-v2 全部通过。
   - 管理员按 (b′) 手算的结果：到 10-15 23:00 UTC 的 15 个文件都过，其中 v1 下只有 e3b_c3 4K 种子 0、e3b 4K 种子 0 通过。夹具里的文件数以仓库为准。
2. **合成的边界**（逐条一个用例）：
   - across_merge 0.979 不过；
   - 受限教师强制 0.9989 不过；
   - 任一比较 gap 最大 2.01 不过、恰好 2.0 通过；
   - 中位数 0.51 不过；
   - 有分歧算不出 gap 不过；
   - error 不空不过；
   - exact_rows 0.5、全词表 merge 0.90 照样通过（只报告）。
3. **tiny 端到端**（CPU fp32）：
   - merge_check.json 是 e5-v2，有 `v1_report_only` 与 `metric_level`，指标层面的差恰好为 0、预测逐行相同；
   - 已有的 v1 文件被改名留存；
   - `--check-only` 不建任务目录、退出码正确；
   - 正常评测的 predictions / metrics / README 与改动前逐字节相同。现有的快照测试（`tests/fixtures/e5/trained_eval_*.py` 一类）照样通过。
4. E1、E3 的核对测试照旧全部通过（`scripts/check_training_interfaces.py` 不改）。

### 5. 文档

- 更新 `src/e5/trained_eval.py` 与 `scripts/eval_e5_trained.py` 的模块说明：e5-v2 的判定、只报告的项、指标层面一项、`--check-only`。
- 估一下每个运行多出的 GPU 时间：
  - 两次结构强制解码，各 40 条；
  - 参照 RUN-0051 的核对耗时，见 merge_check.json 的 `timing_s`。

## 不做

- 不改 `scripts/check_training_interfaces.py` 的判据与阈值。不改 E1、E3 的任何评测路径。不改 `configs/` 的任何数值。
- 不改 `prompts/server_hpc/` 下的驱动脚本（管理员另写补充任务书）。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；e5-v2 判定函数的位置与签名；merge_check.json 新增的字段。
- 真实文件回归的逐个结果表：路径、v1 判定、e5-v2 判定。
- 测试结果原文（通过数 / 用时）。
- 多出的 GPU 时间估计。
- 任何你认为与 D96 原文有出入、或需要管理员定的地方。
