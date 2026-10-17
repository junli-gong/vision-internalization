[DEV-TASK 0048] E5 训练后评测加"不合并"开关（`--unmerged`）：E5 LoRA 挂着、不合并，用正式评测的同一条路径在 test 上评，输出另放一个叶目录（D99-2 的敏感性分析）

## 背景

- D99-2（负责人，10-17）：H5 判定用到的 12 个运行（E3b+C3 与 C3+预热，1K、4K，各 3 个种子），全部用**未合并的 E5 LoRA** 在 test 上再评一次。
- 门报告同时报告合并与未合并两种口径下的 H5 结论；起伏误差的噪声底从这次对比实测。
- 正式评测（`scripts/eval_e5_trained.py`，`src/e5/trained_eval.run_task`）现在要求合并之后才评：`run_task` 里 `ad.lora_state not in ("merged", "none")` 就报错。
- 这次只加一个开关，正式评测的行为与输出一字节都不变。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D96、D97、**D99**；`PREREGISTRATION.md` 的 E5 §8c（含 D99-2 补充）
- `src/e5/trained_eval.py`：`EvalSpec`、`leaf`、`run_identity`、`run_task`、`finalize`、`readme_text`、`stacked_merge_check`、`LoraAudit`（DEV-0046）
- `scripts/eval_e5_trained.py`：`cmd_run`，含 `--check-only` 的写法
- 现有的快照测试：`tests/fixtures/e5/*_4bd7f13.py`、`tests/test_e5_merge_check_e5v2.py`、`tests/test_e5_v3.py`、`tests/test_e5_start_only.py`

## 任务

1. **`scripts/eval_e5_trained.py --unmerged`**（只用于 `--checkpoint`，与 `--start-only`、`--check-only` 互斥）：
   - 加载同正式评测：建 base → 合并起点 LoRA → 核对起点 → `load_trained(E5 检查点, merge=False)`，**不做合并**，E5 LoRA 保持 injected。
   - 不做评测前的合并核对。运行身份与运行记录里写明跳过及理由："D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json"。
   - 评测走 `run_task`、`decode_grids(decode=STRUCTURED)`，与正式评测完全同一条路径。样本、提示、批大小、解析、评分、汇总、门槛都不变。
   - **LoRA 审计**：评测期间用 `LoraAudit` 包住解码，核对 E5 LoRA 层确实被调用，规则同 DEV-0046。审计结果写进 `logs/run_<k>.json`。审计不过就报错，不写 metrics。
   - **输出**：同一个模型目录下另起叶目录 `<任务>[__dev]__structured__unmerged`。
     - 运行身份另记 `lora_merge: "unmerged"` 与依据 D99-2；
     - `metrics.json` 另记 `"merge": "unmerged"`；
     - README 写明。
     - 正式评测的叶目录、`merge_check.json` 都不碰。
   - 支持 `--resume`，规则同正式评测：续跑时运行身份必须相同。
   - `run_task` 的合并检查放宽一种情况：只有 `spec.unmerged` 为真时，允许 `injected`；其余照旧报错。
2. **不改别的**：
   - 不带 `--unmerged` 时，`predictions.jsonl`、`metrics.json`、`config.yaml`、`README.md`、`merge_check.json`、打印行都与改动前逐字节相同，用现有快照测试证明。
   - 不改 `configs/`、驱动脚本、E1、E3。
3. **测试**（`conda run -n visint pytest tests/ -q` 全部通过）：
   - tiny、CPU、fp32 下，同一个 E5 检查点：
     - 合并评测与 `--unmerged` 评测的 `predictions.jsonl` 里，逐条预测与得分完全相同（fp32 下合并前后一致）；
     - `metrics.json` 除 `merge` 字段外的读数完全相同；
     - 叶目录名、运行身份里的 `lora_merge`、README 都对。
   - `--unmerged` 时 LoRA 审计生效：禁用 adapter 时报错（同 DEV-0046 的测试做法）。
   - 参数互斥：`--unmerged` 与 `--start-only`、`--check-only` 一起给时报用法错误。
   - `--unmerged` 的续跑：中途停下再 `--resume`，结果与一次跑完逐字节相同。
   - 正式评测的快照测试照旧通过。
4. **估时**：估一下未合并评测比合并评测每条慢多少（LoRA 多一路计算），给出 12 个运行在 A100 上的总 GPU 时间估计。参照 RUN-0051 的 test 评测墙钟：约 0.4 小时 / 运行。

## 不做

- 不写驱动或队列脚本（管理员另写）。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；`--unmerged` 的行为；新增的运行身份与 metrics 字段。
- 测试列表与结果原文。
- 估时。
- 任何需要管理员定的地方。
