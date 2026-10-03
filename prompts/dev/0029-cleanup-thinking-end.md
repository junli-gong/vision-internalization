[DEV-TASK 0029] 收尾：E5 改用 `thinking_end` 四分类 + 几处遗留（公开停止 token、文档、E2 pre_proj 说明、偶发测试的线程固定、落地源的 script 参数）

## 背景

- DEV-0027（D60）给 `GenerationResult` 加了 `thinking_end`，取值有四种：
  - natural：自然收尾；
  - forced：被强制收尾；
  - not_opened：Gemma 与 InternVL 第一个 token 不是思考开头标记，直接作答；
  - unclosed：思考段里写出了停止 token，没有作答段。

  它提醒 E5 分开报时用 `thinking_end`，因为 `natural_close` 不等于 `not forced_close`。
- DEV-0028（D61）在两者并行时写 E5，假设思维链的 `thinking_unclosed` 一律为 False，`stopped` 指作答段以停止 token 结束。这与上面的四分类不完全一致。
- 其余是 D57、D60 记下的遗留。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D57、D60、D61
- `src/adapters/budget.py`、`src/adapters/base.py`（GenerationResult）、`src/e5/runner.py`、`src/e5/scoring.py`、`src/e1/scoring.py`（thinking_end 一节，可以照着做）

## 任务

1. **E5 用 `thinking_end`**：
   - E5 思维链的记录、metrics 与 README 按四类分开报条数与读数，口径与 E1 的 `thinking_end` 一节一致；
   - 删掉"thinking_unclosed 一律 False"的假设；
   - 测试用真实小模型：DEV-0027 的测试里构造四种结束方式的钩子可以复用，每一类都覆盖到。
2. **公开的停止 token**：`scripts/check_training_interfaces.py` 读私有 `ad._stop_ids()` 的地方，改为 `stop_token_ids()`。全仓库搜一遍，还有别的私有调用的话一起改，只改调用处。
3. **文档与说明**：
   - `scripts/run_e1.py` 的文档字符串里"冻结批大小 16"改为 64（D43）；
   - `scripts/run_e2.py` 的文档字符串，以及 `configs/e2_probes.json` 的 `references.pre_proj` 说明，改为"三家族都已实现"（D60）。E2 配置的 sha256 会因此变化，写进总结。E2 还没有正式运行，所以没有关系。
4. **偶发测试**：`tests/test_e3_train.py::test_master_rows_equal_dense_param_fp32` 在高负载下偶发失败，单独重跑 5/5 通过，推测是 CPU 多线程归约的顺序变了。
   - 在这条测试里固定线程数（例如 `torch.set_num_threads(1)`，测完恢复）；
   - 或者说明为什么应当改为带容差的比较，二选一，写明理由；
   - 测试的本意（两种模式逐位相同）不能放松，除非证明逐位相同本来就不成立。
5. **落地源的 script 参数**：`ade20k.py`、`nyu_depth_train.py`、`nyu_normals.py` 改为 `ctx.manifest(SOURCE, script=SCRIPT)`，不再改写 `m.data["script"]["path"]`（D57-4）。清单内容不变，加一条测试核对。

## 允许改动的文件

`src/e5/`、`scripts/check_training_interfaces.py`、`scripts/run_e1.py`（只改文档字符串）、`scripts/run_e2.py`（只改文档字符串）、`configs/e2_probes.json`（只改那段说明）、`tests/test_e3_train.py`（只改那条测试）、`src/data/land/ade20k.py`、`src/data/land/nyu_depth_train.py`、`src/data/land/nyu_normals.py`，以及对应的测试。

**不许碰**：`src/adapters/`、`src/e1/` 的逻辑、`src/e3/` 的逻辑、`src/synth/`、`configs/e1_prompts.json`、`configs/e5_prompts.json`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网（测试加 `HF_HUB_OFFLINE=1`）。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `run_e5.py --tiny`，思维链走真实小模型
- 总结里写明：E5 四类的报告格式、E2 配置的新 sha256、偶发测试的处理与理由。
