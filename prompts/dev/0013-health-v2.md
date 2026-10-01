[DEV-TASK 0013] 健康检查协议 v2：MMLU-Pro 的标题口径改为 256 token，32 token 留作格式诊断（D28-7）

## 背景

- D26 固定的健康检查协议里，MMLU-Pro 是 `max_new_tokens=32`。
- RUN-0007 发现，Gemma 两个尺寸与 Qwen-0.8B 有 36–54% 的回答在 32 token 内还没给出字母。
- 项目负责人裁定（D28-7）：
  - 标题口径改为 256 token；
  - 32 token 那一列保留，作为格式诊断；
  - 标红规则作用于 256 那一列；
  - 截断率变化超过 10 个点时另行提示。
  
  理由：训练一定会改变输出的长度与格式，32 token 的读数会把"变啰嗦了"误报成"知识退化了"。
- 题目文件与其余协议都不变。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D26、D28-7
- 已合入的 `src/health/`、`scripts/health_check.py`
- `src/adapters/factory.py`（DEV-0011 新加）

## 任务

1. **协议版本升为 2**：
   - MMLU-Pro 有两个读数：
     - `mmlu_pro`：标题，`max_new_tokens=256`；
     - `mmlu_pro_fmt32`：诊断，取同一次生成的前 32 个 token。
   - greedy 下，生成 32 个 token 与生成 256 个 token 的前 32 个完全相同，所以只生成一次 256，再截出 32 的读数：
     - 截出的文本由前 32 个 token 解码；
     - `stopped` 为"前 32 个 token 里出现了停止 token"；
     - 解析规则与 v1 的 32 token 口径完全相同。
   - 写测试：在 `--tiny` 上，截出的读数与真的用 `max_new_tokens=32` 生成的结果逐字段相同。
2. **`compare`**：
   - 标红只看 `mmlu_pro`（256），GSM8K 与 IFEval 照旧；
   - 新增"截断率提示"：任一项在基座与合并后模型之间，写满 max_new_tokens 的比例变化超过 10 个百分点就列出，与标红分开；
   - `fmt32` 的准确率也给出差值与 CI，但只作诊断，不标红。
3. **兼容旧输出**：
   - v1 的输出文件（RUN-0007/0009）可以被读、被 `--rescore`、被 `--compare`；
   - 与 v2 比较时，MMLU-Pro 只比对得上的口径（v1 的 32 对 v2 的 fmt32），并在输出里写明"口径不同、标题不可比"。
4. **协议说明**（`src/health/__init__.py`）与 `protocol.py` 同步更新，写明 D28-7 与版本号。

## 允许改动的文件

- `src/health/`
- `scripts/health_check.py`
- `tests/test_health*.py`

**不许碰**：`src/adapters/`、`src/parsing/`、`src/data/`（另一个 agent 在做）、`configs/health_check_items.json`（题目不变）。

## 环境与验收

- 用 conda 环境 `visint`，不装包。
- 验收：
  - `conda run -n visint pytest tests/ -q` 全绿；
  - `bash scripts/pytest_server_parity.sh` 全绿；
  - 三条 `--tiny --limit 3` 冒烟；
  - 用 `--compare` 对 RUN-0009 的一对 v1 输出跑一次，确认兼容。
- 总结里写明：
  - v2 的输出格式变化；
  - 截出 32 读数的做法与测试；
  - 与 v1 输出比较时的规则。
