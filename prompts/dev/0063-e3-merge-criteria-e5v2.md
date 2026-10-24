[DEV-TASK 0063] E3 的结构强制网格评测可以选用 e5-v2 合并核对判据（N37(b)；D102-2、D113-2；不改默认行为）

## 背景

- 负责人对 N37 的答复（D102-2）：检查点之后，把 E3 的结构强制网格评测统一到 e5-v2 判据，补评被拒的两个种子：4B 的 E3b 种子 2 与 2B 的 E3b 种子 1。连同机内配对所需的重评一起做，标"按 N34 统一判据，决定于看到被拒种子的读数之前"。
- D113-2：N37(b) 在 GPU 空档里做。
- e5-v2 判据（D96，DEV-0046）现在只在 E5 的训练后评测里用：`src/e5/trained_eval.py` 的 `CRITERIA_VERSION = "e5-v2"`，入口是 `scripts/eval_e5_trained.py`。
- E3 的网格评测 `scripts/eval_e3b.py` 用的是 v1 判据（`src/e3/merge_check.py` → `scripts/check_training_interfaces.py`）。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- DECISIONS.md 的 D96、D99、D102
- `src/e5/trained_eval.py`：e5-v2 怎样判，与 v1 的差别是"平局翻转"的容忍
- `src/e3/merge_check.py`、`scripts/check_training_interfaces.py`、`scripts/eval_e3b.py`、`configs/e3b_train.json` 的 eval.merge_check 一节

## 任务

1. `scripts/eval_e3b.py` 加 `--merge-criteria {v1,e5-v2}`，默认 v1，即现有行为逐字节不变。
   - 取 e5-v2 时，网格评测前的合并核对改用 e5-v2 的判法，复用 `src/e5/trained_eval.py` 的同一套函数，不复制。
   - `merge_check.json` 里记 `criteria_version`。
   - 运行身份记判据。
   - 已有的 v1 结果不会被当作 e5-v2 复用，反过来也一样。
2. 结果目录的约定：换了判据的评测必须能写进一个新的 `--out`，不覆盖原评测目录。
   - `scripts/mgr/interference.py` 与论文用图要能读到新目录：通过把新目录放在目录列表的前面来覆盖。写清用法。
3. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_e3*.py tests/test_merge_check*.py -q` 里相关的部分）：
   - 默认 v1 逐字节不变；
   - e5-v2 在 tiny 检查点上能跑通；
   - 构造一个"v1 拒绝、e5-v2 通过"的情形，即平局翻转，核对判据确实不同；
   - 运行身份里记了判据。

## 不做

- 不改 E5 的代码，不改 E3 的训练。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 服务器上补评的命令：
  - 2B 的 E3b 种子 1 与它的机内配对 E3b+C3 种子 1；
  - 4B 的 E3b 种子 2 与它的机内配对 E3b+C3 种子 2；
  - 检查点在 HPC 上的路径按现有台账（`ckpt_ledger.tsv`、RUN-0048、RUN-0052 的报告）找。
- 测试结果原文。
