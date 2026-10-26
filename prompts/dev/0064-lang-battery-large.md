[DEV-TASK 0064] 更大题量的语言读数：MMLU-Pro 与 IFEval 与基座配对，把语言代价测准（D125-2 (c)、D125-3）

## 背景

- G-W2 §4：W2-c 在简报 8.11 的健康检查（固定 200 题）上，MMLU-Pro −6、IFEval −6 两项标红，但 CI 都含 0，n 太小。
- 负责人要求（D125）：
  - 用更大的题量与基座配对，把读数测准；
  - 在 9B 上，用同一套读数把 **E3b+C3、W2-c、W2-c 加回放**三者的语言代价并排列出。
- 现有的健康检查不动（它是简报 8.11 的固定协议）。本任务另写一个"大题量的语言读数"，口径尽量与健康检查相同，只是题更多。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `scripts/health_check.py`、`src/health/`（MMLU-Pro 字母作答、自由作答诊断、IFEval、GSM8K 的提示、解析与打分；`--checkpoint` 的加载方式，DEV-0061 加的）、`configs/health_check_items.json`（固定 200 题从哪来）
- DECISIONS.md 的 D119、D125；`reports/gates/G-W2.md` §4

## 任务

1. **`scripts/lang_battery.py`**：一个模型，可带 `--checkpoint`，跑以下读数，写 `<out>/lang_battery.json`、`predictions.jsonl`、README 与计算记账。
   - **MMLU-Pro 字母作答**：
     - 题目：从 MMLU-Pro 的 test 里，按 `hash_rank(salt='lang_battery_v1')` 按学科分层，取 N 题（默认 2,000）。
     - **排除**健康检查用的那 200 题，在完成总结里写明重叠数。
     - 提示、解析与打分与健康检查的 `mmlu_pro_letter` 完全相同。
   - **IFEval**：
     - 全部 541 条提示，打分用 `src/health/ifeval` 的同一套代码；
     - 报 prompt 级、instruction 级的严格与宽松四个读数。
   - 每个读数给点估计与 bootstrap 95% CI。
   - `--compare base.json other.json`：逐题配对，给差值与配对 bootstrap 95% CI。
   - 数据来源与下载：
     - 先查健康检查的题从哪来，尽量用同一份本地数据；
     - 需要下载完整的 MMLU-Pro 或 IFEval 时，写清在 HPC 上怎么下，放 `$PROJ/data`、不进仓库，开发机上用 fixtures 测；
     - 数据只限非商业研究，不再分发。
   - 模型加载与 `--checkpoint`（合并核对等）同 `scripts/health_check.py`，复用，不复制。思考开关与健康检查相同。
2. **服务器上的命令与用时估计**：写进脚本的文档字符串。
   - 在 hpcgpu06 上，基座 9B 与 W2-c、E3b+C3 两个检查点各跑一次，同节点配对；
   - W2-c 加回放的检查点以后用同一条命令。
   - 检查点路径：W2-c 在 `experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`；9B 的 E3b+C3 按 RUN-0056 的台账找。
3. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_lang_battery.py tests/test_health*.py -q`）：
   - tiny 模型上端到端；
   - 题目抽样可复现、与健康检查的 200 题不重叠；
   - 打分与健康检查同一套函数，在同样的题上逐题相同；
   - `--compare` 的配对。

## 不做

- 不改 `scripts/health_check.py` 的默认行为与它的 200 题。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 题目来源、抽样规则、与健康检查的重叠；
- 服务器命令与用时估计；
- 测试结果原文。
