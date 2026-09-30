# experiments/P0/0005_thinking_probe — RUN-0005 思考行为事实探针

服务器 CC，2026-09-30。Phase 0 事实探针：8 个 E1 模型 × 12 条提示 × 3 种生成设置。不评分，不调提示词，不是 E1 基线。运行报告为 `reports/0005-thinking-probe.md`。

| 文件 | 内容 |
|---|---|
| `records.jsonl` | 每条生成一行（字段见任务书 §3.6；口径见下） |
| `summary.md` | 由 `probe_scripts/p9_summary.py` 从 records.jsonl 生成：表 1–3，以及逐条记录表 |
| `gen_params.json` | 每个模型实际用的生成参数与出处、思考标记与停止 token 的 id（运行时从 tokenizer 查）、种子规则 |
| `image_sha256.txt` | 图 A、图 B 的 sha256 |
| `disk_ledger.tsv` | 每次下载、删除前后的磁盘与内存盘收支 |
| `probe_start_epoch.txt`、`budget_stop.txt` | 第一条生成的时刻（4 小时预算从这里算）；若触发预算停止，记录停在哪一条 |
| `logs/` | 驱动脚本与各模型探针的 stdout/stderr（原样），以及 generation_config 核查 |
| `probe_scripts/` | `p9_thinking_probe.py`（探针，一次一个模型）、`p9_summary.py`（汇总）、`run_models.sh`（驱动：下载 → 探针 → 删除） |

口径（`p9_thinking_probe.py` 的 `analyse`）：
- **停止 token**：取适配器模块的 `STOP_TOKENS`，id 从 tokenizer 查。
- **`hit_max_new_tokens`**：生成满 max_new_tokens，且最后一个不是停止 token。
- **`think_open`**：
  - Qwen 开思考时由提示预填，记为 True，`think_open_source="prompt"`；
  - 其余情形看输出里有没有开头标记（Gemma 为 `<|channel>`，Qwen/InternVL 为 `<think>`）。
- **`n_think_tokens`**：从开头标记的下一个 token 起（Qwen 预填时从输出第一个 token 起），到第一个结束标记之前。没有闭合时数到输出末尾，末尾的停止 token 不计。
  - Gemma 的开头标记后面跟着 "thought\n"，这几个 token 计入思考 token。
- **`n_answer_tokens`**：最后一个结束标记之后的非停止 token 数。没有思考段时，是全部非停止 token；开了思考但没闭合时为 0。
- **`any_think_marker_in_output`**：输出里出现开头标记或结束标记；Gemma 另外检查 `<|think|>`。
