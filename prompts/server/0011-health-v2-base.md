[RUN-TASK 0011] 健康检查协议 v2：8 个基座重跑（MMLU-Pro 标题 256 token，D28-7）

## 为什么做

- 项目负责人裁定（D28-7）：MMLU-Pro 的标题口径改为 256 token，32 token 只作格式诊断。
- DEV-0013 实现了协议 v2（D31）：只生成一次 256 token，32 token 的读数从中截出。
- 这次在 8 个基座上按 v2 重跑，得到以后各门的参照。
- 顺带核对两件事：
  - 同一套题、同样 greedy 下，GSM8K 与 IFEval 的结果是否与 RUN-0007/0009 逐题相同（确定性）；
  - 截出的 32 token 读数，在 GPU bf16 上是否与真的只生成 32 个相同（`--audit-fmt32`）。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D26、D28-7、D31
- `python scripts/health_check.py --help`

本任务只运行，不改任何仓库代码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含本任务书所在的 commit。不包含就停下报告。

## 2. 环境

- 不装包：`pip install --dry-run -r requirements.txt` 应当没有任何 "Would install"。有的话停下报告。
- `export NLTK_DATA=$PROJ/nltk_data`：RUN-0009 已装好并核对过 punkt_tab。

## 3. 运行

- 模型与顺序、内存盘规则同 RUN-0007（每个模型跑完即删，只保留 qwen3.5-0.8b，内存盘 ≤ 25 GB）。权重按 `experiments/P0/0009_ifeval_rescore/weights_manifest.json` 核对。
- 每个模型一条命令：

  ```bash
  python scripts/health_check.py --model <键> --device cuda --dtype bf16 --audit-fmt32 100 \
    --out experiments/P0/0011_health_v2_base/health_<键>.json
  ```

- 每个模型跑完，接着做与 v1 的比较：

  ```bash
  python scripts/health_check.py --compare experiments/P0/0009_ifeval_rescore/health_<键>.json \
    experiments/P0/0011_health_v2_base/health_<键>.json \
    --out experiments/P0/0011_health_v2_base/compare_v1_v2_<键>.json
  ```

- 退出码为 2 时照常推回，report 里贴出报错，继续下一个模型。

## 4. 产出与回传

- `experiments/P0/0011_health_v2_base/`：
  - 8 份 `health_<键>.json`、8 份 `compare_v1_v2_<键>.json`
  - `disk_ledger.tsv`、`README.md`
- `reports/0011-health-v2-base.md`，按 report 模板，第 3 节原样贴：
  - **分数表**：MMLU-Pro 256 与 fmt32、GSM8K flexible 与 strict、IFEval 四个口径；
  - **与 v1 的逐题核对**：
    - GSM8K 与 IFEval 每个模型逐题相同的条数；
    - fmt32 与 RUN-0007 的 32 token 逐题相同的条数；
  - **`--audit-fmt32` 的结果**：100 条里逐字段相同的条数；
  - **截断率**：MMLU-Pro 在 256 与 32 下的截断率；
  - **计算量**：墙钟与生成 token 数。

  第 6 节写磁盘收支。只贴数字与事实，不做解读。
- 分两批：
  1. 前 4 个模型跑完推一次；
  2. 其余模型跑完后，连同 report 一起推。

```bash
git add <逐个路径> && git commit -m "[results] 0011: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0011-health-v2-base.md` 追加进完成清单，重新进入值守。
