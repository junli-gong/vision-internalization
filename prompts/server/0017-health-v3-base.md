[RUN-TASK 0017] 健康检查协议 v3：8 个基座重跑（MMLU-Pro 受限字母读数 + 正确选项概率，D36-4）+ GPU 上的字母读数核对

## 为什么做

- 项目负责人裁定（D36-4）：健康检查里 MMLU-Pro 的标题改为受限字母读数，另记正确选项的平均概率。协议 v3 冻结，G3 之前不再动。
- DEV-0016 实现了 v3（D37）：
  - 适配器的 `option_logits` 取第一个生成位置的 logits，每个字母在 "X" 与 " X" 两种写法里取较大的 logit；
  - v2 的两个读数（256 token 自由作答、前 32 token）保留作诊断。
- 这次在 8 个基座上按 v3 重跑，得到 G3 的参照。
- 顺带核对两件事：
  1. 确定性：v3 里的 free256、fmt32、GSM8K、IFEval，与 RUN-0011（v2）是否逐题相同；
  2. GPU bf16 上，`option_logits` 选出的字母与 `generate(allowed_token_ids=全部写法, max_new_tokens=1)` 是否逐题相同（D37-4 的遗留）。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D31、D36（第 4 条）、D37
- `python scripts/health_check.py --help`
- `src/health/run.py` 的 `letter_text_only`、`spelling_table`

本任务只运行，不改任何仓库代码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `8c7d9b1`（DEV-0016 的合入）与本任务书所在的 commit。不包含就停下报告。

## 2. 环境

- 不装包：`pip install --dry-run -r requirements.txt` 应当没有任何 "Would install"。有的话停下报告。
- `export NLTK_DATA=$PROJ/nltk_data`。
- 模型与顺序、内存盘规则同 RUN-0011：
  - 每个模型跑完即删，只保留 qwen3.5-0.8b；
  - 内存盘 ≤ 25 GB；
  - 权重按 `experiments/P0/0009_ifeval_rescore/weights_manifest.json` 核对。

## 3. 运行

每个模型一条命令：

```bash
python scripts/health_check.py --model <键> --device cuda --dtype bf16 \
  --out experiments/P0/0017_health_v3_base/health_<键>.json
```

跑完接着与 v2 比较：

```bash
python scripts/health_check.py --compare experiments/P0/0011_health_v2_base/health_<键>.json \
  experiments/P0/0017_health_v3_base/health_<键>.json \
  --out experiments/P0/0017_health_v3_base/compare_v2_v3_<键>.json
```

退出码为 2 时照常推回，report 里贴出报错，继续下一个模型。

### 3.1 字母读数核对（探针，只读不改）

在 `experiments/P0/0017_health_v3_base/probe_scripts/audit_letter.py` 写一个探针。只调用仓库里现有的函数，不改仓库代码。每个模型在健康检查之后、删除权重之前跑一次：

- **适配器**：与 `scripts/health_check.py` 相同的方式构造（同一个 `make_adapter`、同样的参数）。
- **题目**：MMLU-Pro 的前 100 题，提示与输入的构造与 `src/health/run.py` 的 `letter_text_only` 相同：零张图、`P.ORDER`、关思考。
- **对每一题**：
  1. 调一次 `option_logits`，得到选中的字母与它所用的 token。
  2. 调一次 `adapter.generate`：
     - 参数：`max_new_tokens=1`、`enable_thinking=False`、greedy（`P.GREEDY_KWARGS`）；
     - `allowed_token_ids` 是本题全部字母的全部单 token 写法，取自 `option_logits` 返回的 spellings。
  3. 记下三项：
     - 生成的 token 与 `option_logits` 选中的 token 是否相同；
     - 两者对应的字母是否相同；
     - 第 1 步选中的字母与 `health_<键>.json` 里同一题的 `mmlu_pro_letter` 是否相同。
- **输出**：写 `audit_letter_<键>.json`，每题一行，另给三项的相同条数。

不计入记账，不改分数。如果不改仓库代码就写不出这个探针，跳过本节，在 report 里说明原因。

## 4. 产出与回传

- `experiments/P0/0017_health_v3_base/`：
  - 8 份 `health_<键>.json`、8 份 `compare_v2_v3_<键>.json`、8 份 `audit_letter_<键>.json`；
  - `probe_scripts/`、`disk_ledger.tsv`、`README.md`。
- `reports/0017-health-v3-base.md`，按 report 模板，第 3 节原样贴：
  - **分数表**：
    - `mmlu_pro_letter` 的准确率与正确选项的平均概率；
    - `mmlu_pro_free256` 与 `mmlu_pro_fmt32`；
    - GSM8K 的 flexible 与 strict；
    - IFEval 的四个口径。
  - **与 v2 的逐题核对**：每个模型 free256、fmt32、GSM8K、IFEval 各自逐题相同的条数。
  - **字母读数核对**：每个模型三项各自的相同条数（/100）。
  - **各家族实际用到的字母写法与 token id**：取自输出的 option_spellings。
  - **计算量**：墙钟与生成 token 数。
- 第 6 节写磁盘收支。只贴数字与事实，不做解读。
- 分两批：
  1. 前 4 个模型跑完推一次；
  2. 其余模型跑完后，连同 report 一起推。

```bash
git add <逐个路径> && git commit -m "[results] 0017: <本批摘要>" && git push
```

预计 1.5–2 GPU 小时：RUN-0011 用了 84 分钟，字母读数与探针每个模型各加不到 2 分钟。累计超过 3 小时，跑完当前模型就停，推回已完成的部分。

完成后按 STANDBY.md，把 `0017-health-v3-base.md` 追加进完成清单，重新进入值守。
