[RUN-TASK 0012] E1 批处理验收（D28-5）与吞吐、算力实测（预注册 §9）

## 为什么做

- E1 预注册交项目负责人批准之前，还缺两样东西：
  1. **批处理验收**：在开发子集上，与 batch-1 greedy 相比，解析后答案的一致率 ≥ 98%，且指标差落在 bootstrap CI 内（D28-5）；
  2. **算力估算**：按真实提示与生成上限，实测吞吐，外推每个任务、每个模型的 GPU 小时。
- 这次的用量计入 E1 的预算（D28-10）。
- 本任务只用开发子集，不碰任何评测子集。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D28（第 3、5、10 条）、D29、D30、D32
- `PREREGISTRATION.md` 的 E1 草稿
- `python scripts/run_e1.py --help`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `a6c0c41` 与本任务书所在的 commit。不包含就停下报告。

## 2. 环境

- `pip install --dry-run -r requirements.txt` 应当没有任何 "Would install"。有的话停下报告。
- 设置 `export VISINT_DATA_ROOT=$PROJ/data`、`export NLTK_DATA=$PROJ/nltk_data`、`export TMPDIR=/dev/shm/visint/tmp`。
- 内存盘规则同前：每个模型跑完即删，只保留 qwen3.5-0.8b，任何时刻 ≤ 25 GB。权重按 `weights_manifest.json` 核对。

## 3. 批处理验收

**模型**：`qwen3.5-0.8b`、`gemma-4-e2b-it`、`internvl3.5-2b-hf`、`qwen3.5-9b`。每个模型一条命令：

```bash
python scripts/run_e1.py --acceptance --model <键> --task all --batch-size 16 --limit 100 \
  --out experiments/E1/0012_acceptance
```

- 开发子集不足 100 条的任务全取。
- 任何一个模型、任何一个任务不达标（一致率 < 98%，或指标差的 CI 不含 0）：
  - 照常推回，report 里写明；
  - 对其余 4 个模型（qwen3.5-2b、qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf）补跑同样的验收。
- 每个模型都另外报告：按任务的截断率与解析失败分布。这是调整 max_new_tokens 的依据。

## 4. 吞吐与算力估算

8 个模型都跑：

```bash
python scripts/run_e1.py --estimate --model <键> --task all --batch-sizes 1 8 16 32 --estimate-think \
  --out experiments/E1/0012_estimate
```

附加口径另跑：
- `qwen3.5-0.8b`、`qwen3.5-9b` 加 `--policy qwen:1024`；
- `internvl3.5-2b-hf`、`internvl3.5-4b-hf` 加 `--policy internvl:native`。

这两种口径只跑 `--task refexp_box ovd`。

## 5. 时间与停止

- 预计合计 3–4 GPU 小时。
- 累计超过 6 小时：跑完当前模型就停，推回已完成的部分。

## 6. 产出与回传

- `experiments/E1/0012_acceptance/` 与 `experiments/E1/0012_estimate/`：run_e1 的全部输出，外加 `disk_ledger.tsv` 与 `README.md`。
- `reports/0012-e1-accept-estimate.md`，按 report 模板：
  - 第 3 节原样贴：
    - 验收汇总表：模型 × 任务的一致率、指标差与 CI、raw_text 相同比例、是否通过；
    - 截断率与解析失败分布；
    - 估算汇总表：每个模型推荐的批大小、按任务与子集外推的 GPU 小时、合计；附加口径单列。
  - 第 6 节写磁盘与内存盘收支。
  - 只贴数字与事实，不做解读。
- 分两批：
  1. 验收完成推一次；
  2. 估算完成后，连同 report 一起推。

```bash
git add <逐个路径> && git commit -m "[results] 0012: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0012-e1-accept-estimate.md` 追加进完成清单，重新进入值守。
