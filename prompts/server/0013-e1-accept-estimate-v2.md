[RUN-TASK 0013] E1 批处理验收与算力估算的重跑（按新的生成上限；三种一致口径都报）+ CountBench 人工问句落地

## 为什么做

- RUN-0012 的验收暴露出两件事：
  - 多选、点对、检测的生成上限太小，大量截断；
  - 框按逐位判一致太严。
- DEV-0015（D33）做了这些修订：
  - 按开发子集的截断率调高了 `max_new_tokens`，调整记录见 `configs/e1_prompts.json`；
  - 验收同时计算三种一致口径（N15 的 (a)(b)(c)）。
- 这次重跑验收与估算，供项目负责人答复 N15、批准 E1 预注册。
- 本任务只用开发子集，不碰任何评测子集。用量计入 E1 预算。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D32、D33；`DECISIONS_NEEDED.md` 的 N15
- `python scripts/run_e1.py --help`、`python scripts/data/land_e1.py --help`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `26e871d` 与本任务书所在的 commit。不包含就停下报告。

## 2. 环境与落地

- `pip install --dry-run -r requirements.txt` 应当没有任何 "Would install"。有的话停下报告。
- 设 `VISINT_DATA_ROOT=$PROJ/data`、`NLTK_DATA=$PROJ/nltk_data`、`TMPDIR=/dev/shm/visint/tmp`。
- 落地 CountBench 的人工问句：

  ```bash
  python scripts/data/land_e1.py countbenchqa_questions --root $PROJ/data
  ```

  清单副本放进运行目录。

## 3. 验收（8 个模型都跑）

```bash
python scripts/run_e1.py --acceptance --model <键> --task all --batch-size 16 --limit 100 \
  --agreement bitwise --out experiments/E1/0013_acceptance
```

- 模型与内存盘规则同 RUN-0012。
- 跑完后，用 `--resume --agreement iou90` 和 `--resume --agreement outcome` 各重算一遍汇总，不重新生成。
- 计数任务另跑一遍 `--task count --count-questions countbenchqa`，同样三种口径。
- 汇总表照原样推回：每种口径的一致率与是否通过、截断率与解析失败、诊断（wrong_shape、refusal、关思考时自发的思考标记）。

## 4. 算力估算（8 个模型都跑）

```bash
python scripts/run_e1.py --estimate --model <键> --task all --batch-sizes 8 16 32 --estimate-think \
  --out experiments/E1/0013_estimate
```

附加口径同 RUN-0012：
- Qwen 0.8B 与 9B 加 `--policy qwen:1024`；
- InternVL 2B 与 4B 加 `--policy internvl:native`；
- 这两种口径只跑 `--task refexp_box ovd`。

## 5. 时间与停止

- 预计合计 4–6 GPU 小时。
- 累计超过 8 小时：跑完当前模型就停，推回已完成的部分。

## 6. 产出与回传

- `experiments/E1/0013_acceptance/`、`experiments/E1/0013_estimate/`：run_e1 的全部输出、`disk_ledger.tsv`、`README.md`。
- `reports/0013-e1-accept-estimate-v2.md`，按 report 模板：
  - 第 3 节原样贴验收汇总（三种口径）、截断率与解析失败、诊断、估算汇总（推荐批大小、按任务与子集的 GPU 小时、合计、附加口径单列）；
  - 只贴数字与事实，不做解读。
- 分两批：
  1. 验收完成推一次；
  2. 估算完成后，连同 report 一起推。

```bash
git add <逐个路径> && git commit -m "[results] 0013: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0013-e1-accept-estimate-v2.md` 追加进完成清单，重新进入值守。
