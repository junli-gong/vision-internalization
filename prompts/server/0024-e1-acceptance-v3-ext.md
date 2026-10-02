[RUN-TASK 0024] E1 批处理验收 v3 扩到其余 4 个模型（qwen3.5-2b、qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf）

## 为什么做

- 项目负责人裁定（D36-1）：
  - E1 的批配置冻结为协议的一部分；
  - 验收看两件事：(i) 同配置重跑逐位相同；(ii) 换一种批组成后，指标的配对差 CI 落在 ±δ 以内。
  - 两种批组成之间的差记作"数值噪声底"。
- DEV-0017（D38）实现了验收 v3。D43 按 RUN-0016 的吞吐把冻结的批大小从 16 改成 64，B 从 12 改成 48。
- 这次是 E1 主评测开跑前的最后一道仪器检查。不通过就不开跑。
- RUN-0020 先跑了 4 个模型：3 个 all_pass。qwen3.5-9b 只有检测组不过，差 +0.0117，CI [−0.0026, +0.0266]，上界超过 +0.02。
- 按 D38-5，不达标就扩到全部 8 个。本任务跑其余 4 个模型，用同一套冻结配置，任务书的其余部分与 RUN-0020 相同。
- 不重跑 RUN-0020 的 4 个模型，也不换别的批组成。
- 只用开发子集。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D36（第 1 条）、D38、D43
- `configs/e1_prompts.json` 的 `batch_protocol` 与 `acceptance`
- `python scripts/run_e1.py --help`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `2257e08`（D43 的批大小改动）与本任务书所在的 commit。不包含就停下报告。

- `configs/e1_prompts.json` 的 sha256 应为 `5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158`。

## 2. 环境

- `pip install --dry-run -r requirements.txt` 应当没有任何 "Would install"。有的话停下报告。
- 设下列环境变量：
  - `VISINT_DATA_ROOT=$PROJ/data`
  - `NLTK_DATA=$PROJ/nltk_data`
  - `TMPDIR=/dev/shm/visint/tmp`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 检测的开发标注（`coco_train2014_e1dev`，RUN-0016）与 CountBench 人工问句（RUN-0013）已经落地，不用重落。
- 模型与内存盘规则同 RUN-0016。

## 3. 运行

模型顺序：qwen3.5-2b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-4b。每个模型一条命令：

```bash
python scripts/run_e1.py --acceptance --model <键> --task all --data-root $PROJ/data \
  --out $REPO/experiments/E1/0024_acceptance_v3_ext
```

- 不加 `--batch-size`、`--alt-batch-size`、`--allow-unfrozen-batch`，用冻结值 64 与 48。
- 中断了用同一条命令加 `--resume` 续跑。
- 退出码为 2 时照常推回，report 里贴出报错，继续下一个模型。

## 4. 时间与停止

- 预计 3–5 GPU 小时：每个模型三遍（A1、A2、B）。
- 累计超过 7 小时，跑完当前模型就停，推回已完成的部分。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E1/0024_acceptance_v3_ext/`：run_e1 的全部输出、`disk_ledger.tsv`、`README.md`、`logs/`。
- `reports/0024-e1-acceptance-v3-ext.md`，按 report 模板，第 3 节原样贴：
  - **总判定**：每个模型的 `all_pass`、`rerun_all_identical`、`groups_failed`、`groups_not_evaluated`。
  - **(i) 逐位重跑**：每个模型 × 任务，A1 与 A2 的 predictions.jsonl 是否逐字节相同。
  - **(ii) 噪声底**：每个模型 × 组，n、A1 与 B 的标题指标、差值、配对 bootstrap 95% CI、δ、是否通过。
  - **三列**：每个模型 × 任务，A1 的严格读数、有效率、条件读数。
  - **拒答诊断**：定位、多边形、网格按 person 与非 person 分列。
  - **v2 的三种一致口径**：A1 与 B 之间的参考列，原样贴。
  - **计算量**：每个模型每遍的墙钟与生成 token，以及 GPU 显存峰值（torch 与 nvidia-smi）。
- 只贴数字与事实，不做解读。
- 一次推回：

```bash
git add <逐个路径> && git commit -m "[results] 0024: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0024-e1-acceptance-v3-ext.md` 追加进完成清单，重新进入值守。
