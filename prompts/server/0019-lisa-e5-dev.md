[RUN-TASK 0019] LISA-7B-v1 在 E5 开发集上的首次运行（外部对照，隔离；val 划分，每档 4 条，a–d）

## 为什么做

- DEV-0019（D40）写好了 LISA 一侧的推理脚本 `scripts/external/lisa_e5_infer.py`。它在 LISA 的独立 venv 里运行，产出外部预测目录，由 `run_e5.py --system external` 评分。
- 真模型那一侧在开发机上测不了，这次在开发集上第一次实跑，核对四件事：
  - 加载与入口检查；
  - 用了哪几处处理；
  - mask 与文本是否合理；
  - 每条的用时。
- 只用 val 划分。test 划分上的 LISA 运行，等 E5 的门槛口径定下来（N18）再发。
- 硬约束 1：LISA 的输出只进本运行目录下的外部预测目录，绝不进 `$PROJ/data`，也不进任何训练或数据目录。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D36（第 5 条）、D39、D40
- `reports/0015-lisa-smoke.md`
- `scripts/external/lisa_e5_infer.py` 的 docstring、`python scripts/run_e5.py --help`

本任务只运行，不改任何仓库代码，也不改 LISA 的源码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `24c4c22`（DEV-0019 的合入）与本任务书所在的 commit。不包含就停下报告。

## 2. 环境与权重

- LISA 的 venv 与代码沿用 RUN-0015 的 `$PROJ/ext/lisa/`，不重装。`$PROJ/ext/lisa/LISA` 的 HEAD 应是 `3cb2d43…` 且没有改动；不是就停下报告。
- 按 `experiments/P0/0015_lisa_smoke/weights_download.json`，把 LISA-7B-v1 与 CLIP 下回内存盘 `HF_HUB_CACHE=/dev/shm/visint/hf-hub`，逐文件核对。可以重用 RUN-0015 的 `p18_download.py`。
  - 内存盘合计 ≤ 25 GB：先删掉其他权重，qwen3.5-0.8b 可以留。
- 设下列环境变量：
  - `HF_HOME=$ROOT/.hf`
  - `TMPDIR=$PROJ/tmp`
  - `PIP_NO_CACHE_DIR=1`

## 3. 运行

运行目录 `experiments/E5/0019_lisa_e5_dev/`。

1. **导出**（项目 venv）：

   ```bash
   $PROJ/venv/bin/python scripts/run_e5.py --export-inputs $PROJ/tmp/e5_inputs_dev_0019 \
     --task a b c d --subset dev --n-per-bin 4
   ```

2. **彩排**（LISA venv，`--limit 5`，单独的 `--out`）：

   ```bash
   $PROJ/ext/lisa/venv/bin/python scripts/external/lisa_e5_infer.py --lisa-src $PROJ/ext/lisa/LISA \
     --inputs $PROJ/tmp/e5_inputs_dev_0019 --out experiments/E5/0019_lisa_e5_dev/lisa_predictions_limit5 \
     --lisa-weights <LISA 快照路径> --clip-weights <CLIP 快照路径> --limit 5
   ```

   - 核对：入口检查通过；`fixes_applied` 记的是哪几处，照实写。
   - 每个任务前 2 条成功的样本，画出原图、mask 叠加图与 `text`，放进 `overlays/`，自己看一眼。只记事实，例如"mask 覆盖输入图里的绿色立方体"，不判断对错。
   - 彩排失败就停下报告，不改代码重试。

3. **全量**：同一条命令去掉 `--limit 5`，`--out experiments/E5/0019_lisa_e5_dev/lisa_predictions`。

4. **评分**（项目 venv）：

   ```bash
   $PROJ/venv/bin/python scripts/run_e5.py --system external --model lisa-7b-v1 \
     --external-dir experiments/E5/0019_lisa_e5_dev/lisa_predictions --task a b c d --subset dev --n-per-bin 4 \
     --out experiments/E5/0019_lisa_e5_dev
   $PROJ/venv/bin/python scripts/run_e5.py --system baseline --task a b c d --subset dev --n-per-bin 4 \
     --out experiments/E5/0019_lisa_e5_dev
   ```

5. **收尾**：删掉内存盘上的 LISA 与 CLIP 权重；删掉 `$PROJ/tmp/e5_inputs_dev_0019`，删之前先记下体积。

## 4. 时间与停止

- 预计 15–30 分钟。
- 如果每条用时超过 10 s，跑满 30 分钟就停，推回已完成的部分（脚本支持 `--resume`）。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E5/0019_lisa_e5_dev/`：
  - 外部预测目录（全量与 limit5 两份）；
  - 评分输出、`overlays/`、`disk_ledger.tsv`、`README.md`、`logs/`。
- `reports/0019-lisa-e5-dev.md`，按 report 模板，第 3 节原样贴：
  - **加载**：入口检查、`fixes_applied`、加载用时、显存峰值。
  - **逐任务**：
    - 条数、成功条数，失败按原因分（`no_seg`、`empty_mask`、异常类名）；
    - `[SEG]` 个数的分布；
    - 平均生成 token、每条墙钟。
  - **评分**：逐任务、逐档的 mask IoU（严格与条件），a 类另列起伏误差的状态。同一批样本上平凡基线的读数并列。n 很小，不作结论。
  - **文本举例**：每个任务前 3 条的 `text`。
  - **目检记录**：`overlays/` 里每张图一句事实描述。
- 只贴数字与事实，不做解读。
- 一次推回：

```bash
git add <逐个路径> && git commit -m "[results] 0019: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0019-lisa-e5-dev.md` 追加进完成清单，重新进入值守。
