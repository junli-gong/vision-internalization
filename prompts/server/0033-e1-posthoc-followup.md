[RUN-TASK 0033] E1 事后读数的后续（D72-2/3）：NYU 点对的"对调标记"去偏（8 个模型）、结构强制的多边形（Qwen 四个尺寸 × 两个变体）、"模型自己的框"基线与新的标题汇总；约 0.6 GPU 小时

## 为什么做

- RUN-0030 发现，点对任务的循环移位去偏会误罚"按标记作答"的模型（N26）。负责人 D72 定了：
  - NYU 点对改用"对调图上的 A、B 标记"去偏；BLINK 的标题改用自由作答；
  - 多边形补结构强制的读数，含"至少 8 个点"变体，报告近矩形占比；
  - 加"模型自己的框"基线：同一句话、同一个模型在 E1 给出的框，光栅化后对比多边形与网格的读数。
- DEV-0036（D74）实现了这些。事后读数的配置是 `configs/e1_posthoc.json`。
- 都是事后的，不改 G1 的结论。规则同 RUN-0030：不改参数、不挑结果、异常照实记下。
- 排在 RUN-0032（E5 test）之后。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`（第 3 步：固定 commit 的工作树）、`.claude/skills/exp-triad-flow/references/server.md`
- `PREREGISTRATION.md` 的 E1 §10；`reports/0030-e1-posthoc.md`
- `python scripts/run_e1.py --help`、`python scripts/summarize_e1_posthoc.py ownbox --help`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

- `git pull --ff-only`，HEAD 必须包含本任务书所在的 commit。
- sha256 必须是下面的值，不是就停下报告：
  - `configs/e1_prompts.json`：`f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452`
  - `configs/e1_posthoc.json`：`9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1`
- 建固定工作树：`git worktree add --detach /dev/shm/visint/tmp/wt0033 HEAD`。
- 确认 `experiments/E1/0027_e1_main`、`0028_e1_think`、`0030_e1_posthoc` 都在工作树里，各 8 个模型。

## 2. 环境

同 RUN-0030：`VISINT_DATA_ROOT`、`NLTK_DATA`、`TMPDIR`、`PYTORCH_CUDA_ALLOC_CONF`；冻结批配置 64；模型与内存盘规则同 RUN-0016。

## 3. 运行

`OUT=$REPO/experiments/E1/0033_e1_posthoc2`。模型顺序同 RUN-0027。

**每个模型**：

```bash
python scripts/run_e1.py --model <键> --task rel_depth_nyu --subset letter_swap --data-root $PROJ/data --out $OUT
```

**Qwen 四个尺寸另加**：

```bash
python scripts/run_e1.py --model <键> --task refexp_poly --subset forced_poly --data-root $PROJ/data --out $OUT
```

- 中断了加 `--resume` 续跑。
- 退出码为 2 时贴出报错，继续下一条。

**全部跑完后，生成汇总**（只用 CPU）：

```bash
python scripts/summarize_e1_posthoc.py validity --debiased experiments/E1/0030_e1_posthoc --letter-swap $OUT \
  --free experiments/E1/0027_e1_main --main experiments/E1/0027_e1_main \
  --out-json $OUT/validity_d72.json --out-md $OUT/validity_d72.md
python scripts/summarize_e1_posthoc.py ownbox --main experiments/E1/0027_e1_main --think experiments/E1/0028_e1_think \
  --forced experiments/E1/0030_e1_posthoc --forced-poly $OUT --data-root $PROJ/data \
  --out-json $OUT/ownbox.json --out-md $OUT/ownbox.md
```

## 4. 时间与停止

- 估算约 0.63 GPU 小时：对调标记 8 个模型约 0.45 小时，结构强制多边形约 0.18 小时。权重下载不算。
- 累计超过 2 小时：跑完当前模型就停，推回已完成的部分，写 `blocked e1-posthoc2-over-budget`。
- 每跑完一个模型推一次，`standby.status` 写 `running 0033 <已完成的模型数>/8`。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E1/0033_e1_posthoc2/`：run_e1 的全部输出、两份汇总、`disk_ledger.tsv`、`README.md`、`logs/`。
- `reports/0033-e1-posthoc2.md`，按 report 模板，第 3 节原样贴：
  - **对调标记的 NYU 点对**：每个模型的去偏读数（全量）与 CI；两张图各自的准确率与最常见字母占比；原图那一遍与 RUN-0027 字母读数逐条一致的比例。
  - **validity_d72.md 全文**：D72-2 的标题表，加有效性检查。
  - **ownbox.md 全文**：Qwen 四个尺寸 × 多边形、网格的各种读法，与"自己的框"基线的配对比较；近矩形占比；其余 4 个模型的"自己的框"基线；id 对不上的条数。
  - **用量**：每个模型的 GPU 墙钟、合计，以及与估算的对照。
- 只贴数字与事实，不做解读。

完成后按 STANDBY.md，把 `0033-e1-posthoc-followup.md` 追加进完成清单，重新进入值守。
