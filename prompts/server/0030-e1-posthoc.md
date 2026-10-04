[RUN-TASK 0030] E1 事后读数：去偏字母读数（8 个模型）与格式强制诊断（Qwen 四个尺寸）；约 1.1 GPU 小时

## 为什么做

- G1 已通过（D66）。负责人要求两项事后读数，口径写在 E1 预注册 §10，写于运行之前：
  - **去偏字母读数**（N22 → (b)）：选择题与点对四个任务的新标题。每题按选项的全部循环移位各读一次，平均概率。
  - **格式强制诊断**：Qwen 的多边形预填 `[[`；网格受限解码，固定 271 个 token，另有结构强制。
- DEV-0033（D68）实现了它们。事后读数的配置是 `configs/e1_posthoc.json`。
- 规则同 RUN-0027：不改参数、不挑结果、异常照实记下。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`（第 3 步：固定 commit 的工作树）、`.claude/skills/exp-triad-flow/references/server.md`
- `PREREGISTRATION.md` 的 E1 §10
- `python scripts/run_e1.py --help`、`python scripts/summarize_e1_posthoc.py --help`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

- `git pull --ff-only`，HEAD 必须包含本任务书所在的 commit。
- sha256 必须是下面的值，不是就停下报告：
  - `configs/e1_prompts.json`：`f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452`
  - `configs/e1_posthoc.json`：`e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc`
- 建固定工作树：`git worktree add --detach /dev/shm/visint/tmp/wt0030 HEAD`。
- 汇总要读 RUN-0027 与 RUN-0028 的结果：在固定工作树里确认 `experiments/E1/0027_e1_main` 与 `experiments/E1/0028_e1_think` 都在，而且是完整的 8 个模型。

## 2. 环境

同 RUN-0027：`VISINT_DATA_ROOT`、`NLTK_DATA`、`TMPDIR`、`PYTORCH_CUDA_ALLOC_CONF`；冻结批配置 64，不加 `--batch-size`。模型与内存盘规则同 RUN-0016：每个模型跑完即删，只保留 qwen3.5-0.8b，内存盘 ≤ 25 GB，权重按清单核对。

## 3. 运行

`OUT=$REPO/experiments/E1/0030_e1_posthoc`。模型顺序同 RUN-0027。

**每个模型**：

```bash
python scripts/run_e1.py --model <键> --task all --subset debiased --data-root $PROJ/data --out $OUT
```

**Qwen 四个尺寸另加**，紧跟在上一条之后：

```bash
python scripts/run_e1.py --model <键> --task all --subset forced --data-root $PROJ/data --out $OUT
```

- 中断了用同一条命令加 `--resume` 续跑。
- 退出码为 2 时贴出报错，继续下一条。

**全部模型跑完后**，生成两份汇总：

```bash
python scripts/summarize_e1_posthoc.py validity --debiased $OUT --free experiments/E1/0027_e1_main \
  --main experiments/E1/0027_e1_main --out-json $OUT/validity.json --out-md $OUT/validity.md
python scripts/summarize_e1_posthoc.py forced --forced $OUT --main experiments/E1/0027_e1_main \
  --think experiments/E1/0028_e1_think --out-json $OUT/forced.json --out-md $OUT/forced.md
```

## 4. 时间与停止

- 估算约 1.1 GPU 小时（DEV-0033 按 RUN-0027 的实测推算）：去偏读数 8 个模型约 0.93 小时，格式强制 ≤ 0.14 小时。权重下载不算 GPU 时间。
- 累计 GPU 时间超过 3 小时：跑完当前模型就停，推回已完成的部分，写 `blocked e1-posthoc-over-budget`。
- 每跑完一个模型推一次，`standby.status` 写 `running 0030 <已完成的模型数>/8`。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E1/0030_e1_posthoc/`：run_e1 的全部输出、两份汇总、`disk_ledger.tsv`、`README.md`、`logs/`。大于 20 MB 的文件按 RUN-0027 的办法压缩。
- `reports/0030-e1-posthoc.md`，按 report 模板，第 3 节原样贴：
  - **去偏读数**：每个模型 × 四个任务，给出
    - 去偏读数（全量）与 CI；
    - 同次运行的原字母读数与配对差；
    - 每种顺序的准确率与最常见字母占比；
    - 随机水平。
  - **validity.md 全文**：在 free 200 条上两条规则的逐格判定与标记；s = 0 与 RUN-0027 字母读数逐条一致的比例。
  - **forced.md 全文**：Qwen 四个尺寸 × 多边形、网格两种读法的严格读数、有效率、条件读数，以及与主评测、思考子集的配对差和 CI。
  - **用量**：每个模型的 GPU 墙钟、合计，以及与估算的对照。
- 只贴数字与事实，不做解读。

完成后按 STANDBY.md，把 `0030-e1-posthoc.md` 追加进完成清单，重新进入值守。
