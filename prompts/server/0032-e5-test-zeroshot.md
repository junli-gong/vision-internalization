[RUN-TASK 0032] E5 test 划分：零样本基座与文本思维链（按优先级，上限 15 GPU 小时）、LISA-7B 对照、平凡基线

## 为什么做

- E5 的判定口径已全部定下：
  - D59-4：两级门槛，三条基线，配对 CI；
  - D72-1：b 类的标题与判定只统计环绕的 [20,180] 三档。代码见 `src/e5/protocol.py` 的 `B_HEADLINE_MIN_DEG = 20`。
- 预注册要求定完口径才在 test 上跑，现在可以发了。预测见 E5 预注册 §6（含 2026-10-05 的补充）。
- 零样本阶段的批准上限是 15 GPU 小时（D59-4）。全部 8 个模型大概跑不完，所以**按优先级跑，到上限就停**。
- 这是正式实验：不改参数、不挑结果、异常照实记下。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`（第 3 步：固定 commit 的工作树）、`.claude/skills/exp-triad-flow/references/server.md`
- `PREREGISTRATION.md` 的 E5（全文）、`prompts/server/0018-e5-dev-check.md` 与 `prompts/server/0019-lisa-e5-dev.md`（命令与环境照它们）
- `python scripts/run_e5.py --help`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

- `git pull --ff-only`，HEAD 必须包含本任务书所在的 commit。
- `configs/e5_prompts.json` 的 sha256 必须是 `c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4`。
- `python -c "from src.e5.protocol import B_HEADLINE_BINS; print(B_HEADLINE_BINS)"` 必须输出 `(1, 2, 3)`。
- 不对就停下报告。
- 建固定工作树：`git worktree add --detach /dev/shm/visint/tmp/wt0032 HEAD`。

## 2. 环境

- 同 RUN-0018（项目 venv、`HF_HOME`、`HF_HUB_CACHE`、`TMPDIR`）。
- LISA 同 RUN-0019：独立 venv，权重放内存盘，用完即删。
- 模型与内存盘规则同 RUN-0016：每个模型跑完即删，只保留 qwen3.5-0.8b；内存盘上的权重 ≤ 25 GB。

## 3. 运行

`OUT=$REPO/experiments/E5/0032_e5_test`。评测集用 test 划分（`--subset eval`）：每档 100 条，b 类每档 200 条（E5 预注册 §2）。所以 b 单独跑一条命令。

**每个模型**（`<键>` 见下面的顺序）：

```bash
python scripts/run_e5.py --system zeroshot cot --model <键> --task a c d e --out $OUT
python scripts/run_e5.py --system zeroshot cot --model <键> --task b --n-per-bin 200 --out $OUT
```

**顺序**（按优先级；累计 GPU 时间超过 15 小时，跑完当前模型就停）：
1. qwen3.5-2b：E5 主比较的尺寸（D72-4）
2. **LISA-7B-v1**（a–d）：照 RUN-0019 的四步，"导出、彩排、全量、评分"，划分改为 test。
   - 导出：`--task a b c d --subset eval`，b 另用 `--n-per-bin 200` 导出一份。
   - 评分：`--system external --model lisa-7b-v1`，a、c、d 每档 100 条，b 每档 200 条。
3. qwen3.5-0.8b
4. qwen3.5-4b
5. qwen3.5-9b
6. internvl3.5-2b-hf
7. internvl3.5-4b-hf
8. gemma-4-e2b-it
9. gemma-4-e4b-it

**平凡基线**（不加载模型，CPU）：

```bash
python scripts/run_e5.py --system baseline --task a c d e --out $OUT
python scripts/run_e5.py --system baseline --task b --n-per-bin 200 --out $OUT
```

- 中断了加 `--resume` 续跑。
- 退出码为 2 时贴出报错，继续下一条。

## 4. 时间与停止

- 上限 15 GPU 小时（D59-4）：零样本加思维链加 LISA 合计。权重下载不计 GPU 时间。
- 每跑完一个模型推一次，`standby.status` 写 `running 0032 <已完成的模型数>/9`。
- 到上限就停，推回已完成的部分，写 `blocked e5-test-cap`。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E5/0032_e5_test/`：run_e5 的全部输出、`disk_ledger.tsv`、`README.md`、`logs/`。大于 20 MB 的文件按 RUN-0027 的办法压缩。
- `reports/0032-e5-test.md`，按 report 模板，第 3 节原样贴：
  - **每个模型 × 系统（零样本、思维链）× 任务 a–e**：
    - n、有效率、解析失败（按原因）、截断率；
    - 思维链的 `thinking_end` 四类条数；
    - mask IoU（网格口径）的严格读数与条件读数；
    - a、b 的起伏误差：严格、条件、中位数。
  - **门槛**（metrics 里的 gate）：
    - 先决条件：最强的非真值基线与配对 CI；
    - 判定条件：平面、真值仿射、凸起先验三条的配对 CI，a、b 两类；
    - imagining 的取值；
    - 标题口径的 n。
  - **LISA**：a–d 的 mask IoU 与门槛，以及对"当前 mask"的 IoU（D59-5）。
  - **平凡基线**：每个任务 × 档的 mask IoU 与起伏误差。
  - **用量**：每个模型的 GPU 墙钟、合计，以及跑到第几个模型。
- 只贴数字与事实，不做解读。

完成后按 STANDBY.md，把 `0032-e5-test-zeroshot.md` 追加进完成清单，重新进入值守。
