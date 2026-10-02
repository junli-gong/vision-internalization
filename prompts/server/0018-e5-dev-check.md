[RUN-TASK 0018] E5 开发集核对：零样本与文本思维链在 8 个基座上的格式、截断率与吞吐（val 划分，每档 4 条）

## 为什么做

- DEV-0018（D39）实现了 E5 评测框架：零样本数字网格、文本思维链、平凡基线、外部预测评分。
- 生成器的 a、b 两类升到版本 4：
  - a 类输出 33 个符号的起伏网格；
  - b 类放大了物体。
- 正式评测之前，先在开发集上核对三件事：
  - 提示格式：零样本是 32×32 网格，a 类用 33 个符号；思维链是多边形 JSON；
  - 生成上限够不够：零样本 2048，思维链 4096；
  - 各模型的实际吞吐，用来修正 E5 的算力估算（N18 第 5 条）。
- 只用 val 划分（`--subset dev`），不碰 test 划分。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D24（d 类以服务器为规范平台）、D39
- `python scripts/run_e5.py --help`

本任务只运行，不改任何仓库代码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `78f9092`（DEV-0018 的合入）与本任务书所在的 commit。不包含就停下报告。

## 2. 环境

- `pip install --dry-run -r requirements.txt` 应当没有任何 "Would install"。有的话停下报告。
- 设下列环境变量：
  - `TMPDIR=/dev/shm/visint/tmp`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（同 RUN-0016）
- 先跑一次 `python scripts/synth_pin_digests.py check`：300 个摘要应当全部相同（a、b 为版本 4）。不相同就停下报告。
- 模型与内存盘规则同 RUN-0016：
  - 每个模型跑完即删，只保留 qwen3.5-0.8b；
  - 内存盘 ≤ 25 GB；
  - 权重按 `experiments/P0/0009_ifeval_rescore/weights_manifest.json` 核对。

## 3. 运行

先跑平凡基线（不加载模型，只用 CPU）：

```bash
python scripts/run_e5.py --system baseline --task all --subset dev --n-per-bin 4 \
  --out experiments/E5/0018_e5_dev
```

再跑 8 个模型，顺序同 RUN-0017。每个模型一条命令：

```bash
python scripts/run_e5.py --system zeroshot cot --model <键> --task all --subset dev --n-per-bin 4 \
  --out experiments/E5/0018_e5_dev
```

批大小用提示配置里的默认值。退出码为 2 时照常推回，report 里贴出报错，继续下一个模型。

## 4. 时间与停止

- 预计 1.5–2.5 GPU 小时：每个模型约 10–20 分钟，大头是思维链。
- 累计超过 4 小时，跑完当前模型就停，推回已完成的部分。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E5/0018_e5_dev/`：run_e5 的全部输出、`disk_ledger.tsv`、`README.md`、`logs/`。
- `reports/0018-e5-dev-check.md`，按 report 模板，第 3 节原样贴：
  - **摘要核对**：`synth_pin_digests.py check` 的结果。
  - **逐模型 × 逐系统 × 逐任务**：
    - 条数、有效率、解析失败按原因分；
    - 截断率；思维链另列思考段未闭合的条数；
    - 平均生成 token、样本/秒、批大小、峰值显存（torch 与 nvidia-smi）。
  - **严格读数与条件读数**：mask IoU；a 类另列起伏误差。只是格式核对，n 很小，不作结论。
  - **平凡基线**：同一批开发样本上的读数。
  - **原始输出举例**：每个模型 × 系统 × 任务，各取第一条解析失败的样本，贴出 `raw_text` 的前 600 个字符。
  - **墙钟与用量合计**。
- 只贴数字与事实，不做解读。
- 一次推回：

```bash
git add <逐个路径> && git commit -m "[results] 0018: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0018-e5-dev-check.md` 追加进完成清单，重新进入值守。
