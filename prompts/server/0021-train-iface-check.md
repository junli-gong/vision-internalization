[RUN-TASK 0021] 训练接口的 GPU 核对：受限 logits 一致率、LoRA 合并一致率（4 个模型，开发提示 200 条）

## 为什么做

- DEV-0020 与 DEV-0021（D42、D45）给适配器加了两样东西：
  - 训练前向、唯一的受限 logits 与受限解码；
  - LoRA 的保存、加载与内存合并。
- CPU fp32 上逐个相同，测试已经钉住。GPU bf16 上不逐位相同：Qwen 预填走 fla chunk、解码走 recurrent。判据写成一致率，需要实测（D21-2）。
- `scripts/check_training_interfaces.py` 是常备仪器，以后每个训练出来的模型评测前都用它核对合并（简报 §12）。这次在 base 加随机 LoRA 上测一遍，给判据定值。
- 只用 E1 开发子集（grid16 的提示），不碰评测子集。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D21（第 2 条）、D42、D43、D45
- `python scripts/check_training_interfaces.py --help`

本任务只运行，不改任何仓库代码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `4e8660e`（DEV-0021 的合入）与本任务书所在的 commit。不包含就停下报告。

## 2. 环境

- `pip install --dry-run -r requirements.txt` 应当没有任何 "Would install"。有的话停下报告。
- 设下列环境变量：
  - `VISINT_DATA_ROOT=$PROJ/data`
  - `TMPDIR=/dev/shm/visint/tmp`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 模型与内存盘规则同 RUN-0020。

## 3. 运行

模型顺序：qwen3.5-0.8b、gemma-4-e2b-it、internvl3.5-2b-hf、qwen3.5-9b。每个模型一条命令：

```bash
python scripts/check_training_interfaces.py --model <键> --checks restricted merge --n 200 \
  --data-root $PROJ/data --out experiments/E3/0021_train_iface_check/<键>.json
```

- 退出码 1 表示有判据不过，JSON 照写，照常推回。退出码 2 是出错，贴报错，继续下一个模型。
- OOM 时可以加 `--batch-size 32 --allow-unfrozen-batch` 重跑一次，report 里写明。除此之外不改参数。

## 4. 时间与停止

- 预计 30–45 分钟：0.8B 约 4 分钟，9B 约 8–12 分钟。
- 累计超过 1.5 小时，跑完当前模型就停。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E3/0021_train_iface_check/`：4 份 JSON、`logs/`、`disk_ledger.tsv`、`README.md`。
- `reports/0021-train-iface-check.md`，按 report 模板，第 3 节原样贴：
  - **总判定**：每个模型的 `pass` 与 `failures`。
  - **restricted**，每个模型 × 思考开关：
    - 受限解码与 `generate_batch` 的 exact_rows 与 token_agreement；
    - 第一个分歧位置的分位数；
    - 教师强制的 agreement；
    - 不一致处 gap 与 top2_gap 的中位数、最大值、< 0.125 的比例。
  - **merge**，每个模型 × 思考开关：
    - `effect_vs_base`，即随机 LoRA 与 base 的差别；
    - 合并前后的 exact_rows、token_agreement；
    - 教师强制的 argmax_agreement、max_abs_logit、max_abs_hidden；
    - gap 统计。
  - **allowed 集合**的 token 表、批配置，以及每个模型的墙钟与显存峰值。
- 只贴数字与事实，不做解读。
- 一次推回：

```bash
git add <逐个路径> && git commit -m "[results] 0021: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0021-train-iface-check.md` 追加进完成清单，重新进入值守。
