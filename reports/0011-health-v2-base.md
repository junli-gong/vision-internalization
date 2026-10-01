# RUN-0011 — 健康检查协议 v2：8 个基座重跑（MMLU-Pro 标题 256 token）

服务器 CC，b200，2026-10-01 07:05–08:29 UTC（值守中接单）。本报告只列数字与事实，不做解读。

## 0. 摘要（只陈述，不解读）

- **退出码：** 8 个模型的 health 和 compare 都是退出码 0，`ok=True`。
- **权重：** 按 RUN-0009 的清单核对，8 个模型共 112/112 个文件一致（13+13+9+18+14+9+20+16）。
- **逐题核对：** 与 v1 相比，每个模型都逐字段相同。
  - GSM8K：50/50；
  - IFEval：50/50；
  - fmt32 对 RUN-0007 的 32 token 记录：100/100。
  - 比的是两边共有的字段，不含 `wall_time_s`。v2 新增了 `stopped` 字段，不在比较范围内。
- **截断审计：** `--audit-fmt32 100` 在 8 个模型上都是 100/100 逐字段相同。
- **compare：** 8 个模型都是 `any_flag=False`。
  - 每个模型有 7 行可比，delta 全为 0，lost 和 gained 全为 0。
  - 截断率的 delta 全为 0。
  - `mmlu_pro` 被标为 incomparable：v1 没有 256 token 的读数。
- **两个 MMLU-Pro 读数不同的模型：** qwen3.5-0.8b 是 15 对 14，gemma-4-e2b-it 是 26 对 25（256 对 fmt32）。其余 6 个模型两者相同。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `1b8751c`。本任务书所在的 commit `59e7b50` 在 HEAD 的历史里（`git merge-base --is-ancestor`）。
- **环境：**
  - `pip install --dry-run -r requirements.txt` 没有任何 "Would install"，见 `pip_dryrun.txt`。
  - 驱动脚本里 export 了 `NLTK_DATA=$PROJ/nltk_data`，其余环境变量来自 `$PROJ/scripts/env.sh`。
- **改动范围：**
  - 没有修改 `src/`、`scripts/`、`tests/`、`configs/`，也没有安装 Python 包。
  - 新写的文件只有 `experiments/P0/0011_health_v2_base/probe_scripts/` 下的三个：`run_health_v2.sh`、`p14_v2_summary.py`、`diskcheck11.sh`。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。
- **运行期间 origin 有新提交。** 推第一批之前 `git pull` 拉到了 `aaccad4`，中间的 commit 是 `3d1e605`、`5477c19`、`a6c0c41`、`aaccad4`，内容是 E1 的 config、DEV-0014 的 E1 harness、D32，以及 RUN-0012 的任务书。
  - 因此各模型 health JSON 的 `run.meta.git.commit` 不全相同：
    - 前 4 个模型是 `1b8751c`；
    - qwen3.5-4b 是 `aaccad4`；
    - 后 3 个是 `99e004b`，即本任务第一批的 commit。
  - `git diff --stat 1b8751c 99e004b -- scripts/health_check.py src/health src/adapters configs/health_check_items.json requirements.txt` 的输出为空。
  - 8 份 JSON 记录的 `configs/health_check_items.json` 的 sha256 都是 `0239675f9c226f17…`，与 RUN-0007 相同。
- **dirty 标记：** `run.meta.git.dirty=True`，`diff_shortstat` 为空，`n_untracked` 为 1–6。未跟踪的文件都在本运行目录下，是还没提交的产物。

## 2. 执行记录

驱动脚本是 `probe_scripts/run_health_v2.sh`，分两批用 nohup 运行：

- 第一批：qwen3.5-0.8b、qwen3.5-2b、gemma-4-e2b-it、internvl3.5-2b-hf；
- 第二批：qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b。

每个模型依次执行：

1. 记一笔磁盘收支。
2. `python experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py <键> --download`。
3. 记一笔磁盘收支。
4. `python scripts/health_check.py --model <键> --device cuda --dtype bf16 --audit-fmt32 100 --out experiments/P0/0011_health_v2_base/health_<键>.json`。
5. `python scripts/health_check.py --compare experiments/P0/0009_ifeval_rescore/health_<键>.json experiments/P0/0011_health_v2_base/health_<键>.json --out experiments/P0/0011_health_v2_base/compare_v1_v2_<键>.json`。
6. 记一笔磁盘收支。
7. 除 qwen3.5-0.8b 外，执行 `scan_cache_dir('/dev/shm/visint/hf-hub').delete_revisions(<revision>).execute()`，再记一笔磁盘收支。

| 模型 | 权重核对 | 下载+核对 | health 墙钟（`time` real） | `wall_time_s` | health rc | compare rc | GPU 显存峰值（按进程，第二批） | 删除释放（expected_freed） |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 13/13 | 1 s（已在内存盘） | 9m59.0s | 597.01 | 0 | 0 | 未采样 | 保留 |
| qwen3.5-2b | 13/13 | 12 s | 7m41.4s | 459.31 | 0 | 0 | 未采样 | 4.558 GB |
| gemma-4-e2b-it | 9/9 | 24 s | 15m18.2s | 916.4 | 0 | 0 | 未采样 | 10.279 GB |
| internvl3.5-2b-hf | 18/18 | 12 s | 6m31.2s | 389.46 | 0 | 0 | 未采样 | 4.715 GB |
| qwen3.5-4b | 14/14 | 22 s | 8m25.7s | 503.73 | 0 | 0 | 9740 MiB | 9.330 GB |
| gemma-4-e4b-it | 9/9 | 38 s | 16m36.5s | 994.66 | 0 | 0 | 16280 MiB | 16.025 GB |
| internvl3.5-4b-hf | 20/20 | 23 s | 7m32.4s | 450.63 | 0 | 0 | 10390 MiB | 9.483 GB |
| qwen3.5-9b | 16/16 | 42 s | 8m47.6s | 525.6 | 0 | 0 | 19078 MiB | 19.317 GB |

- **总时长：** 从 07:05:09 到 08:29:29，共 5060 s（84 min 20 s）。两批之间（07:45:34 到 07:45:55）是我启动第二批与提交第一批的间隔。
- **GPU：** NVIDIA B200，bf16。
  - 第二批每 5 s 采样一次 nvidia-smi，共 510 个样本。
  - 整卡显存峰值 19092 MiB，GPU 利用率均值 58.7%。均值包含下载、加载和比较时的空闲样本。
  - 采样期间 GPU 上只有本任务的 4 个 python 进程：pid 646330、648083、650434、651774，依次对应上表的 4 个模型。
  - 第一批没有采样，见 §4。
- **预热：** 每个模型 0.52–0.58 s，不计入 compute。
- **警告：** 与 RUN-0007 相同，原文见 `logs/health_<键>.log`。
  - 4 个 Qwen：`causal_conv1d_fn`、`causal_conv1d_update` 回退到 PyTorch 参考实现；
  - 2 个 InternVL：tie weights 提示；
  - 2 个 Gemma：没有警告。
  - 没有其他报错。

## 3. 结果表（原样）

3.1 节由 `probe_scripts/p14_v2_summary.py` 从 8 份 health JSON 和 v1 的 JSON 生成，与 `health_summary_v2.md` 相同。逐题核对的口径见 `README.md`。

### 3.1 健康检查 v2

#### 分数表（协议 v2）

| 模型 | ok | MMLU-Pro 256（标题） | MMLU-Pro fmt32（诊断） | GSM8K flexible（标题） | GSM8K strict | IFEval prompt strict（标题） | IFEval inst strict | IFEval prompt loose | IFEval inst loose |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | True | 15.0（15/100） | 14.0（14/100） | 58.0（29/50） | 56.0（28/50） | 64.0（32/50） | 74.0（57/77） | 66.0（33/50） | 76.6（59/77） |
| qwen3.5-2b | True | 27.0（27/100） | 27.0（27/100） | 76.0（38/50） | 72.0（36/50） | 74.0（37/50） | 83.1（64/77） | 80.0（40/50） | 87.0（67/77） |
| gemma-4-e2b-it | True | 26.0（26/100） | 25.0（25/100） | 80.0（40/50） | 78.0（39/50） | 78.0（39/50） | 84.4（65/77） | 86.0（43/50） | 90.9（70/77） |
| internvl3.5-2b-hf | True | 34.0（34/100） | 34.0（34/100） | 80.0（40/50） | 76.0（38/50） | 72.0（36/50） | 79.2（61/77） | 72.0（36/50） | 81.8（63/77） |
| qwen3.5-4b | True | 46.0（46/100） | 46.0（46/100） | 82.0（41/50） | 82.0（41/50） | 88.0（44/50） | 92.2（71/77） | 92.0（46/50） | 94.8（73/77） |
| gemma-4-e4b-it | True | 34.0（34/100） | 34.0（34/100） | 88.0（44/50） | 86.0（43/50） | 82.0（41/50） | 87.0（67/77） | 88.0（44/50） | 92.2（71/77） |
| internvl3.5-4b-hf | True | 46.0（46/100） | 46.0（46/100） | 92.0（46/50） | 80.0（40/50） | 82.0（41/50） | 84.4（65/77） | 84.0（42/50） | 89.6（69/77） |
| qwen3.5-9b | True | 53.0（53/100） | 53.0（53/100） | 86.0（43/50） | 86.0（43/50） | 92.0（46/50） | 94.8（73/77） | 98.0（49/50） | 98.7（76/77） |

#### 与 v1 的逐题核对（共有字段逐字段比较，wall_time_s 除外）

| 模型 | GSM8K vs RUN-0009：raw_text 相同 / 共有字段全同 | IFEval vs RUN-0009：raw_text 相同 / 共有字段全同 | fmt32 vs RUN-0007 32 token：raw_text 相同 / 共有字段全同 | 不同的字段（计数） | 仅一侧有的字段 |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 50/50 / 50/50 | 50/50 / 50/50 | 100/100 / 100/100 | 无 | stopped |
| qwen3.5-2b | 50/50 / 50/50 | 50/50 / 50/50 | 100/100 / 100/100 | 无 | stopped |
| gemma-4-e2b-it | 50/50 / 50/50 | 50/50 / 50/50 | 100/100 / 100/100 | 无 | stopped |
| internvl3.5-2b-hf | 50/50 / 50/50 | 50/50 / 50/50 | 100/100 / 100/100 | 无 | stopped |
| qwen3.5-4b | 50/50 / 50/50 | 50/50 / 50/50 | 100/100 / 100/100 | 无 | stopped |
| gemma-4-e4b-it | 50/50 / 50/50 | 50/50 / 50/50 | 100/100 / 100/100 | 无 | stopped |
| internvl3.5-4b-hf | 50/50 / 50/50 | 50/50 / 50/50 | 100/100 / 100/100 | 无 | stopped |
| qwen3.5-9b | 50/50 / 50/50 | 50/50 / 50/50 | 100/100 / 100/100 | 无 | stopped |

#### `--audit-fmt32 100`（截出的 fmt32 记录 vs 真的 max_new_tokens=32 生成，逐字段，wall_time_s 除外）

| 模型 | 逐字段相同 | 不同的 id（字段） | 审计生成 token / 墙钟 s（不计入 compute） |
|---|---|---|---|
| qwen3.5-0.8b | 100/100 | 无 | 1437 / 17.2 |
| qwen3.5-2b | 100/100 | 无 | 531 / 7.7 |
| gemma-4-e2b-it | 100/100 | 无 | 1820 / 32.0 |
| internvl3.5-2b-hf | 100/100 | 无 | 590 / 6.5 |
| qwen3.5-4b | 100/100 | 无 | 204 / 4.4 |
| gemma-4-e4b-it | 100/100 | 无 | 1280 / 27.9 |
| internvl3.5-4b-hf | 100/100 | 无 | 201 / 3.1 |
| qwen3.5-9b | 100/100 | 无 | 200 / 4.4 |

#### 截断率（写满 max_new_tokens 的条数 / n）

| 模型 | MMLU-Pro @256 | MMLU-Pro fmt32 @32 | GSM8K @512 | IFEval @1024 |
|---|---|---|---|---|
| qwen3.5-0.8b | 39/100（39.0%） | 40/100（40.0%） | 8/50（16.0%） | 15/50（30.0%） |
| qwen3.5-2b | 10/100（10.0%） | 11/100（11.0%） | 10/50（20.0%） | 4/50（8.0%） |
| gemma-4-e2b-it | 51/100（51.0%） | 54/100（54.0%） | 10/50（20.0%） | 4/50（8.0%） |
| internvl3.5-2b-hf | 8/100（8.0%） | 13/100（13.0%） | 2/50（4.0%） | 3/50（6.0%） |
| qwen3.5-4b | 0/100（0.0%） | 0/100（0.0%） | 9/50（18.0%） | 1/50（2.0%） |
| gemma-4-e4b-it | 35/100（35.0%） | 36/100（36.0%） | 4/50（8.0%） | 4/50（8.0%） |
| internvl3.5-4b-hf | 0/100（0.0%） | 0/100（0.0%） | 12/50（24.0%） | 2/50（4.0%） |
| qwen3.5-9b | 0/100（0.0%） | 0/100（0.0%） | 6/50（12.0%） | 1/50（2.0%） |

#### 计算量（compute 字段；fmt32 从同一次生成截出，不单独记账）

| 模型 | 生成 token：MMLU-Pro / GSM8K / IFEval / 合计 | 墙钟 s：MMLU-Pro / GSM8K / IFEval / 合计 | 运行墙钟 s（wall_time_s，含加载与审计） |
|---|---|---|---|
| qwen3.5-0.8b | 10252 / 17753 / 22643 / 50648 | 117.6 / 200.5 / 255.4 / 573.5 | 597.01 |
| qwen3.5-2b | 2796 / 18526 / 16223 / 37545 | 33.9 / 219.1 / 191.3 / 444.3 | 459.31 |
| gemma-4-e2b-it | 13696 / 17305 / 18172 / 49173 | 241.6 / 313.5 / 321.6 / 876.7 | 916.4 |
| internvl3.5-2b-hf | 2728 / 16413 / 15253 / 34394 | 48.1 / 167.9 / 161.8 / 377.8 | 389.46 |
| qwen3.5-4b | 204 / 17619 / 13721 / 31544 | 4.6 / 273.6 / 213.2 / 491.4 | 503.73 |
| gemma-4-e4b-it | 9136 / 16067 / 17951 / 43154 | 210.5 / 361.2 / 386.6 / 958.2 | 994.66 |
| internvl3.5-4b-hf | 201 / 16919 / 15356 / 32476 | 7.0 / 228.3 / 206.1 / 441.5 | 450.63 |
| qwen3.5-9b | 200 / 17170 / 15110 / 32480 | 4.6 / 270.1 / 237.9 / 512.6 | 525.6 |

#### 解析失败

| 模型 | MMLU-Pro 256 | MMLU-Pro fmt32 | GSM8K strict | GSM8K flexible |
|---|---|---|---|---|
| qwen3.5-0.8b | 36/100 {"multiple_choices": 2, "no_choice": 34} | 39/100 {"no_choice": 39} | 8/50 {"no_marker": 8} | 0/50 {} |
| qwen3.5-2b | 10/100 {"no_choice": 10} | 11/100 {"no_choice": 11} | 9/50 {"no_marker": 9} | 0/50 {} |
| gemma-4-e2b-it | 52/100 {"multiple_choices": 3, "no_choice": 49} | 53/100 {"no_choice": 53} | 9/50 {"no_marker": 9} | 0/50 {} |
| internvl3.5-2b-hf | 11/100 {"no_choice": 10, "thinking_unclosed": 1} | 12/100 {"no_choice": 11, "thinking_unclosed": 1} | 4/50 {"no_marker": 4} | 0/50 {} |
| qwen3.5-4b | 0/100 {} | 0/100 {} | 8/50 {"no_marker": 8} | 0/50 {} |
| gemma-4-e4b-it | 34/100 {"multiple_choices": 4, "no_choice": 30} | 36/100 {"no_choice": 36} | 4/50 {"no_marker": 3, "no_number": 1} | 0/50 {} |
| internvl3.5-4b-hf | 0/100 {} | 0/100 {} | 7/50 {"no_marker": 1, "no_number": 6} | 0/50 {} |
| qwen3.5-9b | 0/100 {} | 0/100 {} | 5/50 {"no_marker": 5} | 0/50 {} |

### 3.2 `--compare` 汇总（v1 = RUN-0009 的 JSON，v2 = 本运行；从 `compare_v1_v2_<键>.json` 读取）

可比的 7 行是：

- `mmlu_pro_fmt32/accuracy`；
- `gsm8k/flexible`、`gsm8k/strict`；
- `ifeval` 的 4 个口径。

`mmlu_pro`（256）为 incomparable，原因是 v1 没有这个读数。

| 模型 | 可比的行 | delta≠0 或 lost/gained≠0 的行 | 截断率 delta（fmt32 / GSM8K / IFEval, pp） | any_flag | incomparable |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 7 | 无 | +0.0 / +0.0 / +0.0 | False | mmlu_pro |
| qwen3.5-2b | 7 | 无 | +0.0 / +0.0 / +0.0 | False | mmlu_pro |
| gemma-4-e2b-it | 7 | 无 | +0.0 / +0.0 / +0.0 | False | mmlu_pro |
| internvl3.5-2b-hf | 7 | 无 | +0.0 / +0.0 / +0.0 | False | mmlu_pro |
| qwen3.5-4b | 7 | 无 | +0.0 / +0.0 / +0.0 | False | mmlu_pro |
| gemma-4-e4b-it | 7 | 无 | +0.0 / +0.0 / +0.0 | False | mmlu_pro |
| internvl3.5-4b-hf | 7 | 无 | +0.0 / +0.0 / +0.0 | False | mmlu_pro |
| qwen3.5-9b | 7 | 无 | +0.0 / +0.0 / +0.0 | False | mmlu_pro |

每个模型的 stdout 原文见 `logs/compare_<键>.log`。下面以 qwen3.5-9b 为例：

```
protocol: base v1, merged v2
mmlu_pro        accuracy                  口径不同、标题不可比（见 notes）
mmlu_pro_fmt32  accuracy                  base=53.0 merged=53.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=100)  [诊断，不标红]
gsm8k           flexible                  base=86.0 merged=86.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=50)
gsm8k           strict                    base=86.0 merged=86.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=50)
ifeval          prompt_level_strict       base=92.0 merged=92.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=50)
ifeval          instruction_level_strict  base=94.8 merged=94.8 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=77)
ifeval          prompt_level_loose        base=98.0 merged=98.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=50)
ifeval          instruction_level_loose   base=98.7 merged=98.7 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=77)
truncation mmlu_pro_fmt32  max_new_tokens=32    base=0.0% merged=0.0% delta=+0.0 pp (n=100)
truncation gsm8k           max_new_tokens=512   base=12.0% merged=12.0% delta=+0.0 pp (n=50)
truncation ifeval          max_new_tokens=1024  base=2.0% merged=2.0% delta=+0.0 pp (n=50)
note: 口径不同、标题不可比：mmlu_pro（mmlu_pro，max_new_tokens=256）基座协议 v1，没有这个读数、合并后协议 v2，有这个读数。协议 v1 的 MMLU-Pro 只有 32 token 的读数，只与 v2 的 mmlu_pro_fmt32 比对（D28-7）。
any_flag=False  flagged=[]  truncation_notices=[]  incomparable=['mmlu_pro']  unscored=[]  partial_run=False
```

## 4. 与任务书假设不符的发现，以及我自己的遗漏

- **v2 记录新增 `stopped` 字段。** v2 的每条记录比 v1 多一个 `stopped` 字段（四个读数都有），v1 的记录没有。逐题核对只比两边共有的字段，所以 `stopped` 不在比较范围内。
- **compare 的截断率只有三行。** v1 没有 MMLU-Pro 256 的读数，所以 `--compare` 的截断率只列了 `mmlu_pro_fmt32`、`gsm8k`、`ifeval` 三行。MMLU-Pro 在 256 下的截断率只能从 v2 的 JSON 读，见 §3.1 的截断率表。
- **第一批没有 GPU 采样（我的遗漏）。** 启动第一批时我没有挂 nvidia-smi 采样，第二批开始后才补上。代价是前 4 个模型没有显存峰值和利用率的数据；运行时间和分数不受影响。之后的任务我会在启动时就挂上采样。
- **运行期间 origin 有新提交。** 详见 §1。这些提交没有改动健康检查的代码和题目。RUN-0012（`prompts/server/0012-e1-accept-estimate.md`）已在 origin 上发出，我会在本任务完成后按 STANDBY 接单。

## 5. 生成的文件

`experiments/P0/0011_health_v2_base/` 下的文件：

- 8 份 `health_<键>.json`、8 份 `compare_v1_v2_<键>.json`；
- `health_summary_v2.md`、`summary_v2.json`；
- `pip_dryrun.txt`、`disk_ledger.tsv`、`README.md`；
- `logs/`：8 份 health 日志、8 份 compare 日志、`driver.log`、两份 GPU 采样 csv；
- `probe_scripts/`：3 个脚本。

整个目录约 3.8 MB。其中第一批的 4 份 health、4 份 compare、probe 脚本和 `pip_dryrun.txt` 已在 `99e004b` 推送。

## 6. 磁盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 07:05:07 | 0011 start | 39.362 | 3.103 | 3.853 |
| 07:05:09 | before download qwen3.5-0.8b | 39.362 | 3.103 | 3.853 |
| 07:05:10 | after download qwen3.5-0.8b | 39.362 | 3.103 | 3.853 |
| 07:15:10 | after health qwen3.5-0.8b | 39.361 | 3.103 | 3.854 |
| 07:15:10 | before download qwen3.5-2b | 39.361 | 3.103 | 3.854 |
| 07:15:22 | after download qwen3.5-2b | 39.361 | 7.661 | 3.854 |
| 07:23:05 | after health qwen3.5-2b | 39.360 | 7.661 | 3.854 |
| 07:23:05 | after delete qwen3.5-2b | 39.360 | 3.103 | 3.854 |
| 07:23:05 | before download gemma-4-e2b-it | 39.360 | 3.103 | 3.854 |
| 07:23:29 | after download gemma-4-e2b-it | 39.360 | 13.350 | 3.854 |
| 07:38:48 | after health gemma-4-e2b-it | 39.359 | 13.350 | 3.855 |
| 07:38:49 | after delete gemma-4-e2b-it | 39.359 | 3.071 | 3.855 |
| 07:38:49 | before download internvl3.5-2b-hf | 39.359 | 3.071 | 3.855 |
| 07:39:01 | after download internvl3.5-2b-hf | 39.359 | 7.770 | 3.855 |
| 07:45:34 | after health internvl3.5-2b-hf | 39.359 | 7.770 | 3.855 |
| 07:45:34 | after delete internvl3.5-2b-hf | 39.359 | 3.056 | 3.855 |
| 07:45:55 | before download qwen3.5-4b | 39.358 | 3.056 | 3.855 |
| 07:46:17 | after download qwen3.5-4b | 39.356 | 12.386 | 3.857 |
| 07:54:43 | after health qwen3.5-4b | 39.355 | 12.386 | 3.858 |
| 07:54:44 | after delete qwen3.5-4b | 39.355 | 3.056 | 3.858 |
| 07:54:44 | before download gemma-4-e4b-it | 39.355 | 3.056 | 3.858 |
| 07:55:22 | after download gemma-4-e4b-it | 39.355 | 19.081 | 3.858 |
| 08:11:59 | after health gemma-4-e4b-it | 39.355 | 19.081 | 3.858 |
| 08:12:00 | after delete gemma-4-e4b-it | 39.355 | 3.057 | 3.858 |
| 08:12:00 | before download internvl3.5-4b-hf | 39.355 | 3.057 | 3.858 |
| 08:12:23 | after download internvl3.5-4b-hf | 39.355 | 12.540 | 3.858 |
| 08:19:56 | after health internvl3.5-4b-hf | 39.354 | 12.540 | 3.859 |
| 08:19:57 | after delete internvl3.5-4b-hf | 39.354 | 3.057 | 3.859 |
| 08:19:57 | before download qwen3.5-9b | 39.354 | 3.057 | 3.859 |
| 08:20:39 | after download qwen3.5-9b | 39.354 | 22.374 | 3.859 |
| 08:29:28 | after health qwen3.5-9b | 39.353 | 22.374 | 3.859 |
| 08:29:29 | after delete qwen3.5-9b | 39.353 | 3.058 | 3.859 |

- **根盘：** 可用空间从 39.362 GB 降到 39.353 GB（−9 MB），全程远高于 3 GB 红线。
- **`$PROJ`：** 从 3.853 GB 升到 3.859 GB，在 15 GB 份额以内（D28-8）。本目录的产物和 `$PROJ/logs/0011`（140 KB）都已计入。
- **内存盘：** 峰值 22.374 GB，出现在 qwen3.5-9b，低于 25 GB 上限。
  - 结束时 `/dev/shm/visint` 是 3.058 GB：`hf-hub` 1.770 GB 只剩 Qwen3.5-0.8B；`tmp` 1.277 GB 是 RUN-0010 的 NYU 点对渲染缓存。
- **残留的小条目：** 运行前内存盘里有 InternVL3_5-2B-HF（15.9 MB）、InternVL3_5-4B-HF（43 KB）、gemma-4-E2B-it（32.2 MB）三个以前留下的小缓存条目。它们在对应模型下载、再用 `delete_revisions` 删除后都被清掉了，这就是 `/dev/shm/visint` 从 3.103 GB 降到 3.058 GB 的原因。
