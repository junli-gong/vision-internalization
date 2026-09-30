# RUN-0007 — 环境更新 + 全量测试 + E5 v2 摘要（填入 d 的规范值）+ 8 个基座模型的健康检查

服务器 CC，b200，2026-09-30 约 12:25（接单）–13:57 UTC（值守）。只贴数字与事实，不做解读。原始数据在 `experiments/P0/0007_health_base/`。

## 0. 一句话观察（只陈述，不解读）

- 依赖只新装了 6 个包，没有升降任何已装包；没有下载 punkt_tab，也没有设 NLTK_DATA（N12）。
- 全量测试：`1030 passed, 11 skipped, 23 warnings in 255.84s (0:04:15)`，rc=0。跳过项为 10 个缺 punkt_tab、1 个缺 cv2；没有 xfail。
- E5 v2：c 60/60、d 60/60、e 60/60 与仓库里钉住的摘要相同。本平台（规范平台）上写出的 d 清单，60 个摘要与仓库里开发机的占位值逐个相同，只有平台信息与状态字段不同。
- 8 个基座模型的健康检查全部 `ok=True`、退出码 0，共用约 73 分钟，没有触发 8 小时预算。IFEval 只生成、未评分。

## 1. 前置校验

- `git pull --ff-only` 后 HEAD 为 `64d0569`。本任务书所在的 commit `a566ede` 与 `647b007` 都在 HEAD 历史中。
- 本次没有修改 `src/`、`scripts/`、`tests/`、`configs/`，也没有改 `tests/data/`。新写的只有 `experiments/P0/0007_health_base/probe_scripts/`（驱动 `run_health.sh`、汇总 `p11_health_summary.py`、记账 `diskcheck7.sh`）。
- **接单时根盘可用为 42.759 GB**。RUN-0006 结束时是 5.64 GB，这之间有人释放了约 37 GB，本服务器 CC 没有做任何删除。

## 2. 执行记录

| 步骤 | 命令 | 墙钟 | 结果 |
|---|---|---|---|
| 依赖 dry-run | `pip install --dry-run -r requirements.txt` | 数秒 | Would install cloudpickle-3.1.2 defusedxml-0.7.1 immutabledict-4.3.1 joblib-1.6.0 langdetect-1.0.9 nltk-3.10.3（没有 uninstall） |
| 安装 | `PIP_NO_CACHE_DIR=1 pip install -r requirements.txt` | 1.5 s | `Successfully installed cloudpickle-3.1.2 defusedxml-0.7.1 immutabledict-4.3.1 joblib-1.6.0 langdetect-1.0.9 nltk-3.10.3`（langdetect 从 sdist 构建） |
| 全量测试 | `python -m pytest tests/ -q -rs -rx` | 4 m 18 s | 1030 passed, 11 skipped, 23 warnings in 255.84s (0:04:15) |
| 跳过原因 | `python -m pytest tests/ -q -rsx`（补跑，见 §4） | 4 m 10 s | 同样 1030 passed, 11 skipped |
| E5 核对 | `python scripts/synth_pin_digests.py check` | 1.5 s | rc=0 |
| d 规范值 | `python scripts/synth_pin_digests.py write --only physics --out experiments/P0/0007_health_base/synth_digests_v2.json` | 数秒 | 写出 60 个 d 摘要，状态 canonical |
| 健康检查 ×8 | `python scripts/health_check.py --model <键> --device cuda --dtype bf16 --skip-ifeval-scoring --out experiments/P0/0007_health_base/health_<键>.json`（由 `run_health.sh` 依次执行，每个模型前后下载与删除） | 见下表 | 全部 rc=0 |

| 模型 | 开始 → 结束（UTC） | 退出码 |
|---|---|---|
| qwen3.5-0.8b | 12:42:22 → 12:50:12 | 0 |
| qwen3.5-2b | 12:50:12 → 12:57:15 | 0 |
| gemma-4-e2b-it | 12:57:15 → 13:08:25 | 0 |
| internvl3.5-2b-hf | 13:08:25 → 13:14:35 | 0 |
| qwen3.5-4b | 13:15:37 → 13:24:39 | 0 |
| gemma-4-e4b-it | 13:24:39 → 13:38:30 | 0 |
| internvl3.5-4b-hf | 13:38:30 → 13:46:11 | 0 |
| qwen3.5-9b | 13:46:12 → 13:55:28 | 0 |

- GPU：NVIDIA B200，bf16，sdpa。
- 警告（原样摘要）：
  - 4 个 Qwen：`causal_conv1d_fn` / `causal_conv1d_update` 回退到 torch；
  - 2 个 InternVL：tie weights 提示；
  - 2 个 Gemma：没有警告。
  - 原文见 `health_<键>.log`。

## 3. 结果表（原样）

### 3.1 全量测试

`1030 passed, 11 skipped, 23 warnings in 255.84s (0:04:15)`

跳过项（`pytest_rsx.txt`）：
```
SKIPPED [9] tests/test_health_ifeval.py:177: 找不到 punkt_tab：设 NLTK_DATA 指向含 tokenizers/punkt_tab 的目录（python -m nltk.downloader -d <目录> punkt_tab）
SKIPPED [1] tests/test_health_ifeval.py:279: 找不到 punkt_tab：设 NLTK_DATA 指向含 tokenizers/punkt_tab 的目录（python -m nltk.downloader -d <目录> punkt_tab）
SKIPPED [1] tests/test_metrics_seg.py:129: could not import 'cv2': No module named 'cv2'
1030 passed, 11 skipped, 23 warnings in 249.64s (0:04:09)
```

### 3.2 E5 摘要比对（`synth_check.txt` 原样）

```
platform x86_64 Linux; canonical: True; file version 2, code version 2
occlusion: 60/60 equal 
physics: 60/60 equal pending_canonical
imagery: 60/60 equal 

real	0m1.500s
user	0m6.114s
sys	0m0.097s
```

- `synth_digests_v2.json`（写到运行目录，不改 `tests/data/`）：
  - 只替换了 `tasks.physics`。60 个摘要与仓库文件里开发机（arm64 Darwin）的占位值**逐个相同**。
  - 变化的只有 `generated_on`（x86_64 Linux，Python 3.12.13，numpy 2.5.2，pymunk 7.3.0，Chipmunk 2.0.1-ade7ed7，glibc 2.35）、`status: canonical` 和 `status_note`。
  - `tasks.occlusion`、`tasks.imagery` 照抄仓库文件。

### 3.3 健康检查（由 `probe_scripts/p11_health_summary.py` 从 8 份 JSON 生成，即 `health_summary.md`）

## 分数表（D26 协议；IFEval 按 --skip-ifeval-scoring 只生成、未评分）

| 模型 | ok | MMLU-Pro 准确率 | GSM8K flexible（标题口径） | GSM8K strict | IFEval |
|---|---|---|---|---|---|
| qwen3.5-0.8b | True | 14.0（14/100） | 58.0（29/50） | 56.0（28/50） | 未评分 |
| qwen3.5-2b | True | 27.0（27/100） | 76.0（38/50） | 72.0（36/50） | 未评分 |
| gemma-4-e2b-it | True | 25.0（25/100） | 80.0（40/50） | 78.0（39/50） | 未评分 |
| internvl3.5-2b-hf | True | 34.0（34/100） | 80.0（40/50） | 76.0（38/50） | 未评分 |
| qwen3.5-4b | True | 46.0（46/100） | 82.0（41/50） | 82.0（41/50） | 未评分 |
| gemma-4-e4b-it | True | 34.0（34/100） | 88.0（44/50） | 86.0（43/50） | 未评分 |
| internvl3.5-4b-hf | True | 46.0（46/100） | 92.0（46/50） | 80.0（40/50） | 未评分 |
| qwen3.5-9b | True | 53.0（53/100） | 86.0（43/50） | 86.0（43/50） | 未评分 |

## 解析失败、写满 max_new_tokens、计算量

| 模型 | MMLU-Pro 解析失败（原因） | GSM8K strict 解析失败 | GSM8K flexible 解析失败 | 写满 max_new_tokens：MMLU-Pro(32) / GSM8K(512) / IFEval(1024) | 思考未闭合 / 空回答（三项合计） | 生成 token：MMLU / GSM8K / IFEval / 合计 | 墙钟 s（运行合计 / 三项计时合计） |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 39/100 {"no_choice": 39} | 8/50 {"no_marker": 8} | 0/50 {} | 40 / 8 / 15 | 0 / 0 | 1437 / 17753 / 22643 / 41833 | 468.68 / 463.0 |
| qwen3.5-2b | 11/100 {"no_choice": 11} | 9/50 {"no_marker": 9} | 0/50 {} | 11 / 10 / 4 | 0 / 0 | 531 / 18526 / 16223 / 35280 | 408.64 / 402.0 |
| gemma-4-e2b-it | 53/100 {"no_choice": 53} | 9/50 {"no_marker": 9} | 0/50 {} | 54 / 10 / 4 | 0 / 0 | 1820 / 17305 / 18172 / 37297 | 646.7 / 638.9 |
| internvl3.5-2b-hf | 12/100 {"no_choice": 11, "thinking_unclosed": 1} | 4/50 {"no_marker": 4} | 0/50 {} | 13 / 2 / 3 | 1 / 1 | 590 / 16413 / 15253 / 32256 | 355.87 / 350.6 |
| qwen3.5-4b | 0/100 {} | 8/50 {"no_marker": 8} | 0/50 {} | 0 / 9 / 1 | 0 / 0 | 204 / 17619 / 13721 / 31544 | 519.49 / 512.4 |
| gemma-4-e4b-it | 36/100 {"no_choice": 36} | 4/50 {"no_marker": 3, "no_number": 1} | 0/50 {} | 36 / 4 / 4 | 0 / 0 | 1280 / 16067 / 17951 / 35298 | 801.21 / 792.2 |
| internvl3.5-4b-hf | 0/100 {} | 7/50 {"no_marker": 1, "no_number": 6} | 0/50 {} | 0 / 12 / 2 | 0 / 0 | 201 / 16919 / 15356 / 32476 | 439.83 / 434.3 |
| qwen3.5-9b | 0/100 {} | 5/50 {"no_marker": 5} | 0/50 {} | 0 / 6 / 1 | 0 / 0 | 200 / 17170 / 15110 / 32480 | 520.92 / 512.7 |


- 题目文件 `configs/health_check_items.json`，sha256 `0239675f9c226f17963419e2e12a282dd4ced8ed42452b53a939145372a525df`，与 D26 一致。
- 生成设置（JSON 的 generation 字段）：`instruction_first`，无图像，`enable_thinking=False`，`do_sample=False, num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0`；max_new_tokens 为 MMLU-Pro 32、GSM8K 512、IFEval 1024。
- IFEval 的四个口径在 JSON 里都是 `n_correct: null, pct: null`，并附说明：`--skip-ifeval-scoring: IFEval generated but not scored ... Score later with scripts/health_check.py --rescore.`；原始输出保存在各 JSON 的 `records.ifeval`。

## 4. 与任务书假设不符的发现

1. **`-rs -rx` 没有打印跳过原因**：pytest 里后一个 `-r` 会覆盖前一个，实际只生效 `-rx`。本次没有 xfail，所以 `pytest.txt` 里没有 short test summary。补跑一次 `-rsx`，结果相同（1030 passed, 11 skipped），跳过原因见 §3.1。
2. **d 的比对是 pass 而不是 xfail**：60/60 相同。任务书写明两者都算正常。
3. **InternVL3.5-2B 在关思考的 MMLU-Pro 里有 1 题被记为 `thinking_unclosed`**：解析失败原因 `thinking_unclosed`，diagnostics 的思考未闭合与空回答各 1。只陈述，原始输出在 `health_internvl3.5-2b-hf.json` 的 `records.mmlu_pro`。
4. 根盘可用在接单前从约 5.6 GB 升到 42.76 GB（§1），不是本任务所为。

## 5. 自己造成的错误或浪费

无。补跑 `-rsx` 多花了约 4 分钟，目的是记下跳过原因。

## 6. 生成文件清单与磁盘收支

**仓库内**：`experiments/P0/0007_health_base/`：
- `pip_dryrun.txt`、`pip_install.log`
- `pytest.txt`、`pytest_rsx.txt`
- `synth_check.txt`、`synth_digests_v2.json`
- 8 份 `health_<键>.json` 与 `health_<键>.log`，以及 `health_summary.md`
- `driver.log`、`disk_ledger.tsv`、`README.md`、`probe_scripts/`

另有本 report。

**venv**：新装 cloudpickle 3.1.2、defusedxml 0.7.1、immutabledict 4.3.1、joblib 1.6.0、langdetect 1.0.9、nltk 3.10.3。安装前后，`df /` 可用 42.759 → 42.738 GB，`$PROJ` 0.499 → 0.517 GB。

**内存盘**：
- 峰值 21.094 GB（Qwen3.5-0.8B 加 9B，上限 25 GB）；
- 每个模型跑完即用 `scan_cache_dir().delete_revisions()` 删除；
- 现只剩 Qwen3.5-0.8B（1.770 GB）。

**磁盘收支**（`disk_ledger.tsv`；GB = 1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 12:31:59 | 0007 start / before pip | 42.759 | 1.823 | 0.499 |
| 12:32:12 | after pip install | 42.738 | 1.823 | 0.517 |
| 12:36:30 | after pytest | 42.734 | 1.823 | 0.519 |
| 12:41:55 | before 0007 batch1 commit | 42.696 | 1.823 | 0.519 |
| 12:42:22 | before download qwen3.5-0.8b | 42.696 | 1.823 | 0.520 |
| 12:42:22 | after download qwen3.5-0.8b | 42.696 | 1.823 | 0.520 |
| 12:50:12 | after health qwen3.5-0.8b | 42.683 | 1.823 | 0.520 |
| 12:50:12 | before download qwen3.5-2b | 42.683 | 1.823 | 0.520 |
| 12:50:24 | after download qwen3.5-2b | 42.683 | 6.381 | 0.520 |
| 12:57:15 | after health qwen3.5-2b | 42.681 | 6.381 | 0.521 |
| 12:57:15 | after delete qwen3.5-2b | 42.681 | 1.823 | 0.521 |
| 12:57:15 | before download gemma-4-e2b-it | 42.681 | 1.823 | 0.521 |
| 12:57:36 | after download gemma-4-e2b-it | 42.681 | 12.070 | 0.521 |
| 13:08:25 | after health gemma-4-e2b-it | 42.676 | 12.070 | 0.522 |
| 13:08:25 | after delete gemma-4-e2b-it | 42.676 | 1.791 | 0.522 |
| 13:08:25 | before download internvl3.5-2b-hf | 42.676 | 1.791 | 0.522 |
| 13:08:38 | after download internvl3.5-2b-hf | 42.676 | 6.490 | 0.522 |
| 13:14:35 | after health internvl3.5-2b-hf | 42.671 | 6.490 | 0.523 |
| 13:14:36 | after delete internvl3.5-2b-hf | 42.671 | 1.776 | 0.523 |
| 13:15:37 | before download qwen3.5-4b | 42.671 | 1.776 | 0.523 |
| 13:15:57 | after download qwen3.5-4b | 42.671 | 11.106 | 0.523 |
| 13:24:39 | after health qwen3.5-4b | 42.667 | 11.106 | 0.524 |
| 13:24:39 | after delete qwen3.5-4b | 42.667 | 1.776 | 0.524 |
| 13:24:39 | before download gemma-4-e4b-it | 42.667 | 1.776 | 0.524 |
| 13:25:07 | after download gemma-4-e4b-it | 42.667 | 17.801 | 0.524 |
| 13:38:30 | after health gemma-4-e4b-it | 42.663 | 17.801 | 0.524 |
| 13:38:30 | after delete gemma-4-e4b-it | 42.663 | 1.777 | 0.524 |
| 13:38:31 | before download internvl3.5-4b-hf | 42.663 | 1.777 | 0.524 |
| 13:38:50 | after download internvl3.5-4b-hf | 42.663 | 11.260 | 0.524 |
| 13:46:12 | after health internvl3.5-4b-hf | 42.661 | 11.260 | 0.525 |
| 13:46:12 | after delete internvl3.5-4b-hf | 42.661 | 1.777 | 0.525 |
| 13:46:12 | before download qwen3.5-9b | 42.661 | 1.777 | 0.525 |
| 13:46:45 | after download qwen3.5-9b | 42.660 | 21.094 | 0.525 |
| 13:55:28 | after health qwen3.5-9b | 42.657 | 21.094 | 0.525 |
| 13:55:29 | after delete qwen3.5-9b | 42.657 | 1.778 | 0.525 |
| 13:56:44 | after all health checks | 42.657 | 1.778 | 0.525 |
| 13:57:33 | before 0007 report | 42.656 | 1.778 | 0.525 |

- 根盘最低 42.656 GB（红线 3 GB）。
- 没有删除、移动或修改任何非本项目文件；没有下载 nltk 数据。
