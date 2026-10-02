# RUN-0013 — E1 批处理验收与算力估算重跑（新的生成上限；三种一致口径）＋ CountBench 人工问句落地

服务器 CC，b200，2026-10-01 14:50–23:02 UTC，值守中接单。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- **落地：** `countbenchqa_questions` 落地 1 个文件（31,595 B，540 条问句），sha256 与钉住的值相同，rc=0。
- **验收（§3，8 个模型都跑）：** 全部 48 次 run_e1 的 rc 都是 0。8 个模型在三种口径下 all_pass 都是 False。
  - 不通过的格子数（共 72 个"模型 × 任务"格子）：
    - (a) 逐位 51 个；
    - (b) IoU≥0.9 46 个；
    - (c) 判定结果 36 个，另有 8 个 ovd 格子不判（开发子集没有 GT）。
  - (c) 口径下不通过的任务：
    - spatial_embspatial 8/8，rel_depth_nyu 8/8，spatial_cvbench 7/8，rel_depth_blink 5/8；
    - refexp_poly 3/8，count 3/8，grid16 2/8；
    - refexp_box 0/8。
  - 配对差 95% CI 不含 0 的只有 1 格：qwen3.5-2b 的 rel_depth_nyu，差 +0.07，CI [0.01, 0.13]。其余不通过的格子都是因为一致率 < 0.98。
- **截断（新的上限下）：**
  - 多选与点对四个任务的截断率每个模型 ≤ 0.083，最大值是 rel_depth_blink 的 1/12。
  - grid16 在 qwen3.5-0.8b、qwen3.5-2b 上仍是 1.00 / 0.99。
  - ovd 在 qwen3.5-0.8b 上是 0.61 / 0.65（bs1 / 批量）。
- **计数换成 CountBenchQA 人工问句（n=49）：** 三种口径下 5/8 通过。不通过的 qwen3.5-2b、internvl3.5-2b-hf、internvl3.5-4b-hf 都是 48/49 一致（0.9796），配对差 CI 都含 0。
- **估算（§4）：**
  - internvl3.5-2b-hf 跑完时累计已到 29,492 s，超过 8 小时，按 §5 没有启动 qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b。完成的范围与 RUN-0012 相同。
  - 完成的部分：主口径 4 个模型（qwen3.5-0.8b、qwen3.5-2b、gemma-4-e2b-it、internvl3.5-2b-hf），附加口径 qwen3.5-0.8b qwen:1024 和 internvl3.5-2b-hf internvl:native。
  - 已测 4 个模型的主口径合计：eval 12.102、think 11.191、think8192（上界）5.864 GPU 小时。RUN-0012 的同 4 个模型是 7.238 / 11.025 / 5.779。
- **本次用量：** 29,492 s（8.19 h）。其中验收 22,977 s（6.38 h），估算 6,515 s（1.81 h）。按 D28-10 计入 E1 预算。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `0721f83`。`26e871d` 与本任务书所在的 `de9fd4f` 都在 HEAD 的历史里（`git merge-base --is-ancestor`）。
- **环境：**
  - `pip install --dry-run -r requirements.txt` 没有任何 "Would install"，见 `experiments/E1/0013_acceptance/pip_dryrun.txt`。
  - 驱动脚本里 export 了 `VISINT_DATA_ROOT=$PROJ/data`、`NLTK_DATA=$PROJ/nltk_data`、`TMPDIR=/dev/shm/visint/tmp`。
- **提示配置：** sha256 是 `e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6`（各 acceptance.json 的 header），与 D33 一致。
- **改动范围：**
  - 没有修改 `src/`、`scripts/`、`tests/`、`configs/`，也没有安装 Python 包。
  - 新写的文件只在运行目录的 `probe_scripts/` 下：`run_accept13.sh`、`chain13.sh`、`run_estimate13.sh`、`p17_accept_summary.py`、`gpu_sampler.sh`、`diskcheck13.sh`；`p16_estimate_summary.py` 从 RUN-0012 原样复制。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。
- **运行期间 origin 有新提交：**
  - 推第一批之前 `git pull` 拉到了 `8337b82`，中间的 commit 是 `169cbb6`（RUN-0012 关闭，预注册的句数改为 56,857）和 `8337b82`（D34、N16）。
  - 只改了 `DECISIONS.md`、`DECISIONS_NEEDED.md`、`PREREGISTRATION.md`、`RESEARCH_LOG.md`；`git diff --stat 0721f83 8337b82 -- src scripts configs requirements.txt` 的输出为空。
  - D34 的检查点规则与本任务无关：本任务不存权重或检查点。
- **只用了开发子集。** 没有运行任何 eval/think/think8192/extra 子集，估算只按 n 外推。

## 2. 执行记录

### 2.1 落地

`python scripts/data/land_e1.py countbenchqa_questions --root $PROJ/data` 的执行情况：

- 0.54 s，rc=0；
- 估算体积 0.000 GB，根盘前后都剩 39.27 GB；
- 1 个文件，失败 0，核对不通过 0；
- `stats.n_questions=540`；
- 上游是 `google-research/big_vision@ec104fd760a1272796e12e5fcde42f6bc3f78f9b`，许可 Apache-2.0。

清单副本见 `experiments/E1/0013_acceptance/countbenchqa_questions_MANIFEST.json`，日志见 `land_countbenchqa.log`。

### 2.2 验收与估算

- **驱动：** 由 `chain13.sh` 启动，先跑 `run_accept13.sh`（8 个模型），再跑 `run_estimate13.sh`（8 个模型）。每个模型开始前都检查累计耗时是否超过 28,800 s。
- **每个模型的验收命令，共 6 条：**
  1. `--task all --agreement bitwise`，生成两遍；
  2. `--resume --agreement iou90`，只重算汇总；
  3. `--resume --agreement outcome`，只重算汇总；
  4. `--task count --count-questions countbenchqa --agreement bitwise`，用另一个 `--out`：`experiments/E1/0013_acceptance_countbenchqa`，原因见 §4；
  5. 第 4 条的 `--resume --agreement iou90`；
  6. 第 4 条的 `--resume --agreement outcome`。
- **汇总副本：** 每次运行之后，驱动把 `acceptance.json` 和 `acceptance.md` 复制成 `acceptance__<口径>.*`，因为它们每次都会被覆盖。
- **`--resume` 各遍：** 日志里没有生成进度行，所有 id 都已在 `predictions.jsonl` 里。

| # | 阶段 | 模型 | 权重核对 | 下载+核对 | bitwise 那遍（`time` real） | 其余 5 遍（iou90、outcome、cbqa bitwise、cbqa iou90、cbqa outcome） | rc | GPU 显存峰值（nvidia-smi 按进程，MiB） |
|---|---|---|---|---|---|---|---|---|
| 1 | 验收 | qwen3.5-0.8b | 13/13 | 1 s（已在内存盘） | 65m42.0s | 11.6 / 11.7 / 11.8 / 7.3 / 7.3 s | 全 0 | 4446 |
| 2 | 验收 | qwen3.5-2b | 13/13 | 12 s | 56m44.6s | 12.0 / 12.2 / 12.4 / 7.7 / 7.7 s | 全 0 | 7108 |
| 3 | 验收 | gemma-4-e2b-it | 9/9 | 31 s | 40m47.4s | 13.1 / 13.0 / 13.8 / 8.6 / 8.5 s | 全 0 | 13848 |
| 4 | 验收 | internvl3.5-2b-hf | 18/18 | 16 s | 25m22.1s | 10.5 / 10.4 / 36.8 / 5.7 / 5.9 s | 全 0 | 40984 |
| 5 | 验收 | qwen3.5-4b | 14/14 | 22 s | 46m35.2s | 12.6 / 12.7 / 13.6 / 8.1 / 8.1 s | 全 0 | 21914 |
| 6 | 验收 | gemma-4-e4b-it | 9/9 | 37 s | 48m45.4s | 13.7 / 13.8 / 15.2 / 9.2 / 9.4 s | 全 0 | 20028 |
| 7 | 验收 | internvl3.5-4b-hf | 20/20 | 24 s | 41m00.8s | 10.9 / 10.7 / 15.7 / 6.2 / 6.2 s | 全 0 | 34552 |
| 8 | 验收 | qwen3.5-9b | 16/16 | 43 s | 47m14.1s | 13.4 / 13.5 / 14.7 / 8.9 / 8.9 s | 全 0 | 23962 |
| 9 | 估算 | qwen3.5-0.8b（default） | 13/13 | 1 s | 23m23.5s | — | 0 | 20236 |
| 10 | 估算 | qwen3.5-0.8b（qwen:1024） | — | — | 6m57.2s | — | 0 | 9898 |
| 11 | 估算 | qwen3.5-2b | 13/13 | 12 s | 23m10.5s | — | 0 | 38698 |
| 12 | 估算 | gemma-4-e2b-it | 9/9 | 23 s | 17m19.8s | — | 0 | 16660 |
| 13 | 估算 | internvl3.5-2b-hf（default） | 18/18 | 12 s | 24m32.8s | — | 0 | 162442 |
| 14 | 估算 | internvl3.5-2b-hf（internvl:native） | — | — | 12m20.1s | — | 0 | 40490 |
| — | 估算 | qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b | — | — | 未开始（累计 29,492 s > 28,800 s） | — | — | — |

- **GPU：** NVIDIA B200，bf16。
  - nvidia-smi 每 5 s 采样一次，共 5,865 个样本：验收 4,570 个，估算 1,295 个。
  - 整卡显存峰值：验收阶段 40,999 MiB，估算阶段 162,456 MiB。
  - GPU 利用率均值：验收 55.2%，估算 58.1%。
  - 采样期间 GPU 上依次出现 54 个 python 进程，时间区间互不重叠，对应 48 次验收和 6 次估算。
- **第 13 行的显存（只陈述）：** nvidia-smi 看到这个进程占 162,442 MiB，但同一运行的 estimate.json 里，`torch.cuda.max_memory_allocated` 的最大值是 14.27 GB（ovd、bs=32）。所有批大小都是 `mem_ok=True`，没有 OOM。
- **权重：** 12 次核对全部一致。
- **报错：** 54 次 run_e1 都没有 Traceback。
- **警告：** 与 RUN-0012 相同（Qwen 的 `causal_conv1d` 回退、InternVL 的 tie weights 提示），原文见 `logs/`。
- **累计耗时：** 驱动日志里记的是各模型验收开始时的累计秒数：0、3993、7462、9998、11606、14479、17504、20039；验收结束时为 22977。估算各模型开始时：22977、24798、26202、27265；internvl3.5-2b-hf 结束时为 29492。

## 3. 结果表（原样）

### 3.1 批处理验收（三种口径；`experiments/E1/0013_acceptance/accept_summary.md`）

本节由 `p17_accept_summary.py` 从 `acceptance__outcome.json` 生成；三种口径每一遍都会算，所以三份 `acceptance__*.json` 里每行的 `agreement_by_mode` 相同。另外用到了两遍的 `metrics.json` 与 `compute.json`。诊断原样取自 `metrics.json` 的 `diagnostics.failure_detail` 和 `diagnostics.think_marker_when_off`。

#### 验收汇总（三种一致口径；判据：一致率 ≥ 0.98 且配对差 95% CI 含 0；计数用默认的启发式问句）

| 模型 | 任务 | n | (a) 逐位 一致率 / 通过 | (b) IoU≥0.9 一致率 / 通过 | (c) 判定结果 一致率 / 通过 | 指标 | 逐条 | 批量 | 差（逐条−批量） | 配对差 95% CI | CI 含 0 | raw_text 逐字相同 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 100 | 0.3000 / 否 | 0.9700 / 否 | 0.9800 / 是 | acc@0.5 | 0.8000 | 0.8200 | -0.0200 | [-0.0500, 0.0000] | 是 | 0.2800 |
| qwen3.5-0.8b | refexp_poly | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3300 |
| qwen3.5-0.8b | grid16 | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-0.8b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9592 |
| qwen3.5-0.8b | ovd | 100 | 0.5900 / 否 | 0.8100 / 否 | — / — | — | — | — | — | — | — | 0.0000 |
| qwen3.5-0.8b | rel_depth_nyu | 100 | 0.7800 / 否 | 0.7800 / 否 | 0.7800 / 否 | accuracy | 0.6300 | 0.5700 | 0.0600 | [-0.0300, 0.1500] | 是 | 0.1400 |
| qwen3.5-0.8b | rel_depth_blink | 12 | 0.7500 / 否 | 0.7500 / 否 | 0.7500 / 否 | accuracy | 0.7500 | 0.7500 | 0.0000 | [-0.2500, 0.2500] | 是 | 0.1667 |
| qwen3.5-0.8b | spatial_cvbench | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.7200 | 0.7000 | 0.0200 | [-0.0400, 0.1000] | 是 | 0.3200 |
| qwen3.5-0.8b | spatial_embspatial | 50 | 0.8400 / 否 | 0.8400 / 否 | 0.8400 / 否 | accuracy | 0.3600 | 0.4000 | -0.0400 | [-0.1200, 0.0400] | 是 | 0.4000 |
| qwen3.5-2b | refexp_box | 100 | 0.2900 / 否 | 0.9800 / 是 | 1.0000 / 是 | acc@0.5 | 0.8600 | 0.8600 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2800 |
| qwen3.5-2b | refexp_poly | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3000 |
| qwen3.5-2b | grid16 | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9900 |
| qwen3.5-2b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7143 | 0.7143 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-2b | ovd | 100 | 0.3500 / 否 | 0.7200 / 否 | — / — | — | — | — | — | — | — | 0.0200 |
| qwen3.5-2b | rel_depth_nyu | 100 | 0.9100 / 否 | 0.9100 / 否 | 0.9100 / 否 | accuracy | 0.7500 | 0.6800 | 0.0700 | [0.0100, 0.1300] | 否 | 0.2000 |
| qwen3.5-2b | rel_depth_blink | 12 | 0.8333 / 否 | 0.8333 / 否 | 0.8333 / 否 | accuracy | 0.6667 | 0.7500 | -0.0833 | [-0.2500, 0.0000] | 是 | 0.0000 |
| qwen3.5-2b | spatial_cvbench | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.8200 | 0.8200 | 0.0000 | [-0.0600, 0.0600] | 是 | 0.5400 |
| qwen3.5-2b | spatial_embspatial | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.5400 | 0.5000 | 0.0400 | [0.0000, 0.1000] | 是 | 0.3800 |
| gemma-4-e2b-it | refexp_box | 100 | 0.9100 / 否 | 0.9800 / 是 | 1.0000 / 是 | acc@0.5 | 0.0800 | 0.0800 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7000 |
| gemma-4-e2b-it | refexp_poly | 100 | 0.6900 / 否 | 0.6900 / 否 | 0.6900 / 否 | cIoU | 0.2413 | 0.2340 | 0.0073 | [-0.0016, 0.0163] | 是 | 0.5400 |
| gemma-4-e2b-it | grid16 | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9500 |
| gemma-4-e2b-it | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.3673 | 0.3673 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 |
| gemma-4-e2b-it | ovd | 100 | 0.4500 / 否 | 0.8000 / 否 | — / — | — | — | — | — | — | — | 0.4100 |
| gemma-4-e2b-it | rel_depth_nyu | 100 | 0.8400 / 否 | 0.8400 / 否 | 0.8400 / 否 | accuracy | 0.6400 | 0.6200 | 0.0200 | [-0.0500, 0.0900] | 是 | 0.0600 |
| gemma-4-e2b-it | rel_depth_blink | 12 | 0.9167 / 否 | 0.9167 / 否 | 0.9167 / 否 | accuracy | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.0833 |
| gemma-4-e2b-it | spatial_cvbench | 50 | 0.8600 / 否 | 0.8600 / 否 | 0.8600 / 否 | accuracy | 0.7200 | 0.7400 | -0.0200 | [-0.1200, 0.0600] | 是 | 0.4200 |
| gemma-4-e2b-it | spatial_embspatial | 50 | 0.7600 / 否 | 0.7600 / 否 | 0.7600 / 否 | accuracy | 0.1800 | 0.2000 | -0.0200 | [-0.1200, 0.0800] | 是 | 0.1400 |
| internvl3.5-2b-hf | refexp_box | 100 | 0.0900 / 否 | 0.9100 / 否 | 0.9800 / 是 | acc@0.5 | 0.7100 | 0.7300 | -0.0200 | [-0.0500, 0.0000] | 是 | 0.0900 |
| internvl3.5-2b-hf | refexp_poly | 100 | 0.4300 / 否 | 0.4300 / 否 | 0.4300 / 否 | cIoU | 0.1031 | 0.0918 | 0.0113 | [-0.0111, 0.0344] | 是 | 0.1800 |
| internvl3.5-2b-hf | grid16 | 100 | 0.9900 / 是 | 0.9900 / 是 | 0.9900 / 是 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.8700 |
| internvl3.5-2b-hf | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.4694 | 0.4694 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6939 |
| internvl3.5-2b-hf | ovd | 100 | 1.0000 / 是 | 1.0000 / 是 | — / — | — | — | — | — | — | — | 0.3400 |
| internvl3.5-2b-hf | rel_depth_nyu | 100 | 0.8900 / 否 | 0.8900 / 否 | 0.8900 / 否 | accuracy | 0.5500 | 0.6000 | -0.0500 | [-0.1100, 0.0200] | 是 | 0.3100 |
| internvl3.5-2b-hf | rel_depth_blink | 12 | 0.6667 / 否 | 0.6667 / 否 | 0.6667 / 否 | accuracy | 0.5833 | 0.5833 | 0.0000 | [-0.3333, 0.3333] | 是 | 0.1667 |
| internvl3.5-2b-hf | spatial_cvbench | 50 | 0.9200 / 否 | 0.9200 / 否 | 0.9200 / 否 | accuracy | 0.7600 | 0.8000 | -0.0400 | [-0.1200, 0.0400] | 是 | 0.3800 |
| internvl3.5-2b-hf | spatial_embspatial | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.3400 | 0.3400 | 0.0000 | [-0.0800, 0.0800] | 是 | 0.4400 |
| qwen3.5-4b | refexp_box | 100 | 0.3300 / 否 | 0.9800 / 是 | 1.0000 / 是 | acc@0.5 | 0.8700 | 0.8700 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3000 |
| qwen3.5-4b | refexp_poly | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2700 |
| qwen3.5-4b | grid16 | 100 | 0.9900 / 是 | 0.9900 / 是 | 0.9900 / 是 | gIoU | 0.0111 | 0.0111 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9900 |
| qwen3.5-4b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-4b | ovd | 100 | 0.1300 / 否 | 0.5500 / 否 | — / — | — | — | — | — | — | — | 0.0100 |
| qwen3.5-4b | rel_depth_nyu | 100 | 0.8700 / 否 | 0.8700 / 否 | 0.8700 / 否 | accuracy | 0.7100 | 0.7700 | -0.0600 | [-0.1200, 0.0000] | 是 | 0.2900 |
| qwen3.5-4b | rel_depth_blink | 12 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.9167 | 0.9167 | 0.0000 | [0.0000, 0.0000] | 是 | 0.1667 |
| qwen3.5-4b | spatial_cvbench | 50 | 0.9600 / 否 | 0.9600 / 否 | 0.9600 / 否 | accuracy | 0.9000 | 0.9000 | 0.0000 | [-0.0600, 0.0600] | 是 | 0.5200 |
| qwen3.5-4b | spatial_embspatial | 50 | 0.8600 / 否 | 0.8600 / 否 | 0.8600 / 否 | accuracy | 0.3400 | 0.4200 | -0.0800 | [-0.1800, 0.0005] | 是 | 0.4000 |
| gemma-4-e4b-it | refexp_box | 100 | 0.9200 / 否 | 0.9900 / 是 | 1.0000 / 是 | acc@0.5 | 0.1300 | 0.1300 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6900 |
| gemma-4-e4b-it | refexp_poly | 100 | 0.9800 / 是 | 0.9800 / 是 | 0.9800 / 是 | cIoU | 0.0381 | 0.0336 | 0.0045 | [-0.0000, 0.0122] | 是 | 0.1600 |
| gemma-4-e4b-it | grid16 | 100 | 0.9800 / 是 | 0.9800 / 是 | 0.9800 / 是 | gIoU | 0.0112 | 0.0122 | -0.0010 | [-0.0038, 0.0010] | 是 | 0.9700 |
| gemma-4-e4b-it | count | 49 | 0.9592 / 否 | 0.9592 / 否 | 0.9592 / 否 | accuracy | 0.5102 | 0.4898 | 0.0204 | [0.0000, 0.0612] | 是 | 0.9592 |
| gemma-4-e4b-it | ovd | 100 | 0.1700 / 否 | 0.7500 / 否 | — / — | — | — | — | — | — | — | 0.1700 |
| gemma-4-e4b-it | rel_depth_nyu | 100 | 0.9400 / 否 | 0.9400 / 否 | 0.9400 / 否 | accuracy | 0.7100 | 0.7000 | 0.0100 | [-0.0300, 0.0500] | 是 | 0.2200 |
| gemma-4-e4b-it | rel_depth_blink | 12 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.9167 | 0.9167 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2500 |
| gemma-4-e4b-it | spatial_cvbench | 50 | 0.9400 / 否 | 0.9400 / 否 | 0.9400 / 否 | accuracy | 0.7000 | 0.6600 | 0.0400 | [0.0000, 0.1000] | 是 | 0.5600 |
| gemma-4-e4b-it | spatial_embspatial | 50 | 0.8600 / 否 | 0.8600 / 否 | 0.8600 / 否 | accuracy | 0.2200 | 0.2400 | -0.0200 | [-0.0800, 0.0400] | 是 | 0.2800 |
| internvl3.5-4b-hf | refexp_box | 100 | 0.4800 / 否 | 0.9500 / 否 | 0.9900 / 是 | acc@0.5 | 0.8700 | 0.8600 | 0.0100 | [0.0000, 0.0400] | 是 | 0.4800 |
| internvl3.5-4b-hf | refexp_poly | 100 | 0.3700 / 否 | 0.3700 / 否 | 0.3700 / 否 | cIoU | 0.1630 | 0.1639 | -0.0009 | [-0.0311, 0.0276] | 是 | 0.2500 |
| internvl3.5-4b-hf | grid16 | 100 | 0.9700 / 否 | 0.9700 / 否 | 0.9700 / 否 | gIoU | 0.0044 | 0.0040 | 0.0004 | [0.0000, 0.0012] | 是 | 0.9700 |
| internvl3.5-4b-hf | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.6735 | 0.6735 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 |
| internvl3.5-4b-hf | ovd | 100 | 0.1400 / 否 | 0.3100 / 否 | — / — | — | — | — | — | — | — | 0.0200 |
| internvl3.5-4b-hf | rel_depth_nyu | 100 | 0.8700 / 否 | 0.8700 / 否 | 0.8700 / 否 | accuracy | 0.7100 | 0.7200 | -0.0100 | [-0.0800, 0.0600] | 是 | 0.3400 |
| internvl3.5-4b-hf | rel_depth_blink | 12 | 0.8333 / 否 | 0.8333 / 否 | 0.8333 / 否 | accuracy | 0.6667 | 0.5000 | 0.1667 | [0.0000, 0.4167] | 是 | 0.1667 |
| internvl3.5-4b-hf | spatial_cvbench | 50 | 0.9200 / 否 | 0.9200 / 否 | 0.9200 / 否 | accuracy | 0.8000 | 0.7800 | 0.0200 | [-0.0400, 0.0800] | 是 | 0.5000 |
| internvl3.5-4b-hf | spatial_embspatial | 50 | 0.9600 / 否 | 0.9600 / 否 | 0.9600 / 否 | accuracy | 0.5800 | 0.6200 | -0.0400 | [-0.1000, 0.0000] | 是 | 0.8200 |
| qwen3.5-9b | refexp_box | 100 | 0.3500 / 否 | 0.9900 / 是 | 1.0000 / 是 | acc@0.5 | 0.8900 | 0.8900 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3300 |
| qwen3.5-9b | refexp_poly | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3800 |
| qwen3.5-9b | grid16 | 100 | 0.9500 / 否 | 0.9500 / 否 | 0.9500 / 否 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9400 |
| qwen3.5-9b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7959 | 0.7959 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-9b | ovd | 100 | 0.1000 / 否 | 0.4900 / 否 | — / — | — | — | — | — | — | — | 0.0000 |
| qwen3.5-9b | rel_depth_nyu | 100 | 0.9700 / 否 | 0.9700 / 否 | 0.9700 / 否 | accuracy | 0.7100 | 0.7100 | 0.0000 | [-0.0300, 0.0300] | 是 | 0.5300 |
| qwen3.5-9b | rel_depth_blink | 12 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.9167 | 0.9167 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2500 |
| qwen3.5-9b | spatial_cvbench | 50 | 0.9800 / 是 | 0.9800 / 是 | 0.9800 / 是 | accuracy | 0.8600 | 0.8600 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5800 |
| qwen3.5-9b | spatial_embspatial | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.6000 | 0.5800 | 0.0200 | [-0.0400, 0.1000] | 是 | 0.5400 |

#### 每种口径的总判定

| 模型 | (a) 逐位 all_pass（不判的任务） | (b) IoU≥0.9 all_pass（不判的任务） | (c) 判定结果 all_pass（不判的任务） | (a) 逐位 不通过的任务 | (b) IoU≥0.9 不通过的任务 | (c) 判定结果 不通过的任务 |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 否（—） | 否（—） | 否（ovd） | refexp_box, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_box, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| qwen3.5-2b | 否（—） | 否（—） | 否（ovd） | refexp_box, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| gemma-4-e2b-it | 否（—） | 否（—） | 否（ovd） | refexp_box, refexp_poly, count, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_poly, count, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_poly, count, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| internvl3.5-2b-hf | 否（—） | 否（—） | 否（ovd） | refexp_box, refexp_poly, count, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_box, refexp_poly, count, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_poly, count, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| qwen3.5-4b | 否（—） | 否（—） | 否（ovd） | refexp_box, ovd, rel_depth_nyu, spatial_cvbench, spatial_embspatial | ovd, rel_depth_nyu, spatial_cvbench, spatial_embspatial | rel_depth_nyu, spatial_cvbench, spatial_embspatial |
| gemma-4-e4b-it | 否（—） | 否（—） | 否（ovd） | refexp_box, count, ovd, rel_depth_nyu, spatial_cvbench, spatial_embspatial | count, ovd, rel_depth_nyu, spatial_cvbench, spatial_embspatial | count, rel_depth_nyu, spatial_cvbench, spatial_embspatial |
| internvl3.5-4b-hf | 否（—） | 否（—） | 否（ovd） | refexp_box, refexp_poly, grid16, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_box, refexp_poly, grid16, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_poly, grid16, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| qwen3.5-9b | 否（—） | 否（—） | 否（ovd） | refexp_box, grid16, ovd, rel_depth_nyu, spatial_embspatial | grid16, ovd, rel_depth_nyu, spatial_embspatial | grid16, rel_depth_nyu, spatial_embspatial |

#### 计数：CountBenchQA 人工问句（--task count --count-questions countbenchqa）

| 模型 | 任务 | n | (a) 逐位 一致率 / 通过 | (b) IoU≥0.9 一致率 / 通过 | (c) 判定结果 一致率 / 通过 | 指标 | 逐条 | 批量 | 差（逐条−批量） | 配对差 95% CI | CI 含 0 | raw_text 逐字相同 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7959 | 0.7959 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-2b | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.8367 | 0.8163 | 0.0204 | [0.0000, 0.0612] | 是 | 0.9796 |
| gemma-4-e2b-it | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.5306 | 0.5306 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| internvl3.5-2b-hf | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.5714 | 0.5714 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9592 |
| qwen3.5-4b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.8571 | 0.8571 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| gemma-4-e4b-it | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.5714 | 0.5714 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| internvl3.5-4b-hf | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 |
| qwen3.5-9b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.9184 | 0.9184 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |

| 模型 | (a) 逐位 all_pass | (b) IoU≥0.9 all_pass | (c) 判定结果 all_pass |
|---|---|---|---|
| qwen3.5-0.8b | 是 | 是 | 是 |
| qwen3.5-2b | 否 | 否 | 否 |
| gemma-4-e2b-it | 是 | 是 | 是 |
| internvl3.5-2b-hf | 否 | 否 | 否 |
| qwen3.5-4b | 是 | 是 | 是 |
| gemma-4-e4b-it | 是 | 是 | 是 |
| internvl3.5-4b-hf | 否 | 否 | 否 |
| qwen3.5-9b | 是 | 是 | 是 |

#### 截断率与解析失败（逐条 bs=1 / 批量；截断 = stopped=False；启发式问句）

| 模型 | 任务 | max_new_tokens | 截断率 bs1 / 批量 | 解析失败 bs1 | 解析失败 批量 | 平均生成 token bs1 / 批量 |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 256 | 0.01 / 0.01 | 2/100 {"multiple_boxes": 1, "truncated": 1} | 2/100 {"multiple_boxes": 1, "truncated": 1} | 49.2 / 49.2 |
| qwen3.5-0.8b | refexp_poly | 1024 | 0.01 / 0.01 | 100/100 {"malformed": 99, "truncated": 1} | 100/100 {"malformed": 99, "truncated": 1} | 66.3 / 67.5 |
| qwen3.5-0.8b | grid16 | 1024 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 1024.0 / 1024.0 |
| qwen3.5-0.8b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.3 / 2.6 |
| qwen3.5-0.8b | ovd | 2048 | 0.61 / 0.65 | 61/100 {"truncated": 61} | 65/100 {"truncated": 65} | 1361.8 / 1413.2 |
| qwen3.5-0.8b | rel_depth_nyu | 1024 | 0.02 / 0.02 | 2/100 {"truncated": 2} | 4/100 {"multiple_choices": 2, "truncated": 2} | 262.3 / 251.1 |
| qwen3.5-0.8b | rel_depth_blink | 1024 | 0.08 / 0.00 | 2/12 {"multiple_choices": 1, "truncated": 1} | 2/12 {"multiple_choices": 2} | 346.1 / 329.0 |
| qwen3.5-0.8b | spatial_cvbench | 1024 | 0.00 / 0.04 | 0/50 {} | 2/50 {"truncated": 2} | 136.2 / 163.2 |
| qwen3.5-0.8b | spatial_embspatial | 1024 | 0.00 / 0.00 | 17/50 {"multiple_choices": 17} | 18/50 {"multiple_choices": 18} | 133.8 / 131.2 |
| qwen3.5-2b | refexp_box | 256 | 0.00 / 0.00 | 1/100 {"multiple_boxes": 1} | 1/100 {"multiple_boxes": 1} | 47.4 / 47.4 |
| qwen3.5-2b | refexp_poly | 1024 | 0.00 / 0.00 | 100/100 {"malformed": 100} | 100/100 {"malformed": 100} | 43.4 / 42.3 |
| qwen3.5-2b | grid16 | 1024 | 0.99 / 0.99 | 100/100 {"truncated": 99, "wrong_row_count": 1} | 100/100 {"truncated": 99, "wrong_row_count": 1} | 1016.3 / 1016.3 |
| qwen3.5-2b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| qwen3.5-2b | ovd | 2048 | 0.38 / 0.37 | 38/100 {"truncated": 38} | 37/100 {"truncated": 37} | 965.2 / 957.6 |
| qwen3.5-2b | rel_depth_nyu | 1024 | 0.00 / 0.02 | 0/100 {} | 2/100 {"truncated": 2} | 183.9 / 200.1 |
| qwen3.5-2b | rel_depth_blink | 1024 | 0.00 / 0.00 | 2/12 {"multiple_choices": 2} | 1/12 {"multiple_choices": 1} | 255.3 / 253.7 |
| qwen3.5-2b | spatial_cvbench | 1024 | 0.02 / 0.04 | 4/50 {"multiple_choices": 3, "truncated": 1} | 2/50 {"truncated": 2} | 182.0 / 182.6 |
| qwen3.5-2b | spatial_embspatial | 1024 | 0.00 / 0.00 | 20/50 {"multiple_choices": 20} | 19/50 {"multiple_choices": 19} | 123.0 / 124.7 |
| gemma-4-e2b-it | refexp_box | 256 | 0.03 / 0.04 | 88/100 {"multiple_boxes": 3, "no_box": 82, "truncated": 3} | 87/100 {"multiple_boxes": 3, "no_box": 80, "truncated": 4} | 55.6 / 55.1 |
| gemma-4-e2b-it | refexp_poly | 1024 | 0.00 / 0.00 | 2/100 {"malformed": 1, "too_few_points": 1} | 2/100 {"malformed": 1, "too_few_points": 1} | 84.5 / 88.2 |
| gemma-4-e2b-it | grid16 | 1024 | 0.00 / 0.00 | 100/100 {"empty": 3, "wrong_row_count": 97} | 100/100 {"empty": 3, "wrong_row_count": 97} | 245.3 / 244.7 |
| gemma-4-e2b-it | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| gemma-4-e2b-it | ovd | 2048 | 0.00 / 0.00 | 35/100 {"no_box": 35} | 35/100 {"no_box": 35} | 158.2 / 159.6 |
| gemma-4-e2b-it | rel_depth_nyu | 1024 | 0.00 / 0.00 | 1/100 {"multiple_choices": 1} | 6/100 {"multiple_choices": 6} | 281.4 / 280.0 |
| gemma-4-e2b-it | rel_depth_blink | 1024 | 0.00 / 0.00 | 2/12 {"multiple_choices": 2} | 1/12 {"multiple_choices": 1} | 288.8 / 270.5 |
| gemma-4-e2b-it | spatial_cvbench | 1024 | 0.00 / 0.00 | 8/50 {"multiple_choices": 7, "no_choice": 1} | 4/50 {"multiple_choices": 3, "no_choice": 1} | 190.8 / 196.5 |
| gemma-4-e2b-it | spatial_embspatial | 1024 | 0.00 / 0.02 | 36/50 {"multiple_choices": 35, "no_choice": 1} | 35/50 {"multiple_choices": 33, "no_choice": 1, "truncated": 1} | 337.6 / 346.4 |
| internvl3.5-2b-hf | refexp_box | 256 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 28.4 / 28.4 |
| internvl3.5-2b-hf | refexp_poly | 1024 | 0.02 / 0.02 | 22/100 {"ambiguous": 3, "malformed": 2, "too_few_points": 15, "truncated": 2} | 21/100 {"ambiguous": 1, "malformed": 4, "too_few_points": 14, "truncated": 2} | 254.8 / 265.2 |
| internvl3.5-2b-hf | grid16 | 1024 | 0.00 / 0.00 | 100/100 {"wrong_row_count": 100} | 99/100 {"wrong_row_count": 99} | 252.2 / 253.0 |
| internvl3.5-2b-hf | count | 256 | 0.08 / 0.14 | 13/49 {"multiple_numbers": 9, "thinking_unclosed": 2, "truncated": 2} | 14/49 {"multiple_numbers": 7, "thinking_unclosed": 3, "truncated": 4} | 61.6 / 68.6 |
| internvl3.5-2b-hf | ovd | 2048 | 0.02 / 0.03 | 100/100 {"no_box": 98, "truncated": 2} | 100/100 {"no_box": 97, "truncated": 3} | 198.2 / 214.0 |
| internvl3.5-2b-hf | rel_depth_nyu | 1024 | 0.00 / 0.00 | 2/100 {"multiple_choices": 2} | 0/100 {} | 164.8 / 167.6 |
| internvl3.5-2b-hf | rel_depth_blink | 1024 | 0.00 / 0.00 | 1/12 {"multiple_choices": 1} | 1/12 {"multiple_choices": 1} | 198.1 / 204.1 |
| internvl3.5-2b-hf | spatial_cvbench | 1024 | 0.00 / 0.00 | 2/50 {"multiple_choices": 2} | 1/50 {"multiple_choices": 1} | 144.8 / 144.3 |
| internvl3.5-2b-hf | spatial_embspatial | 1024 | 0.00 / 0.00 | 19/50 {"multiple_choices": 19} | 19/50 {"multiple_choices": 19} | 132.0 / 122.9 |
| qwen3.5-4b | refexp_box | 256 | 0.00 / 0.00 | 3/100 {"multiple_boxes": 3} | 3/100 {"multiple_boxes": 3} | 49.1 / 49.1 |
| qwen3.5-4b | refexp_poly | 1024 | 0.00 / 0.00 | 100/100 {"malformed": 100} | 100/100 {"malformed": 100} | 40.8 / 41.1 |
| qwen3.5-4b | grid16 | 1024 | 0.00 / 0.00 | 1/100 {"wrong_row_count": 1} | 0/100 {} | 272.3 / 272.0 |
| qwen3.5-4b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| qwen3.5-4b | ovd | 2048 | 0.14 / 0.13 | 14/100 {"truncated": 14} | 13/100 {"truncated": 13} | 623.4 / 613.8 |
| qwen3.5-4b | rel_depth_nyu | 1024 | 0.02 / 0.01 | 5/100 {"multiple_choices": 3, "truncated": 2} | 3/100 {"multiple_choices": 2, "truncated": 1} | 225.3 / 210.2 |
| qwen3.5-4b | rel_depth_blink | 1024 | 0.00 / 0.00 | 0/12 {} | 0/12 {} | 191.5 / 184.3 |
| qwen3.5-4b | spatial_cvbench | 1024 | 0.00 / 0.00 | 0/50 {} | 1/50 {"multiple_choices": 1} | 145.1 / 163.6 |
| qwen3.5-4b | spatial_embspatial | 1024 | 0.02 / 0.04 | 30/50 {"multiple_choices": 28, "no_choice": 1, "truncated": 1} | 27/50 {"multiple_choices": 25, "truncated": 2} | 194.1 / 194.2 |
| gemma-4-e4b-it | refexp_box | 256 | 0.02 / 0.03 | 82/100 {"multiple_boxes": 3, "no_box": 77, "truncated": 2} | 83/100 {"multiple_boxes": 3, "no_box": 77, "truncated": 3} | 49.5 / 52.6 |
| gemma-4-e4b-it | refexp_poly | 1024 | 0.00 / 0.00 | 93/100 {"malformed": 93} | 93/100 {"malformed": 93} | 71.5 / 71.5 |
| gemma-4-e4b-it | grid16 | 1024 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 272.0 / 272.0 |
| gemma-4-e4b-it | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| gemma-4-e4b-it | ovd | 2048 | 0.00 / 0.00 | 1/100 {"no_box": 1} | 1/100 {"no_box": 1} | 197.5 / 199.5 |
| gemma-4-e4b-it | rel_depth_nyu | 1024 | 0.00 / 0.00 | 3/100 {"multiple_choices": 3} | 0/100 {} | 265.2 / 258.5 |
| gemma-4-e4b-it | rel_depth_blink | 1024 | 0.00 / 0.00 | 0/12 {} | 0/12 {} | 227.8 / 271.0 |
| gemma-4-e4b-it | spatial_cvbench | 1024 | 0.00 / 0.00 | 8/50 {"multiple_choices": 7, "no_choice": 1} | 10/50 {"multiple_choices": 9, "no_choice": 1} | 132.1 / 134.9 |
| gemma-4-e4b-it | spatial_embspatial | 1024 | 0.02 / 0.02 | 33/50 {"multiple_choices": 27, "no_choice": 5, "truncated": 1} | 32/50 {"multiple_choices": 24, "no_choice": 7, "truncated": 1} | 263.1 / 254.1 |
| internvl3.5-4b-hf | refexp_box | 256 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 28.4 / 28.4 |
| internvl3.5-4b-hf | refexp_poly | 1024 | 0.04 / 0.06 | 5/100 {"out_of_range": 1, "truncated": 4} | 6/100 {"truncated": 6} | 117.7 / 135.1 |
| internvl3.5-4b-hf | grid16 | 1024 | 0.00 / 0.00 | 2/100 {"multiple_blocks": 2} | 1/100 {"multiple_blocks": 1} | 272.0 / 272.0 |
| internvl3.5-4b-hf | count | 256 | 0.00 / 0.00 | 1/49 {"multiple_numbers": 1} | 1/49 {"multiple_numbers": 1} | 7.6 / 7.6 |
| internvl3.5-4b-hf | ovd | 2048 | 0.19 / 0.15 | 19/100 {"truncated": 19} | 15/100 {"truncated": 15} | 761.5 / 736.6 |
| internvl3.5-4b-hf | rel_depth_nyu | 1024 | 0.00 / 0.00 | 4/100 {"multiple_choices": 4} | 2/100 {"multiple_choices": 2} | 150.3 / 151.9 |
| internvl3.5-4b-hf | rel_depth_blink | 1024 | 0.00 / 0.00 | 0/12 {} | 0/12 {} | 232.2 / 249.9 |
| internvl3.5-4b-hf | spatial_cvbench | 1024 | 0.00 / 0.00 | 2/50 {"multiple_choices": 2} | 1/50 {"multiple_choices": 1} | 126.3 / 123.5 |
| internvl3.5-4b-hf | spatial_embspatial | 1024 | 0.00 / 0.00 | 7/50 {"multiple_choices": 7} | 6/50 {"multiple_choices": 6} | 43.7 / 40.8 |
| qwen3.5-9b | refexp_box | 256 | 0.00 / 0.00 | 2/100 {"multiple_boxes": 2} | 2/100 {"multiple_boxes": 2} | 47.7 / 47.7 |
| qwen3.5-9b | refexp_poly | 1024 | 0.01 / 0.01 | 100/100 {"malformed": 99, "truncated": 1} | 100/100 {"malformed": 99, "truncated": 1} | 50.0 / 49.7 |
| qwen3.5-9b | grid16 | 1024 | 0.00 / 0.00 | 14/100 {"wrong_row_count": 14} | 13/100 {"wrong_row_count": 13} | 274.4 / 274.2 |
| qwen3.5-9b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |
| qwen3.5-9b | ovd | 2048 | 0.16 / 0.13 | 16/100 {"truncated": 16} | 13/100 {"truncated": 13} | 688.1 / 642.7 |
| qwen3.5-9b | rel_depth_nyu | 1024 | 0.00 / 0.01 | 1/100 {"multiple_choices": 1} | 2/100 {"multiple_choices": 1, "truncated": 1} | 176.1 / 185.6 |
| qwen3.5-9b | rel_depth_blink | 1024 | 0.00 / 0.00 | 0/12 {} | 0/12 {} | 205.1 / 191.0 |
| qwen3.5-9b | spatial_cvbench | 1024 | 0.00 / 0.00 | 1/50 {"multiple_choices": 1} | 2/50 {"multiple_choices": 2} | 148.1 / 156.2 |
| qwen3.5-9b | spatial_embspatial | 1024 | 0.02 / 0.02 | 16/50 {"multiple_choices": 15, "truncated": 1} | 16/50 {"multiple_choices": 15, "truncated": 1} | 155.1 / 152.8 |

#### 诊断（failure_detail：wrong_shape、refusal；关思考时自发的思考标记；思考未闭合；启发式问句）

| 模型 | 任务 | failure_detail bs1 | failure_detail 批量 | 自发思考标记 bs1 / 批量 | 思考未闭合 bs1 / 批量 |
|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | refexp_poly | {"wrong_shape:bbox_2d": 98, "wrong_shape:point_2d": 1} | {"wrong_shape:bbox_2d": 98, "wrong_shape:point_2d": 1} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | ovd | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_nyu | {"refusal": 1} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_blink | {"refusal": 1} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | refexp_poly | {"wrong_shape:bbox_2d": 15, "wrong_shape:point_2d": 85} | {"wrong_shape:bbox_2d": 14, "wrong_shape:point_2d": 86} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | ovd | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | rel_depth_nyu | {} | {"refusal": 1} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | refexp_box | {"refusal": 60} | {"refusal": 61} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | refexp_poly | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | count | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | ovd | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | spatial_cvbench | {"refusal": 1} | {"refusal": 1} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | spatial_embspatial | {"refusal": 1} | {"refusal": 2} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | refexp_poly | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | count | {} | {} | 3 / 4 | 2 / 3 |
| internvl3.5-2b-hf | ovd | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | refexp_poly | {"wrong_shape:point_2d": 100} | {"wrong_shape:point_2d": 100} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | ovd | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | refexp_box | {"refusal": 63} | {"refusal": 63} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | refexp_poly | {"wrong_shape:box_2d": 93} | {"wrong_shape:box_2d": 93} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | count | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | ovd | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | spatial_cvbench | {"refusal": 1} | {"refusal": 1} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | refexp_poly | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | count | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | ovd | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | refexp_poly | {"wrong_shape:bbox_2d": 1, "wrong_shape:point_2d": 98} | {"wrong_shape:bbox_2d": 1, "wrong_shape:point_2d": 98} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | ovd | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | rel_depth_nyu | {} | {"refusal": 1} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |

#### 截断率与解析失败（逐条 bs=1 / 批量；截断 = stopped=False；CountBenchQA 问句（仅计数））

| 模型 | 任务 | max_new_tokens | 截断率 bs1 / 批量 | 解析失败 bs1 | 解析失败 批量 | 平均生成 token bs1 / 批量 |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |
| qwen3.5-2b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| gemma-4-e2b-it | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| internvl3.5-2b-hf | count | 256 | 0.00 / 0.00 | 5/49 {"multiple_numbers": 5} | 4/49 {"multiple_numbers": 4} | 25.6 / 23.4 |
| qwen3.5-4b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |
| gemma-4-e4b-it | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |
| internvl3.5-4b-hf | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 9.1 / 9.1 |
| qwen3.5-9b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |

#### 诊断（failure_detail：wrong_shape、refusal；关思考时自发的思考标记；思考未闭合；CountBenchQA 问句（仅计数））

| 模型 | 任务 | failure_detail bs1 | failure_detail 批量 | 自发思考标记 bs1 / 批量 | 思考未闭合 bs1 / 批量 |
|---|---|---|---|---|---|
| qwen3.5-0.8b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | count | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | count | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | count | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | count | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | count | {} | {} | 0 / 0 | 0 / 0 |

#### 每遍的墙钟与生成 token（compute.json；墙钟只含 generate；启发式问句）

| 模型 | 任务 | 墙钟 s bs1 / 批量 | 生成 token bs1 / 批量 |
|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 59.9 / 9.8 | 4921 / 4920 |
| qwen3.5-0.8b | refexp_poly | 79.9 / 23.7 | 6631 / 6747 |
| qwen3.5-0.8b | grid16 | 1217.2 / 94.1 | 102400 / 102400 |
| qwen3.5-0.8b | count | 2.5 / 0.6 | 111 / 128 |
| qwen3.5-0.8b | ovd | 1626.6 / 186.7 | 136180 / 141319 |
| qwen3.5-0.8b | rel_depth_nyu | 311.8 / 44.2 | 26230 / 25109 |
| qwen3.5-0.8b | rel_depth_blink | 49.3 / 12.8 | 4153 / 3948 |
| qwen3.5-0.8b | spatial_cvbench | 81.2 / 22.1 | 6808 / 8160 |
| qwen3.5-0.8b | spatial_embspatial | 79.7 / 13.7 | 6691 / 6562 |
| qwen3.5-0.8b | 合计 | 3508.1 / 407.7 | |
| qwen3.5-2b | refexp_box | 62.4 / 7.4 | 4741 / 4741 |
| qwen3.5-2b | refexp_poly | 55.2 / 9.9 | 4342 / 4234 |
| qwen3.5-2b | grid16 | 1240.6 / 95.8 | 101631 / 101631 |
| qwen3.5-2b | count | 2.5 / 0.4 | 104 / 104 |
| qwen3.5-2b | ovd | 1181.7 / 177.5 | 96522 / 95764 |
| qwen3.5-2b | rel_depth_nyu | 229.6 / 45.0 | 18394 / 20013 |
| qwen3.5-2b | rel_depth_blink | 37.4 / 8.7 | 3064 / 3044 |
| qwen3.5-2b | spatial_cvbench | 110.9 / 23.1 | 9100 / 9128 |
| qwen3.5-2b | spatial_embspatial | 75.3 / 13.3 | 6152 / 6234 |
| qwen3.5-2b | 合计 | 2995.6 / 381.2 | |
| gemma-4-e2b-it | refexp_box | 101.7 / 27.4 | 5557 / 5506 |
| gemma-4-e2b-it | refexp_poly | 154.2 / 56.8 | 8448 / 8815 |
| gemma-4-e2b-it | grid16 | 447.9 / 35.4 | 24530 / 24470 |
| gemma-4-e2b-it | count | 2.9 / 0.7 | 103 / 103 |
| gemma-4-e2b-it | ovd | 295.3 / 49.2 | 15823 / 15957 |
| gemma-4-e2b-it | rel_depth_nyu | 515.1 / 59.4 | 28141 / 28000 |
| gemma-4-e2b-it | rel_depth_blink | 63.4 / 9.6 | 3465 / 3246 |
| gemma-4-e2b-it | spatial_cvbench | 180.3 / 47.1 | 9538 / 9823 |
| gemma-4-e2b-it | spatial_embspatial | 307.7 / 62.5 | 16882 / 17318 |
| gemma-4-e2b-it | 合计 | 2068.5 / 348.1 | |
| internvl3.5-2b-hf | refexp_box | 32.5 / 5.3 | 2836 / 2838 |
| internvl3.5-2b-hf | refexp_poly | 285.8 / 88.8 | 25482 / 26523 |
| internvl3.5-2b-hf | grid16 | 257.0 / 22.0 | 25218 / 25299 |
| internvl3.5-2b-hf | count | 33.0 / 9.7 | 3017 / 3361 |
| internvl3.5-2b-hf | ovd | 229.7 / 121.5 | 19824 / 21399 |
| internvl3.5-2b-hf | rel_depth_nyu | 171.5 / 26.8 | 16482 / 16760 |
| internvl3.5-2b-hf | rel_depth_blink | 24.8 / 11.9 | 2377 / 2449 |
| internvl3.5-2b-hf | spatial_cvbench | 77.6 / 19.8 | 7241 / 7217 |
| internvl3.5-2b-hf | spatial_embspatial | 68.9 / 11.0 | 6599 / 6146 |
| internvl3.5-2b-hf | 合计 | 1180.8 / 316.7 | |
| qwen3.5-4b | refexp_box | 82.2 / 13.1 | 4908 / 4907 |
| qwen3.5-4b | refexp_poly | 68.6 / 10.4 | 4080 / 4113 |
| qwen3.5-4b | grid16 | 445.8 / 34.9 | 27234 / 27200 |
| qwen3.5-4b | count | 3.2 / 0.6 | 105 / 105 |
| qwen3.5-4b | ovd | 1036.3 / 232.2 | 62337 / 61375 |
| qwen3.5-4b | rel_depth_nyu | 376.2 / 59.9 | 22534 / 21015 |
| qwen3.5-4b | rel_depth_blink | 37.7 / 5.1 | 2298 / 2212 |
| qwen3.5-4b | spatial_cvbench | 118.8 / 30.7 | 7254 / 8179 |
| qwen3.5-4b | spatial_embspatial | 158.7 / 52.1 | 9705 / 9709 |
| qwen3.5-4b | 合计 | 2327.5 / 439.0 | |
| gemma-4-e4b-it | refexp_box | 113.5 / 30.0 | 4950 / 5261 |
| gemma-4-e4b-it | refexp_poly | 163.4 / 24.3 | 7151 / 7150 |
| gemma-4-e4b-it | grid16 | 627.4 / 48.6 | 27200 / 27200 |
| gemma-4-e4b-it | count | 3.4 / 0.8 | 104 / 104 |
| gemma-4-e4b-it | ovd | 454.9 / 96.1 | 19748 / 19950 |
| gemma-4-e4b-it | rel_depth_nyu | 604.0 / 84.7 | 26521 / 25848 |
| gemma-4-e4b-it | rel_depth_blink | 61.8 / 15.5 | 2734 / 3252 |
| gemma-4-e4b-it | spatial_cvbench | 153.1 / 28.4 | 6607 / 6746 |
| gemma-4-e4b-it | spatial_embspatial | 307.8 / 75.6 | 13156 / 12705 |
| gemma-4-e4b-it | 合计 | 2489.3 / 404.0 | |
| internvl3.5-4b-hf | refexp_box | 38.6 / 5.9 | 2839 / 2842 |
| internvl3.5-4b-hf | refexp_poly | 173.6 / 117.0 | 11766 / 13509 |
| internvl3.5-4b-hf | grid16 | 344.0 / 37.6 | 27200 / 27200 |
| internvl3.5-4b-hf | count | 5.2 / 1.2 | 372 / 373 |
| internvl3.5-4b-hf | ovd | 1006.2 / 285.4 | 76150 / 73661 |
| internvl3.5-4b-hf | rel_depth_nyu | 193.5 / 32.8 | 15032 / 15188 |
| internvl3.5-4b-hf | rel_depth_blink | 37.8 / 11.4 | 2787 / 2999 |
| internvl3.5-4b-hf | spatial_cvbench | 84.9 / 23.3 | 6313 / 6176 |
| internvl3.5-4b-hf | spatial_embspatial | 29.9 / 6.2 | 2186 / 2038 |
| internvl3.5-4b-hf | 合计 | 1913.6 / 520.7 | |
| qwen3.5-9b | refexp_box | 80.8 / 11.3 | 4770 / 4771 |
| qwen3.5-9b | refexp_poly | 84.4 / 28.5 | 4999 / 4972 |
| qwen3.5-9b | grid16 | 451.9 / 36.9 | 27442 / 27425 |
| qwen3.5-9b | count | 3.1 / 0.8 | 106 / 106 |
| qwen3.5-9b | ovd | 1144.2 / 236.3 | 68815 / 64274 |
| qwen3.5-9b | rel_depth_nyu | 294.3 / 56.4 | 17607 / 18560 |
| qwen3.5-9b | rel_depth_blink | 42.1 / 6.8 | 2461 / 2292 |
| qwen3.5-9b | spatial_cvbench | 125.5 / 33.5 | 7406 / 7810 |
| qwen3.5-9b | spatial_embspatial | 132.0 / 33.8 | 7754 / 7641 |
| qwen3.5-9b | 合计 | 2358.3 / 444.2 | |

### 3.2 吞吐与算力估算（`experiments/E1/0013_estimate/estimate_summary.md`）

本节由 `p16_estimate_summary.py` 从各 `estimate.json` 生成。逐批大小的明细（样本/秒、token/秒、平均生成 token、截断率、峰值显存）在该文件里，共 120 行，这里不重贴。

#### 估算汇总（主口径；推荐批大小 = 可行的批大小里样本/秒最高者；GPU 小时 = n / 样本每秒，由 run_e1 外推）

| 模型 | 任务 | 推荐批大小 | 样本/秒 | n(eval) | GPU h(eval) | 思考实测批大小 | 思考样本/秒 | n(think) | GPU h(think) | n(8192) | GPU h(8192, 上界) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 32 | 17.08 | 56857 | 0.925 | 8 | 4.52 | 200 | 0.012 | 50 | 0.006 |
| qwen3.5-0.8b | refexp_poly | 32 | 12.50 | 8000 | 0.178 | 8 | 0.15 | 200 | 0.368 | 50 | 0.184 |
| qwen3.5-0.8b | grid16 | 32 | 2.25 | 500 | 0.062 | 8 | 0.15 | 200 | 0.374 | 50 | 0.187 |
| qwen3.5-0.8b | count | 32 | 128.81 | 442 | 0.001 | 8 | 0.14 | 200 | 0.393 | 50 | 0.196 |
| qwen3.5-0.8b | ovd | 32 | 1.17 | 963 | 0.229 | 8 | 0.15 | 200 | 0.368 | 50 | 0.184 |
| qwen3.5-0.8b | rel_depth_nyu | 32 | 3.28 | 3270 | 0.277 | 8 | 0.16 | 200 | 0.351 | 50 | 0.176 |
| qwen3.5-0.8b | rel_depth_blink | 8 | 1.29 | 112 | 0.024 | 8 | 0.15 | 112 | 0.205 | 50 | 0.183 |
| qwen3.5-0.8b | spatial_cvbench | 32 | 2.67 | 2588 | 0.270 | 8 | 0.14 | 200 | 0.392 | 50 | 0.196 |
| qwen3.5-0.8b | spatial_embspatial | 32 | 6.57 | 3590 | 0.152 | 8 | 0.15 | 200 | 0.375 | 50 | 0.187 |
| qwen3.5-0.8b | **合计** | | | | **2.116** | | | | **2.837** | | **1.499** |
| qwen3.5-2b | refexp_box | 32 | 17.57 | 56857 | 0.899 | 8 | 3.79 | 200 | 0.015 | 50 | 0.007 |
| qwen3.5-2b | refexp_poly | 32 | 15.33 | 8000 | 0.145 | 8 | 0.15 | 200 | 0.378 | 50 | 0.189 |
| qwen3.5-2b | grid16 | 32 | 2.27 | 500 | 0.061 | 8 | 0.14 | 200 | 0.397 | 50 | 0.198 |
| qwen3.5-2b | count | 32 | 143.00 | 442 | 0.001 | 8 | 0.22 | 200 | 0.248 | 50 | 0.124 |
| qwen3.5-2b | ovd | 32 | 1.15 | 963 | 0.233 | 8 | 0.21 | 200 | 0.262 | 50 | 0.131 |
| qwen3.5-2b | rel_depth_nyu | 32 | 4.68 | 3270 | 0.194 | 8 | 0.15 | 200 | 0.383 | 50 | 0.191 |
| qwen3.5-2b | rel_depth_blink | 32 | 1.31 | 112 | 0.024 | 8 | 0.14 | 112 | 0.224 | 50 | 0.200 |
| qwen3.5-2b | spatial_cvbench | 32 | 2.42 | 2588 | 0.298 | 8 | 0.14 | 200 | 0.400 | 50 | 0.200 |
| qwen3.5-2b | spatial_embspatial | 32 | 5.49 | 3590 | 0.182 | 8 | 0.14 | 200 | 0.398 | 50 | 0.199 |
| qwen3.5-2b | **合计** | | | | **2.037** | | | | **2.704** | | **1.440** |
| gemma-4-e2b-it | refexp_box | 32 | 6.03 | 56857 | 2.617 | 8 | 0.56 | 200 | 0.100 | 50 | 0.050 |
| gemma-4-e2b-it | refexp_poly | 32 | 4.45 | 8000 | 0.500 | 8 | 1.88 | 200 | 0.030 | 50 | 0.015 |
| gemma-4-e2b-it | grid16 | 32 | 6.07 | 500 | 0.023 | 8 | 0.46 | 200 | 0.122 | 50 | 0.061 |
| gemma-4-e2b-it | count | 32 | 77.69 | 442 | 0.002 | 8 | 0.66 | 200 | 0.085 | 50 | 0.042 |
| gemma-4-e2b-it | ovd | 32 | 4.08 | 963 | 0.066 | 8 | 0.35 | 200 | 0.160 | 50 | 0.080 |
| gemma-4-e2b-it | rel_depth_nyu | 32 | 2.38 | 3270 | 0.381 | 8 | 0.46 | 200 | 0.121 | 50 | 0.060 |
| gemma-4-e2b-it | rel_depth_blink | 32 | 1.33 | 112 | 0.023 | 8 | 0.33 | 112 | 0.096 | 50 | 0.085 |
| gemma-4-e2b-it | spatial_cvbench | 32 | 1.76 | 2588 | 0.408 | 8 | 0.37 | 200 | 0.152 | 50 | 0.076 |
| gemma-4-e2b-it | spatial_embspatial | 32 | 1.55 | 3590 | 0.643 | 8 | 0.20 | 200 | 0.279 | 50 | 0.139 |
| gemma-4-e2b-it | **合计** | | | | **4.663** | | | | **1.142** | | **0.609** |
| internvl3.5-2b-hf | refexp_box | 32 | 31.89 | 56857 | 0.495 | 8 | 0.05 | 200 | 1.093 | 50 | 0.547 |
| internvl3.5-2b-hf | refexp_poly | 32 | 1.07 | 8000 | 2.078 | 8 | 0.13 | 200 | 0.424 | 50 | 0.212 |
| internvl3.5-2b-hf | grid16 | 32 | 8.98 | 500 | 0.015 | 8 | 0.13 | 200 | 0.417 | 50 | 0.209 |
| internvl3.5-2b-hf | count | 16 | 4.83 | 442 | 0.025 | 8 | 0.13 | 200 | 0.412 | 50 | 0.206 |
| internvl3.5-2b-hf | ovd | 8 | 1.40 | 963 | 0.191 | 8 | 0.06 | 200 | 0.899 | 50 | 0.450 |
| internvl3.5-2b-hf | rel_depth_nyu | 32 | 6.40 | 3270 | 0.142 | 8 | 0.19 | 200 | 0.286 | 50 | 0.143 |
| internvl3.5-2b-hf | rel_depth_blink | 32 | 3.34 | 112 | 0.009 | 8 | 0.19 | 112 | 0.160 | 50 | 0.143 |
| internvl3.5-2b-hf | spatial_cvbench | 16 | 4.35 | 2588 | 0.165 | 8 | 0.14 | 200 | 0.407 | 50 | 0.204 |
| internvl3.5-2b-hf | spatial_embspatial | 32 | 6.06 | 3590 | 0.164 | 8 | 0.14 | 200 | 0.408 | 50 | 0.204 |
| internvl3.5-2b-hf | **合计** | | | | **3.286** | | | | **4.507** | | **2.316** |

#### 每个模型的合计（GPU 小时，主口径）

| 模型 | eval | think | think8192（上界） | eval+think+think8192 |
|---|---|---|---|---|
| qwen3.5-0.8b | 2.116 | 2.837 | 1.499 | 6.452 |
| qwen3.5-2b | 2.037 | 2.704 | 1.440 | 6.181 |
| gemma-4-e2b-it | 4.663 | 1.142 | 0.609 | 6.413 |
| internvl3.5-2b-hf | 3.286 | 4.507 | 2.316 | 10.110 |
| 已测 4 个模型之和 | 12.102 | 11.191 | 5.864 | 29.156 |

未测的模型：qwen3.5-4b, gemma-4-e4b-it, internvl3.5-4b-hf, qwen3.5-9b

#### 附加口径（只跑 refexp_box、ovd；只外推 extra 子集）

| 模型 | 口径 | 任务 | 推荐批大小 | 样本/秒 | n(extra) | GPU h(extra) | 主口径同任务样本/秒 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | qwen:1024 | refexp_box | 32 | 14.93 | 500 | 0.009 | 17.08 |
| qwen3.5-0.8b | qwen:1024 | ovd | 32 | 1.11 | 100 | 0.025 | 1.17 |
| qwen3.5-0.8b | qwen:1024 | **合计** | | | | **0.034** | |
| internvl3.5-2b-hf | internvl:native | refexp_box | 32 | 10.18 | 500 | 0.014 | 31.89 |
| internvl3.5-2b-hf | internvl:native | ovd | 8 | 0.37 | 100 | 0.075 | 1.40 |
| internvl3.5-2b-hf | internvl:native | **合计** | | | | **0.089** | |

#### 开思考的实测（预算 4096；开发样本 ≤ 8 条）

| 模型 | 任务 | 批大小 | 样本数 | ok | 样本/秒 | 平均生成 token | 截断率 | 峰值显存 GB |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 8 | 8 | True | 4.52 | 118.2 | 0.000 | 2.23 |
| qwen3.5-0.8b | refexp_poly | 8 | 8 | True | 0.15 | 2265.8 | 0.500 | 2.71 |
| qwen3.5-0.8b | grid16 | 8 | 8 | True | 0.15 | 3253.5 | 0.750 | 2.71 |
| qwen3.5-0.8b | count | 8 | 8 | True | 0.14 | 1482.4 | 0.250 | 2.70 |
| qwen3.5-0.8b | ovd | 8 | 8 | True | 0.15 | 1788.1 | 0.250 | 2.74 |
| qwen3.5-0.8b | rel_depth_nyu | 8 | 8 | True | 0.16 | 2731.2 | 0.500 | 2.44 |
| qwen3.5-0.8b | rel_depth_blink | 8 | 8 | True | 0.15 | 1421.8 | 0.250 | 2.71 |
| qwen3.5-0.8b | spatial_cvbench | 8 | 8 | True | 0.14 | 2446.5 | 0.500 | 2.71 |
| qwen3.5-0.8b | spatial_embspatial | 8 | 8 | True | 0.15 | 1288.0 | 0.250 | 2.72 |
| qwen3.5-2b | refexp_box | 8 | 8 | True | 3.79 | 126.5 | 0.000 | 4.99 |
| qwen3.5-2b | refexp_poly | 8 | 8 | True | 0.15 | 2271.0 | 0.500 | 5.43 |
| qwen3.5-2b | grid16 | 8 | 8 | True | 0.14 | 3758.9 | 0.875 | 5.44 |
| qwen3.5-2b | count | 8 | 8 | True | 0.22 | 749.4 | 0.000 | 5.19 |
| qwen3.5-2b | ovd | 8 | 8 | True | 0.21 | 1488.1 | 0.000 | 5.25 |
| qwen3.5-2b | rel_depth_nyu | 8 | 8 | True | 0.15 | 3168.5 | 0.750 | 5.16 |
| qwen3.5-2b | rel_depth_blink | 8 | 8 | True | 0.14 | 1905.8 | 0.250 | 5.43 |
| qwen3.5-2b | spatial_cvbench | 8 | 8 | True | 0.14 | 1330.6 | 0.125 | 5.43 |
| qwen3.5-2b | spatial_embspatial | 8 | 8 | True | 0.14 | 1337.2 | 0.250 | 5.44 |
| gemma-4-e2b-it | refexp_box | 8 | 8 | True | 0.56 | 560.4 | 0.000 | 10.92 |
| gemma-4-e2b-it | refexp_poly | 8 | 8 | True | 1.88 | 180.5 | 0.000 | 10.93 |
| gemma-4-e2b-it | grid16 | 8 | 8 | True | 0.46 | 676.1 | 0.000 | 10.93 |
| gemma-4-e2b-it | count | 8 | 8 | True | 0.66 | 323.8 | 0.000 | 10.92 |
| gemma-4-e2b-it | ovd | 8 | 8 | True | 0.35 | 412.1 | 0.000 | 10.95 |
| gemma-4-e2b-it | rel_depth_nyu | 8 | 8 | True | 0.46 | 583.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | rel_depth_blink | 8 | 8 | True | 0.33 | 707.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | spatial_cvbench | 8 | 8 | True | 0.37 | 497.1 | 0.000 | 10.92 |
| gemma-4-e2b-it | spatial_embspatial | 8 | 8 | True | 0.20 | 755.8 | 0.000 | 10.93 |
| internvl3.5-2b-hf | refexp_box | 8 | 8 | True | 0.05 | 3104.4 | 0.625 | 9.19 |
| internvl3.5-2b-hf | refexp_poly | 8 | 8 | True | 0.13 | 3443.6 | 0.625 | 9.24 |
| internvl3.5-2b-hf | grid16 | 8 | 8 | True | 0.13 | 4096.0 | 1.000 | 9.25 |
| internvl3.5-2b-hf | count | 8 | 8 | True | 0.13 | 3153.8 | 0.750 | 9.18 |
| internvl3.5-2b-hf | ovd | 8 | 8 | True | 0.06 | 3331.1 | 0.750 | 9.18 |
| internvl3.5-2b-hf | rel_depth_nyu | 8 | 8 | True | 0.19 | 4096.0 | 1.000 | 8.98 |
| internvl3.5-2b-hf | rel_depth_blink | 8 | 8 | True | 0.19 | 4096.0 | 1.000 | 8.98 |
| internvl3.5-2b-hf | spatial_cvbench | 8 | 8 | True | 0.14 | 3180.6 | 0.750 | 9.21 |
| internvl3.5-2b-hf | spatial_embspatial | 8 | 8 | True | 0.14 | 4096.0 | 1.000 | 9.25 |

#### 与 RUN-0012 同模型的合计对照（GPU 小时，主口径与附加口径；数字取自两次运行的 estimate_summary.json）

| 模型 / 口径 | eval RUN-0012 → RUN-0013 | think RUN-0012 → RUN-0013 | think8192 RUN-0012 → RUN-0013 | extra RUN-0012 → RUN-0013 |
|---|---|---|---|---|
| qwen3.5-0.8b/estimate | 1.154 → 2.116 | 2.731 → 2.837 | 1.445 → 1.499 | — → — |
| qwen3.5-2b/estimate | 1.031 → 2.037 | 2.628 → 2.704 | 1.396 → 1.440 | — → — |
| gemma-4-e2b-it/estimate | 2.217 → 4.663 | 1.152 → 1.142 | 0.616 → 0.609 | — → — |
| internvl3.5-2b-hf/estimate | 2.835 → 3.286 | 4.514 → 4.507 | 2.322 → 2.316 | — → — |
| qwen3.5-0.8b/estimate__qwen-1024 | — → — | — → — | — → — | 0.015 → 0.034 |
| internvl3.5-2b-hf/estimate__internvl-native | — → — | — → — | — → — | 0.040 → 0.089 |

## 4. 与任务书假设不符的发现

- **用时超出预计，估算又只完成了一半。** 任务书预计合计 4–6 GPU 小时。
  - 验收 8 个模型用了 6.38 h，RUN-0012 是 3.82 h。主要是 batch=1 那遍更长：qwen3.5-0.8b 从 37 min 增加到 66 min，qwen3.5-2b 从 36 min 增加到 57 min，各任务的墙钟见 §3.1 最后一张表。
  - 估算从累计 6.38 h 开始，第 4 个模型结束时到了 8.19 h，按 §5 停止。
  - 两次运行里，qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b 的估算都没有跑；qwen3.5-9b 的 qwen:1024 与 internvl3.5-4b-hf 的 internvl:native 也都没有跑。
  - 执行顺序沿用 RUN-0012：先做完 §3，再做 §4。任务书写的是先推验收、再推估算。
- **CountBenchQA 那遍用了单独的 `--out`。** 任务书写的是"计数任务另跑一遍 `--task count --count-questions countbenchqa`"，没有给 `--out`。
  - `src/e1/runner.leaf_dir` 的叶子目录名只含任务、子集与口径，不含问句来源。写进同一个 `experiments/E1/0013_acceptance` 会与启发式问句的 `count__dev` 撞名：不加 `--resume` 会报 FileExistsError；加了 `--resume`，又会因为配置中的 count_questions 不同，被拒绝混写。
  - 所以另用 `experiments/E1/0013_acceptance_countbenchqa`，目录名符合 `RUN_DIR_NAME`。
- **`--resume` 每遍都会改写 `acceptance.json` 和 `acceptance.md`。** 同一个 `<键>/acceptance/` 下只会留下最后一遍。驱动在每遍之后把它们复制成 `acceptance__<口径>.*`，`acceptance.json` 本身与 `__outcome` 版相同。
- **一条不一致就会让小样本任务不达标（只陈述算术）。** 一致率门槛是 0.98：
  - count（n=49）：1 条不一致是 0.9796，2 条是 0.9592；
  - rel_depth_blink（n=12）：1 条不一致是 0.9167；
  - spatial_cvbench、spatial_embspatial（n=50）：1 条不一致是 0.98，可以通过；2 条是 0.96。
- **ovd 在 (c) 口径下不判（D33-2）。** 8 个模型都列进了 `not_evaluated`。原因是开发子集没有 GT（N15 的补充事实）。
- **诊断（原样取自 metrics.json，两遍合计）：**
  - refexp_poly 的 wrong_shape：
    - qwen3.5-0.8b：`bbox_2d` 196、`point_2d` 2；
    - qwen3.5-2b：`point_2d` 171、`bbox_2d` 29；
    - qwen3.5-4b：`point_2d` 200；
    - qwen3.5-9b：`point_2d` 196、`bbox_2d` 2；
    - gemma-4-e4b-it：`box_2d` 186。
  - refexp_box 的 refusal：gemma-4-e2b-it 121、gemma-4-e4b-it 126（两遍合计，n=100×2）。
  - 关思考时自发出现思考标记：只有 internvl3.5-2b-hf 的 count（启发式问句）有，两遍合计 7 条；CountBenchQA 那遍是 0。
- **推荐批大小大多落在测试范围的上端。** 40 格里 32 占 35 个，8 占 3 个，16 占 2 个。

## 5. 生成的文件

- `experiments/E1/0013_acceptance/`（31 MB）与 `experiments/E1/0013_acceptance_countbenchqa/`（1.9 MB），共 1,790 个文件，第一批在 `6dcf31a` 推送：
  - 8 个模型的 `acceptance/`：三种口径的汇总，以及 bs1、bs16 两遍的全部输出；
  - `accept_summary.md/json`、`README.md`、`pip_dryrun.txt`；
  - `countbenchqa_questions_MANIFEST.json`、`land_countbenchqa.log`；
  - `logs/`、`probe_scripts/`。
  - 本批又更新了 `logs/driver.log`，新增了 `disk_ledger.tsv`。
- `experiments/E1/0013_estimate/`（本批推送）：
  - 4 个模型的 `estimate/`，另有 2 个附加口径目录；
  - `estimate_summary.md/json`、`README.md`、`disk_ledger.tsv`、`logs/`、`probe_scripts/`。

## 6. 磁盘与内存盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 14:50:25 | 0013 start | 39.269 | 3.062 | 3.909 |
| 14:51:04 | before download qwen3.5-0.8b | 39.268 | 3.062 | 3.909 |
| 14:51:05 | after download qwen3.5-0.8b | 39.268 | 3.062 | 3.909 |
| 15:57:37 | after accept qwen3.5-0.8b | 39.254 | 3.062 | 3.913 |
| 15:57:37 | before download qwen3.5-2b | 39.254 | 3.062 | 3.913 |
| 15:57:49 | after download qwen3.5-2b | 39.254 | 7.620 | 3.913 |
| 16:55:26 | after accept qwen3.5-2b | 39.248 | 7.620 | 3.917 |
| 16:55:26 | after delete qwen3.5-2b | 39.248 | 3.062 | 3.917 |
| 16:55:26 | before download gemma-4-e2b-it | 39.248 | 3.062 | 3.917 |
| 16:55:57 | after download gemma-4-e2b-it | 39.248 | 13.341 | 3.917 |
| 17:37:41 | after accept gemma-4-e2b-it | 39.243 | 13.341 | 3.921 |
| 17:37:42 | after delete gemma-4-e2b-it | 39.243 | 3.062 | 3.921 |
| 17:37:42 | before download internvl3.5-2b-hf | 39.243 | 3.062 | 3.921 |
| 17:37:58 | after download internvl3.5-2b-hf | 39.243 | 7.777 | 3.921 |
| 18:04:30 | after accept internvl3.5-2b-hf | 39.239 | 7.777 | 3.924 |
| 18:04:30 | after delete internvl3.5-2b-hf | 39.239 | 3.063 | 3.924 |
| 18:04:30 | before download qwen3.5-4b | 39.239 | 3.063 | 3.924 |
| 18:04:52 | after download qwen3.5-4b | 39.239 | 12.393 | 3.924 |
| 18:52:23 | after accept qwen3.5-4b | 39.234 | 12.393 | 3.928 |
| 18:52:23 | after delete qwen3.5-4b | 39.234 | 3.063 | 3.928 |
| 18:52:24 | before download gemma-4-e4b-it | 39.234 | 3.063 | 3.928 |
| 18:53:01 | after download gemma-4-e4b-it | 39.234 | 19.089 | 3.928 |
| 19:42:47 | after accept gemma-4-e4b-it | 39.229 | 19.089 | 3.931 |
| 19:42:48 | after delete gemma-4-e4b-it | 39.229 | 3.064 | 3.931 |
| 19:42:48 | before download internvl3.5-4b-hf | 39.229 | 3.064 | 3.931 |
| 19:43:12 | after download internvl3.5-4b-hf | 39.229 | 12.547 | 3.931 |
| 20:25:03 | after accept internvl3.5-4b-hf | 39.223 | 12.547 | 3.934 |
| 20:25:03 | after delete internvl3.5-4b-hf | 39.223 | 3.064 | 3.934 |
| 20:25:03 | before download qwen3.5-9b | 39.223 | 3.064 | 3.934 |
| 20:25:46 | after download qwen3.5-9b | 39.223 | 22.381 | 3.934 |
| 21:14:00 | after accept qwen3.5-9b | 39.296 | 22.381 | 3.938 |
| 21:14:01 | after delete qwen3.5-9b | 39.296 | 3.065 | 3.938 |
| 21:14:01 | before download qwen3.5-0.8b (estimate) | 39.296 | 3.065 | 3.938 |
| 21:14:02 | after download qwen3.5-0.8b (estimate) | 39.296 | 3.065 | 3.938 |
| 21:44:22 | after estimate qwen3.5-0.8b | 39.273 | 3.065 | 3.943 |
| 21:44:23 | before download qwen3.5-2b (estimate) | 39.273 | 3.065 | 3.943 |
| 21:44:35 | after download qwen3.5-2b (estimate) | 39.273 | 7.623 | 3.943 |
| 22:07:45 | after estimate qwen3.5-2b | 39.273 | 7.623 | 3.943 |
| 22:07:46 | after delete qwen3.5-2b (estimate) | 39.273 | 3.065 | 3.943 |
| 22:07:46 | before download gemma-4-e2b-it (estimate) | 39.273 | 3.065 | 3.943 |
| 22:08:09 | after download gemma-4-e2b-it (estimate) | 39.273 | 13.344 | 3.943 |
| 22:25:29 | after estimate gemma-4-e2b-it | 39.272 | 13.344 | 3.943 |
| 22:25:29 | after delete gemma-4-e2b-it (estimate) | 39.272 | 3.066 | 3.943 |
| 22:25:30 | before download internvl3.5-2b-hf (estimate) | 39.272 | 3.066 | 3.943 |
| 22:25:42 | after download internvl3.5-2b-hf (estimate) | 39.272 | 7.781 | 3.943 |
| 23:02:35 | after estimate internvl3.5-2b-hf | 39.272 | 7.781 | 3.943 |
| 23:02:36 | after delete internvl3.5-2b-hf (estimate) | 39.272 | 3.066 | 3.943 |
| 23:03:19 | 0013 end | 39.272 | 3.066 | 3.943 |

- **根盘：** 可用空间从 39.269 GB 到 39.272 GB，全程远高于 3 GB 红线。根盘与其他用户共用。
- **`$PROJ`：** 从 3.909 GB 升到 3.943 GB（+34 MB），在 15 GB 份额以内（D28-8）。
  - 增量是本运行的产物（都在仓库的运行目录里，第一批已提交）、`$PROJ/logs/0013`，以及 `$PROJ/data/countbenchqa_questions`（39 KB）。
- **内存盘：** 峰值 22.381（qwen3.5-9b 验收时） GB，低于 25 GB 上限。
  - 结束时 `/dev/shm/visint` 是 3.066 GB：Qwen3.5-0.8B 的权重、NYU 点对渲染缓存与 xet 缓存。
  - `TMPDIR` 下没有留下新文件。
