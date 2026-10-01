# RUN-0012 — E1 批处理验收（D28-5）与吞吐、算力实测（预注册 §9）

服务器 CC，b200，2026-10-01 08:34–14:44 UTC，值守中接单。本报告只列数字与事实，不做解读。

## 0. 摘要（只陈述，不解读）

- **验收（§3）：**
  - 先跑任务书指定的 4 个模型。qwen3.5-0.8b 有任务不达标，触发 §3 的补跑，其余 4 个模型也做了同样的验收。
  - 8 个模型都是 rc=0，`all_pass` 都是 False。
  - 72 个"模型 × 任务"格子里有 29 个不通过：refexp_box 8/8、ovd 7/8、spatial_embspatial 4/8、refexp_poly 3/8、count 3/8、spatial_cvbench 2/8、grid16 2/8。
  - 配对差 95% CI 不含 0 的只有 1 格：internvl3.5-2b-hf 的 spatial_embspatial，差 −0.08，CI [−0.16, −0.02]。其余不通过的格子都是因为答案一致率 < 0.98。
- **截断（只陈述）：**
  - 8 个模型的 rel_depth_nyu（100 条）和 rel_depth_blink（12 条）在 max_new_tokens=64 下，两遍都 100% 截断。
  - 72 格里有 24 格两遍全部解析失败：rel_depth_nyu 8、rel_depth_blink 8、refexp_poly 4、grid16 3、ovd 1。按 D32-2 的判据，两遍都失败算一致，所以这些格子的一致率都是 1.0。
- **估算（§4）：**
  - 主口径完成 4 个模型：qwen3.5-0.8b、qwen3.5-2b、gemma-4-e2b-it、internvl3.5-2b-hf。附加口径完成 qwen3.5-0.8b 的 qwen:1024 和 internvl3.5-2b-hf 的 internvl:native。
  - internvl3.5-2b-hf 跑完时累计已到 22,224 s，超过 6 小时，按 §5 没有启动 qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b。
  - 已测 4 个模型的主口径外推合计：eval 7.238、think 11.025、think8192（上界）5.779 GPU 小时。
  - 40 个"模型 × 任务"格子里，推荐批大小 32 有 37 个，16 有 3 个。32 是测过的最大批大小。没有任何 OOM。
- **本次用量：** 墙钟 22,224 s（6.17 h），全程 GPU 上同一时刻只有本任务的一个进程。其中验收 13,735 s（3.82 h），估算 8,489 s（2.36 h）。按 D28-10，这些都计入 E1 的预算。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `dbaa58e`。`a6c0c41` 与本任务书所在的 `aaccad4` 都在 HEAD 的历史里（`git merge-base --is-ancestor`）。
- **环境：**
  - `pip install --dry-run -r requirements.txt` 没有任何 "Would install"，见 `experiments/E1/0012_acceptance/pip_dryrun.txt`。
  - 驱动脚本先 `source $PROJ/scripts/env.sh`，再 export `VISINT_DATA_ROOT=$PROJ/data`、`NLTK_DATA=$PROJ/nltk_data`、`TMPDIR=/dev/shm/visint/tmp`。
  - NYU 点对渲染图用的是 `$TMPDIR/visint_nyu_pairs_rendered`，即 RUN-0010 留在内存盘的 1.2 GB 缓存。
- **改动范围：**
  - 没有修改 `src/`、`scripts/`、`tests/`、`configs/`，也没有安装 Python 包。
  - 新写的文件只在两个运行目录的 `probe_scripts/` 下：
    - 驱动：`run_accept.sh`、`run_estimate.sh`、`chain.sh`；
    - 采样：`gpu_sampler.sh`；
    - 汇总：`p15_accept_summary.py`、`p16_estimate_summary.py`；
    - 磁盘：`diskcheck12.sh`。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。
- **运行期间 origin 有新提交：**
  - 推第一批之前 `git pull` 拉到了 `cc81faa`（管理员关闭 RUN-0011，提出 N14）。
  - 它只改了 `DECISIONS_NEEDED.md` 和 `RESEARCH_LOG.md`；`git diff --stat dbaa58e cc81faa -- src scripts configs requirements.txt` 的输出为空。
- **只用了开发子集。** 没有运行任何 `--subset eval/think/think8192/extra`。估算只按各子集的 n 外推。

## 2. 执行记录

每个模型的流程如下（驱动日志见 `logs/driver.log`）：

1. 记一笔磁盘收支。
2. `p12_verify_weights.py <键> --download`，按 RUN-0009 的清单核对。
3. 记一笔磁盘收支。
4. 跑 run_e1。
5. 记一笔磁盘收支。
6. 除 qwen3.5-0.8b 外，执行 `delete_revisions` 删除权重，再记一笔磁盘收支。
7. 检查累计耗时。

执行顺序：

- 第一个驱动跑任务书指定的 4 个模型；
- `chain.sh` 等第一个驱动结束后，补跑其余 4 个模型的验收，然后跑估算；
- 每一步开始前都检查累计耗时是否超过 6 小时。

| # | 阶段 | 模型 | 权重核对 | 下载+核对 | run_e1 墙钟（`time` real） | rc | GPU 显存峰值（nvidia-smi 按进程，MiB） | 删除释放 |
|---|---|---|---|---|---|---|---|---|
| 1 | 验收 | qwen3.5-0.8b | 13/13 | 1 s（已在内存盘） | 37m06.7s | 0 | 4038 | 保留 |
| 2 | 验收 | gemma-4-e2b-it | 9/9 | 24 s | 24m00.3s | 0 | 13848 | 10.279 GB |
| 3 | 验收 | internvl3.5-2b-hf | 18/18 | 14 s | 18m25.7s | 0 | 10364 | 4.715 GB |
| 4 | 验收 | qwen3.5-9b | 16/16 | 42 s | 26m50.1s | 0 | 22302 | 19.317 GB |
| 5 | 验收（补跑） | qwen3.5-2b | 13/13 | 18 s | 36m19.0s | 0 | 7108 | 4.558 GB |
| 6 | 验收（补跑） | qwen3.5-4b | 14/14 | 23 s | 25m40.7s | 0 | 12422 | 9.330 GB |
| 7 | 验收（补跑） | gemma-4-e4b-it | 9/9 | 41 s | 31m46.6s | 0 | 20018 | 16.025 GB |
| 8 | 验收（补跑） | internvl3.5-4b-hf | 20/20 | 24 s | 25m24.5s | 0 | 15360 | 9.483 GB |
| 9 | 估算 | qwen3.5-0.8b（default） | 13/13 | 1 s | 35m01.3s | 0 | 12256 | — |
| 10 | 估算 | qwen3.5-0.8b（qwen:1024） | — | — | 7m34.8s | 0 | 9932 | 保留 |
| 11 | 估算 | qwen3.5-2b | 13/13 | 13 s | 36m41.0s | 0 | 14798 | 4.558 GB |
| 12 | 估算 | gemma-4-e2b-it | 9/9 | 42 s | 23m56.6s | 0 | 16668 | 10.279 GB |
| 13 | 估算 | internvl3.5-2b-hf（default） | 18/18 | 25 s | 29m48.8s | 0 | 67732 | — |
| 14 | 估算 | internvl3.5-2b-hf（internvl:native） | — | — | 7m02.8s | 0 | 29382 | 4.715 GB |
| — | 估算 | qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b | — | — | 未开始（累计 22,224 s > 21,600 s） | — | — | — |

- **GPU：** NVIDIA B200，bf16。
  - nvidia-smi 每 5 s 采样一次，共 4,420 个样本：验收 2,731 个，估算 1,689 个。
  - 整卡显存峰值：验收阶段 22,316 MiB，估算阶段 67,746 MiB。
  - GPU 利用率均值：验收 55.0%，估算 51.4%。均值包含下载与加载时的空闲样本。
  - 采样期间 GPU 上依次出现 14 个 python 进程，时间区间互不重叠，与上表的 14 次运行一一对应。
- **权重：** 12 次核对全部一致。
- **报错：** 14 次 run_e1 都没有 Traceback。
- **警告：** 与 RUN-0007、RUN-0011 相同。4 个 Qwen 是 `causal_conv1d` 回退到 PyTorch 参考实现；2 个 InternVL 是 tie weights 提示。原文见各 `logs/*.log`。
- **累计耗时：** 驱动日志记的是每个模型结束时的累计秒数：2228、3693、4814、6467、8673、10237、12186、13735（验收结束）。估算各模型开始时的累计秒数是：13735、16292、18507、19987；internvl3.5-2b-hf 结束时为 22224。

## 3. 结果表（原样）

### 3.1 批处理验收

本节由 `experiments/E1/0012_acceptance/probe_scripts/p15_accept_summary.py` 从各模型的 `acceptance.json`，以及两遍的 `metrics.json`、`compute.json`、`predictions.jsonl` 生成，与 `accept_summary.md` 相同。判据原样取自 `acceptance.json`。

#### 验收汇总（模型 × 任务；判据：答案一致率 ≥ 0.98 且配对差 95% CI 含 0；检测无 GT，只判一致率）

| 模型 | 任务 | n | 答案一致率 | 不一致条数 | 指标 | 逐条 | 批量 | 差（逐条−批量） | 配对差 95% CI | CI 含 0 | raw_text 逐字相同 | 通过 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 100 | 0.3000 | 70 | acc@0.5 | 0.8000 | 0.8200 | -0.0200 | [-0.0500, 0.0000] | 是 | 0.2800 | 否 |
| qwen3.5-0.8b | refexp_poly | 100 | 1.0000 | 0 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3300 | 是 |
| qwen3.5-0.8b | grid16 | 100 | 1.0000 | 0 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 | 是 |
| qwen3.5-0.8b | count | 49 | 1.0000 | 0 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9592 | 是 |
| qwen3.5-0.8b | ovd | 100 | 0.6300 | 37 | — | — | — | — | — | — | 0.0000 | 否 |
| qwen3.5-0.8b | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6300 | 是 |
| qwen3.5-0.8b | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5833 | 是 |
| qwen3.5-0.8b | spatial_cvbench | 50 | 0.9800 | 1 | accuracy | 0.0200 | 0.0400 | -0.0200 | [-0.0600, 0.0000] | 是 | 0.5600 | 是 |
| qwen3.5-0.8b | spatial_embspatial | 50 | 0.9600 | 2 | accuracy | 0.0200 | 0.0200 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5200 | 否 |
| qwen3.5-2b | refexp_box | 100 | 0.2900 | 71 | acc@0.5 | 0.8600 | 0.8600 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2800 | 否 |
| qwen3.5-2b | refexp_poly | 100 | 1.0000 | 0 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3000 | 是 |
| qwen3.5-2b | grid16 | 100 | 1.0000 | 0 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9900 | 是 |
| qwen3.5-2b | count | 49 | 1.0000 | 0 | accuracy | 0.7143 | 0.7143 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 | 是 |
| qwen3.5-2b | ovd | 100 | 0.4500 | 55 | — | — | — | — | — | — | 0.0200 | 否 |
| qwen3.5-2b | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6200 | 是 |
| qwen3.5-2b | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.4167 | 是 |
| qwen3.5-2b | spatial_cvbench | 50 | 0.9800 | 1 | accuracy | 0.0400 | 0.0600 | -0.0200 | [-0.0600, 0.0000] | 是 | 0.7400 | 是 |
| qwen3.5-2b | spatial_embspatial | 50 | 0.9800 | 1 | accuracy | 0.1000 | 0.1200 | -0.0200 | [-0.0600, 0.0000] | 是 | 0.5400 | 是 |
| gemma-4-e2b-it | refexp_box | 100 | 0.9100 | 9 | acc@0.5 | 0.0800 | 0.0800 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7200 | 否 |
| gemma-4-e2b-it | refexp_poly | 100 | 0.6900 | 31 | cIoU | 0.2413 | 0.2340 | 0.0073 | [-0.0016, 0.0163] | 是 | 0.5400 | 否 |
| gemma-4-e2b-it | grid16 | 100 | 1.0000 | 0 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9500 | 是 |
| gemma-4-e2b-it | count | 49 | 0.9796 | 1 | accuracy | 0.3673 | 0.3673 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 | 否 |
| gemma-4-e2b-it | ovd | 100 | 0.4500 | 55 | — | — | — | — | — | — | 0.4100 | 否 |
| gemma-4-e2b-it | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7000 | 是 |
| gemma-4-e2b-it | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.4167 | 是 |
| gemma-4-e2b-it | spatial_cvbench | 50 | 1.0000 | 0 | accuracy | 0.2600 | 0.2600 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6800 | 是 |
| gemma-4-e2b-it | spatial_embspatial | 50 | 1.0000 | 0 | accuracy | 0.0200 | 0.0200 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6000 | 是 |
| internvl3.5-2b-hf | refexp_box | 100 | 0.0900 | 91 | acc@0.5 | 0.7100 | 0.7300 | -0.0200 | [-0.0500, 0.0000] | 是 | 0.0900 | 否 |
| internvl3.5-2b-hf | refexp_poly | 100 | 0.4300 | 57 | cIoU | 0.1031 | 0.0918 | 0.0113 | [-0.0111, 0.0344] | 是 | 0.1800 | 否 |
| internvl3.5-2b-hf | grid16 | 100 | 0.9900 | 1 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.8700 | 是 |
| internvl3.5-2b-hf | count | 49 | 0.9796 | 1 | accuracy | 0.4490 | 0.4490 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7755 | 否 |
| internvl3.5-2b-hf | ovd | 100 | 1.0000 | 0 | — | — | — | — | — | — | 0.3400 | 是 |
| internvl3.5-2b-hf | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5800 | 是 |
| internvl3.5-2b-hf | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6667 | 是 |
| internvl3.5-2b-hf | spatial_cvbench | 50 | 0.9600 | 2 | accuracy | 0.1000 | 0.1000 | 0.0000 | [-0.0600, 0.0600] | 是 | 0.7200 | 否 |
| internvl3.5-2b-hf | spatial_embspatial | 50 | 0.9200 | 4 | accuracy | 0.1400 | 0.2200 | -0.0800 | [-0.1600, -0.0200] | 否 | 0.6800 | 否 |
| qwen3.5-4b | refexp_box | 100 | 0.3300 | 67 | acc@0.5 | 0.8700 | 0.8700 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3000 | 否 |
| qwen3.5-4b | refexp_poly | 100 | 1.0000 | 0 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2700 | 是 |
| qwen3.5-4b | grid16 | 100 | 0.9900 | 1 | gIoU | 0.0111 | 0.0111 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9900 | 是 |
| qwen3.5-4b | count | 49 | 1.0000 | 0 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 | 是 |
| qwen3.5-4b | ovd | 100 | 0.3000 | 70 | — | — | — | — | — | — | 0.0100 | 否 |
| qwen3.5-4b | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6400 | 是 |
| qwen3.5-4b | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6667 | 是 |
| qwen3.5-4b | spatial_cvbench | 50 | 0.9800 | 1 | accuracy | 0.0400 | 0.0400 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6800 | 是 |
| qwen3.5-4b | spatial_embspatial | 50 | 0.8800 | 6 | accuracy | 0.3000 | 0.3800 | -0.0800 | [-0.1800, 0.0005] | 是 | 0.6600 | 否 |
| gemma-4-e4b-it | refexp_box | 100 | 0.9200 | 8 | acc@0.5 | 0.1300 | 0.1300 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6900 | 否 |
| gemma-4-e4b-it | refexp_poly | 100 | 0.9800 | 2 | cIoU | 0.0381 | 0.0336 | 0.0045 | [-0.0000, 0.0122] | 是 | 0.1600 | 是 |
| gemma-4-e4b-it | grid16 | 100 | 0.9800 | 2 | gIoU | 0.0112 | 0.0122 | -0.0010 | [-0.0038, 0.0010] | 是 | 0.9700 | 是 |
| gemma-4-e4b-it | count | 49 | 0.9592 | 2 | accuracy | 0.5102 | 0.4898 | 0.0204 | [0.0000, 0.0612] | 是 | 0.9592 | 否 |
| gemma-4-e4b-it | ovd | 100 | 0.1900 | 81 | — | — | — | — | — | — | 0.1700 | 否 |
| gemma-4-e4b-it | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6700 | 是 |
| gemma-4-e4b-it | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5833 | 是 |
| gemma-4-e4b-it | spatial_cvbench | 50 | 0.9600 | 2 | accuracy | 0.3400 | 0.3800 | -0.0400 | [-0.1000, 0.0000] | 是 | 0.7000 | 否 |
| gemma-4-e4b-it | spatial_embspatial | 50 | 1.0000 | 0 | accuracy | 0.0800 | 0.0800 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6600 | 是 |
| internvl3.5-4b-hf | refexp_box | 100 | 0.4800 | 52 | acc@0.5 | 0.8700 | 0.8600 | 0.0100 | [0.0000, 0.0400] | 是 | 0.4800 | 否 |
| internvl3.5-4b-hf | refexp_poly | 100 | 0.3700 | 63 | cIoU | 0.1630 | 0.1639 | -0.0009 | [-0.0311, 0.0276] | 是 | 0.2500 | 否 |
| internvl3.5-4b-hf | grid16 | 100 | 0.9700 | 3 | gIoU | 0.0044 | 0.0040 | 0.0004 | [0.0000, 0.0012] | 是 | 0.9700 | 否 |
| internvl3.5-4b-hf | count | 49 | 1.0000 | 0 | accuracy | 0.6735 | 0.6735 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 | 是 |
| internvl3.5-4b-hf | ovd | 100 | 0.4500 | 55 | — | — | — | — | — | — | 0.0200 | 否 |
| internvl3.5-4b-hf | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5300 | 是 |
| internvl3.5-4b-hf | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3333 | 是 |
| internvl3.5-4b-hf | spatial_cvbench | 50 | 1.0000 | 0 | accuracy | 0.2400 | 0.2400 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7200 | 是 |
| internvl3.5-4b-hf | spatial_embspatial | 50 | 0.9200 | 4 | accuracy | 0.4600 | 0.4600 | 0.0000 | [-0.0600, 0.0600] | 是 | 0.8400 | 否 |
| qwen3.5-9b | refexp_box | 100 | 0.3500 | 65 | acc@0.5 | 0.8900 | 0.8900 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3300 | 否 |
| qwen3.5-9b | refexp_poly | 100 | 1.0000 | 0 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3800 | 是 |
| qwen3.5-9b | grid16 | 100 | 0.9500 | 5 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9400 | 否 |
| qwen3.5-9b | count | 49 | 1.0000 | 0 | accuracy | 0.7959 | 0.7959 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 | 是 |
| qwen3.5-9b | ovd | 100 | 0.3200 | 68 | — | — | — | — | — | — | 0.0000 | 否 |
| qwen3.5-9b | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.8200 | 是 |
| qwen3.5-9b | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.4167 | 是 |
| qwen3.5-9b | spatial_cvbench | 50 | 0.9800 | 1 | accuracy | 0.1200 | 0.1000 | 0.0200 | [0.0000, 0.0600] | 是 | 0.7400 | 是 |
| qwen3.5-9b | spatial_embspatial | 50 | 0.9800 | 1 | accuracy | 0.3200 | 0.3200 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6800 | 是 |

| 模型 | all_pass | 不通过的任务 |
|---|---|---|
| qwen3.5-0.8b | 否 | refexp_box, ovd, spatial_embspatial |
| qwen3.5-2b | 否 | refexp_box, ovd |
| gemma-4-e2b-it | 否 | refexp_box, refexp_poly, count, ovd |
| internvl3.5-2b-hf | 否 | refexp_box, refexp_poly, count, spatial_cvbench, spatial_embspatial |
| qwen3.5-4b | 否 | refexp_box, ovd, spatial_embspatial |
| gemma-4-e4b-it | 否 | refexp_box, count, ovd, spatial_cvbench |
| internvl3.5-4b-hf | 否 | refexp_box, refexp_poly, grid16, ovd, spatial_embspatial |
| qwen3.5-9b | 否 | refexp_box, grid16, ovd |

#### 截断率与解析失败分布（逐条 bs=1 / 批量；截断 = stopped=False）

| 模型 | 任务 | max_new_tokens | 截断率 bs1 / 批量 | 解析失败 bs1 | 解析失败 批量 | 平均生成 token bs1 / 批量 | 思考未闭合 bs1 / 批量 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 128 | 0.02 / 0.02 | 2/100 {"truncated": 2} | 2/100 {"truncated": 2} | 47.0 / 47.0 | 0 / 0 |
| qwen3.5-0.8b | refexp_poly | 1024 | 0.01 / 0.01 | 100/100 {"malformed": 99, "truncated": 1} | 100/100 {"malformed": 99, "truncated": 1} | 66.3 / 67.5 | 0 / 0 |
| qwen3.5-0.8b | grid16 | 1024 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 1024.0 / 1024.0 | 0 / 0 |
| qwen3.5-0.8b | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.3 / 2.6 | 0 / 0 |
| qwen3.5-0.8b | ovd | 512 | 0.66 / 0.66 | 66/100 {"truncated": 66} | 66/100 {"truncated": 66} | 416.1 / 413.6 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-0.8b | spatial_cvbench | 64 | 0.96 / 0.94 | 48/50 {"truncated": 48} | 47/50 {"truncated": 47} | 63.8 / 64.0 | 0 / 0 |
| qwen3.5-0.8b | spatial_embspatial | 64 | 0.92 / 0.96 | 46/50 {"truncated": 46} | 48/50 {"truncated": 48} | 59.9 / 61.6 | 0 / 0 |
| qwen3.5-2b | refexp_box | 128 | 0.01 / 0.01 | 1/100 {"truncated": 1} | 1/100 {"truncated": 1} | 46.7 / 46.7 | 0 / 0 |
| qwen3.5-2b | refexp_poly | 1024 | 0.00 / 0.00 | 100/100 {"malformed": 100} | 100/100 {"malformed": 100} | 43.4 / 42.3 | 0 / 0 |
| qwen3.5-2b | grid16 | 1024 | 0.99 / 0.99 | 100/100 {"truncated": 99, "wrong_row_count": 1} | 100/100 {"truncated": 99, "wrong_row_count": 1} | 1016.3 / 1016.3 | 0 / 0 |
| qwen3.5-2b | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 | 0 / 0 |
| qwen3.5-2b | ovd | 512 | 0.47 / 0.47 | 47/100 {"truncated": 47} | 47/100 {"truncated": 47} | 367.8 / 365.7 | 0 / 0 |
| qwen3.5-2b | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-2b | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-2b | spatial_cvbench | 64 | 0.94 / 0.92 | 47/50 {"truncated": 47} | 46/50 {"truncated": 46} | 62.9 / 62.4 | 0 / 0 |
| qwen3.5-2b | spatial_embspatial | 64 | 0.86 / 0.84 | 43/50 {"truncated": 43} | 42/50 {"truncated": 42} | 57.2 / 56.3 | 0 / 0 |
| gemma-4-e2b-it | refexp_box | 128 | 0.13 / 0.12 | 88/100 {"multiple_boxes": 3, "no_box": 72, "truncated": 13} | 87/100 {"multiple_boxes": 3, "no_box": 72, "truncated": 12} | 44.0 / 43.8 | 0 / 0 |
| gemma-4-e2b-it | refexp_poly | 1024 | 0.00 / 0.00 | 2/100 {"malformed": 1, "too_few_points": 1} | 2/100 {"malformed": 1, "too_few_points": 1} | 84.5 / 88.2 | 0 / 0 |
| gemma-4-e2b-it | grid16 | 1024 | 0.00 / 0.00 | 100/100 {"empty": 3, "wrong_row_count": 97} | 100/100 {"empty": 3, "wrong_row_count": 97} | 245.3 / 244.7 | 0 / 0 |
| gemma-4-e2b-it | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 | 0 / 0 |
| gemma-4-e2b-it | ovd | 512 | 0.01 / 0.01 | 35/100 {"no_box": 34, "truncated": 1} | 35/100 {"no_box": 34, "truncated": 1} | 156.9 / 159.0 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| gemma-4-e2b-it | spatial_cvbench | 64 | 0.70 / 0.70 | 35/50 {"truncated": 35} | 35/50 {"truncated": 35} | 56.5 / 56.5 | 0 / 0 |
| gemma-4-e2b-it | spatial_embspatial | 64 | 0.98 / 0.98 | 49/50 {"truncated": 49} | 49/50 {"truncated": 49} | 63.4 / 63.4 | 0 / 0 |
| internvl3.5-2b-hf | refexp_box | 128 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 28.4 / 28.4 | 0 / 0 |
| internvl3.5-2b-hf | refexp_poly | 1024 | 0.02 / 0.02 | 22/100 {"ambiguous": 3, "malformed": 2, "too_few_points": 15, "truncated": 2} | 21/100 {"ambiguous": 1, "malformed": 4, "too_few_points": 14, "truncated": 2} | 254.8 / 265.2 | 0 / 0 |
| internvl3.5-2b-hf | grid16 | 1024 | 0.00 / 0.00 | 100/100 {"wrong_row_count": 100} | 99/100 {"wrong_row_count": 99} | 252.2 / 253.0 | 0 / 0 |
| internvl3.5-2b-hf | count | 64 | 0.29 / 0.31 | 14/49 {"thinking_unclosed": 3, "truncated": 11} | 15/49 {"thinking_unclosed": 4, "truncated": 11} | 28.0 / 29.1 | 3 / 4 |
| internvl3.5-2b-hf | ovd | 512 | 0.04 / 0.05 | 100/100 {"no_box": 96, "truncated": 4} | 100/100 {"no_box": 95, "truncated": 5} | 159.4 / 159.7 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| internvl3.5-2b-hf | spatial_cvbench | 64 | 0.88 / 0.88 | 44/50 {"truncated": 44} | 44/50 {"truncated": 44} | 61.1 / 61.7 | 0 / 0 |
| internvl3.5-2b-hf | spatial_embspatial | 64 | 0.62 / 0.56 | 31/50 {"truncated": 31} | 28/50 {"truncated": 28} | 44.3 / 42.6 | 0 / 0 |
| qwen3.5-4b | refexp_box | 128 | 0.03 / 0.03 | 3/100 {"truncated": 3} | 3/100 {"truncated": 3} | 48.0 / 48.0 | 0 / 0 |
| qwen3.5-4b | refexp_poly | 1024 | 0.00 / 0.00 | 100/100 {"malformed": 100} | 100/100 {"malformed": 100} | 40.8 / 41.1 | 0 / 0 |
| qwen3.5-4b | grid16 | 1024 | 0.00 / 0.00 | 1/100 {"wrong_row_count": 1} | 0/100 {} | 272.3 / 272.0 | 0 / 0 |
| qwen3.5-4b | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 | 0 / 0 |
| qwen3.5-4b | ovd | 512 | 0.29 / 0.30 | 29/100 {"truncated": 29} | 30/100 {"truncated": 30} | 339.2 / 339.1 | 0 / 0 |
| qwen3.5-4b | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-4b | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-4b | spatial_cvbench | 64 | 0.94 / 0.96 | 47/50 {"truncated": 47} | 48/50 {"truncated": 48} | 63.2 / 63.6 | 0 / 0 |
| qwen3.5-4b | spatial_embspatial | 64 | 0.68 / 0.62 | 35/50 {"no_choice": 1, "truncated": 34} | 31/50 {"truncated": 31} | 48.4 / 44.2 | 0 / 0 |
| gemma-4-e4b-it | refexp_box | 128 | 0.08 / 0.09 | 82/100 {"multiple_boxes": 1, "no_box": 73, "truncated": 8} | 83/100 {"multiple_boxes": 1, "no_box": 73, "truncated": 9} | 44.6 / 46.3 | 0 / 0 |
| gemma-4-e4b-it | refexp_poly | 1024 | 0.00 / 0.00 | 93/100 {"malformed": 93} | 93/100 {"malformed": 93} | 71.5 / 71.5 | 0 / 0 |
| gemma-4-e4b-it | grid16 | 1024 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 272.0 / 272.0 | 0 / 0 |
| gemma-4-e4b-it | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 | 0 / 0 |
| gemma-4-e4b-it | ovd | 512 | 0.02 / 0.04 | 3/100 {"no_box": 1, "truncated": 2} | 5/100 {"no_box": 1, "truncated": 4} | 193.6 / 193.4 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| gemma-4-e4b-it | spatial_cvbench | 64 | 0.62 / 0.58 | 32/50 {"no_choice": 1, "truncated": 31} | 30/50 {"no_choice": 1, "truncated": 29} | 52.6 / 52.2 | 0 / 0 |
| gemma-4-e4b-it | spatial_embspatial | 64 | 0.86 / 0.86 | 43/50 {"truncated": 43} | 43/50 {"truncated": 43} | 61.2 / 61.2 | 0 / 0 |
| internvl3.5-4b-hf | refexp_box | 128 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 28.4 / 28.4 | 0 / 0 |
| internvl3.5-4b-hf | refexp_poly | 1024 | 0.04 / 0.06 | 5/100 {"out_of_range": 1, "truncated": 4} | 6/100 {"truncated": 6} | 117.7 / 135.1 | 0 / 0 |
| internvl3.5-4b-hf | grid16 | 1024 | 0.00 / 0.00 | 2/100 {"multiple_blocks": 2} | 1/100 {"multiple_blocks": 1} | 272.0 / 272.0 | 0 / 0 |
| internvl3.5-4b-hf | count | 64 | 0.00 / 0.00 | 1/49 {"multiple_numbers": 1} | 1/49 {"multiple_numbers": 1} | 7.6 / 7.6 | 0 / 0 |
| internvl3.5-4b-hf | ovd | 512 | 0.47 / 0.46 | 47/100 {"truncated": 47} | 46/100 {"truncated": 46} | 388.7 / 392.9 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| internvl3.5-4b-hf | spatial_cvbench | 64 | 0.70 / 0.70 | 35/50 {"truncated": 35} | 35/50 {"truncated": 35} | 53.8 / 53.9 | 0 / 0 |
| internvl3.5-4b-hf | spatial_embspatial | 64 | 0.28 / 0.28 | 17/50 {"multiple_choices": 3, "truncated": 14} | 17/50 {"multiple_choices": 3, "truncated": 14} | 34.6 / 34.4 | 0 / 0 |
| qwen3.5-9b | refexp_box | 128 | 0.02 / 0.02 | 2/100 {"truncated": 2} | 2/100 {"truncated": 2} | 46.9 / 47.0 | 0 / 0 |
| qwen3.5-9b | refexp_poly | 1024 | 0.01 / 0.01 | 100/100 {"malformed": 99, "truncated": 1} | 100/100 {"malformed": 99, "truncated": 1} | 50.0 / 49.7 | 0 / 0 |
| qwen3.5-9b | grid16 | 1024 | 0.00 / 0.00 | 14/100 {"wrong_row_count": 14} | 13/100 {"wrong_row_count": 13} | 274.4 / 274.2 | 0 / 0 |
| qwen3.5-9b | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 | 0 / 0 |
| qwen3.5-9b | ovd | 512 | 0.36 / 0.34 | 36/100 {"truncated": 36} | 34/100 {"truncated": 34} | 346.4 / 346.2 | 0 / 0 |
| qwen3.5-9b | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-9b | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-9b | spatial_cvbench | 64 | 0.86 / 0.88 | 43/50 {"truncated": 43} | 44/50 {"truncated": 44} | 63.3 / 63.3 | 0 / 0 |
| qwen3.5-9b | spatial_embspatial | 64 | 0.68 / 0.66 | 34/50 {"truncated": 34} | 33/50 {"truncated": 33} | 46.9 / 45.7 | 0 / 0 |

#### 两遍不一致的细目（只描述两遍 parsed 的差别）

| 模型 | 任务 | 一遍失败一遍成功 | 两遍都失败 | 两遍都成功但 parsed 不同 | 失败原因不同 | 生成 token 数相同 | 框/检测的细目 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 0 | 2 | 70/98 | 0 | 97/100 | 两框 IoU {"n": 70, "min": 0.0, "median": 0.9937, "max": 0.9988}；坐标最大绝对差 px {"n": 70, "min": 0.426, "median": 1.28, "max": 391.68}；IoU<0.9 的 3 条 |
| qwen3.5-0.8b | refexp_poly | 0 | 100 | 0/0 | 0 | 98/100 | — |
| qwen3.5-0.8b | grid16 | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-0.8b | count | 0 | 0 | 0/49 | 0 | 47/49 | — |
| qwen3.5-0.8b | ovd | 6 | 63 | 31/31 | 6 | 85/100 | 标签序列相同 22；框数相同 24；标签与框数都相同时逐框 IoU 的最小值 {"n": 22, "min": 0.7075, "median": 0.9851, "max": 0.9982} |
| qwen3.5-0.8b | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-0.8b | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| qwen3.5-0.8b | spatial_cvbench | 1 | 47 | 0/2 | 1 | 49/50 | — |
| qwen3.5-0.8b | spatial_embspatial | 2 | 46 | 0/2 | 2 | 48/50 | — |
| qwen3.5-2b | refexp_box | 0 | 1 | 71/99 | 0 | 100/100 | 两框 IoU {"n": 71, "min": 0.7975, "median": 0.9935, "max": 0.9989}；坐标最大绝对差 px {"n": 71, "min": 0.404, "median": 1.125, "max": 23.077}；IoU<0.9 的 2 条 |
| qwen3.5-2b | refexp_poly | 0 | 100 | 0/0 | 0 | 96/100 | — |
| qwen3.5-2b | grid16 | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-2b | count | 0 | 0 | 0/49 | 0 | 49/49 | — |
| qwen3.5-2b | ovd | 8 | 43 | 47/49 | 8 | 85/100 | 标签序列相同 42；框数相同 42；标签与框数都相同时逐框 IoU 的最小值 {"n": 42, "min": 0.0, "median": 0.9805, "max": 0.9983} |
| qwen3.5-2b | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-2b | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| qwen3.5-2b | spatial_cvbench | 1 | 46 | 0/3 | 1 | 49/50 | — |
| qwen3.5-2b | spatial_embspatial | 1 | 42 | 0/7 | 1 | 49/50 | — |
| gemma-4-e2b-it | refexp_box | 1 | 87 | 8/12 | 2 | 95/100 | 两框 IoU {"n": 8, "min": 0.384, "median": 0.9963, "max": 0.9987}；坐标最大绝对差 px {"n": 8, "min": 0.544, "median": 0.853, "max": 64.0}；IoU<0.9 的 1 条 |
| gemma-4-e2b-it | refexp_poly | 0 | 2 | 46/98 | 0 | 85/100 | — |
| gemma-4-e2b-it | grid16 | 0 | 100 | 0/0 | 0 | 96/100 | — |
| gemma-4-e2b-it | count | 0 | 0 | 1/49 | 0 | 49/49 | — |
| gemma-4-e2b-it | ovd | 2 | 34 | 53/64 | 2 | 81/100 | 标签序列相同 40；框数相同 42；标签与框数都相同时逐框 IoU 的最小值 {"n": 40, "min": 0.7563, "median": 0.9888, "max": 0.9989} |
| gemma-4-e2b-it | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| gemma-4-e2b-it | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| gemma-4-e2b-it | spatial_cvbench | 0 | 35 | 0/15 | 0 | 50/50 | — |
| gemma-4-e2b-it | spatial_embspatial | 0 | 49 | 0/1 | 0 | 50/50 | — |
| internvl3.5-2b-hf | refexp_box | 0 | 0 | 91/100 | 0 | 96/100 | 两框 IoU {"n": 91, "min": 0.3645, "median": 0.9776, "max": 0.9986}；坐标最大绝对差 px {"n": 91, "min": 0.375, "median": 3.416, "max": 114.56}；IoU<0.9 的 9 条 |
| internvl3.5-2b-hf | refexp_poly | 13 | 15 | 45/72 | 15 | 34/100 | — |
| internvl3.5-2b-hf | grid16 | 1 | 99 | 0/0 | 1 | 87/100 | — |
| internvl3.5-2b-hf | count | 1 | 14 | 0/34 | 2 | 47/49 | — |
| internvl3.5-2b-hf | ovd | 0 | 100 | 0/0 | 5 | 38/100 | 标签序列相同 0；框数相同 0；标签与框数都相同时逐框 IoU 的最小值 null |
| internvl3.5-2b-hf | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| internvl3.5-2b-hf | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| internvl3.5-2b-hf | spatial_cvbench | 2 | 43 | 0/5 | 2 | 48/50 | — |
| internvl3.5-2b-hf | spatial_embspatial | 3 | 28 | 2/19 | 3 | 46/50 | — |
| qwen3.5-4b | refexp_box | 0 | 3 | 67/97 | 0 | 99/100 | 两框 IoU {"n": 67, "min": 0.7705, "median": 0.9945, "max": 0.9988}；坐标最大绝对差 px {"n": 67, "min": 0.375, "median": 0.72, "max": 49.0}；IoU<0.9 的 2 条 |
| qwen3.5-4b | refexp_poly | 0 | 100 | 0/0 | 0 | 93/100 | — |
| qwen3.5-4b | grid16 | 1 | 0 | 0/99 | 1 | 99/100 | — |
| qwen3.5-4b | count | 0 | 0 | 0/49 | 0 | 49/49 | — |
| qwen3.5-4b | ovd | 1 | 29 | 69/70 | 1 | 75/100 | 标签序列相同 55；框数相同 57；标签与框数都相同时逐框 IoU 的最小值 {"n": 55, "min": 0.0, "median": 0.9667, "max": 0.9985} |
| qwen3.5-4b | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-4b | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| qwen3.5-4b | spatial_cvbench | 1 | 47 | 0/2 | 1 | 48/50 | — |
| qwen3.5-4b | spatial_embspatial | 6 | 30 | 0/14 | 6 | 44/50 | — |
| gemma-4-e4b-it | refexp_box | 1 | 82 | 7/17 | 1 | 86/100 | 两框 IoU {"n": 7, "min": 0.9858, "median": 0.9955, "max": 0.9985}；坐标最大绝对差 px {"n": 7, "min": 0.478, "median": 0.64, "max": 1.28}；IoU<0.9 的 0 条 |
| gemma-4-e4b-it | refexp_poly | 0 | 93 | 6/7 | 0 | 97/100 | — |
| gemma-4-e4b-it | grid16 | 0 | 0 | 3/100 | 0 | 100/100 | — |
| gemma-4-e4b-it | count | 0 | 0 | 2/49 | 0 | 49/49 | — |
| gemma-4-e4b-it | ovd | 2 | 3 | 79/95 | 2 | 82/100 | 标签序列相同 64；框数相同 65；标签与框数都相同时逐框 IoU 的最小值 {"n": 64, "min": 0.0747, "median": 0.9873, "max": 0.9984} |
| gemma-4-e4b-it | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| gemma-4-e4b-it | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| gemma-4-e4b-it | spatial_cvbench | 2 | 30 | 0/18 | 2 | 46/50 | — |
| gemma-4-e4b-it | spatial_embspatial | 0 | 43 | 0/7 | 0 | 49/50 | — |
| internvl3.5-4b-hf | refexp_box | 0 | 0 | 52/100 | 0 | 97/100 | 两框 IoU {"n": 52, "min": 0.5045, "median": 0.9905, "max": 0.9979}；坐标最大绝对差 px {"n": 52, "min": 0.426, "median": 1.92, "max": 99.2}；IoU<0.9 的 5 条 |
| internvl3.5-4b-hf | refexp_poly | 5 | 3 | 67/92 | 6 | 63/100 | — |
| internvl3.5-4b-hf | grid16 | 1 | 1 | 2/98 | 1 | 100/100 | — |
| internvl3.5-4b-hf | count | 0 | 1 | 0/48 | 0 | 48/49 | — |
| internvl3.5-4b-hf | ovd | 7 | 43 | 48/50 | 7 | 75/100 | 标签序列相同 33；框数相同 35；标签与框数都相同时逐框 IoU 的最小值 {"n": 33, "min": 0.4091, "median": 0.9279, "max": 0.9963} |
| internvl3.5-4b-hf | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| internvl3.5-4b-hf | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| internvl3.5-4b-hf | spatial_cvbench | 0 | 35 | 0/15 | 0 | 48/50 | — |
| internvl3.5-4b-hf | spatial_embspatial | 4 | 15 | 0/31 | 4 | 45/50 | — |
| qwen3.5-9b | refexp_box | 0 | 2 | 65/98 | 0 | 99/100 | 两框 IoU {"n": 65, "min": 0.7456, "median": 0.9955, "max": 0.9988}；坐标最大绝对差 px {"n": 65, "min": 0.335, "median": 0.956, "max": 12.296}；IoU<0.9 的 1 条 |
| qwen3.5-9b | refexp_poly | 0 | 100 | 0/0 | 0 | 98/100 | — |
| qwen3.5-9b | grid16 | 5 | 11 | 0/84 | 5 | 94/100 | — |
| qwen3.5-9b | count | 0 | 0 | 0/49 | 0 | 49/49 | — |
| qwen3.5-9b | ovd | 6 | 32 | 62/62 | 6 | 84/100 | 标签序列相同 52；框数相同 55；标签与框数都相同时逐框 IoU 的最小值 {"n": 52, "min": 0.0, "median": 0.975, "max": 0.9985} |
| qwen3.5-9b | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-9b | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| qwen3.5-9b | spatial_cvbench | 1 | 43 | 0/6 | 1 | 48/50 | — |
| qwen3.5-9b | spatial_embspatial | 1 | 33 | 0/16 | 1 | 49/50 | — |

#### 每遍的墙钟与生成 token（compute.json；墙钟只含 generate）

| 模型 | 任务 | 墙钟 s bs1 / 批量 | 生成 token bs1 / 批量 |
|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 58.0 / 7.6 | 4700 / 4699 |
| qwen3.5-0.8b | refexp_poly | 80.3 / 23.7 | 6631 / 6747 |
| qwen3.5-0.8b | grid16 | 1211.4 / 92.7 | 102400 / 102400 |
| qwen3.5-0.8b | count | 2.4 / 0.5 | 111 / 128 |
| qwen3.5-0.8b | ovd | 494.8 / 46.8 | 41608 / 41361 |
| qwen3.5-0.8b | rel_depth_nyu | 77.4 / 5.9 | 6400 / 6400 |
| qwen3.5-0.8b | rel_depth_blink | 9.3 / 0.9 | 768 / 768 |
| qwen3.5-0.8b | spatial_cvbench | 38.5 / 3.5 | 3189 / 3198 |
| qwen3.5-0.8b | spatial_embspatial | 36.2 / 3.5 | 2996 / 3081 |
| qwen3.5-0.8b | 合计 | 2008.4 / 185.2 | |
| qwen3.5-2b | refexp_box | 59.3 / 6.6 | 4672 / 4672 |
| qwen3.5-2b | refexp_poly | 55.1 / 9.9 | 4342 / 4234 |
| qwen3.5-2b | grid16 | 1253.5 / 94.8 | 101631 / 101631 |
| qwen3.5-2b | count | 2.5 / 0.4 | 104 / 104 |
| qwen3.5-2b | ovd | 444.0 / 47.7 | 36781 / 36573 |
| qwen3.5-2b | rel_depth_nyu | 79.1 / 6.2 | 6400 / 6400 |
| qwen3.5-2b | rel_depth_blink | 9.5 / 0.9 | 768 / 768 |
| qwen3.5-2b | spatial_cvbench | 39.0 / 3.6 | 3145 / 3118 |
| qwen3.5-2b | spatial_embspatial | 35.6 / 3.7 | 2858 / 2813 |
| qwen3.5-2b | 合计 | 1977.7 / 173.9 | |
| gemma-4-e2b-it | refexp_box | 80.8 / 17.4 | 4395 / 4379 |
| gemma-4-e2b-it | refexp_poly | 159.9 / 58.7 | 8448 / 8815 |
| gemma-4-e2b-it | grid16 | 452.1 / 35.7 | 24530 / 24470 |
| gemma-4-e2b-it | count | 2.9 / 0.7 | 103 / 103 |
| gemma-4-e2b-it | ovd | 291.0 / 47.5 | 15688 / 15900 |
| gemma-4-e2b-it | rel_depth_nyu | 117.1 / 9.1 | 6400 / 6400 |
| gemma-4-e2b-it | rel_depth_blink | 14.1 / 1.4 | 768 / 768 |
| gemma-4-e2b-it | spatial_cvbench | 51.7 / 5.4 | 2823 / 2823 |
| gemma-4-e2b-it | spatial_embspatial | 57.5 / 5.4 | 3172 / 3172 |
| gemma-4-e2b-it | 合计 | 1227.0 / 181.5 | |
| internvl3.5-2b-hf | refexp_box | 31.0 / 7.5 | 2836 / 2838 |
| internvl3.5-2b-hf | refexp_poly | 289.6 / 94.8 | 25482 / 26523 |
| internvl3.5-2b-hf | grid16 | 262.5 / 22.0 | 25218 / 25299 |
| internvl3.5-2b-hf | count | 15.1 / 2.8 | 1371 / 1425 |
| internvl3.5-2b-hf | ovd | 161.5 / 46.6 | 15941 / 15973 |
| internvl3.5-2b-hf | rel_depth_nyu | 66.3 / 8.2 | 6400 / 6400 |
| internvl3.5-2b-hf | rel_depth_blink | 7.9 / 2.1 | 768 / 768 |
| internvl3.5-2b-hf | spatial_cvbench | 32.0 / 4.9 | 3057 / 3084 |
| internvl3.5-2b-hf | spatial_embspatial | 24.1 / 2.9 | 2217 / 2130 |
| internvl3.5-2b-hf | 合计 | 890.0 / 191.7 | |
| qwen3.5-4b | refexp_box | 79.5 / 11.1 | 4800 / 4799 |
| qwen3.5-4b | refexp_poly | 68.6 / 10.4 | 4080 / 4113 |
| qwen3.5-4b | grid16 | 442.2 / 34.2 | 27234 / 27200 |
| qwen3.5-4b | count | 3.0 / 0.6 | 105 / 105 |
| qwen3.5-4b | ovd | 562.1 / 67.1 | 33919 / 33913 |
| qwen3.5-4b | rel_depth_nyu | 108.2 / 8.3 | 6400 / 6400 |
| qwen3.5-4b | rel_depth_blink | 12.8 / 1.2 | 768 / 768 |
| qwen3.5-4b | spatial_cvbench | 52.7 / 5.0 | 3161 / 3180 |
| qwen3.5-4b | spatial_embspatial | 40.8 / 5.1 | 2418 / 2208 |
| qwen3.5-4b | 合计 | 1369.8 / 143.0 | |
| gemma-4-e4b-it | refexp_box | 102.0 / 19.4 | 4462 / 4630 |
| gemma-4-e4b-it | refexp_poly | 165.3 / 24.8 | 7151 / 7150 |
| gemma-4-e4b-it | grid16 | 663.3 / 50.8 | 27200 / 27200 |
| gemma-4-e4b-it | count | 3.5 / 0.8 | 104 / 104 |
| gemma-4-e4b-it | ovd | 451.4 / 79.9 | 19363 / 19336 |
| gemma-4-e4b-it | rel_depth_nyu | 143.7 / 11.1 | 6400 / 6400 |
| gemma-4-e4b-it | rel_depth_blink | 17.2 / 1.7 | 768 / 768 |
| gemma-4-e4b-it | spatial_cvbench | 59.2 / 6.7 | 2630 / 2608 |
| gemma-4-e4b-it | spatial_embspatial | 68.8 / 6.9 | 3062 / 3061 |
| gemma-4-e4b-it | 合计 | 1674.4 / 202.1 | |
| internvl3.5-4b-hf | refexp_box | 38.6 / 5.9 | 2839 / 2842 |
| internvl3.5-4b-hf | refexp_poly | 175.7 / 117.5 | 11766 / 13509 |
| internvl3.5-4b-hf | grid16 | 363.7 / 38.3 | 27200 / 27200 |
| internvl3.5-4b-hf | count | 5.5 / 1.2 | 372 / 373 |
| internvl3.5-4b-hf | ovd | 507.5 / 72.0 | 38874 / 39290 |
| internvl3.5-4b-hf | rel_depth_nyu | 84.0 / 9.2 | 6400 / 6400 |
| internvl3.5-4b-hf | rel_depth_blink | 10.3 / 2.3 | 768 / 768 |
| internvl3.5-4b-hf | spatial_cvbench | 35.5 / 5.9 | 2692 / 2693 |
| internvl3.5-4b-hf | spatial_embspatial | 23.0 / 3.7 | 1728 / 1721 |
| internvl3.5-4b-hf | 合计 | 1243.8 / 256.0 | |
| qwen3.5-9b | refexp_box | 80.4 / 10.6 | 4694 / 4695 |
| qwen3.5-9b | refexp_poly | 84.7 / 34.2 | 4999 / 4972 |
| qwen3.5-9b | grid16 | 455.3 / 36.4 | 27442 / 27425 |
| qwen3.5-9b | count | 3.2 / 0.8 | 106 / 106 |
| qwen3.5-9b | ovd | 579.5 / 62.1 | 34644 / 34616 |
| qwen3.5-9b | rel_depth_nyu | 105.8 / 8.5 | 6400 / 6400 |
| qwen3.5-9b | rel_depth_blink | 12.7 / 1.3 | 768 / 768 |
| qwen3.5-9b | spatial_cvbench | 52.4 / 5.6 | 3167 / 3167 |
| qwen3.5-9b | spatial_embspatial | 39.2 / 5.0 | 2345 / 2285 |
| qwen3.5-9b | 合计 | 1413.3 / 164.4 | |

### 3.2 吞吐与算力估算

本节由 `experiments/E1/0012_estimate/probe_scripts/p16_estimate_summary.py` 从各 `estimate.json` 生成，摘自 `estimate_summary.md`。逐批大小的明细（样本/秒、token/秒、平均生成 token、截断率、峰值显存）在该文件的"逐批大小明细"表里，共 160 行，这里不重贴。

#### 估算汇总（主口径；推荐批大小 = 可行的批大小里样本/秒最高者；GPU 小时 = n / 样本每秒，由 run_e1 外推）

| 模型 | 任务 | 推荐批大小 | 样本/秒 | n(eval) | GPU h(eval) | 思考实测批大小 | 思考样本/秒 | n(think) | GPU h(think) | n(8192) | GPU h(8192, 上界) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 32 | 20.44 | 56857 | 0.773 | 8 | 4.52 | 200 | 0.012 | 50 | 0.006 |
| qwen3.5-0.8b | refexp_poly | 32 | 12.79 | 8000 | 0.174 | 8 | 0.14 | 200 | 0.383 | 50 | 0.192 |
| qwen3.5-0.8b | grid16 | 32 | 2.47 | 500 | 0.056 | 8 | 0.15 | 200 | 0.359 | 50 | 0.179 |
| qwen3.5-0.8b | count | 16 | 91.80 | 442 | 0.001 | 8 | 0.15 | 200 | 0.360 | 50 | 0.180 |
| qwen3.5-0.8b | ovd | 32 | 4.88 | 963 | 0.055 | 8 | 0.16 | 200 | 0.358 | 50 | 0.179 |
| qwen3.5-0.8b | rel_depth_nyu | 32 | 36.77 | 3270 | 0.025 | 8 | 0.16 | 200 | 0.340 | 50 | 0.170 |
| qwen3.5-0.8b | rel_depth_blink | 32 | 14.35 | 112 | 0.002 | 8 | 0.15 | 112 | 0.201 | 50 | 0.180 |
| qwen3.5-0.8b | spatial_cvbench | 32 | 21.56 | 2588 | 0.033 | 8 | 0.15 | 200 | 0.359 | 50 | 0.179 |
| qwen3.5-0.8b | spatial_embspatial | 32 | 28.25 | 3590 | 0.035 | 8 | 0.15 | 200 | 0.358 | 50 | 0.179 |
| qwen3.5-0.8b | **合计** | | | | **1.154** | | | | **2.731** | | **1.445** |
| qwen3.5-2b | refexp_box | 32 | 23.76 | 56857 | 0.665 | 8 | 3.79 | 200 | 0.015 | 50 | 0.007 |
| qwen3.5-2b | refexp_poly | 32 | 15.53 | 8000 | 0.143 | 8 | 0.15 | 200 | 0.374 | 50 | 0.187 |
| qwen3.5-2b | grid16 | 32 | 2.20 | 500 | 0.063 | 8 | 0.14 | 200 | 0.399 | 50 | 0.200 |
| qwen3.5-2b | count | 32 | 141.66 | 442 | 0.001 | 8 | 0.21 | 200 | 0.262 | 50 | 0.131 |
| qwen3.5-2b | ovd | 32 | 4.34 | 963 | 0.062 | 8 | 0.21 | 200 | 0.262 | 50 | 0.131 |
| qwen3.5-2b | rel_depth_nyu | 32 | 32.11 | 3270 | 0.028 | 8 | 0.16 | 200 | 0.357 | 50 | 0.178 |
| qwen3.5-2b | rel_depth_blink | 32 | 13.29 | 112 | 0.002 | 8 | 0.15 | 112 | 0.210 | 50 | 0.187 |
| qwen3.5-2b | spatial_cvbench | 32 | 25.55 | 2588 | 0.028 | 8 | 0.15 | 200 | 0.375 | 50 | 0.188 |
| qwen3.5-2b | spatial_embspatial | 32 | 25.54 | 3590 | 0.039 | 8 | 0.15 | 200 | 0.374 | 50 | 0.187 |
| qwen3.5-2b | **合计** | | | | **1.031** | | | | **2.628** | | **1.396** |
| gemma-4-e2b-it | refexp_box | 32 | 10.72 | 56857 | 1.474 | 8 | 0.52 | 200 | 0.107 | 50 | 0.054 |
| gemma-4-e2b-it | refexp_poly | 32 | 4.41 | 8000 | 0.504 | 8 | 1.84 | 200 | 0.030 | 50 | 0.015 |
| gemma-4-e2b-it | grid16 | 32 | 5.61 | 500 | 0.025 | 8 | 0.42 | 200 | 0.131 | 50 | 0.066 |
| gemma-4-e2b-it | count | 32 | 78.52 | 442 | 0.002 | 8 | 0.70 | 200 | 0.080 | 50 | 0.040 |
| gemma-4-e2b-it | ovd | 32 | 4.32 | 963 | 0.062 | 8 | 0.37 | 200 | 0.151 | 50 | 0.075 |
| gemma-4-e2b-it | rel_depth_nyu | 32 | 21.49 | 3270 | 0.042 | 8 | 0.43 | 200 | 0.129 | 50 | 0.064 |
| gemma-4-e2b-it | rel_depth_blink | 32 | 8.29 | 112 | 0.004 | 8 | 0.30 | 112 | 0.102 | 50 | 0.091 |
| gemma-4-e2b-it | spatial_cvbench | 32 | 16.14 | 2588 | 0.045 | 8 | 0.37 | 200 | 0.150 | 50 | 0.075 |
| gemma-4-e2b-it | spatial_embspatial | 32 | 16.51 | 3590 | 0.060 | 8 | 0.20 | 200 | 0.271 | 50 | 0.136 |
| gemma-4-e2b-it | **合计** | | | | **2.217** | | | | **1.152** | | **0.616** |
| internvl3.5-2b-hf | refexp_box | 32 | 30.51 | 56857 | 0.518 | 8 | 0.05 | 200 | 1.100 | 50 | 0.550 |
| internvl3.5-2b-hf | refexp_poly | 32 | 1.07 | 8000 | 2.068 | 8 | 0.13 | 200 | 0.425 | 50 | 0.213 |
| internvl3.5-2b-hf | grid16 | 32 | 8.98 | 500 | 0.015 | 8 | 0.13 | 200 | 0.418 | 50 | 0.209 |
| internvl3.5-2b-hf | count | 16 | 16.87 | 442 | 0.007 | 8 | 0.14 | 200 | 0.408 | 50 | 0.204 |
| internvl3.5-2b-hf | ovd | 32 | 2.83 | 963 | 0.094 | 8 | 0.06 | 200 | 0.880 | 50 | 0.440 |
| internvl3.5-2b-hf | rel_depth_nyu | 32 | 21.65 | 3270 | 0.042 | 8 | 0.19 | 200 | 0.296 | 50 | 0.148 |
| internvl3.5-2b-hf | rel_depth_blink | 32 | 17.14 | 112 | 0.002 | 8 | 0.19 | 112 | 0.166 | 50 | 0.148 |
| internvl3.5-2b-hf | spatial_cvbench | 16 | 16.25 | 2588 | 0.044 | 8 | 0.14 | 200 | 0.409 | 50 | 0.204 |
| internvl3.5-2b-hf | spatial_embspatial | 32 | 22.46 | 3590 | 0.044 | 8 | 0.14 | 200 | 0.411 | 50 | 0.205 |
| internvl3.5-2b-hf | **合计** | | | | **2.835** | | | | **4.514** | | **2.322** |

#### 每个模型的合计（GPU 小时，主口径）

| 模型 | eval | think | think8192（上界） | eval+think+think8192 |
|---|---|---|---|---|
| qwen3.5-0.8b | 1.154 | 2.731 | 1.445 | 5.330 |
| qwen3.5-2b | 1.031 | 2.628 | 1.396 | 5.056 |
| gemma-4-e2b-it | 2.217 | 1.152 | 0.616 | 3.986 |
| internvl3.5-2b-hf | 2.835 | 4.514 | 2.322 | 9.671 |
| 已测 4 个模型之和 | 7.238 | 11.025 | 5.779 | 24.042 |

未测的模型：qwen3.5-4b, gemma-4-e4b-it, internvl3.5-4b-hf, qwen3.5-9b

#### 附加口径（只跑 refexp_box、ovd；只外推 extra 子集）

| 模型 | 口径 | 任务 | 推荐批大小 | 样本/秒 | n(extra) | GPU h(extra) | 主口径同任务样本/秒 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | qwen:1024 | refexp_box | 32 | 20.84 | 500 | 0.007 | 20.44 |
| qwen3.5-0.8b | qwen:1024 | ovd | 32 | 3.23 | 100 | 0.009 | 4.88 |
| qwen3.5-0.8b | qwen:1024 | **合计** | | | | **0.015** | |
| internvl3.5-2b-hf | internvl:native | refexp_box | 32 | 10.18 | 500 | 0.014 | 30.51 |
| internvl3.5-2b-hf | internvl:native | ovd | 32 | 1.06 | 100 | 0.026 | 2.83 |
| internvl3.5-2b-hf | internvl:native | **合计** | | | | **0.040** | |

#### 开思考的实测（预算 4096；开发样本 ≤ 8 条）

| 模型 | 任务 | 批大小 | 样本数 | ok | 样本/秒 | 平均生成 token | 截断率 | 峰值显存 GB |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 8 | 8 | True | 4.52 | 118.2 | 0.000 | 2.23 |
| qwen3.5-0.8b | refexp_poly | 8 | 8 | True | 0.14 | 2265.8 | 0.500 | 2.71 |
| qwen3.5-0.8b | grid16 | 8 | 8 | True | 0.15 | 3253.5 | 0.750 | 2.71 |
| qwen3.5-0.8b | count | 8 | 8 | True | 0.15 | 1482.4 | 0.250 | 2.70 |
| qwen3.5-0.8b | ovd | 8 | 8 | True | 0.16 | 1788.1 | 0.250 | 2.74 |
| qwen3.5-0.8b | rel_depth_nyu | 8 | 8 | True | 0.16 | 2731.2 | 0.500 | 2.44 |
| qwen3.5-0.8b | rel_depth_blink | 8 | 8 | True | 0.15 | 1421.8 | 0.250 | 2.71 |
| qwen3.5-0.8b | spatial_cvbench | 8 | 8 | True | 0.15 | 2446.5 | 0.500 | 2.71 |
| qwen3.5-0.8b | spatial_embspatial | 8 | 8 | True | 0.15 | 1288.0 | 0.250 | 2.72 |
| qwen3.5-2b | refexp_box | 8 | 8 | True | 3.79 | 126.5 | 0.000 | 4.99 |
| qwen3.5-2b | refexp_poly | 8 | 8 | True | 0.15 | 2271.0 | 0.500 | 5.43 |
| qwen3.5-2b | grid16 | 8 | 8 | True | 0.14 | 3758.9 | 0.875 | 5.44 |
| qwen3.5-2b | count | 8 | 8 | True | 0.21 | 749.4 | 0.000 | 5.19 |
| qwen3.5-2b | ovd | 8 | 8 | True | 0.21 | 1488.1 | 0.000 | 5.25 |
| qwen3.5-2b | rel_depth_nyu | 8 | 8 | True | 0.16 | 3168.5 | 0.750 | 5.16 |
| qwen3.5-2b | rel_depth_blink | 8 | 8 | True | 0.15 | 1905.8 | 0.250 | 5.43 |
| qwen3.5-2b | spatial_cvbench | 8 | 8 | True | 0.15 | 1330.6 | 0.125 | 5.43 |
| qwen3.5-2b | spatial_embspatial | 8 | 8 | True | 0.15 | 1337.2 | 0.250 | 5.44 |
| gemma-4-e2b-it | refexp_box | 8 | 8 | True | 0.52 | 560.4 | 0.000 | 10.92 |
| gemma-4-e2b-it | refexp_poly | 8 | 8 | True | 1.84 | 180.5 | 0.000 | 10.93 |
| gemma-4-e2b-it | grid16 | 8 | 8 | True | 0.42 | 676.1 | 0.000 | 10.93 |
| gemma-4-e2b-it | count | 8 | 8 | True | 0.70 | 323.8 | 0.000 | 10.92 |
| gemma-4-e2b-it | ovd | 8 | 8 | True | 0.37 | 412.1 | 0.000 | 10.95 |
| gemma-4-e2b-it | rel_depth_nyu | 8 | 8 | True | 0.43 | 583.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | rel_depth_blink | 8 | 8 | True | 0.30 | 707.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | spatial_cvbench | 8 | 8 | True | 0.37 | 497.1 | 0.000 | 10.92 |
| gemma-4-e2b-it | spatial_embspatial | 8 | 8 | True | 0.20 | 755.8 | 0.000 | 10.93 |
| internvl3.5-2b-hf | refexp_box | 8 | 8 | True | 0.05 | 3104.4 | 0.625 | 9.18 |
| internvl3.5-2b-hf | refexp_poly | 8 | 8 | True | 0.13 | 3443.6 | 0.625 | 9.24 |
| internvl3.5-2b-hf | grid16 | 8 | 8 | True | 0.13 | 4096.0 | 1.000 | 9.25 |
| internvl3.5-2b-hf | count | 8 | 8 | True | 0.14 | 3153.8 | 0.750 | 9.18 |
| internvl3.5-2b-hf | ovd | 8 | 8 | True | 0.06 | 3331.1 | 0.750 | 9.18 |
| internvl3.5-2b-hf | rel_depth_nyu | 8 | 8 | True | 0.19 | 4096.0 | 1.000 | 8.98 |
| internvl3.5-2b-hf | rel_depth_blink | 8 | 8 | True | 0.19 | 4096.0 | 1.000 | 8.98 |
| internvl3.5-2b-hf | spatial_cvbench | 8 | 8 | True | 0.14 | 3180.6 | 0.750 | 9.21 |
| internvl3.5-2b-hf | spatial_embspatial | 8 | 8 | True | 0.14 | 4096.0 | 1.000 | 9.25 |

## 4. 与任务书假设不符的发现

- **用时超出预计，估算只完成一半。** 任务书预计合计 3–4 GPU 小时，但：
  - 补跑之后验收是 8 个模型，单验收就用了 3.82 h；
  - 估算在第 4 个模型结束时累计到 6.17 h，按 §5 停止，推回已完成的部分。
  - 未完成的有：qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b 的主口径估算，以及 qwen3.5-9b 的 qwen:1024、internvl3.5-4b-hf 的 internvl:native。
  - 执行顺序是我定的：先做完 §3（含补跑），再做 §4。任务书没有规定补跑与估算的先后。
  - 验收中最耗时的是 batch=1 那遍的长输出任务。例如 qwen3.5-0.8b 的 grid16 那遍 1211 s，100 条全部写满 1024 token；ovd 那遍 495 s。逐任务的墙钟见 §3.1 最后一张表。
- **多选与点对任务被 64 token 的上限截断。**
  - rel_depth_nyu、rel_depth_blink：8 个模型两遍都 100% 截断。抽看了 qwen3.5-0.8b、qwen3.5-9b、internvl3.5-4b-hf、gemma-4-e4b-it 这 4 个模型的 raw_text，开头都是先描述两点的位置，例如 "Looking at the image: - Point **A** is circled on …"，写到 64 token 时还没给出选项。
  - spatial_cvbench、spatial_embspatial 的截断率也很高，见 §3.1 的截断表。
  - 任务书说"按任务的截断率与解析失败分布，是调整 max_new_tokens 的依据"，相应数字都在 §3.1 的表里。
- **两遍都失败算一致（D32-2）。** 两遍都 100% 解析失败的 24 格，一致率都是 1.0，配对差都是 0。
- **refexp_box 的不一致主要是坐标上的小差。** 8 个模型都没有通过。不一致的条目中，两遍的框大多在原生 0–1000 坐标上相差 1 个单位，例如 `[458, 228, 894, 988]` 对 `[458, 227, 894, 988]`。两框 IoU 的中位数、坐标最大差的中位数与最大值，以及 IoU < 0.9 的条数，逐模型见 §3.1 的"不一致的细目"表；这一判据要求逐位相等（D32-2）。
- **refexp_box 的 eval 句数与预注册不一致。** `src/e1/tasks.select` 数出的 eval n 是 56,857，等于 D30-4 各划分之和：
  - RefCOCO：10,634 + 5,657 + 5,095；
  - RefCOCO+：10,558 + 5,726 + 4,889；
  - RefCOCOg：4,696 + 9,602。
  - PREREGISTRATION.md 的 E1 草稿 §4 写的是 57,457 句。
- **推荐批大小大多落在测试范围的上端。** 37/40 格的推荐批大小是 32，也就是 `--batch-sizes 1 8 16 32` 里最大的那个。
- **开思考的实测样本很少。** 每个任务只有 8 条，批大小为 min(推荐批大小, 8) = 8。按 `src/e1/estimate.py`，think8192 是由 4096 的实测按"每条耗时 × 2"算出的上界，没有实测。
- **检测的开发子集没有 GT（D32-2）。** 检测的验收只比一致率与 raw_text，指标差记为 null；估算用的开发图是 RefCOCO 的图。

## 5. 生成的文件

- `experiments/E1/0012_acceptance/`（23 MB，1,190 个文件，第一批 `db215b8` 推送）：
  - 8 个模型的 `acceptance/`，含 `acceptance.json`、`acceptance.md`，以及 `bs1/`、`bs16/` 两遍的全部输出；
  - `accept_summary.md/json`、`README.md`、`pip_dryrun.txt`、`logs/`、`probe_scripts/`。
  - 本批又更新了 `logs/driver.log`，新增了 `disk_ledger.tsv`。
- `experiments/E1/0012_estimate/`（本批推送）：
  - 4 个模型的 `estimate/`，另有 2 个附加口径目录；
  - `estimate_summary.md/json`、`README.md`、`disk_ledger.tsv`、`logs/`、`probe_scripts/`。

## 6. 磁盘与内存盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 08:33:58 | 0012 start | 39.352 | 3.058 | 3.860 |
| 08:34:17 | before download qwen3.5-0.8b | 39.352 | 3.058 | 3.860 |
| 08:34:18 | after download qwen3.5-0.8b | 39.352 | 3.058 | 3.860 |
| 09:11:25 | after accept qwen3.5-0.8b | 39.341 | 3.058 | 3.869 |
| 09:11:25 | before download gemma-4-e2b-it | 39.341 | 3.058 | 3.869 |
| 09:11:49 | after download gemma-4-e2b-it | 39.341 | 13.337 | 3.869 |
| 09:35:50 | after accept gemma-4-e2b-it | 39.337 | 13.337 | 3.871 |
| 09:35:50 | after delete gemma-4-e2b-it | 39.337 | 3.058 | 3.871 |
| 09:35:50 | before download internvl3.5-2b-hf | 39.337 | 3.058 | 3.871 |
| 09:36:04 | after download internvl3.5-2b-hf | 39.337 | 7.773 | 3.871 |
| 09:54:30 | after accept internvl3.5-2b-hf | 39.329 | 7.773 | 3.874 |
| 09:54:31 | after delete internvl3.5-2b-hf | 39.329 | 3.058 | 3.874 |
| 09:54:31 | before download qwen3.5-9b | 39.329 | 3.058 | 3.874 |
| 09:55:13 | after download qwen3.5-9b | 39.329 | 22.376 | 3.874 |
| 10:22:03 | after accept qwen3.5-9b | 39.318 | 22.376 | 3.883 |
| 10:22:04 | after delete qwen3.5-9b | 39.318 | 3.059 | 3.883 |
| 10:22:13 | before download qwen3.5-2b | 39.318 | 3.059 | 3.883 |
| 10:22:31 | after download qwen3.5-2b | 39.318 | 7.618 | 3.883 |
| 10:58:50 | after accept qwen3.5-2b | 39.314 | 7.618 | 3.885 |
| 10:58:50 | after delete qwen3.5-2b | 39.314 | 3.059 | 3.885 |
| 10:58:50 | before download qwen3.5-4b | 39.314 | 3.059 | 3.885 |
| 10:59:13 | after download qwen3.5-4b | 39.314 | 12.389 | 3.885 |
| 11:24:54 | after accept qwen3.5-4b | 39.311 | 12.389 | 3.888 |
| 11:24:54 | after delete qwen3.5-4b | 39.311 | 3.059 | 3.888 |
| 11:24:55 | before download gemma-4-e4b-it | 39.311 | 3.059 | 3.888 |
| 11:25:36 | after download gemma-4-e4b-it | 39.311 | 19.085 | 3.888 |
| 11:57:23 | after accept gemma-4-e4b-it | 39.307 | 19.085 | 3.890 |
| 11:57:23 | after delete gemma-4-e4b-it | 39.307 | 3.060 | 3.890 |
| 11:57:23 | before download internvl3.5-4b-hf | 39.307 | 3.060 | 3.890 |
| 11:57:47 | after download internvl3.5-4b-hf | 39.307 | 12.544 | 3.890 |
| 12:23:11 | after accept internvl3.5-4b-hf | 39.304 | 12.544 | 3.893 |
| 12:23:12 | after delete internvl3.5-4b-hf | 39.304 | 3.061 | 3.893 |
| 12:23:12 | before download qwen3.5-0.8b (estimate) | 39.304 | 3.061 | 3.893 |
| 12:23:13 | after download qwen3.5-0.8b (estimate) | 39.304 | 3.061 | 3.893 |
| 13:05:49 | after estimate qwen3.5-0.8b | 39.272 | 3.061 | 3.908 |
| 13:05:49 | before download qwen3.5-2b (estimate) | 39.272 | 3.061 | 3.908 |
| 13:06:02 | after download qwen3.5-2b (estimate) | 39.272 | 7.619 | 3.908 |
| 13:42:43 | after estimate qwen3.5-2b | 39.272 | 7.619 | 3.908 |
| 13:42:44 | after delete qwen3.5-2b (estimate) | 39.272 | 3.061 | 3.908 |
| 13:42:44 | before download gemma-4-e2b-it (estimate) | 39.272 | 3.061 | 3.908 |
| 13:43:26 | after download gemma-4-e2b-it (estimate) | 39.272 | 13.340 | 3.908 |
| 14:07:23 | after estimate gemma-4-e2b-it | 39.271 | 13.340 | 3.908 |
| 14:07:24 | after delete gemma-4-e2b-it (estimate) | 39.271 | 3.062 | 3.908 |
| 14:07:24 | before download internvl3.5-2b-hf (estimate) | 39.271 | 3.062 | 3.908 |
| 14:07:49 | after download internvl3.5-2b-hf (estimate) | 39.271 | 7.776 | 3.908 |
| 14:44:40 | after estimate internvl3.5-2b-hf | 39.270 | 7.776 | 3.908 |
| 14:44:41 | after delete internvl3.5-2b-hf (estimate) | 39.270 | 3.062 | 3.908 |
| 14:46:40 | 0012 end | 39.270 | 3.062 | 3.908 |

- **根盘：** 可用空间从 39.352 GB 降到 39.270 GB（−82 MB），全程远高于 3 GB 红线。根盘与其他用户共用，变化不全是本项目造成的。
- **`$PROJ`：** 从 3.860 GB 升到 3.908 GB（+48 MB），在 15 GB 份额以内（D28-8）。增量主要是 `experiments/E1/0012_acceptance` 的 23 MB 和 `0012_estimate`。
- **内存盘：** 峰值 22.376 GB，出现在 qwen3.5-9b 验收时，低于 25 GB 上限。
  - 结束时 `/dev/shm/visint` 是 3.062 GB：`hf-hub` 1.770 GB（只剩 Qwen3.5-0.8B）、NYU 点对渲染缓存 1.277 GB、`xet` 0.015 GB。
  - `TMPDIR=/dev/shm/visint/tmp` 下没有留下新文件。
