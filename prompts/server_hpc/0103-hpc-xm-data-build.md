[RUN-TASK 0103] X-M 的两份清单、训练目标与长度统计（D165、D167）。只用 CPU；排在 RUN-0102 的数据落地之后；约 1 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 第 2 步借 06 或 07 的 CPU（`srun --overlap`），不占 GPU，不碰两台节点上在跑的任何东西。

## 为什么

- DEV-0079 已合入（D167）。这一步在 HPC 上用真数据生成：
  - `xm_splits.json`：三个 COCO 划分、LVIS 子集、ReasonSeg val；
  - `xm_train_exclusions.json`：去重清单。
- 两份清单由管理员审过后再提交到 `configs/`。去重清单与哈希要写进预注册 XM-2a，早于任何训练（D165 第二节）。
- 再生成 COCO 训练目标（框列表与每实例样条）与长度统计，给可行性实测与 DEV-0081 的训练样本用。
- **判定集与最终报告集**：这里只读真值（划分、统计），不跑任何系统。

## 文件（工作树 wt0103 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `scripts/xm/build_configs.py` | `41a10c1f0c4850dab73777c0da785c1cb2c11f753ad45f9444d608aba0b4cc50` |
| `scripts/xm/build_targets.py` | `aef0fdda700a1090570f6ae52741061ed835b8afc1253b1902b543afd78bfb04` |
| `scripts/xm/data_stats.py` | `28d21f0779dbfeef5dcb90a81e2a81acfc1b8deb93349a367f1cfc48cbcd72a4` |
| `src/xm/data.py` | `f51759702f8dc5b67c4775859df97d64334fb0b9df03e5c7a430c01ba7a73870` |
| `src/xm/exclusions.py` | `01e87ed4bc01187da660e48c2f15c721ad912e8df5ae732c9bd243f117ec37c0` |
| `src/xm/targets.py` | `c88d94a7d7530bf1cb7040e1160c372f71ae551700fcf6d31e15a47f7a9d3eae` |
| `src/explore/oracle_sim.py` | `5ada0675c52384b67c2c463deacde7c9209cff057ee847eb0424259c3499cbbd` |
| `src/explore/guidance.py` | `40c52ebc0cd6eabe0e6d51c9f6906530fb8fb4a0c5a63b8cef5d912de7ee5ef5` |

## 做法

先设好：`VI=/home/mail/chenyuma/visint`、`export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1`，`cd $VI/wt/wt0103`。

1. **两份清单**（hpcmgt，nice，约 10–20 分钟；大头是 train2017 逐张算 sha256）：
   ```bash
   nice -n 10 $VI/env/bin/python scripts/xm/build_configs.py --data-root $VI/data --refcoco-root $VI/b200/data \
     --out $VI/repo/experiments/XM/0103_xm_configs --workers 12
   ```
   - 贴两份清单的摘要：每份的对数、图数、sha256，以及去重各类的条数与和 train2017 的交集大小。
   - 推回仓库，管理员审。
2. **训练目标**（在 07 上借 CPU：`srun --jobid=157297 --overlap -n1 -c 14 … < /dev/null`）：
   - 先试 2,000 个实例量吞吐：
     ```bash
     $VI/env/bin/python scripts/xm/build_targets.py --data-root $VI/data \
       --splits $VI/repo/experiments/XM/0103_xm_configs/xm_splits.json \
       --exclusions $VI/repo/experiments/XM/0103_xm_configs/xm_train_exclusions.json \
       --out $VI/repo/experiments/XM/0103_xm_targets_probe --limit 2000 --workers 14
     ```
   - 再全量：同一条命令，`--out $VI/repo/experiments/XM/0103_xm_targets`，去掉 `--limit`。估计 10–20 分钟，内存 6–8 GB，输出 0.5–1 GB。
   - 输出较大的部分只留在 HPC，记进 `LARGE_FILES.manifest.tsv`，照同步循环的大小上限处理。
3. **统计**（hpcmgt，约 10 分钟；分词器离线读 qwen3.5-9b）：
   ```bash
   nice -n 10 $VI/env/bin/python scripts/xm/data_stats.py --data-root $VI/data \
     --splits $VI/repo/experiments/XM/0103_xm_configs/xm_splits.json \
     --exclusions $VI/repo/experiments/XM/0103_xm_configs/xm_train_exclusions.json \
     --targets $VI/repo/experiments/XM/0103_xm_targets --out $VI/repo/experiments/XM/0103_xm_data_stats --workers 12
   ```
   - 贴 md 全文。
4. 写 `reports/0103-hpc-xm-data-build.md`，再跑一次 `squeue`。
