[RUN-TASK 0064] HPC，只用 CPU：W1 错误分析，重放 RUN-0063 开发-检查集的每条样本并分类（D113-4；D114）。约 10–30 分钟，不占 GPU、不让路

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0056、07 上的 RUN-0057；不写任何暂停文件。
- 不往 `$VI/repo` 里 pull。

## 为什么

- 负责人要求 10-24 前把 W1 开发-检查集的失败样本分为四类，用于设计 W2 的候选生成器：种子点放错、修改把好的结果改坏、没有提交、贴边算法本身的问题。
- W1 的工作台逐位确定，所以可以按记录的调用序列重放，得到每一步的 mask。
- 只读 RUN-0063 的结果，只用 CPU。

## 文件（commit `7ab1e1d1` 或更新；工作树 wt0064 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `src/w1/replay.py` | `757e898cb43e129ac31a7008b2afe054e7b4ef620b649e0283af4200e066b0da` |
| `scripts/w1_replay.py` | `331f76dc0e62a3d58a817ee77f8f82d3cb6132d1360e06d9e44df13c840eca3c` |
| `src/w1/workbench.py` | `775e0dca6901662ddd2e1880c01dbf848efaad73127f6d594308fcf797d46591`（与 RUN-0063 相同） |
| `src/w1/sandbox.py` | `13ed14dc0fe0e9b52b953d66e6038a397d77b8920af0a4f2c382eadf52e87cc2` |

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0064（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试，不在旧的 origin/main 上建。核对上表 4 个 sha256。
3. 在 07 上经 srun 跑（只用 CPU，不碰 GPU；`-c 8` 被拒就去掉，改用 `--workers 1`）：

   ```bash
   VI=/home/mail/chenyuma/visint
   ssh -n hpcgpu07 "srun --jobid=157297 --overlap -n1 -c 8 env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=$VI/tmp PYTHONHASHSEED=0 \
       PATH=$VI/env/bin:/usr/bin:/bin \
       bash -c 'cd $VI/wt/wt0064 && $VI/env/bin/python scripts/w1_replay.py \
         --run $VI/repo/experiments/W1/0063_w1_devcheck_hpcgpu07 --data-root $VI/b200/data \
         --out $VI/repo/experiments/W1/0064_w1_error_analysis --workers 8 --classify' \
       > $VI/logs/replay_0064.log 2>&1 < /dev/null"
   ```

   - 前台跑，约 10–30 分钟。日志在 `$VI/logs/replay_0064.log`。
4. 跑完后在 hpcmgt 上用 `sync_push.sh` 推一次（`ONCE=1`，TAG 0064，路径 `experiments/W1/0064_w1_error_analysis`）。不起常驻循环。
5. 删掉工作树 wt0064，再跑一次 `squeue`。

## 报告

推送 `reports/0064-hpc-w1-error-analysis.md`，内容：
- 命令、退出码、墙钟；
- `replay_meta.json` 里不一致的条数与 id；
- `error_analysis.md` 原文；
- 任何报错的 traceback。

只贴数字，不解读。
