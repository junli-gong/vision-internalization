[RUN-TASK 0060] hpcgpu07：W1 冒烟测试与单位用时实测（D103、D107；W1 预算内）。9B 完整工作台跑开发-调参集 32 句，四种基线跑同样的句子，定位与计数各 16 条，演练一次去留计算。上限 3 小时。**排在 RUN-0054 跑完之后、RUN-0057 之前**

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0056。
- 只用 07 自己的暂停文件：脚本的 `claim_gpu` 写入，退出时删掉。

## 什么时候开始与顺序

1. 等 RUN-0054 全部跑完：`$VI/logs/run_0054.log` 有 `done`，07 上没有 0054 的进程，GPU 空。
2. **先跑本任务**，跑完（心跳 `done 0060`）再按原计划启动 RUN-0057。

RUN-0057 因此晚开约 2–3 小时，仍在 10-25 之前跑完。

## 为什么

- W1 的三个开发任务都已合入（D104、D106、D107），HPC 上的测试也全过（RUN-0059）。
- 但回路还没在真的 9B 上跑过。开发-调参（10-23 起）之前，先查清三件事：
  - 能不能跑通；
  - 每句要多久、显存多少；
  - 批参数是否合适：`batch_size 16`、`max_batch_tokens 48000`。
- 只用开发-调参集（devtune），**不碰开发-检查集**（devcheck）。devcheck 留给去留判定。

## 文件（commit `08ba07cd` 或更新；工作树 wt0060 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0060-hpc-07-w1-smoke/smoke.sh` | `be100d15f99ac8a6fb9842ebfa40536d95920b590608a095bc9c7668ab65c855` |
| `configs/w1.json` | `6a4704590d47abf76b92ac566b38ac522cb735f2d3298ee7334db3b72eed7c19`（协议初稿，`frozen: false`） |
| `configs/w1_splits.json` | `a1371851555e854fac6f9d0e267f19a23602ee9cf31d5c303ef08432d1f7f0f7` |
| `scripts/run_w1.py` | `bacb38e46b7e4dbe8b6872a4a33a911b4910150f189d75521493653978f78535` |
| `scripts/w1_gate.py` | `0826f7822886173913fad93db86be7901f752c384a2eae1b98d58ce24c83c208` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |

开发机 tiny 彩排：9 步都 rc=0。

## 内容（全部在 smoke.sh 里）

| 步 | 内容 |
|---|---|
| `full_refseg_devtune32` | 9B 完整工作台（W1-1），refseg，devtune 前 32 句 |
| `baseline_e1_refexp_box_devtune32` | W1-0 的定位，"自己的框"要用 |
| `baseline_ownbox_devtune32` | 自己的框光栅化，只用 CPU |
| `baseline_poly_structured_devtune32` | E1 结构强制多边形 |
| `w1f_devtune32` | 自己的框交给 GrabCut，只用 CPU |
| `full_refexp_box_devtune16` | 完整工作台的定位 |
| `full_count_devtune_countbench16` | 完整工作台的计数 |
| `gate_rehearsal` | `w1_gate.py` 演练。devtune 不是判定点，结论标"彩排，不作数" |

全程每秒采样一次显存，写 `<OUT>/gpu_mem.csv`。结果目录 `experiments/W1/0060_w1_smoke_hpcgpu07/`。

## 做法

1. hpcmgt 上 `squeue`；确认"什么时候开始"的条件。
2. 建工作树 wt0060（稀疏，同 RUN-0054 的规则），建在 origin/main。核对上表 6 个文件的 sha256。
3. 启动（方案 B）：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=3 \
     bash /home/mail/chenyuma/visint/wt/wt0060/prompts/server_hpc/0060-hpc-07-w1-smoke/smoke.sh \
     > /home/mail/chenyuma/visint/logs/eval_0060.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳 `$VI/logs/eval_0060.status`。
   - 另起一个 v3 推送循环，TAG 0060，路径 `experiments/W1/0060_w1_smoke_hpcgpu07`。跑完、推完后按 PID 停掉。
4. 第一步跑完时先报一次：
   - 墙钟；
   - 32 句里完成、各失败码的条数；
   - 轮数与工具调用次数的分布；
   - `gpu_mem.csv` 的峰值。
   - 显存不够就照实报告，不改批参数重试。
5. 全部跑完：报告，然后按原计划启动 RUN-0057。

## 报告

推送 `reports/0060-hpc-07-w1-smoke.md`：
- `steps.log` 原文；
- 每步的墙钟；
- 回路各步的统计：完成数、失败码分布、轮数与工具调用次数的分布、每句的生成 token 数中位与最大、子批数；
- 显存峰值；
- `gate_rehearsal.md` 原文；
- 任何报错的 traceback。

devtune 的分数可以照贴，这是调参集。
