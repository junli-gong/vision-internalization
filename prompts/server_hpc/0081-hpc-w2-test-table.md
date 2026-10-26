[RUN-TASK 0081] 只用 CPU、几分钟：在 HPC 上跑 W2 test 的汇总脚本，补齐 W2-a 的配对差（W2 预注册 §11）。现在就可以发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不用 GPU、不写暂停文件；不碰在跑的 RUN-0076、0078、0080。

## 为什么

- W2-a 的 `predictions.jsonl` 约 10 MB，同步循环没有推回仓库，管理员在开发机上算不了 W2-a 的配对差。
- 本任务只读 `$VI/repo` 里 RUN-0072、0073 的结果目录，不跑模型。

## 文件（工作树 wt0081 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `scripts/mgr/w2_test_table.py` | `1f085a7c7f1068cdd3a81bc91bbd0dd23345df291f872f4e80f6332dff6a7b02` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0081（稀疏），建在 origin/main，核对 sha256。
2. 在 07 上用 `srun --jobid=157297 --overlap -n1 -c 1` 跑，几秒到一分钟：
   ```bash
   cd $VI/wt/wt0081 && env PATH=$VI/env/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0 \
     python scripts/mgr/w2_test_table.py --root $VI/repo --out $VI/repo/experiments/W2/0081_w2_test_table/w2_test_table
   ```
3. 先 `--check` 一次（同一条命令加 `--check`），应输出 `check same`。
4. 用 `sync_push.sh` 的 ONCE=1 推 `experiments/W2/0081_w2_test_table`，里面只有 .json 与 .md。
5. 再跑一次 `squeue`。

## 报告

推送 `reports/0081-hpc-w2-test-table.md`：命令、退出码；`w2_test_table.md` 的原文；表里还有没有"缺逐句数据"的格子。只贴数字，不解读。
