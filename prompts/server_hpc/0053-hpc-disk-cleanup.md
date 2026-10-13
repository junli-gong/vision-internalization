[RUN-TASK 0053] 共享盘清理第一步：没在运行的工作树改成稀疏签出（不含 experiments/）、清掉测试临时目录、git gc（D94）。只用 CPU，不碰训练驱动

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**

- 前后各跑一次 `squeue`。
- 本任务不碰任何训练或评测驱动；它们正在用的工作树（wt0045、wt0050）一个都不动。

## 为什么

- 份额差值到了 68.4 GB（上限 80 GB）。D94 要求现在就清理，降回 60 GB 以下。
- 管理员在开发机上估算：
  - 每个完整的工作树约 3.4 GB，几乎全是 `experiments/` 下的历史结果；
  - HPC 上有 7 个工作树（push、sync、wt0045、wt0049、wt0050、wt0051、wt0052），约 24 GB，是份额的大头；
  - git 对象（`count-objects`）只有约 0.7 GB。
- 代码用的工作树不需要 `experiments/`，**唯一的例外是 `experiments/P0/`**：
  - `src/data/train_dedupe.py` 在运行时读 `experiments/P0/0001_env_probe/raw/p6/A/coco_val2017/coco_val2017_coco_urls.txt`，用来做训练数据去重；
  - 这个文件不在时，代码不报错，只是悄悄少掉 COCO val2017 那一组，训练数据就变了；
  - 所以稀疏规则必须保留 `experiments/P0/`（约 79 MB）。管理员在开发机上验证过：规则下 P0 在，sha256 与钉住的一致（`60267210…`），`git status` 干净，工作树 91 MB。
- 推送工作树 `wt/push` 正被 0045、0050 的 v2 循环用着，这一步不动。推送循环 v3 不用工作树，下一步上线后再删 push。

## 做法（在 hpcmgt 上）

1. `squeue`，两条 RUNNING。记一次份额差值：`timeout 60 lfs quota -u chenyuma /home | sed -n 3p`，第 2 列减 7,059,403,936 KB。
2. **改成稀疏签出**：只改这四个，它们现在都没有进程在用。
   - 先用 `pgrep -af` 确认没有进程的命令行里有这个路径；
   - 再确认 `git -C <工作树> status --short` 为空。

   | 工作树 | 用途 |
   |---|---|
   | `$VI/wt/wt0049` | RUN-0049 的 06 那一次要用 |
   | `$VI/wt/wt0051` | E5，等 RUN-0050 跑完启动 |
   | `$VI/wt/wt0052` | 4B，等 RUN-0045 的 06 队列跑完启动 |
   | `$VI/wt/sync` | 只放推送脚本；正在跑的 v2 循环只读 `scripts/server/sync_push.sh`，那个文件留着 |

   对每一个：

   ```bash
   W=<工作树>
   git -C $W sparse-checkout set --no-cone '/*' '!/experiments/*/' '/experiments/P0/'
   ls $W/experiments/                     # 应只剩 P0
   sha256sum $W/experiments/P0/0001_env_probe/raw/p6/A/coco_val2017/coco_val2017_coco_urls.txt   # 应以 60267210 开头
   git -C $W status --short               # 应为空
   ```

   - 照任务书再核对一次 sha256：`wt0051` 的 `queue.sh`；`wt0052` 的 `run.sh`、`eval.sh`、`queue.sh`；`wt/sync` 的 `sync_push.sh`。应与原值相同，因为这些文件不在被删的范围里。
3. **清掉测试与测量留下的临时目录**：都在 `$VI/tmp` 下，可以重新生成。
   - 先 `ls -la $VI/tmp` 把清单记进报告。
   - 删掉这些：pytest 的 basetemp（`pt_*`、`pytest-of-*`、`tests_r*` 一类）、RUN-0044 的 `pt_failtest`、`msp_*.json` 以外的测量残留。
   - 不确定的不删，写进报告，问管理员。
   - **正在跑的驱动用 `$VI/tmp` 当 TMPDIR**，修改时间在最近 2 小时内的文件一律不动。
4. **git gc**（拿 `git.lock`，与推送循环互斥）：

   ```bash
   flock $VI/logs/git.lock -c 'git -C $VI/repo gc --prune=now'
   git -C $VI/repo count-objects -vH
   ```

   `count-objects` 报的 garbage（`tmp_pack_*`）由 gc 一并清掉。
5. 再记一次份额差值，并核对：
   - `squeue` 两条都是 RUNNING；
   - 两台驱动的心跳在更新；
   - 0045、0050 推送循环的下一轮是 ok。
6. 推送 `reports/0053-hpc-disk-cleanup.md`，只记事实：
   - 清理前后的份额差值；
   - 每个工作树稀疏前后的检查结果；
   - 删掉的 tmp 清单与大小；
   - gc 前后的 `count-objects`。

## 之后（不在本任务里）

- wt0045、wt0050 的驱动跑完后（约 10-14 上午），按原任务书删掉，每个约 3.4 GB。
- 推送循环 v3（D94）：
  - 不用工作树，直接从 `$VI/repo` 读文件，在单独的索引文件里拼提交；
  - 大的结果文件不进 git，用清单（路径、字节数、sha256）管理。
  - 管理员写好、彩排完另发。上线后删 `wt/push`。
