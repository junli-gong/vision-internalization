[RUN-TASK 0053-addendum-1] 共享盘清理第二步：推送循环换成 v3（不用工作树；大文件不进 git，用清单管理），然后删掉 wt/push（D94）

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。** 前后各跑一次 `squeue`。不碰任何训练或评测驱动。

## v3 改了什么（`scripts/server/sync_push.sh`，sha256 `21e5892ef539b5095b21974978a36d44f28f197c7bd000ab6ddf1240a16092d4`）

- **不用工作树**：不碰源仓库 `$VI/repo` 的工作区与它自己的索引，只 fetch。每轮的做法：
  - 用一个单独的索引文件（`$VI/logs/sync_<TAG>.index`），以 origin/main 为底；
  - 把给定目录换成源目录里的现状，然后 write-tree、commit-tree，推到 main；
  - 推送被拒就重新取 origin/main，整轮重来，最多 3 次。
- **大的结果文件不进 git**（D94）：
  - 每个目录下大于 5 MB 的文件（`LARGE_MB`）留在 Lustre 上原处，不加进提交；
  - 目录里另写 `LARGE_FILES.manifest.tsv`，记相对路径、字节数、sha256、修改时间，这份清单进 git；
  - sha256 按（大小、修改时间）缓存在 `$VI/logs/sync_<TAG>.shacache`。
- 以前的提交里已有的大文件，在第一轮里从树上删掉（历史里还在），之后只出现在清单里。
- 不同步：`*.cur`、`*.tmp`、`*.part`，以及任何 `ckpt/` 下的文件。
- 用法与 v2 相同，只是不再需要 `SYNC_WT`：

  ```bash
  sync_push.sh <源仓库> <锁文件> <间隔秒> <TAG> <相对路径>…
  ```

- **开发机彩排**：
  - 大文件只进清单，sha256 与文件一致；以前提交过的大文件从树上删掉；`.cur`、`ckpt/` 不进；
  - 一个整段开着写的文件 inode 不变、内容不丢；
  - 没有变化时不产生提交；推送被拒一次后照样推上去；
  - 源仓库的索引与工作区不变。

## 做法（在 hpcmgt 上；先做完 RUN-0053 第一步）

1. `squeue`，两条 RUNNING。
2. 建放 v3 的稀疏工作树：

   ```bash
   flock $VI/logs/git.lock -c 'git -C $VI/repo fetch -q origin main'
   git -C $VI/repo worktree add --no-checkout --detach $VI/wt/sync3 origin/main
   git -C $VI/wt/sync3 sparse-checkout set --no-cone '/*' '!/experiments/*/' '/experiments/P0/'
   sha256sum $VI/wt/sync3/scripts/server/sync_push.sh     # 应为 21e5892ef539b5095b21974978a36d44f28f197c7bd000ab6ddf1240a16092d4
   ```

3. **停掉 v2 的循环**（0045、0050，以及别的还在跑的）：
   - 等它在 sleep、`git.lock` 没人拿着的时候，按 PID 停掉，同 0045-addendum-1 的做法；
   - 不要用 `git.lock.stop`。
4. **用 v3 重启同样的循环**，TAG 与路径不变，脚本换成 `$VI/wt/sync3/scripts/server/sync_push.sh`，去掉 `SYNC_WT=`：

   ```bash
   setsid nohup bash /home/mail/chenyuma/visint/wt/sync3/scripts/server/sync_push.sh \
     /home/mail/chenyuma/visint/repo /home/mail/chenyuma/visint/logs/git.lock 1800 0045 \
     experiments/E3/0045_hpc_train_hpcgpu06 experiments/E3/0045_hpc_train_hpcgpu07 \
     > /home/mail/chenyuma/visint/logs/sync_0045.log 2>&1 < /dev/null &
   ```

   0050 的循环同理，路径是 `experiments/E3/0050_hpc_eval_08b_hpcgpu07` 和 `experiments/E3/0050_hpc_e3b_hpcgpu07`。
5. **核对第一轮**：
   - 两个 status 都是 ok；
   - origin 上的新提交里，这些目录下没有大于 5 MB 的文件，各有一份 `LARGE_FILES.manifest.tsv`；
   - 任挑一个清单里的文件，在 Lustre 上算 sha256，与清单相同；
   - `$VI/repo` 的 `git status --short` 行数与切换前相同。
6. **删掉 v2 的推送工作树与旧的 sync**：

   ```bash
   git -C $VI/repo worktree remove --force $VI/wt/push
   git -C $VI/repo worktree remove --force $VI/wt/sync
   git -C $VI/repo worktree prune
   ```

   之后还要起的推送循环（RUN-0051、0052），一律用 `$VI/wt/sync3` 里的 v3。
7. 记一次 lfs quota 的差值，`squeue` 两条 RUNNING。把以上结果追加进 `reports/0053-hpc-disk-cleanup.md`。
