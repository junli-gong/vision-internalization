[RUN-TASK 0045-addendum-1] 推送循环换成 v2：在专用的推送工作树里提交，不碰训练正在写的源仓库工作区

## 为什么

- A100 的 CC 在 10-11 06:03 UTC 报告，并已停掉 v1 的推送循环。
- 原因：
  - v1 在 `$VI/repo` 里提交后，执行 `git pull --rebase --autostash`；
  - autostash 把训练期间改动过的已跟踪文件暂存，再写回成新文件（新 inode）；
  - 驱动整段开着写的 `driver/logs/c3_s0.cur` 被换掉，段结束时读到截断的内容，把正常停下的段判成"没有完成也没有按段停下"。
- 两台的 c3_s0 各误判过一次。训练本身没有丢数据：train_log.jsonl 的行数等于步数，误判后都从最后的检查点接着训。
- 这是管理员设计 v1 时的错，不是执行的问题。

## 改了什么（本 addendum 所在的 commit）

- `scripts/server/sync_push.sh` v2：
  - **不碰源仓库的工作区与索引**，在源仓库里只 fetch；
  - 每轮把给定目录 rsync 到一个专用的推送工作树 `SYNC_WT`（detached，只有本脚本写），在那里提交和推送；推送被拒就 fetch、rebase 后再推；
  - `*.cur` 不同步，origin 上已有的两个 `.cur` 会被删掉；
  - 开发机彩排：一个写入进程整段开着文件，同步前后 inode 不变、内容不丢；推送被拒一次后照样推上去；源仓库的 `git status` 不变。
- `scripts/server/train_queue.sh`：段正常续上后，重试计数清零。只对以后启动的驱动生效。正在跑的 RUN-0045 驱动不重启，原因：
  - 推送循环换成 v2 以后，不会再有误判；
  - c3_s0 各剩 1 次机会：07 的 c3_s0 快训完了，06 的约在 11:30 UTC 训完；
  - 万一再出一次真的失败，驱动会记 FAILED 并跳到下一条，以后可以补跑，检查点不丢。
- `.gitignore` 加 `experiments/**/driver/logs/*.cur`。

## 做法（在 hpcmgt 上）

1. 确认 v1 循环已经不在：`pgrep -u chenyuma -f 'scripts/server/sync_push.sh'` 为空。
2. **从现在起不要在 `$VI/repo` 里 pull、stash、checkout 或 commit。** 它的工作区由两台的驱动在写，落后于 origin 也没关系。
   - 要新代码时：`flock $VI/logs/git.lock -c 'git -C $VI/repo fetch -q origin main'`；
   - 新任务的工作树用 `git -C $VI/repo worktree add --detach $VI/wt/<名字> origin/main` 建。
3. 拿 v2 脚本：
   - `flock $VI/logs/git.lock -c 'git -C $VI/repo fetch -q origin main'`；
   - `git -C $VI/repo worktree add --detach $VI/wt/sync origin/main`，这个工作树只用来放脚本；
   - 确认 `$VI/wt/sync` 的 HEAD 包含本 addendum 所在的 commit，`sha256sum $VI/wt/sync/scripts/server/sync_push.sh` 是 `4d50e7a17f23b6a8241d779d14ef6102fa97a03f4f3df9f7ef17097ddda8c41d`。
4. 删掉 v1 留下的 `$VI/logs/git.lock.stop`，不然新循环跑完第一轮就退出。
5. 启动 v2 循环。推送工作树 `$VI/wt/push` 第一轮自动建：

   ```bash
   SYNC_WT=/home/mail/chenyuma/visint/wt/push setsid nohup bash /home/mail/chenyuma/visint/wt/sync/scripts/server/sync_push.sh \
     /home/mail/chenyuma/visint/repo /home/mail/chenyuma/visint/logs/git.lock 1800 0045 \
     experiments/E3/0045_hpc_train_hpcgpu06 experiments/E3/0045_hpc_train_hpcgpu07 \
     > /home/mail/chenyuma/visint/logs/sync_0045.log 2>&1 < /dev/null &
   ```

6. 第一轮之后核对：
   - `sync_0045.status` 是 `ok`；
   - origin 上有一条新的 `[results] 0045: sync …`，里面删掉了两个 `.cur`；
   - 两台驱动的下一次段边界正常识别：驱动日志里没有新的"没有完成也没有按段停下"；
   - `$VI/repo` 的 `git status --short` 与启动前相同，只有那些结果目录的改动。
7. 把核对结果追加进 `reports/0045-hpc-train.md`，加一节"推送循环 v1 的事故与 v2"，包括：
   - 两次误判的时刻；
   - 停 v1 的方法；
   - v2 的启动时刻与第一轮的 commit；
   - 第 6 步的核对结果。

## 以后的任务

- RUN-0047、RUN-0048 及以后的 HPC 任务，都用 v2 的推送方式。
- 每个任务的推送循环可以共用 `$VI/wt/push`，因为都拿 `git.lock`；也可以各用一个，由任务书写明。
- 你自己的报告照旧从开发机上的 `A100/repo` 推送。
