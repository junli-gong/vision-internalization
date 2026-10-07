# 服务器 CC 值守说明（模式 B，D8）

从现在起你在值守模式下工作：自己发现新任务、执行、回传，再等下一个。用户可能不在电脑前。
通用规则见 `.claude/skills/exp-triad-flow/references/server.md`，含末尾"值守模式"一节。
项目纪律照旧：CLAUDE.md、D3 服务器布局、D6 磁盘红线、D28-8 根盘份额（本项目最多 15 GB）。

## 常量

```bash
# 每行一个 export：写在同一行时，$ROOT 会在赋值之前就被展开（RUN-0001 实测 PROJ 变成 /vision-internalization）
export ROOT=/home/ubuntu/workspace/wzg/gjl
export PROJ=$ROOT/vision-internalization
REPO=$PROJ/repo
STATUS=$PROJ/logs/standby.status
DONE=$PROJ/logs/server_tasks_done.txt
WAIT=$REPO/.claude/skills/exp-triad-flow/scripts/server_standby_wait.sh
```

## 进入值守

1. 当前任务全部推送完后，把已经执行过的任务书文件名逐行写进 `$DONE`。已有的行保留。
   - 第一次进入时，至少写入 `0001-phase0-env.md`。
   - 文件名必须和 `prompts/server/` 里的一字不差。
2. 用 Bash 的 `run_in_background` 运行：

   ```bash
   bash $WAIT $REPO $STATUS $DONE 120 21600
   ```

   不要前台 sleep，也不要自己循环轮询。脚本退出时你会被自动唤醒。

## 被唤醒后

- **退出码 0**（stdout 是任务书路径）：
  1. `cd $REPO && git pull --ff-only`
  2. `echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) running <文件名>" > $STATUS`
  3. **从固定 commit 的工作树运行**（D46；RUN-0017 的教训：推中间批时 pull 会把新代码带进还没跑的模型）：
     - `git -C $REPO worktree add --detach /dev/shm/visint/tmp/wt<NNNN> HEAD`，其中 HEAD 就是第 1 步之后、开跑之前的 commit；
     - 所有模型与配置都从这个工作树启动；`--out` 写成主仓库运行目录的绝对路径（`$REPO/experiments/...`）；
     - 主 checkout 只用来 pull 与推结果；
     - 任务结束后 `git worktree remove`；
     - report 里写明运行用的 commit。
  4. 按 server.md 与该任务书执行：校验 hash、守 D6 磁盘红线、分批 push。
  5. 完成后，或按规定停止并已推送 report 后，把文件名追加进 `$DONE`，回到"进入值守"第 2 步。
- **退出码 3**（等满 6 小时）：看一眼 `df /`。
  - 剩余低于 3 GB：写 `blocked disk` 到 `$STATUS` 并推送一份简短 report。
  - 否则立即重挂。
- **其他退出码**：输出记进 `$PROJ/logs/standby.err` 后重挂。连续三次异常，就写 `blocked standby-error` 到 `$STATUS` 并停止值守。

## 纪律

- 一次只执行一个任务。addendum（`NNNN-addendum-K.md`）在当前任务完成后，按文件名顺序接。
- 任务书里的"停下报告"：推送 report 后写 `blocked <原因>` 到 `$STATUS`，然后重新进入值守，等管理员的 addendum 或新任务书。不要自己决定绕过去。
- 仓库根出现 `HOLD` 时脚本只写心跳、不接单。不要自己删 `HOLD`。
- 等长任务结束也用 `run_in_background` 的 until 循环，不要前台等待。每个配置跑完照常单独 push。
- 某一步被权限拦住、需要人工批准：推送 report 说明卡点，写 `blocked permission` 到 `$STATUS`。
- `$STATUS` 只由值守脚本和你写。管理员通过 ssh 只读查看它，来判断你是否还在。

## 长时间运行的驱动（2026-10-07 加，RUN-0032 的教训）

- 超过 30 分钟的驱动脚本一律与 CC 的进程脱钩：`setsid nohup bash <驱动> > <日志> 2>&1 &`，或者放进 tmux 的独立会话。
- CC 自己的后台任务（run_in_background）只用来等待和轮询（`$WAIT`、看日志），不要用它直接跑实验。CC 的会话一结束，挂在它下面的进程会被一起杀掉。RUN-0032 的驱动就是这样在下载 InternVL-4B 时断掉的，GPU 空了 12 小时。
- 驱动脚本自己负责每个模型完成后推送结果、更新心跳，CC 重连之后照样能接上。
- 心跳在单个长运行里也要定期更新（每 30 分钟一次，写当前在跑什么），不要只在模型之间更新：RUN-0032 的 InternVL-4B 一个模型跑了 9.4 小时，心跳一直不动，管理员误以为驱动断了。
