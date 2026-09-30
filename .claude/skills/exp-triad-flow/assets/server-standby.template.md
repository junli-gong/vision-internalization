# 服务器 CC 值守说明（模式 B）

从现在起你在值守模式下工作：自己发现新任务、执行、回传，再等下一个。用户可能不在电脑前。
通用规则见 `.claude/skills/exp-triad-flow/references/server.md`（含末尾"值守模式"一节）。

## 常量

```bash
REPO=<服务器上的仓库路径>
STATUS=<项目日志目录>/standby.status             # 心跳，管理员只读查看
DONE=<项目日志目录>/server_tasks_done.txt         # 已执行的任务书文件名，每行一个
WAIT=$REPO/.claude/skills/exp-triad-flow/scripts/server_standby_wait.sh
```

## 进入值守

1. 当前任务全部推送完后，把已经执行过的任务书文件名逐行写进 `$DONE`，例如 `0001-xxx.md`。已有的行保留。
2. 用 Bash 的 `run_in_background` 运行：

   ```bash
   bash $WAIT $REPO $STATUS $DONE 120 21600
   ```

   不要前台 sleep，也不要自己循环轮询。脚本退出时你会被自动唤醒。

## 被唤醒后

按脚本的退出码处理：

- **0**（stdout 是任务书路径）：
  1. `cd $REPO && git pull --ff-only`
  2. `echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) running <文件名>" > $STATUS`
  3. 按 server.md 与该任务书执行：校验 hash、守磁盘红线、分批 push。
  4. 完成后，或按规定停止并已推送 report 后，把文件名追加进 `$DONE`，回到"进入值守"第 2 步。
- **3**（等满）：检查 `df /` 与磁盘红线，然后立即重挂。
- **其他**：记进日志后重挂。连续三次异常，就写 `blocked <原因>` 到 `$STATUS`，停止值守。

## 纪律

- 一次只执行一个任务。addendum 在当前任务完成后，按文件名顺序接。
- 任务书里的"停下报告"：推送 report 后写 `blocked <原因>` 到 `$STATUS`，然后重新进入值守。不要自己绕过去。
- 仓库根出现 `HOLD` 时脚本只写心跳、不接单。不要自己删 `HOLD`。
- 等长任务结束也用 `run_in_background` 的 until 循环，不要前台等待。每个配置跑完照常单独 push。
- 某一步被权限拦住：推送 report 说明卡点，写 `blocked permission` 到 `$STATUS`。
