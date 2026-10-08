[RUN-TASK 0036-addendum-1] 训练队列换到新 commit 重启：补上 CPU 线程数（OMP_NUM_THREADS=16）

## 为什么

- RUN-0036 的种子 0 里，E3a 系列比 RUN-0034 的吞吐实测慢得多：

  | 变体 | 每步中位 s：RUN-0034 | 每步中位 s：RUN-0036 | 其中 build_inputs：RUN-0034 → RUN-0036 |
  |---|---:|---:|---|
  | E3a full | 1.155 | 2.146 | 0.218 → 0.795 |
  | C1 | 0.779 | 2.045 | 0.219 → 1.083 |
  | C2 | 0.382 | 1.868 | 0.215 → 1.628 |

  E3b、C3 不受影响：它们以 GPU 为主，build_inputs 只要 0.15–0.2 s。
- 原因：RUN-0034 的驱动设了 `OMP_NUM_THREADS=MKL_NUM_THREADS=16`，与 A800 的 `common.sh` 相同；管理员写的队列脚本漏了这一项。这台机器 30 核、多人共用，torch 与 numpy 按全部核开线程，互相争抢。
- 修正：`scripts/server/train_queue.sh` 与 `eval_lib.sh` 默认设 16（调用方已设的照用），见本 addendum 所在的 commit。剩下两个种子的 E3a 系列预计省约 8 小时。
- 训练结果不受影响：线程数只改变 CPU 上准备输入的速度，不改变数据与计算。已完成的 8 条（种子 0 的全部 7 条，加 warmup_s1）照用。

## 做法

在 RUN-0038 跑完、推送报告之后执行（STANDBY：addendum 在当前任务完成后接）。

1. `flock $PROJ/logs/git.lock -c 'git -C $REPO pull --ff-only'`，HEAD 必须包含本 addendum 所在的 commit。
2. 让队列停在段的边界：`echo "0036-addendum-1.md" > $PROJ/logs/train.pause`。
   - 用后台 until 循环等，直到 `train_0036.status` 以 `paused` 开头，并且没有训练进程：`pgrep -f 'scripts/train_e3' || echo none` 输出 none。
3. 停驱动：`ps -o pid,sid,cmd -p $(pgrep -f 'wt0036/prompts/server/0036-b200-e3-train-2b/queue.sh')` 记下会话号 SID，然后 `kill -- -<SID>`。
   - 确认驱动与它的 sleep 都没了。
4. 把工作树切到新 commit：`git -C /dev/shm/visint/tmp/wt0036 checkout --detach <第 1 步之后的 HEAD>`。
   - 工作树里不应有任何改动：`git -C … status --short` 为空。
5. 照 RUN-0036 任务书 §3 的命令重新启动，与 CC 脱钩。
6. 等驱动日志里出现 8 条"已完成（driver 记录），跳过"，并且 `train_0036.status` 显示 `paused (pause file: 0036-addendum-1.md …)`。
7. `rm $PROJ/logs/train.pause`，队列一分钟内接着跑。
8. 下一个训练子进程（e3b_s1）开始后，确认它的环境里有线程设置：`tr '\0' '\n' < /proc/<子进程 PID>/environ | grep -E 'OMP|MKL'`，应为 16。
   - E3a 系列（e3a_full_s1）要在 e3b_s1 之后约 2.3 小时才开始，不用等它；它的 build_inputs_s 由管理员从推送的日志里看。
9. 推送一份短报告 `reports/0036-addendum-1.md`，只记事实：
   - 第 2–8 步各自的时刻；
   - 驱动的新旧进程号与会话号；
   - 工作树的新 commit；
   - 训练子进程 environ 里的 OMP 与 MKL。
10. 把 `0036-addendum-1.md` 追加进 `$DONE`，回到值守。

不改任何代码与配置。
