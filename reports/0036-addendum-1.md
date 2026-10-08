# RUN-0036 addendum 1 — 训练队列换到新 commit 重启（补上 OMP/MKL 线程数 16）

服务器 CC，b200，2026-10-08。只记事实。在 RUN-0038 的报告推送（`adc54a5`）之后执行。

## 各步时刻（UTC）

| 步 | 内容 | 时刻 |
|---|---|---|
| 1 | `flock … git pull --ff-only`，HEAD `adc54a5`，含本 addendum 的 `842b5ca` | 17:31 |
| 2 | 写 `train.pause`（"0036-addendum-1.md"） | 17:31:04 |
| 2 | `train_0036.status` 变成 `paused (pause file: 0036-addendum-1.md )`，`pgrep -f scripts/train_e3` 为 none。e3a_full_s1 的第 1 段（1,000 步）先跑完 | 18:06:30 / 18:06:44 |
| 3 | `kill -- -2169926`。会话 2169926 里原有驱动 2169926 与它的 `sleep 60`（3661033），之后这个会话里没有进程 | 18:07:46 |
| 4 | `git -C /dev/shm/visint/tmp/wt0036 checkout --detach adc54a5`，`git status --short` 为空 | 18:07:47 |
| 5 | 照任务书 §3 用 `setsid nohup` 重启，日志追加到 `train_0036.driver.log` | 18:07:55 |
| 6 | 驱动日志出现 9 条"已完成（driver 记录），跳过"，`train_0036.status` 是 `paused (pause file: 0036-addendum-1.md )` | 18:07:55 |
| 7 | `rm train.pause` | 18:08:11 |
| 8 | 下一个训练子进程开始，读它的 environ | 18:08:56 开始，18:09:21 读取 |

## 驱动

| | 进程号 | 会话号 | commit |
|---|---|---|---|
| 旧 | 2169926 | 2169926 | `18bf339` |
| 新 | 3661308 | 3661308 | `adc54a5` |

- 启动新驱动的 CC shell 的会话号是 3661304，与驱动不同。
- 跳过的 9 条：种子 0 的 7 条、warmup_s1、e3b_s1。addendum 写的是 8 条；e3b_s1 在 addendum 发出之后、执行之前完成（17:28:52），所以多了一条。

## 训练子进程的线程设置（第 8 步）

- 下一个子进程是续跑的 `e3a_full_s1`，不是 addendum 写的 e3b_s1，因为 e3b_s1 已经完成。
  - 它从第 1,000 步的检查点接着跑。
  - 进程号 3661665，父进程是新驱动 3661308。

```
    PID    PPID     SID     ELAPSED CMD
3661665 3661308 3661308       00:25 /home/ubuntu/workspace/wzg/gjl/vision-internalization/venv/bin/python scripts/train_e3a.py --model qwen3.5-2b --variant full --seed 1 --data-root /dev/shm/visint/tr
OMP_NUM_THREADS=16
MKL_NUM_THREADS=16
```

- 续跑后训练日志（`experiments/E3/0036_e3_train_2b/driver/logs/e3a_full_s1.cur`）的第一行进度：`第 1010/4000 步 loss 1.0355（1.16 s/步）`。
- 本 addendum 列的 RUN-0036 种子 0 的 E3a full 每步中位是 2.146 s。
