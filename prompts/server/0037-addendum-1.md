[RUN-TASK 0037-addendum-1] 根盘被别的用户写满（剩 1.7 GB）：腾出我们自己的空间，E5 的最终检查点改放内存盘，重启 E5 队列

## 为什么

- RUN-0037 的驱动在 2026-10-10 21:46 UTC 停下：`disk free 1GB<3GB (D6)`。
- 管理员只读查过（10-11 02:55 UTC）：
  - 根盘 485 GB 已用 438 GB，剩 1.7 GB；
  - 本项目只占 12 GB，在 15 GB 的份额之内，是别的用户写满的，此前还有约 27 GB 空闲；
  - 内存盘剩 66 GB，可用内存 132 GB。
- 处理办法（D90）：
  - 只删、只挪**我们自己的、可以重新生成的**东西；
  - 后面约 5 GB 的 E5 最终检查点改写到 `/dev/shm`。每个训完紧接着就评 test，评完的结果已推送；整个队列跑完后，由管理员归档到开发机。
- **不要动任何别人的文件**，也不要清系统目录。

## 做法

1. `flock $PROJ/logs/git.lock -c 'git -C $REPO pull --ff-only'`，HEAD 必须包含本 addendum 所在的 commit。
2. 记一次 `df -h / /dev/shm` 与 `free -g`。
3. 腾空间，每项先 `du -sh` 记下再动：
   - 删 `$PROJ/ckpt/0039/`：RUN-0039 缩放诊断的派生检查点，用 `scripts/scale_lora_ckpt.py` 一秒就能重新生成，结果已入库；
   - 删 `$PROJ/tmp/triton-cache/`、`$PROJ/tmp/pytest-of-root/`：都是缓存；
   - 把 `$PROJ/tmp/visint_nyu_pairs_rendered/` 整个 `mv` 到 `/dev/shm/visint/tmp/`，它是 NYU 渲染图的缓存，以后要用会自己重建；
   - `$PROJ/tmp/` 里其余的小文件不动。
4. 再记一次 `df -h /`。剩余低于 3 GB 就停下报告，不要重启队列。
5. **重启 E5 队列**：
   - 确认旧驱动已经退出：`pgrep -f 'wt0037/prompts/server/0037-b200-e5-main/queue.sh'` 应为空；
   - 把工作树切到新 commit：`git -C /dev/shm/visint/tmp/wt0037 checkout --detach <第 1 步之后的 HEAD>`，并确认 `status --short` 为空；
   - `mkdir -p /dev/shm/visint/ckpt_0037_final`；
   - 照 RUN-0037 任务书 §3 的命令启动，加一个环境变量：

     ```bash
     CK_FINAL_OVERRIDE=/dev/shm/visint/ckpt_0037_final setsid nohup bash /dev/shm/visint/tmp/wt0037/prompts/server/0037-b200-e5-main/queue.sh \
       > $PROJ/logs/train_0037.driver.log 2>&1 < /dev/null &
     ```

   - 等驱动日志里出现已完成的 N = 0 那几条"已完成（driver 记录），跳过"，并开始下一条。
6. 推送一份短报告 `reports/0037-addendum-1.md`，只记事实：
   - 第 2–4 步的 df、free 与每项腾出的大小；
   - 驱动的新进程号与会话号；
   - 工作树的新 commit；
   - `train_0037.status` 原文。
7. 把 `0037-addendum-1.md` 追加进 `$DONE`，回到值守。

不改任何实验代码与配置。
