# RUN-0037 启动报告 — B200 后台队列：E5 主比较 v3（2B）

服务器 CC，b200，2026-10-09 17:59 UTC。只贴数字与事实，不做解读。

## 1. 校验

- **HEAD：** `flock … git pull --ff-only` 之后 HEAD 是 `cd55851`，本任务书就在这个 commit 里。
- **worktree：** `/dev/shm/visint/tmp/wt0037`，HEAD 是 `cd55851`。
- **配置：** sha256 都与任务书相同：

  ```
  e7f28997935f7281a6d0f4fe1d29132cc2d14e4b500000f00882dd7b2b2b94c3  configs/e5_train.json
  c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4  configs/e5_prompts.json
  b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375  configs/e3b_train.json
  ```

- **起点检查点：**
  - base_warmup 的 3 个（RUN-0036）都在，`.sha256.txt` 逐文件核对通过，叶目录有 `config.yaml`：

    ```
    base_warmup s0 model=y sha256=ok config.yaml=y
    base_warmup s1 model=y sha256=ok config.yaml=y
    base_warmup s2 model=y sha256=ok config.yaml=y
    ```

  - e3b_c3 与 c3_warmup 的 6 个（RUN-0040）还没产出，RUN-0040 在等 RUN-0036。
    - 照任务书在 RUN-0040 跑完之后核对。我启动了一个与 CC 脱钩的核对脚本（`experiments/E5/0037_e5_main/probe_scripts/check_starts_after_0040.sh`，进程号 3890263）。
    - 它等 `train_0040.done`，核对 9 个起点，推送 `reports/0037-b200-e5-main-progress-1.md`。
- **环境：** `pip install --dry-run -r requirements.txt` 没有任何 "Would install"。

## 2. 预估的复核

| 依据 | 实测 |
|---|---|
| 2B E3b 每步中位（RUN-0036） | s0 2.038 s、s1 2.036 s、s2 1.921 s |
| 结构强制满长生成（RUN-0039，NYU 654 条，含加载与合并核对之外的开销） | 177 s，约 0.27 s/条 |

- **训练：**
  - 1K、4K（1,000 步）约 0.55 h，加 val 选择；
  - 16K（2,000 步）约 1.1 h；
  - 64K（8,000 步）约 4.4 h。
  - 与任务书表里的 0.75、1.15、4.5 h 一致或略低。
- **test 评测：** 约 2,300 条 × 0.27 s ≈ 0.17 h，加合并核对与加载，与表里的 0.35 h 相符。
- **合计：** 按任务书的表约 59 h，按上面的实测不会更高，低于上限 70 h，没有预计超出。
- E5 训练在 B200 上还没有实测，第一条 E5 运行之后可以核对。
- 队列自己的上限：`QUEUE_CAP_H=70`，每条开始前按"已用 + 预估"判断。
- **根盘份额：** 现在 $PROJ 9.3 GB。E5 的 39 个最终检查点约 5 GB，加 RUN-0040、RUN-0041 的约 1.3 GB，可能接近 15 GB，驱动触线会停（任务书 §6）。

## 3. 启动

- **时刻：** 2026-10-09T17:59:44Z，`setsid nohup bash /dev/shm/visint/tmp/wt0037/prompts/server/0037-b200-e5-main/queue.sh > $PROJ/logs/train_0037.driver.log 2>&1 < /dev/null &`。
- **驱动：** 进程号与会话号都是 3890017；启动它的 CC shell 的会话号是 3890012。
- **`train_0037.status` 原文：** `2026-10-09T18:00:44Z waiting for queue 0040`
- **队列链：** 0036（running c3_s2）→ 0040（waiting for queue 0036）→ 0037（waiting for queue 0040）→ 0041（waiting for queue 0037）。
