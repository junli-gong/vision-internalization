# RUN-0061 报告：hpcgpu07，W1 开发-调参 T1–T4 —— 12 步全部 rc=2，没有跑起来

任务书 `prompts/server_hpc/0061-hpc-07-w1-tune.md`（origin 141d8f4d）。只用 devtune；没有碰 devcheck；没有碰 06 的 RUN-0056。

## 启动

- 10-21 23:17:17Z 第一次启动脚本（`$VI/tmp/start0061.sh`）：squeue 两条 RUNNING；07 没有暂停文件；`git fetch` 失败（rc=128，`Connection to 20.205.243.166 port 22 timed out`），工作树建在旧的 origin/main 1aecbebd，缺 tune.sh 与 T1–T4 配置；12 个 sha256 对不上，**脚本在启动前中止**，什么都没启动。
- 23:20:37Z 重试 fetch 成功（origin/main 含 141d8f4d）；删掉旧的（干净的）wt0061，23:20:43Z 重跑启动脚本：
  - wt0061：HEAD `141d8f4d1fa1936b9e5f6d2082dd85577c852c0b`，status 空，`experiments/` 只有 P0，含 6038d755。
  - 12 个 sha256 与任务书逐字相同；tune.sh 用 `EV_PAUSE=$LOGS/train.pause.$NODE`。
  - 23:20:56Z 按任务书命令启动 tune.sh（`EV_CAP_H=6`）；推送循环 0061 由脚本起。
- 心跳：`2026-10-21T23:20:57Z waiting for gpu (running: train_0057-hpcgpu07.status; pids: 1137806)`；暂停文件 `train.pause.hpcgpu07` 内容 `0061-hpc-07-w1-tune.md`。
- （另：为备用的 443 路线做过只读核对：`ssh-keyscan -p 443 ssh.github.com` 得 `256 SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU [ssh.github.com]:443 (ED25519)`，与 GitHub 公布的一致；没用上，没写进 known_hosts。）

## 让路与交还

- RUN-0057 在第 2 段之后停住：驱动 `=== 2026-10-22T00:11:28Z chunk rc=0 wall_s=6661`，心跳 `2026-10-22T00:11:28Z paused (pause file: 0061-hpc-07-w1-tune.md )`。
- RUN-0061 00:11:30Z 拿到 GPU（`steps.log` 的 start 行）。
- 00:12:20Z `done 0061 (failed steps: 12)`；暂停文件已删（00:13Z 查，`train.pause*` 都不存在）；RUN-0057 心跳 `2026-10-22T00:12:28Z running e3b_c3_n4000_s0 (chunk 3)`。
- 07 的 GPU 被借用约 1 分钟。`gpu_mem.csv` 49 行，最大 0 MiB（没有加载模型）。

## steps.log 原文

```
2026-10-22T00:11:30Z start commit 141d8f4d1fa1936b9e5f6d2082dd85577c852c0b node hpcgpu07 model qwen3.5-9b n 96
2026-10-22T00:11:49Z base_e1_refexp_box_devtune96 rc=2 wall_s=18
2026-10-22T00:11:52Z base_ownbox_devtune96 rc=2 wall_s=3
2026-10-22T00:11:56Z base_poly_structured_devtune96 rc=2 wall_s=4
2026-10-22T00:11:59Z base_w1f_devtune96 rc=2 wall_s=3
2026-10-22T00:12:02Z arm_T1_refseg_devtune96 rc=2 wall_s=3
2026-10-22T00:12:04Z arm_T2_refseg_devtune96 rc=2 wall_s=2
2026-10-22T00:12:07Z arm_T3_refseg_devtune96 rc=2 wall_s=3
2026-10-22T00:12:09Z arm_T4_refseg_devtune96 rc=2 wall_s=2
2026-10-22T00:12:12Z gate_tune_T1 rc=2 wall_s=3
2026-10-22T00:12:15Z gate_tune_T2 rc=2 wall_s=3
2026-10-22T00:12:17Z gate_tune_T3 rc=2 wall_s=2
2026-10-22T00:12:20Z gate_tune_T4 rc=2 wall_s=3
2026-10-22T00:12:20Z done; failed steps: 12
```

## 报错原文

- 8 个 run_w1.py 步（`logs/base_*.log`、`logs/arm_T*_refseg_devtune96.log`）都是参数检查报错，例如：

```
run_w1.py: error: --out 的目录名必须带运行编号，形如 0060_w1_devcheck，收到 '/home/mail/chenyuma/visint/repo/experiments/W1/0061_w1_tune_hpcgpu07/base'
run_w1.py: error: --out 的目录名必须带运行编号，形如 0060_w1_devcheck，收到 '/home/mail/chenyuma/visint/repo/experiments/W1/0061_w1_tune_hpcgpu07/T1'
```

- 4 个 gate 步：`不能判定：w1：/home/mail/chenyuma/visint/repo/experiments/W1/0061_w1_tune_hpcgpu07/T1/qwen3.5-9b/refseg/full/devtune 里没有 predictions.jsonl`（T2–T4 同）。
- 相关代码（只读，未改）：
  - `prompts/server_hpc/0061-hpc-07-w1-tune/tune.sh` 第 36 行 `OUT=$REPO/experiments/W1/0061_w1_tune_$NODE`，第 44 行 `--out "$OUT/$sub"`，`$sub` 是 `base`、`T1` … `T4`。
  - `scripts/run_w1.py` 第 98–99 行：`if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name): p.error(...)`。这条检查在 `--tiny` 时跳过，所以开发机的 tiny 彩排没有触发。

## 推送

- 推送循环 0061 在跑；23:53:33Z 一轮 `push failed (will retry)`（GitHub 22 端口超时，同前）。先不停，等管理员决定是否重发。
