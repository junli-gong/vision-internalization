[RUN-TASK 0066 补充 1] W2-a 调参改在 hpcgpu06 上**立即**开始（06 已空；D113 第五节"不让卡空等"；负责人的排期原本就把 W2-a 调参放在 06）

## 改什么

- **停掉 07 上等待 RUN-0057 的自动启动脚本**（`$VI/tmp/start0066.sh`）：按 PID 停。它只是 hpcmgt 上一个轮询的脚本，没有启动任何东西。
  - 不碰 RUN-0057，不碰任何 Slurm 任务。
  - 停之前确认 07 上没有 0066 的进程，`eval_0066.status` 不存在。
- **在 06 上启动 RUN-0066**：用已建好、已核对 12 个 sha256 的工作树 wt0066（130d2314），驱动 `tune_w2.sh` 不变。只把节点参数换成 06：

  ```bash
  ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
    env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=10 NODE=hpcgpu06 TQ_STATUS=train_0056-hpcgpu06.status \
    bash /home/mail/chenyuma/visint/wt/wt0066/prompts/server_hpc/0066-hpc-w2-tune/tune_w2.sh \
    > /home/mail/chenyuma/visint/logs/eval_0066.driver.log 2>&1 < /dev/null &'
  ```

  - `-c 8` 被拒就去掉，并在 env 里加 `CW=1`。
  - 结果目录因此是 `experiments/W2/0066_w2_tune_hpcgpu06/`。
  - 推送循环的 TAG 仍是 0066，路径照改。
- 其余照原任务书：每当 `stepN chosen` 出现，就把 `stepN.md` 发给管理员；中途 ABORT 照实报告、不要自己重跑；跑完推报告。

## 之后

- 07 在 RUN-0057 跑完后先空着。调参选定、管理员冻结之后，会发 RUN-0068（devcheck2 上 W2-a 的判定）到 07，发 RUN-0069（W2-c 全流程）到 06。

## 铁律

照旧：任何情况下都不能取消或停止 Slurm 任务 157296 与 157297；只用本节点自己的暂停文件；前后各跑一次 `squeue`。
