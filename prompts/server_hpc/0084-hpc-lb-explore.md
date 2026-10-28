[RUN-TASK 0084] 探索候选的大题量语言读数（D145）：07 上基座、X-2、X-3（约 4.5 GPU 小时，现在就发）；06 上 X-3L（约 1.5 GPU 小时，排在 RUN-0083 之后）

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上在跑的 RUN-0083。
- 只用本节点自己的暂停文件。

## 为什么

- 健康检查的 IFEval 只有 50 题，噪声大：X-2 是 −4、不标红，X-3 是 −10、标红，两者只差 3 道题。
- 选候选需要更可靠的语言读数，即 MMLU-Pro 2,000 题与 IFEval 541 条，与同节点的基座配对。
- 判定本身仍按健康检查（预注册 X §3），这里只描述。

## 文件（工作树 wt0084 建在 origin/main；06、07 两次启动共用）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0084-hpc-lb-explore/lb_explore.sh` | `f0cea3fbb69d94fc2ae841b4a4a4e9bc27ba038f8bb3ab7f6f5ba9531f0d88e3` |
| `scripts/lang_battery.py` | `c6cee3ff3b610f1203c6a7e00ca7c85fbeb5b1f2a9c5d3f844641fce28d062e7` |
| `scripts/health_check.py` | `4728a0ca26433f5c02b8b3c064b637a437a32de4c1a1847056614aee2ec5c926` |
| `src/health/run.py` | `ca4a995857d69fa080ba6a01aac4225c076e69ef0b4150c6f84cf3a5dfad1437` |
| `src/health/scoring.py` | `05d11f230fe66b67abcabdff2064bc1cb72bb8401cbdc042e0515378871111be` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

- 数据：`$VI/data/lang_battery_raw`，RUN-0074 已下载。
- 检查点：
  - X-2：`experiments/X/0080_x2_hpcgpu07/0080_train/…/ckpt/final/model`，`a49cf4e5…`；
  - X-3：`0082_x3_hpcgpu07/0082_train/…`，`837bb679…`；
  - X-3L：`0083_x3l_hpcgpu06/0083_train/…`，`eb200b24…`。
- 驱动开头核对 manifest 存在，并把 sha256 记进 `steps.log`。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0084（稀疏），建在 origin/main，核对上表 6 个 sha256。
2. **07**（现在发）：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 NODE=hpcgpu07 EV_CAP_H=7 \
     bash /home/mail/chenyuma/visint/wt/wt0084/prompts/server_hpc/0084-hpc-lb-explore/lb_explore.sh \
     > /home/mail/chenyuma/visint/logs/eval_0084-hpcgpu07.driver.log 2>&1 < /dev/null &'
   ```
   - 跑基座、X-2、X-3 三个，再 `--compare` 基座对 X-2、基座对 X-3。
   - 输出 `experiments/X/0084_lb_explore_hpcgpu07`。
3. **06**（同时发；驱动等 `eval_0083.status` 以 `done 0083` 开头）：命令同上，换成 `--jobid=157296`、`NODE=hpcgpu06`、`EV_CAP_H=3`、日志 `eval_0084-hpcgpu06.driver.log`。
   - 跑 X-3L，再与 RUN-0074 在 06 上的 `lb_base` 做 `--compare`。
   - 输出 `experiments/X/0084_lb_explore_hpcgpu06`。
4. 心跳是 `eval_0084-hpcgpu07.status` 与 `eval_0084-hpcgpu06.status`。推送循环各一个，TAG 0084-07、0084-06。
5. 每个 `--compare` 跑完，原文照贴。
6. 跑完写 `reports/0084-hpc-lb-explore.md`，再跑一次 `squeue`。
