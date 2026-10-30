[RUN-TASK 0098 补充 1] hpcgpu07：补跑 4B 的健康检查基座与对比（D161）。约 25 分钟，上限 1 小时；07 空闲，现在就发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 只用 07 自己的暂停文件。

## 为什么

- RUN-0098 的 `health_base_4b` 一步报 `--merge-check-n、--data-root 只配 --checkpoint 用`，退出码 2。原因是驱动给基座也带了 `--data-root`。这是管理员写驱动时的错；RUN-0080 的基座那一步没带。
- 驱动已改：
  - 基座不带 `--data-root`；
  - 已有的 `health_x2_4b.json` 不重跑（RUN-0098 里那一步 rc=0，读数有效）。
- 只用 `STAGES=health` 补跑；训练与评测都不动。

## 文件（工作树 wt0098 更新到 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0098-hpc-07-x2-4b/x2_4b.sh` | `3d8abe66250151ed0f92aaab0e125a78bb0d5ce461c3b90fec5021c50beddf1d` |

其余文件与 RUN-0098 的任务书相同，不用重新核对。

## 做法

1. hpcmgt 上 `squeue`。把 wt0098 更新到 origin/main（`git -C $VI/wt/wt0098 fetch && git -C $VI/wt/wt0098 checkout --detach origin/main`），核对上表的 sha256。
2. 启动：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 12 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=1 STAGES=health \
     bash /home/mail/chenyuma/visint/wt/wt0098/prompts/server_hpc/0098-hpc-07-x2-4b/x2_4b.sh \
     > /home/mail/chenyuma/visint/logs/eval_0098a1.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳仍在 `eval_0098.status`。推送循环 TAG 0098，同一路径。
3. 跑完贴 `health_compare` 原文，在 `reports/0098-hpc-07-x2-4b.md` 末尾补一节，再跑一次 `squeue`。

管理员在开发机上用 tiny 彩排过 `STAGES=health`：已有的 4B 读数被跳过，基座重跑 rc=0，对比 rc=0。基座的命令已逐项对照 `health_check.py` 的参数规则，与 RUN-0080 的 `health_base` 一步相同。
