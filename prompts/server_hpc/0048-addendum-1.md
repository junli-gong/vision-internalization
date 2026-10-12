[RUN-TASK 0048-addendum-1] 补测 T 文本的单位耗时：measure.sh 里 run_e1 的输出目录名不带运行编号，被拒绝了

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。** 前后各跑一次 `squeue`。只用 hpcgpu07；不写 `$VI/logs/train.pause`。

## 为什么

- RUN-0048 的 `ttext_e3b_c3_s0_dev20`、`ttext_e3b_c3_s0_free10`、`e1_base_dev20` 三步都是 rc=2（3–4 秒）。原因：
  - `run_e1` 要求 `--out` 的目录名带运行编号，例如 `0011_e1_main`；
  - measure.sh 给的是 `<OUT>/e1`、`<OUT>/e1base`。
- 这是管理员写的错。开发机彩排用 tiny 模式，tiny 模式不查这一条，所以没发现。
- RUN-0050 的预估要用到 T 文本的单位耗时，所以补测。RUN-0050 的输出目录名已经核对过，符合要求。

## 做法

1. 等 measure.sh 跑完，即 `$VI/logs/eval_0048.status` 是 `done 0048 (failed steps: N)`。N 里会包括这三步。
2. `flock $VI/logs/git.lock -c 'git -C $VI/repo fetch -q origin main'`；`git -C $VI/repo worktree add --detach $VI/wt/wt0048a origin/main`，origin/main 必须包含本补充所在的 commit。
3. 核对 sha256：
   - `prompts/server_hpc/0048-hpc-measure-07/measure_e1.sh`：`6cf0f673c96a66878acf9e015769d015f28b8f6e6c8ebbfca43be1e88e3e77b2`
   - `scripts/server/eval_lib.sh`：`2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59`
4. 启动（方案 B）：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     bash /home/mail/chenyuma/visint/wt/wt0048a/prompts/server_hpc/0048-hpc-measure-07/measure_e1.sh \
     > /home/mail/chenyuma/visint/logs/eval_0048e1.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳：`$VI/logs/eval_0048e1.status`。
   - 三步合计约 30–45 分钟，上限 2 小时。
5. 结果在 `experiments/E3/0048_hpc_measure_e1_hpcgpu07/`。把这个路径加进 0048 的推送循环：
   - 按 PID 停掉旧的 0048 循环；
   - 照原来的命令重启，路径多给这一个。
6. 报告：在 `reports/0048-hpc-measure-07.md` 里加一节"补充 1"，内容照任务书 §4 的"0.8B 评测、基座 E1"一项：每条命令的加载与合并核对、每条样本的耗时（按任务）、合计墙钟。
7. 删工作树 `$VI/wt/wt0048a`。之后 07 等 RUN-0050，管理员拿到报告就发。

另外，在报告里照实记下 RUN-0048 的 E3b 种子 1：A100 上的合并核对 exact_rows 是 0.940，没过，拒绝训练。按 E5 预注册 §8b，E3b 起点只用种子 0、2。
