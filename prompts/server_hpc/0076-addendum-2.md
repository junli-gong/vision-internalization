[RUN-TASK 0076 补充 2] hpcgpu06：W2-c 加回放在 5 个 test 划分上的评测，重跑。排在 B 段的其余步骤之后，约 1.5 GPU 小时，上限 4 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 B 段主驱动与 RUN-0080。
- 只用 06 自己的暂停文件。

## 为什么

- B 段的 5 个 test 划分全部 rc=2：`replay.sh` 给 `run_w2.py` 的 `--out` 是 `$OUT/test`，目录名没有运行编号。
- 这是管理员的驱动错误，D140；tiny 彩排不查这条规则。B 段的健康检查、T 文本、lb 照常在跑。

## 文件（工作树 wt0076t 建在 origin/main，只取驱动；python 的代码用 RUN-0072 的工作树 wt0072）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0076-addendum-2/test_replay.sh` | `dbd89144806a1b096eb17eb0d2989c512a47a7d3a01314c677bd3ce6743881ae` |

- `CODE=$VI/wt/wt0072`：`run_w2.py` 等与 RUN-0072 的 W2-c test 同一个 commit（d38a5205），配对更干净。开跑前再核对 wt0072 的那几个文件。
- 检查点：`experiments/W2/0076_w2c_replay_hpcgpu07/0076_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`（manifest `ecad7667…`）。
- 输出：`experiments/W2/0076_w2c_replay_test_hpcgpu06/`，目录名带运行编号；驱动开头会自查。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0076t（稀疏），建在 origin/main，核对上表的 sha256 与 wt0072 的文件。
2. 启动（方案 B）：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=4 \
     bash /home/mail/chenyuma/visint/wt/wt0076t/prompts/server_hpc/0076-addendum-2/test_replay.sh \
     > /home/mail/chenyuma/visint/logs/eval_0076t.driver.log 2>&1 < /dev/null &'
   ```
   - 驱动先等 `eval_0076.status` 以 `done 0076 [eval]` 开头，即 B 段跑完；再 `claim_gpu`。
   - 心跳在 `eval_0076t.status`。
   - 推送循环 TAG 0076t，路径 `experiments/W2/0076_w2c_replay_test_hpcgpu06`。
3. 每个划分报 cIoU、gIoU 与 CI。
4. 跑完：在 `reports/0076b-hpc-06-w2c-replay-eval.md` 里加一节"test 重跑（补充 2）"，再跑一次 `squeue`。
