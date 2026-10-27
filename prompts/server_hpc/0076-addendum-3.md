[RUN-TASK 0076 补充 3] hpcgpu06：W2-c 加回放的定位与计数 T 文本读数，重跑。排在补充 2 之后，约 2 GPU 小时，上限 4 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 B 段主驱动、补充 2 与 RUN-0080。
- 只用 06 自己的暂停文件。

## 为什么

- B 段的 `ttext_w2c_replay` 一步 rc=2，与 test 划分同一个错：`run_e1.py` 的 `--out` 是 `$OUT/ttext`，没有运行编号（D140、D141）。
- 这一步给 D125-3 的语言代价并排表提供定位与计数两行。

## 文件（工作树 wt0076x 建在 origin/main，只取驱动；python 的代码用 wt0072）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0076-addendum-3/ttext_replay.sh` | `89f6d9ef278573abcacd9eedef4a0ae1ded4311bace954318479473f6e005434` |

- `CODE=$VI/wt/wt0072`：从 RUN-0069 的 7f53f240 到 d38a5205，`run_e1` / E1 / 适配器都没有改动；只有 DEV-0063 给 E3 网格评测加的可选判据，默认不变。所以与 RUN-0069 在 06 上的 T 文本基座可比。
- 输出：`experiments/W2/0076_w2c_replay_ttext_hpcgpu06/`，标签 `t_text/w2c_replay/seed0`。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0076x（稀疏），建在 origin/main，核对 sha256。
2. 启动（方案 B）：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=4 \
     bash /home/mail/chenyuma/visint/wt/wt0076x/prompts/server_hpc/0076-addendum-3/ttext_replay.sh \
     > /home/mail/chenyuma/visint/logs/eval_0076x.driver.log 2>&1 < /dev/null &'
   ```
   - 驱动先等 `eval_0076t.status` 以 `done 0076t` 开头，即补充 2 跑完；再 `claim_gpu`。
   - 心跳在 `eval_0076x.status`。
   - 推送循环 TAG 0076x，路径 `experiments/W2/0076_w2c_replay_ttext_hpcgpu06`。
3. 跑完报 refexp_box 的 acc@0.5 与 count 的准确率，标题读数加 CI。在 `reports/0076b` 里加一节"T 文本重跑（补充 3）"。再跑一次 `squeue`。
