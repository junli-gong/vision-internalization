[RUN-TASK 0091] hpcgpu06：X-2（C1）与 X-3（C2）的定位与计数 T 文本读数（D147-4：9B 语言代价四方对照表）。约 3.5 GPU 小时，上限 5 小时，06 空闲，现在就发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 07 上的 RUN-0090。
- 只用 06 自己的暂停文件。

## 为什么

- 负责人要求（D147-4）：9B 上用同一套检查，把 E3b+C3、W2-c、W2-c+回放、X-2 四者的语言代价并排列出。
- X-2 已有健康检查与大题量读数，缺定位与计数的 T 文本读数。X-3 一并读，作消融。
- 与 RUN-0069 在 06 上的 T 文本基座配对。
- T 文本用的是 E1 的 eval 子集（RefCOCO val 等），不碰 devcheck3 与 test。

## 文件（工作树 wt0091 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0091-hpc-06-x-ttext/x_ttext.sh` | `1c4a80fe9a38705f7bd2d8950bc9ba2d0363c486e2c714628ab89986b2ba466f` |
| `scripts/run_e1.py` | `5f46ed7c28b4653221fa5ef6eb6b8dc8eec952a53360b30ffa444fa1e72bfd4d` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

- 检查点：
  - X-2：`experiments/X/0080_x2_hpcgpu07/0080_train/…/ckpt/final/model`（`a49cf4e5…`）；
  - X-3：`0082_x3_hpcgpu07/0082_train/…`（`837bb679…`）。
- 输出：`experiments/X/0091_x_ttext_hpcgpu06/`，标签 `t_text/x2/seed0`、`t_text/x3/seed0`。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0091（稀疏），建在 origin/main，核对 3 个 sha256。
2. **启动**：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=5 \
     bash /home/mail/chenyuma/visint/wt/wt0091/prompts/server_hpc/0091-hpc-06-x-ttext/x_ttext.sh \
     > /home/mail/chenyuma/visint/logs/eval_0091.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0091.status`；推送循环 TAG 0091。
3. 跑完报：每个候选的 refexp_box acc@0.5、count 准确率，标题读数加 CI，以及合并核对的结论。写 `reports/0091-hpc-06-x-ttext.md`，再跑一次 `squeue`。
