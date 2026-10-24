[RUN-TASK 0071] hpcgpu07：N37(b)，E3 的结构强制网格评测按 e5-v2 判据补评两个被拒的种子，连同机内配对（D102-2、D113-2、D123）。约 10 GPU 小时，上限 14 小时，现在就可以发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0069（W2-c）。
- 只用 07 自己的暂停文件：由 `claim_gpu` 写入，退出时删除。

## 为什么

- 负责人要求（D102-2）：E3 的结构强制网格评测统一到 e5-v2 判据，补评被 v1 拒绝的两个种子：
  - 2B 的 E3b 种子 1；
  - 4B 的 E3b 种子 2。
- 连同机内配对所需的重评，即同一种子的 E3b+C3 也在 07 上评。
- 标"按 N34 统一判据，决定于看到被拒种子的读数之前"：被拒种子的读数至今没有人看过。
- D113-2：在 GPU 空档里做。07 现在空着。

## 文件（commit 见发出时的启动语；工作树 wt0071 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0071-hpc-07-e3-reeval-e5v2/reeval.sh` | `37cbf6b130b1473888451aacc444c11b5927940129fd0305b69da63afc06fe87` |
| `scripts/eval_e3b.py` | `9820ab447eac5d58f6a601a7ca2e9935569aee29fc3fbc81b3a646c8e076ecbf` |
| `src/e3/merge_check.py` | `a1ac7d22424a0bfc62b05a0dfdf5ae476e87484944856e4efa86991eef7be99d` |
| `src/e3/evaluate.py` | `80ccd48b920f0b7694417fd66562d09671186d5f53637d37b22716d90a4115cb` |
| `src/e5/trained_eval.py` | `6695b08b98633b3f4eff76a8ddb3a17d05013ab4bf7e814d67347fde75366b05` |
| `scripts/check_training_interfaces.py` | `525b3b05d993c5fb8fcf172836fa1ce5bc1971ea921d2e68859084b68e71bef6` |
| `configs/e3b_train.json` | `b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |

## 内容（reeval.sh）

- **检查点**：开头按台账核对 4 个检查点 `manifest.json` 的 sha256，对不上就拒跑。
  - 2B 的检查点在 `$VI/b200/ckpt/...`，RUN-0048 在 07 上用过；
  - 4B 的在 `$VI/ckpt/hpcgpu06/...`。
- **每个种子**：先评 E3b（refseg、depth；结构强制，k = 1，`--merge-criteria e5-v2`），再评 E3b+C3。
  - E3b 按 e5-v2 也被拒（退出码 1）时，跳过那个种子的 E3b+C3，并在 `steps.log` 里写明。
- **结果目录**：`experiments/E3/0071_e3_reeval_e5v2_hpcgpu07/`，是新目录，不覆盖原评测。
- **预计**：4B 每臂约 3.3 小时，2B 两臂约 3 小时，合计约 10 小时。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0071（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试。核对上表 8 个 sha256。
3. 启动（方案 B）：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=14 \
     bash /home/mail/chenyuma/visint/wt/wt0071/prompts/server_hpc/0071-hpc-07-e3-reeval-e5v2/reeval.sh \
     > /home/mail/chenyuma/visint/logs/eval_0071.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0071.status`。
   - 另起一个 v3 推送循环，TAG 0071，路径 `experiments/E3/0071_e3_reeval_e5v2_hpcgpu07`。跑完、推完后按 PID 停掉。
4. 每个种子的 E3b 评完报一次：e5-v2 通过与否，以及 `merge_check.json` 的 `pass` 与 `failures`。**不要贴网格读数**，读数由管理员按规则读。
5. 全部跑完：写报告，再跑一次 `squeue`。

## 报告

推送 `reports/0071-hpc-07-e3-reeval-e5v2.md`，内容：
- `steps.log` 原文与每步墙钟；
- 4 个臂的合并核对：判据、`pass`、`failures`，以及 `v1_report_only` 的判定；
- 显存峰值、任何 traceback。

**网格的标题读数不进报告正文**，管理员从结果目录读。
