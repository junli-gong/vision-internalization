[RUN-TASK 0067] hpcgpu06：W2-c 训练数据的第一步，用 base 9B 在 train 抽样的 16,000 句上出框（D118、D119；W2 预注册 §9；W2 预算内，约 0.5–1 GPU 小时）。**等 RUN-0056 全部跑完再开始**。上限 3 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 07（RUN-0057、RUN-0066）。
- 只用 06 自己的暂停文件：由脚本的 `claim_gpu` 写入，退出时删除。

## 什么时候开始

- RUN-0056 全部跑完：`train_0056-hpcgpu06.status` 与 `$VI/logs/run_0056.log` 都显示 done，06 上没有 0056 的进程，GPU 空着。约 10-24 上午（UTC）。
- 请挂一个监视，条件满足后立即启动，不用等我。

## 为什么

- W2-c 训练 LLM 在候选里挑选。训练数据的第一步：base 9B 按 E1 的定位协议，在 RefCOCO、RefCOCO+、RefCOCOg 的 train 里出框。
  - 句子先按图像排除 W2 全部评测集的图，再按 `hash_rank(w2c_train_v1)` 抽 16,000 句。
- 这一步不读候选清单，不要求 W2 的配置冻结（D118-3）。候选与标签一步要等 W2-a 在 07 上调参、冻结之后，另发任务书。

## 文件（commit 见发出时的启动语；工作树 wt0067 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0067-hpc-06-w2c-propose/propose.sh` | `f0cfb0f933cb55efd9568467053e31c33331a13dd8e9630688d1555feae09de2` |
| `scripts/build_w2c_data.py` | `15d09131f2fbb72717a5f0d30a3868342c2ea6727556c0afa0876caf5053d4e6` |
| `src/w2/w2c_data.py` | `4fa7f9cd929592dd29a200907ba8ca07881c8702a2464df720e78bd6a5ff4c03` |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |
| `configs/e3b_train.json` | `b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |

开发机 tiny 彩排 rc=0；目录名 `0067_w2c_data_hpcgpu06` 带运行编号，合规。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0067（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试。核对上表 6 个 sha256。
3. 等"什么时候开始"的条件满足后启动（方案 B）：

   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=3 \
     bash /home/mail/chenyuma/visint/wt/wt0067/prompts/server_hpc/0067-hpc-06-w2c-propose/propose.sh \
     > /home/mail/chenyuma/visint/logs/eval_0067.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0067.status`。
   - 结果目录：`experiments/W2/0067_w2c_data_hpcgpu06/`。
4. 跑完后用 `sync_push.sh` 推一次（`ONCE=1`，TAG 0067，路径 `experiments/W2/0067_w2c_data_hpcgpu06`）。这一步只有出框，没有分片；出框的 `predictions.jsonl` 约 16 MB，照 v3 的规则推。
5. **之后 06 空着等**：W2-c 的候选与标签一步，要等 W2-a 冻结后另发任务书。

## 报告

推送 `reports/0067-hpc-06-w2c-propose.md`，内容：
- `steps.log` 原文与墙钟；
- `manifest.json` 的去重与抽样统计：各步条数、`w2_eval_exclusion.n_removed`、`checks.selected_on_w2_eval_images`，后者必须为 0；
- 出框的失败率，即 `propose/metrics.json` 的标题读数；
- 显存峰值；任何 traceback。

只贴数字，不解读。
