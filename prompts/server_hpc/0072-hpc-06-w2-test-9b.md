[RUN-TASK 0072] hpcgpu06：W2 的 test 主实验，9B 的全部设置（W2 预注册 §11；D125 确认 G-W2；冻结的配置 D122）。约 10 GPU 小时，上限 16 小时，现在就可以发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 07（RUN-0073）。
- 只用 06 自己的暂停文件：由 `claim_gpu` 写入，退出时删除。

## 为什么

- 负责人确认 G-W2（D125）：W2-c 进入 test，按三段式投 CVPR。
- test 只报告、不判定。只用冻结的 `configs/w2.json` 与 RUN-0069 训好的 W2-c 检查点。

## 文件（commit 见发出时的启动语；工作树 wt0072 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0072-hpc-w2-test/test_w2.sh` | `3d75782e41532d068a75f9a58b96c7033d5092ed42c2759466868f3904e5c1b4` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |
| `scripts/run_w2.py` | `4ff69ad10a5b4a6c589fe96f5eb4a8f6ac36fc446af9f0dcb7e6746a04b46c27` |
| `src/w2/loop.py` | `f0b077649d1e4495b86da0d5e872261598106356e19239332b9d01c2b800d662` |
| `src/w2/runner.py` | `640092bb9cb08d74a58ec289315a1c3708367b3ab376502b6da76802291786ce` |
| `src/w2/config.py` | `46a00f288d949b3c093af57ec4c5fd015d8ea42cb870c588a83b1b9804fa546c` |
| `src/w2/candidates.py` | `f03bbbfce130d937ace7d5555c5fcbc898e76c88880a2d9eb30d289207a635d5` |
| `src/w2/render.py` | `559ee42461ee2be7bdd37ea12407b2d95c89493f97a02615fd7c22c611ed4ad2` |
| `src/w2/w2c.py` | `61b5c9b166c98c7e9a6693763b33c0243bf7dd86eb1334a9c199be0115a8dcf9` |
| `src/adapters/qwen35.py` | `64d2c6e0e4deb7cf41a44dbf1173742944a4c2d691ffd8d213b6ed3d61bdca99` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |

- W2-c 检查点：`$VI/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`，在 Lustre 上，开跑前与 RUN-0069 的 train_summary 核对它的 `manifest.json` 的 sha256。

## 内容（`test_w2.sh`，`RUN_ID=0072 MODELS=qwen3.5-9b`）

- 5 个 test 划分：refcoco testA、testB，refcoco+ testA、testB，refcocog test，各 1,000 句。
- 每个划分依次跑：
  - E1 的框；
  - 自己的框、W1-F；
  - 随机挑选、最优候选；
  - E1 零样本多边形（`baseline_e1` 的 refseg，参照）；
  - **W2-a**；
  - **W2-c**。
- 结果目录：`experiments/W2/0072_w2_test_hpcgpu06/`。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0072（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试。核对上表的 sha256 与 W2-c 检查点。
3. 启动（方案 B；`-c 8` 被拒就去掉，并在 env 里加 `CW=1`）：

   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=16 NODE=hpcgpu06 RUN_ID=0072 MODELS=qwen3.5-9b \
     bash /home/mail/chenyuma/visint/wt/wt0072/prompts/server_hpc/0072-hpc-w2-test/test_w2.sh \
     > /home/mail/chenyuma/visint/logs/eval_0072.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0072.status`。
   - 另起一个 v3 推送循环，TAG 0072，路径 `experiments/W2/0072_w2_test_hpcgpu06`。跑完、推完后按 PID 停掉。
4. 每跑完一个划分报一次：w1f、w2a、w2c 的 cIoU。
5. 全部跑完：写报告，再跑一次 `squeue`。之后 06 接 LLMSeg 的 9B 基线，任务书另发。

## 报告

推送 `reports/0072-hpc-06-w2-test-9b.md`，内容：
- `steps.log` 原文与每步墙钟；
- 每个划分各设置的 cIoU、gIoU 与 CI；
- w2a、w2c 的挑选统计；
- 显存峰值、任何 traceback。

只贴数字，不解读。
