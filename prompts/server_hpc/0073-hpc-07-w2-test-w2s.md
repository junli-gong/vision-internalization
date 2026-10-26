[RUN-TASK 0073] hpcgpu07：W2 的 test 主实验之 W2-S，即 2B、4B 的 W2-a（W2 预注册 §11；D125；冻结的配置 D122）。约 8 GPU 小时，上限 14 小时，现在就可以发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06（RUN-0072）。
- 只用 07 自己的暂停文件：由 `claim_gpu` 写入，退出时删除。

## 为什么

- 负责人确认 G-W2（D125）：W2-c 进入 test，按三段式投 CVPR。W2-S（D113 第二节第 6 条）看 W2-a 的能力是否随 LLM 变强。
- test 只报告、不判定。只用冻结的 `configs/w2.json` 与 RUN-0069 训好的 W2-c 检查点。

## 文件（commit 见发出时的启动语；工作树 wt0073 建在 origin/main）

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

- 2B、4B 的权重在 `$VI/hf/hub`（E3 用过），离线可用。

## 内容（`test_w2.sh`，`RUN_ID=0073 MODELS="qwen3.5-2b qwen3.5-4b"`）

- 5 个 test 划分：refcoco testA、testB，refcoco+ testA、testB，refcocog test，各 1,000 句。
- 两个模型依次跑，每个模型、每个划分：E1 的框 → 自己的框、W1-F → **W2-a**。
- W2-c 只有 9B，这里不跑；随机、最优候选与多边形只在 9B 上跑。
- 结果目录：`experiments/W2/0073_w2_test_hpcgpu07/`。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0073（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试。核对上表的 sha256 与 W2-c 检查点。
3. 启动（方案 B；`-c 8` 被拒就去掉，并在 env 里加 `CW=1`）：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=14 NODE=hpcgpu07 RUN_ID=0073 MODELS="qwen3.5-2b qwen3.5-4b" \
     bash /home/mail/chenyuma/visint/wt/wt0073/prompts/server_hpc/0072-hpc-w2-test/test_w2.sh \
     > /home/mail/chenyuma/visint/logs/eval_0073.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0073.status`。
   - 另起一个 v3 推送循环，TAG 0073，路径 `experiments/W2/0073_w2_test_hpcgpu07`。跑完、推完后按 PID 停掉。
4. 每个模型跑完报一次：各划分 w1f、w2a 的 cIoU。
5. 全部跑完：写报告，再跑一次 `squeue`。之后 07 接 (b) 缓解变体，任务书另发。

## 报告

推送 `reports/0073-hpc-07-w2-test-w2s.md`，内容：
- `steps.log` 原文与每步墙钟；
- 每个划分各设置的 cIoU、gIoU 与 CI；
- w2a 的挑选统计；
- 显存峰值、任何 traceback。

只贴数字，不解读。
