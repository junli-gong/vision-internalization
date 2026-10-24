[RUN-TASK 0068] hpcgpu07：W2-a 的去留判定（D113 第三节；W2 预注册 §6；冻结的配置 D122）。devcheck2 500 句、Qwen3.5-9B。约 1–2 GPU 小时，上限 4 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06（RUN-0069）。
- 只用 07 自己的暂停文件：由 `claim_gpu` 写入，退出时删除。

## 为什么

- W2-a 的协议已在 devtune800 上调参、冻结（D121、D122）：`configs/w2.json` 是 `frozen: true`。
- 判定规则写死在 `scripts/w2_gate.py`：W2-a 减 W1-F 的 cIoU 差，配对 bootstrap 97.5% CI（重抽样 10,000 次），下界 > 0 才算通过。
- devcheck2 至今没有任何运行。

## 文件（commit 见发出时的启动语；工作树 wt0068 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0068-hpc-w2a-devcheck2/gate_w2a.sh` | `9019ae55f398bed6a1c8594ec2de624b64fe5f46a254946ed22302238dcbb8be` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29`（冻结，D122） |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |
| `scripts/run_w2.py` | `4ff69ad10a5b4a6c589fe96f5eb4a8f6ac36fc446af9f0dcb7e6746a04b46c27` |
| `scripts/w2_gate.py` | `90f7102937d63f6615155ccffdcbd53a2abcd7f38a06995c69dd5b41dde40f86` |
| `src/w2/loop.py` | `f0b077649d1e4495b86da0d5e872261598106356e19239332b9d01c2b800d662` |
| `src/w2/runner.py` | `640092bb9cb08d74a58ec289315a1c3708367b3ab376502b6da76802291786ce` |
| `src/w2/config.py` | `46a00f288d949b3c093af57ec4c5fd015d8ea42cb870c588a83b1b9804fa546c` |
| `src/w2/candidates.py` | `f03bbbfce130d937ace7d5555c5fcbc898e76c88880a2d9eb30d289207a635d5` |
| `src/w2/render.py` | `559ee42461ee2be7bdd37ea12407b2d95c89493f97a02615fd7c22c611ed4ad2` |
| `src/adapters/qwen35.py` | `64d2c6e0e4deb7cf41a44dbf1173742944a4c2d691ffd8d213b6ed3d61bdca99` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |

## 内容（`prompts/server_hpc/0068-hpc-w2a-devcheck2/gate_w2a.sh`）

| 步 | 内容 |
|---|---|
| `e1_box` | base 9B 的 E1 框，devcheck2 500 句 |
| `w1f` | W1-F（1 号候选），判定的对照 |
| `w2a` | 主配置 |
| `gate` | `w2_gate.py --main-name W2-a`，写 `gate_w2a/gate.md`、`gate.json` |
| `ownbox`、`random_pick`、`oracle_pick` | 参照，只报告 |

- 结果目录：`experiments/W2/0068_w2a_devcheck2_hpcgpu07/`。
- 驱动开头核对 `configs/w2.json` 已冻结，没冻结就拒跑。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0068（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试。核对上表的 sha256。
3. 启动（方案 B；`-c 8` 被拒就去掉，并在 env 里加 `CW=1`）：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=4 NODE=hpcgpu07 TQ_STATUS=nonexistent_queue.status \
     bash /home/mail/chenyuma/visint/wt/wt0068/prompts/server_hpc/0068-hpc-w2a-devcheck2/gate_w2a.sh \
     > /home/mail/chenyuma/visint/logs/eval_0068.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0068.status`。
   - 另起一个 v3 推送循环，TAG 0068；跑完、推完后按 PID 停掉。
4. **`gate` 一步跑完，立即把 `gate_w2a/gate.md` 原文发给管理员**，不等后面的参照。
5. 全部跑完：写报告，再跑一次 `squeue`。

## 报告

推送 `reports/0068-hpc-07-w2a-devcheck2.md`，内容：
- `steps.log` 原文；
- 各设置的标题读数（cIoU、gIoU 与 CI）；
- `gate.md` 原文；
- w2a 的挑选统计：挑中最优、挑 1 号、都不对、解析失败的比例；
- 显存峰值、墙钟、任何 traceback。

只贴数字，不解读。
