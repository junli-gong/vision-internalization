[RUN-TASK 0066] hpcgpu07：W2 在 devtune800 上调参，三步共 9 个臂（W2 预注册 §5 与"补记"；D113、D117；W2 预算内，约 5 GPU 小时）。**等 RUN-0057 全部跑完再开始**（D113-2：先跑完 E5 事后探索）。上限 10 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0056。
- 只用 07 自己的暂停文件：由脚本的 `claim_gpu` 写入，退出时删除。
- 不写全局的 `$VI/logs/train.pause`。

## 什么时候开始

- RUN-0057 全部跑完：`train_0057-hpcgpu07.status` 是 done，07 上没有 0057 的进程，GPU 空着。约 10-24 06:00Z。
- 请挂一个监视，条件满足后立即按下面的步骤启动，不用等我。
- 本任务不让 RUN-0057 让路，所以 `claim_gpu` 只是确认 GPU 空着。

## 为什么

- W2-a 的方法：LLM 用 E1 协议给框，工具在框上生成 K 个候选，再把候选编号后叠回原图，让 LLM 只回答一个编号。
- 跑判定集 devcheck2 之前，要在 devtune800（W1 用过的 800 句）上定下：布局、挑选时开不开思考、K、提出时给不给点。
- 三步的臂与选法都事先写进了 W2 预注册 §5。驱动脚本 `tune_w2.sh` 每一步跑完后，用 `scripts/mgr/w2_tune.py select` 按规则自动选出这一步的臂，再进下一步，中间不用等管理员。
- 冻结由管理员在调参之后做，不在本任务里。
- **只用 devtune800，不碰 devcheck2**：未冻结的配置碰 devcheck2 会被 `run_w2.py` 拒绝。

## 文件（commit 见发出时的启动语；工作树 wt0066 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0066-hpc-w2-tune/tune_w2.sh` | `f9e0a3735ee5df8e7cbd49494c8767c99664035380aa12a9a8e94b62128a396c` |
| `scripts/mgr/w2_tune.py` | `1af19c6ae7c4f438f2b048c8bdcae1fe3edea0f49e01907e1b1e972ae7d2fbc9` |
| `configs/w2.json` | `a4c013c344c83f1885c82db345be11f252e706d1a833316396302aeb4abbc6fd`（`frozen: false`） |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |
| `scripts/run_w2.py` | `8b9a345ce8190735af8723eccf66449fe287d50957d7b27f7e92af5ac890b8e4` |
| `src/w2/loop.py` | `7cf956494b17f95d85af2dfb88394193cc08250d55832503725c0acb081d5dec` |
| `src/w2/runner.py` | `f8b74ab679a15061328a78b084566b1bfa0cd865ff168942d3073db0af013d67` |
| `src/w2/config.py` | `e317fdd47cc29c03ae34bb479d490d86abd2c8c9f52c1e01847a9c766b2689e4` |
| `src/w2/candidates.py` | `f03bbbfce130d937ace7d5555c5fcbc898e76c88880a2d9eb30d289207a635d5` |
| `src/w2/render.py` | `559ee42461ee2be7bdd37ea12407b2d95c89493f97a02615fd7c22c611ed4ad2` |
| `src/adapters/qwen35.py` | `64d2c6e0e4deb7cf41a44dbf1173742944a4c2d691ffd8d213b6ed3d61bdca99` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |

开发机 tiny 彩排：全部步骤 rc=0，三步都按规则选出了臂，并写出 `final.json`。非 tiny 的目录名 `0066_s0`、`0066_<臂名>` 都带运行编号，合规。

## 内容（全部在 tune_w2.sh 里）

| 步 | 内容 |
|---|---|
| 第 0 步 | E1 的框、DEFAULT_SPEC 的 8 个候选；`w1f`、`ownbox`、`random_pick`、`oracle_pick` 四个基线；按 `oracle_pick` 的 `best_index`（真值）给生成器排序 |
| 第 1 步 | grid、separate 两种布局 × 挑选时关、开思考，共 4 个臂，K = 8 |
| 第 2 步 | 第 1 步选中的臂，K 取 4 和 6；K = 8 就是第 1 步的臂 |
| 第 3 步 | 第 2 步选中的臂，提出改为 `box_points` |
| 输出 | `step1.md`–`step3.md`（各臂的读数与选中的臂）、`ranking.json`、`final.json`、`cfg/` 下各臂的配置 |

- 结果目录：`experiments/W2/0066_w2_tune_hpcgpu07/`。
- 各臂复制第 0 步的框，候选各自重新生成，用 `--candidate-workers 8`。
- 预计：
  - 第 0 步约 25 分钟；
  - 每个臂约 8 分钟生成候选，加上挑选：关思考约 10 分钟，开思考约 30–45 分钟；
  - 合计约 4–6 小时。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0066（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试。核对上表 12 个 sha256。
3. 等"什么时候开始"的条件满足后启动（方案 B；`-c 8` 被拒就去掉，并在 env 里加 `CW=1`）：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=10 NODE=hpcgpu07 TQ_STATUS=train_0057-hpcgpu07.status \
     bash /home/mail/chenyuma/visint/wt/wt0066/prompts/server_hpc/0066-hpc-w2-tune/tune_w2.sh \
     > /home/mail/chenyuma/visint/logs/eval_0066.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0066.status`。
   - 另起一个 v3 推送循环，TAG 0066，路径 `experiments/W2/0066_w2_tune_hpcgpu07`。跑完、推完后按 PID 停掉。
4. 每一步选完臂（`steps.log` 里出现 `stepN chosen`），把 `stepN.md` 原文发给管理员。
5. 全部跑完：写报告，再跑一次 `squeue`。
   - 中途 ABORT（某个臂失败）：驱动会写 blocked 心跳并推送已有的结果。照实报告，不要自己重跑。

## 报告

推送 `reports/0066-hpc-07-w2-tune.md`，内容：
- `steps.log` 原文；
- `step1.md`–`step3.md` 原文；
- `ranking.json`、`final.json` 原文；
- 第 0 步四个基线与各臂 `metrics.json` 的标题读数（cIoU 与 CI）；
- 显存峰值、每步墙钟、任何 traceback。

只贴数字，不解读。
