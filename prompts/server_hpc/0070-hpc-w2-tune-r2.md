[RUN-TASK 0070] hpcgpu06 + hpcgpu07：W2-a 调参第 2 步重做，以 sep_on 为底跑 K = 4、6 两个臂，06、07 各跑一个；之后在 hpcmgt 上带仪器有效性过滤重新选臂（W2 预注册 §5 补记 2；D121；约 2 GPU 小时）。现在就可以发，两张卡都空着

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 只用各节点自己的暂停文件：由脚本的 `claim_gpu` 写入，退出时删除。

## 为什么

- RUN-0066 按原规则选中了 `sep_off_k4`，但这个臂的挑选几乎都被截断、回退到 1 号，等于没有在挑选。
- D121 加了"仪器有效性"过滤：挑选解析失败率或出框失败率超过 10% 的臂不参选。
- 据此第 1 步改选 sep_on，第 2 步要以 sep_on 为底重跑 K = 4、6。K = 8 就是 RUN-0066 的 sep_on。
- 只用 devtune800，不碰 devcheck2。

## 文件（commit 见发出时的启动语；工作树 wt0070 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0070-hpc-w2-tune-r2/arm.sh` | `085a750132c4924f6522772f2c942ad3ddcd2471f2a3f104b963943a2004a5f6` |
| `scripts/mgr/w2_tune.py` | `de578bb48a6ac61ab9c8afa0cd9ce39a494ffc0219133800fedeeb65827eb476` |
| `scripts/run_w2.py` | `4ff69ad10a5b4a6c589fe96f5eb4a8f6ac36fc446af9f0dcb7e6746a04b46c27` |
| `src/w2/loop.py` | `f0b077649d1e4495b86da0d5e872261598106356e19239332b9d01c2b800d662` |
| `src/w2/runner.py` | `640092bb9cb08d74a58ec289315a1c3708367b3ab376502b6da76802291786ce` |
| `src/w2/config.py` | `46a00f288d949b3c093af57ec4c5fd015d8ea42cb870c588a83b1b9804fa546c` |
| `src/w2/candidates.py` | `f03bbbfce130d937ace7d5555c5fcbc898e76c88880a2d9eb30d289207a635d5` |
| `src/w2/render.py` | `559ee42461ee2be7bdd37ea12407b2d95c89493f97a02615fd7c22c611ed4ad2` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |

- W2 的回路代码比 RUN-0066 用的新：DEV-0061 加了 w2c 设置，按它的完成总结，现有设置的输出不变。
- 底配置 `cfg/sep_on.json`、`ranking.json` 与第 0 步的框，都取 RUN-0066 在 Lustre 上的 `$VI/repo/experiments/W2/0066_w2_tune_hpcgpu06/`。脚本开头会核对它们在。
- 开发机 tiny 彩排：两个臂都 rc=0。选臂的过滤用手造的记录核对过：截断的臂与出框失败的臂都被排除，在其余的臂里选差最大的。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0070（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试。核对上表 9 个 sha256。
3. 两个臂同时启动（方案 B；`-c 8` 被拒就去掉，并在 env 里加 `CW=1`）：

   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=3 NODE=hpcgpu06 K=4 \
     bash /home/mail/chenyuma/visint/wt/wt0070/prompts/server_hpc/0070-hpc-w2-tune-r2/arm.sh \
     > /home/mail/chenyuma/visint/logs/eval_0070_k4.driver.log 2>&1 < /dev/null &'
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=3 NODE=hpcgpu07 K=6 \
     bash /home/mail/chenyuma/visint/wt/wt0070/prompts/server_hpc/0070-hpc-w2-tune-r2/arm.sh \
     > /home/mail/chenyuma/visint/logs/eval_0070_k6.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0070_k4.status` 与 `eval_0070_k6.status`。
   - 结果目录：`experiments/W2/0070_w2_tune_r2_hpcgpu06/`（K = 4）与 `0070_w2_tune_r2_hpcgpu07/`（K = 6）。
4. 两个都 `done` 之后，在 hpcmgt 的 wt0070 里选臂。只用 CPU，读 Lustre 上的文件：

   ```bash
   VI=/home/mail/chenyuma/visint; E=$VI/repo/experiments/W2; R=$E/0066_w2_tune_hpcgpu06; M=qwen3.5-9b; L=refseg/w2a/devtune800/predictions.jsonl
   O=$E/0070_w2_tune_r2_hpcgpu06
   COMMON=(--w1f $R/0066_s0/$M/refseg/w1f/devtune800/predictions.jsonl --random $R/0066_s0/$M/refseg/random_pick/devtune800/predictions.jsonl \
           --oracle $R/0066_s0/$M/refseg/oracle_pick/devtune800/predictions.jsonl --max-parse-failure 0.10 --max-propose-failure 0.10)
   # 第 1 步按 D121 重选（只为记录；用 RUN-0066 的结果）
   $VI/env/bin/python scripts/mgr/w2_tune.py select "${COMMON[@]}" --arm grid_off=$R/0066_grid_off/$M/$L --arm sep_off=$R/0066_sep_off/$M/$L \
     --arm grid_on=$R/0066_grid_on/$M/$L --arm sep_on=$R/0066_sep_on/$M/$L --out $O/step1r.json --md $O/step1r.md --step "第 1 步（D121 重选）"
   # 第 2 步重做（K4 < K6 < K8）
   $VI/env/bin/python scripts/mgr/w2_tune.py select "${COMMON[@]}" --arm sep_on_k4=$O/0070_sep_on_k4/$M/$L \
     --arm sep_on_k6=$E/0070_w2_tune_r2_hpcgpu07/0070_sep_on_k6/$M/$L --arm sep_on=$R/0066_sep_on/$M/$L \
     --out $O/step2r.json --md $O/step2r.md --step "第 2 步（D121 重做）"
   ```

   - 第 3 步不重跑：`box_points` 的出框失败 88%，按 D121 无效。
5. 两个目录各用 `sync_push.sh` 推一次（`ONCE=1`，TAG 0070）。
6. 把 `step1r.md`、`step2r.md` 原文发给管理员。第 2 步选中的臂就是最终的臂：
   - K = 4 → `0070_w2_tune_r2_hpcgpu06/cfg/sep_on_k4.json`；
   - K = 6 → `0070_w2_tune_r2_hpcgpu07/cfg/sep_on_k6.json`；
   - K = 8 → `0066_w2_tune_hpcgpu06/cfg/sep_on.json`。
7. 再跑一次 `squeue`。之后两张卡空着等管理员冻结，然后 07 接 RUN-0068，06 接 RUN-0069。

## 报告

推送 `reports/0070-hpc-w2-tune-r2.md`，内容：
- 两个 `steps.log` 原文与墙钟；
- 两个臂 `metrics.json` 的标题读数；
- `step1r.md`、`step2r.md` 原文；
- 显存峰值、任何 traceback。

只贴数字，不解读。
