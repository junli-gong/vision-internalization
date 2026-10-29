[RUN-TASK 0096] 主方法 C1 的 seed 1、seed 2 的语言读数（D153）。07 现在就发；06 同时发，驱动等 RUN-0094 完成。约 7.5 GPU 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上在跑的 RUN-0094；06 的驱动自己等它完成。
- 只用各节点自己的暂停文件。

## 为什么

- seed 1（RUN-0093）的健康检查在 IFEval 上标红：50 条里 −12。seed 0 是 −4，大题量 541 条上是 −3.5。
- 论文的语言代价用的是 seed 0。这里用大题量和 T 文本测 seed 1、2，看语言代价换种子后是否站得住。
- 只描述。不碰 test 与 devcheck3，不改任何已有的结论。

## 文件（工作树 wt0096 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0096-hpc-lang-seeds/lang_seeds.sh` | `bc97d8e44e48d00099927d19578b8390a156383c7803cb4d8ddf86f6d58e1f43` |
| `scripts/lang_battery.py` | `c6cee3ff3b610f1203c6a7e00ca7c85fbeb5b1f2a9c5d3f844641fce28d062e7` |
| `scripts/health_check.py` | `4728a0ca26433f5c02b8b3c064b637a437a32de4c1a1847056614aee2ec5c926` |
| `src/health/run.py` | `ca4a995857d69fa080ba6a01aac4225c076e69ef0b4150c6f84cf3a5dfad1437` |
| `src/health/scoring.py` | `05d11f230fe66b67abcabdff2064bc1cb72bb8401cbdc042e0515378871111be` |
| `scripts/run_e1.py` | `5f46ed7c28b4653221fa5ef6eb6b8dc8eec952a53360b30ffa444fa1e72bfd4d` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

除驱动外，其余 6 个与 RUN-0084、RUN-0091 的任务书逐字相同。命令行的参数与那两次逐项相同，只换了检查点、标签与 `--out`。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0096（稀疏），建在 origin/main，核对上表 7 个 sha256。
2. **07**（现在发；07 空闲）：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 NODE=hpcgpu07 EV_CAP_H=4 \
     bash /home/mail/chenyuma/visint/wt/wt0096/prompts/server_hpc/0096-hpc-lang-seeds/lang_seeds.sh \
     > /home/mail/chenyuma/visint/logs/eval_0096-hpcgpu07.driver.log 2>&1 < /dev/null &'
   ```
   - 跑 seed 1 的大题量，再 `--compare` RUN-0084 在 07 上的 `lb_base`。
   - 输出在 `experiments/X/0096_lang_seeds_hpcgpu07`。
3. **06**（同时发；驱动等 `eval_0094.status` 以 `done 0094` 开头）：命令同上，换成 `--jobid=157296`、`NODE=hpcgpu06`、`EV_CAP_H=8`、日志 `eval_0096-hpcgpu06.driver.log`。
   - 依次跑 seed 1、seed 2 的 T 文本定位与计数（`run_e1 --task refexp_box count`），再跑 seed 2 的大题量，并 `--compare` RUN-0074 在 06 上的 `lb_base`。
   - 输出在 `experiments/X/0096_lang_seeds_hpcgpu06`。
4. 心跳是 `eval_0096-hpcgpu07.status` 与 `eval_0096-hpcgpu06.status`。推送循环各一个，TAG 分别是 0096-07、0096-06。
5. 每个 `--compare` 跑完，原文照贴；T 文本贴两个任务的 headline。
6. 两个节点都跑完后写 `reports/0096-hpc-lang-seeds.md`，再跑一次 `squeue`。

上限 EV_CAP_H 从拿到 GPU 时开始计，06 等待 RUN-0094 的时间不算在内。管理员在开发机上用 tiny 彩排过：目录名与两个节点的命令行都正确；NODE 写错时拒绝。
