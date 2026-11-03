[RUN-TASK 0107] hpcgpu07：X-M 判定第 3、4 条的评测：2,300 句开发数据上的指代分割与大题量语言，与 X-2 seed 0 同节点配对（XM-4b、XM-5、XM-5a）。约 4.5 GPU 小时，上限 8 小时；**排在 RUN-0104 训完之后**

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰别的项目（lly-mle）的进程。
- 只用 07 自己的暂停文件。

## 为什么

- 判定第 3 条：X-M 在 2,300 句开发数据上的 cIoU 对 X-2 seed 0（RUN-0080，07）。
- 判定第 4 条：大题量对同节点基座（RUN-0084 的 lb_base，07），D170。
- 两项都必须在 07 上跑。命令同 RUN-0093 的指代分割、RUN-0101 的大题量，只换 X-M 的检查点。
- **这些读数在门报告出来之前也不看**：报告里只贴 complete、失败数与用时。

## 顺序

- RUN-0106 在 06 上时，本任务在 RUN-0104 训完后立即在 07 上开始（缺省 `WAIT_DONE`）。
- RUN-0106 也在 07 上时，启动时给 `WAIT_DONE="eval_0106.status:done 0106"`，排在它之后。

## 文件（工作树 wt0107 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0107-hpc-07-xm-refseg-lb/xm_refseg_lb.sh` | `a421561d8edfe4275c6248f6ae879a1270ebe3decc61bd42d8ce06bad8de3ea0` |
| `scripts/run_x2.py` | `08f9962bea2ae6b60a70ea0940a9bbf53581ead9bf2ef76e6b9c24042d577bcf` |
| `src/explore/x2_runner.py` | `1ef0f768e11bbde99b569634ef47fc1d386f84a3fe64e3da62788d54ca64eff7` |
| `scripts/lang_battery.py` | `c6cee3ff3b610f1203c6a7e00ca7c85fbeb5b1f2a9c5d3f844641fce28d062e7` |
| `configs/x2_run.json` | `c92ad838a9e698236e8a34d17f6ce187ee185ffaa7df96df5ee99a04f47dc061` |
| `configs/explore_splits.json` | `df9f6c19141cee8c012dd3b6606e5b12f00db92557d43f07c7ac2214a3ab7892` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0107（稀疏），核对上表 8 个 sha256。
2. 启动：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 12 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=8 [WAIT_DONE=…] \
     bash /home/mail/chenyuma/visint/wt/wt0107/prompts/server_hpc/0107-hpc-07-xm-refseg-lb/xm_refseg_lb.sh \
     > /home/mail/chenyuma/visint/logs/eval_0107.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0107.status`。推送循环 TAG 0107，路径 `experiments/XM/0107_xm_refseg_lb_hpcgpu07`。
3. 跑完写 `reports/0107-hpc-07-xm-refseg-lb.md`：只贴 complete、失败数与用时，不贴 cIoU 与语言读数。再跑一次 `squeue`。

管理员在开发机上用 tiny 彩排过：指代分割与大题量都是 rc=0。正式分支的参数对过 run_x2 与 lang_battery 的限制：大题量的 `--data-root` 只配 `--checkpoint`（D161）。
