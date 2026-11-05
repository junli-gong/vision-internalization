[RUN-TASK 0114] hpcgpu06：X-M 的定位与计数 T 文本读数，以及大题量的 --compare（D182 第 3 点：把 X-M 的 CountBench 补进语言表）。约 1.8 GPU 小时，上限 4 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰别的项目的进程。
- 只用 06 自己的暂停文件。

## 为什么

负责人要求把 X-M 的计数补进语言表（D182）。X-2 的计数代价是 −8.1 到 −16.1，X-M 也要如实报告。

**定位与计数**
- 写法同 RUN-0091：`run_e1 --task refexp_box count --subset eval --checkpoint <X-M>`。
- 在 06 上跑，与 RUN-0069 在 06 上的 T 文本基座逐题配对。自 7f53f240 以来，run_e1、src/e1、src/adapters 没有改动。

**大题量**
- `lang_battery --compare`：07 的基座（RUN-0084 lb_base）对 X-M（RUN-0107 lb_xm），只用 CPU。
- 与四方表其余各列同一种格式。

## 文件（工作树 wt0114 建在本任务书所在的 commit）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0114-hpc-06-xm-ttext/xm_ttext.sh` | `cc6987e3b756221d0d0ba527022867d26a61b984bb05b01359875ca96a0a97f5` |
| `scripts/run_e1.py` | `5f46ed7c28b4653221fa5ef6eb6b8dc8eec952a53360b30ffa444fa1e72bfd4d` |
| `scripts/lang_battery.py` | `c6cee3ff3b610f1203c6a7e00ca7c85fbeb5b1f2a9c5d3f844641fce28d062e7` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0114，核对上表 4 个 sha256。
2. **06 上没有别的计算进程时才启动**；被占就每分钟查一次，空出来再启动。
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=4 \
     bash /home/mail/chenyuma/visint/wt/wt0114/prompts/server_hpc/0114-hpc-06-xm-ttext/xm_ttext.sh \
     > /home/mail/chenyuma/visint/logs/eval_0114.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0114.status`。
   - 推送循环 TAG 0114，路径 `experiments/XM/0114_xm_ttext_hpcgpu06`。
3. 报告 `reports/0114-hpc-06-xm-ttext.md`：
   - 定位 acc@0.5 与计数准确率；
   - 合并核对；
   - `lb_compare_xm.json` 的标题几行：MMLU-Pro 字母、IFEval prompt 严格、指令级严格；
   - 只贴，不解读。
4. 再跑一次 `squeue`。

管理员做了 tiny 彩排，只核对目录名与命令行：run_e1 那一行与 RUN-0091 逐字相同。`--compare` 在开发机上用入库的两个运行目录试过，rc=0。
