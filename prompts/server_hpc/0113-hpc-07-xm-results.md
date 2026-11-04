[RUN-TASK 0113] hpcgpu07（只用 CPU）：X-M 结果汇总的配对 CI（管理员的分析脚本 `scripts/mgr/xm_results_table.py`，只读结果目录）。约 0.2 小时，上限 1 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不占 GPU，不写暂停文件。

## 为什么

最终报告集与 LVIS 的配对差要用逐对的 predictions.jsonl。这些文件有一部分只在 HPC 上，列在 LARGE_FILES.manifest.tsv 里，开发机算不了。脚本只读 `$REPO/experiments/` 下的结果，写一个新目录，不改任何已有结果。要算的配对差：
- 检测：X-M − 基座；X-M − Rex-Omni，joint 与逐类两种。
- 实例分割：主设置分别减 X-M 的框交给 GrabCut、基座的框交给 GrabCut、框内椭圆、样条直接光栅化。
- RefCOCO test 上 X-M 对 X-2，以及 ReasonSeg 上 X-M 对 X-2。

## 文件（工作树 wt0113 建在本任务书所在的 commit，稀疏签出 `src/`、`scripts/`、`configs/`、`prompts/`）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0113-hpc-07-xm-results/xm_results.sh` | `1766a4c350058edd19869125ca578ff2036e11f361cea6cf5e7a22ea15de93d3` |
| `scripts/mgr/xm_results_table.py` | `bcb5c148713c51363b402ce9eec7f7d89279213fec1ea99fa57f03ead2a6b572` |
| `scripts/mgr/x_test_table.py` | `5e660fe3a13a91b1449f6c8fdad6ea19a03baa45971d15f0a70bb61e0af26e92` |
| `scripts/mgr/w2_test_table.py` | `1f085a7c7f1068cdd3a81bc91bbd0dd23345df291f872f4e80f6332dff6a7b02` |
| `src/xm/metrics.py` | `448ebb26139d0df2f3555d67108b0316254cda77021ce526105e852b5bd85df9` |
| `src/stats.py` | `3f011218898c38aae18c22ca8bf704c59fbdc2f74d795b8c3c20a1828a4bf9b1` |
| `src/constants.py` | `1fe1d46df8158812f32686358892afbfd3e3283c5cd11c452804e80555180ba9` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0113，核对上表 8 个 sha256。
2. 确认 `$REPO` 里有 RUN-0106、0109、0111、0112 的全部叶目录，包括只在 HPC 上的 predictions.jsonl。
3. 启动：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=1 \
     bash /home/mail/chenyuma/visint/wt/wt0113/prompts/server_hpc/0113-hpc-07-xm-results/xm_results.sh \
     > /home/mail/chenyuma/visint/logs/eval_0113.driver.log 2>&1 < /dev/null &'
   ```
4. 一次 ONCE 推送 `experiments/XM/0113_xm_results_hpcgpu07`。
5. 报告 `reports/0113-hpc-07-xm-results.md`：steps.log 与 `xm_results.md` 全文照贴。表里写"缺"的格子要逐个列出缺哪个文件。
6. 再跑一次 `squeue`。

管理员在开发机上彩排过：TINY=1，`--root` 指向含 experiments/ 的目录，rc=0。开发机上能算的格子（test、ReasonSeg、各叶目录的点估计）与直接运行逐字相同。
