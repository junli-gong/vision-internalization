[RUN-TASK 0108] hpcgpu07（只用 CPU）：X-M 的去留判定 G-XM。用 gate_xm 判一次（D165 第五节；预注册 XM-5、XM-5a（D170）、XM-5b；D176）。约 0.2 小时，上限 2 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上 lly-mle 的进程。
- 本任务不占 GPU，不写暂停文件。

## 为什么

RUN-0106（重跑）与 RUN-0107 都跑完了，判定的输入齐了，三处配对都在 07 上：
- 判定集：`0106_xm_dec_r2`（X-M）与 `0106_base_dec`（基座），按 D176 的选法；
- 指代分割：RUN-0107 的 X-M 对 RUN-0080 的 X-2 seed 0，共 2,300 句、5 个划分；
- 大题量：RUN-0107 的 `0107_lb_xm` 对 RUN-0084 的 `lb_base`（第 4 条，D170），以及对 `lb_x2`（D165 原文，只作诊断）。

gate_xm 的规则与阈值写死在脚本里，不带任何可调参数。只判一次：`gate_xm.json` 已存在时，驱动会拒绝再跑。

## 文件（工作树 wt0108 建在 origin/main，稀疏签出 `src/`、`scripts/`、`configs/`、`prompts/`）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0108-hpc-07-gate-xm/gate_xm.sh` | `9ac70829031ae2fbca40f7a3fc7d81a30dda54d7a450480fa1535e9596fe5726` |
| `scripts/gate_xm.py` | `d4971e8c88b011f499b735a7e17b413eb955a5627a998159699b8427dc4ed15b` |
| `src/xm/data.py` | `f51759702f8dc5b67c4775859df97d64334fb0b9df03e5c7a430c01ba7a73870` |
| `src/xm/metrics.py` | `448ebb26139d0df2f3555d67108b0316254cda77021ce526105e852b5bd85df9` |
| `src/stats.py` | `3f011218898c38aae18c22ca8bf704c59fbdc2f74d795b8c3c20a1828a4bf9b1` |
| `src/constants.py` | `1fe1d46df8158812f32686358892afbfd3e3283c5cd11c452804e80555180ba9` |
| `src/e3/train.py` | `9abc2c504db9be2293ae883e783bf332fb4f95b038f896d09f51b7ec7c655280` |
| `scripts/lang_battery.py` | `c6cee3ff3b610f1203c6a7e00ca7c85fbeb5b1f2a9c5d3f844641fce28d062e7` |
| `configs/xm_splits.json` | `995345f470111e3ac73c9681754dd46eca66da7d9ca0b648873cc1429219db8d` |
| `configs/xm_train_exclusions.json` | `b50395674de3a33d23e028ca98cf18f06a62f982c7847005a40cbe2955546dc3` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0108，核对上表 11 个 sha256。
2. 启动：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=2 \
     bash /home/mail/chenyuma/visint/wt/wt0108/prompts/server_hpc/0108-hpc-07-gate-xm/gate_xm.sh \
     > /home/mail/chenyuma/visint/logs/eval_0108.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0108.status`。
   - 驱动先逐个核对输入目录与文件，缺任何一个就写 blocked、不判定。
   - gate_xm 自己的核对不过时，退出码 2，什么都不写。这时把 `logs/gate_xm.log` 原文发回，不要重跑，等管理员处理。
3. 推送 `experiments/XM/0108_gate_xm_hpcgpu07`，用一次 ONCE 推送即可，不必开同步循环。
4. 写 `reports/0108-hpc-07-gate-xm.md`：
   - steps.log 原文、gate_xm 的退出码；
   - `0108_gate/gate_xm.md` 全文照贴（判定就是揭盲，可以贴数字）；
   - `gate_xm.json` 的 verdict、rehearsal、checks 三项原样；
   - 只贴，不解读。
5. 再跑一次 `squeue`。

管理员在开发机上做了布局彩排（`TINY=1 REPO=<本地仓库>`）：除 `0106_xm_dec_r2/…/xm_spline_trimap/xm_decision/predictions.jsonl` 外，所有输入文件都在 git 里，那一个列在 LARGE_FILES.manifest.tsv 里，只在 HPC 上。gate_xm 的参数逐条对过：`--gate G-XM`、`--criterion4 d170`（缺省值，也是唯一取值）、指代分割两边各 5 个叶目录，没给 `--splits`（用入库清单）。gate_xm 的逻辑由 `tests/test_gate_xm.py` 覆盖（DEV-0081，main 上 164 条测试通过）。
