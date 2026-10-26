# RUN-0079 报告：hpcgpu07（只用 CPU），探索 X-1b 的噪声敏感度，noise 网格（devtune800 + devcheck2）

任务书 `prompts/server_hpc/0079-hpc-07-explore-noise.md`。不用 GPU，没有写暂停文件，没有调 `claim_gpu`；没有碰 RUN-0072、0073、0074。只用 devtune800 与 devcheck2。只贴数字，不解读。

## 工作树

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0079.sh`（10-26 06:21:32Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0079（稀疏）HEAD `68c983eb676434b258fac6a53102c1acc1a8635f`，status 空，`experiments/` 只有 P0；6 个 sha256 与任务书逐字相同（`configs/explore_ceiling.json` `81c10fdd…`）。
- 第 3 步（bench）按任务书跳过，沿用 RUN-0077 的换算。

## 正式运行

- `$VI/tmp/run0079.sh`（sha256 `fca4e2bb…`）`cat` 核对：路径都已展开为绝对路径，四个结果目录都存在（与 RUN-0077 相同的四个）。原文：

```bash
set -e
cd /home/mail/chenyuma/visint/wt/wt0079
C="python scripts/explore_ceiling.py --data-root /home/mail/chenyuma/visint/b200/data --grid noise --workers 12 --out /home/mail/chenyuma/visint/repo/experiments/X/0079_ceiling_hpcgpu07"
$C --splits devtune800 --box-run base=/home/mail/chenyuma/visint/repo/experiments/W2/0070_w2_tune_r2_hpcgpu06/0070_sep_on_k4/qwen3.5-9b/refexp_box/baseline_e1/devtune800 $RES
$C --splits devcheck2 --box-run base=/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refexp_box/baseline_e1/devcheck2 \
  --box-run w2c=/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refexp_box/propose_w2c/devcheck2 \
  --start-mask w2c_final=w2c:/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refseg/w2c/devcheck2 $RES
```

- 启动：06:21:42Z，`ssh -n hpcgpu07 "setsid nohup srun --jobid=157297 --overlap -n1 -c 12 env $ENVV RES= bash $VI/tmp/run0079.sh > $VI/logs/run_0079.log …"`。
- 退出码与墙钟：srun 步 `157297.109`，sacct `COMPLETED`、ExitCode `0:0`，10-26 06:21:43Z – 07:17:31Z（3,348 s，0.93 h）。

| 划分 | job | wall_s | workers | CPU 秒合计 | 每 job CPU 秒均值 | learned_modules_in_workers |
|---|---:|---:|---:|---:|---:|---|
| devtune800 | 1600 | 1,623 | 12 | 13,477 | 8.42（gen 2.257，pts 0.000，spl_raster 0.172，spl_gc 3.871，cor_grabcut 2.110） | [] |
| devcheck2 | 1500 | 1,708 | 12 | 14,628 | 9.75（gen 2.445，pts 0.000，spl_raster 0.172，spl_gc 4.076，cor_grabcut 3.046） | [] |

- **负载与核数**：没有降过核数（一直 12）。
  - 06:22:44Z load 8.78 / 3.70 / 5.66（48 核）。
  - 06:30:28Z load 11.32 / 9.59 / 7.82；devtune800 512/1600 job（490 s）；07 的 GPU 上只有 RUN-0073（pid 667543，9,796 MiB）。
  - 07:32:47Z（跑完后）load 1.17。
  - RUN-0073 2B w2a 各划分墙钟：refcoco testA 3,907 s（03:44Z 完），refcoco testB 4,434 s（05:05Z 完，大部分与 RUN-0077 重叠），refcoco+ testA 4,789 s（05:16:50Z – 06:36:39Z，后 15 分钟与本任务重叠）。
  - RUN-0076A 整个期间都是 "waiting for gpu (running: eval_0073.status)"，data 还没开始。
- `run_0079.log` 只有进度行与两行"写出 …metrics.json、summary.md、compute.json"；grep `warn|traceback|error` 为 0。

## 输出目录 `experiments/X/0079_ceiling_hpcgpu07/`

| 文件 | 字节 |
|---|---:|
| `devcheck2/compute.json` | 1168 |
| `devcheck2/config.yaml` | 1744 |
| `devcheck2/logs/run_1.json` | 1018 |
| `devcheck2/logs/timing.jsonl` | 425528 |
| `devcheck2/meta.json` | 890 |
| `devcheck2/metrics.json` | 93843 |
| `devcheck2/predictions.jsonl` | 5135576 |
| `devcheck2/summary.md` | 18809 |
| `devtune800/compute.json` | 1176 |
| `devtune800/config.yaml` | 1211 |
| `devtune800/logs/run_1.json` | 745 |
| `devtune800/logs/timing.jsonl` | 452656 |
| `devtune800/meta.json` | 890 |
| `devtune800/metrics.json` | 52399 |
| `devtune800/predictions.jsonl` | 4790148 |
| `devtune800/summary.md` | 11342 |

- 两个 `predictions.jsonl`（4.8 MB、5.1 MB）都在推送的大文件线（20 MB）以下，没有压缩。
- 推送：`ONCE=1`（TAG 0079）`2026-10-26T07:32:46Z ok`。后 squeue：157296、157297 RUNNING。

## 局部 GrabCut 退回 direct 的计数（`info.cor.fallback_direct`）

由 `experiments/X/0077_ceiling_probe/p77_counts.py`（RUN-0077 的探针，原样复用）读出：

```
devtune800 box_kind=base: jobs 800, failure 14, 没有 info.cor 14, 有退回的 job 43, 退回 direct 的格点合计 96
devtune800 box_kind=gt: jobs 800, failure 0, 没有 info.cor 0, 有退回的 job 18, 退回 direct 的格点合计 36
devtune800 合计: jobs 1600, failure 14, 没有 info.cor 14, 有退回的 job 61, 退回 direct 的格点合计 132
devcheck2 box_kind=base: jobs 500, failure 4, 没有 info.cor 4, 有退回的 job 32, 退回 direct 的格点合计 72
devcheck2 box_kind=gt: jobs 500, failure 0, 没有 info.cor 0, 有退回的 job 16, 退回 direct 的格点合计 44
devcheck2 box_kind=w2c: jobs 500, failure 0, 没有 info.cor 0, 有退回的 job 30, 退回 direct 的格点合计 91
devcheck2 合计: jobs 1500, failure 4, 没有 info.cor 4, 有退回的 job 78, 退回 direct 的格点合计 207
```

## devtune800 的 `summary.md` 原文

# X-1 上限（devtune800，格点 noise）

- 800 句；框 IoU ≥ 0.9 记好框；CI 为逐句 bootstrap 95%（1000 次，种子 20260929）。
- token：Qwen/Qwen3.5-9B@c20223623576，格式 x1-text-v1（只数这个通道的输出，框不计）。

## 要点：每种框上的参照与各通道、各噪声档里 cIoU 最高的格点

| 框 | 行 | 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | token 均值 |
|---|---|---|---|---|---|---|
| gt | 参照 | `box` | 0.550 [0.534, 0.566] | 0.550 | — | 0.0 |
| gt | 参照 | `gen.gc_default` | 0.621 [0.599, 0.644] | 0.621 | — | 0.0 |
| gt | 参照 | `genK.4` | 0.700 [0.682, 0.718] | 0.700 | — | 1.0 |
| gt | spl 最高（none） | `spl.n24.raster.none` | 0.943 [0.938, 0.947] | 0.943 | — | 236.5 |
| gt | spl 最高（light） | `spl.n24.raster.light` | 0.932 [0.928, 0.937] | 0.932 | — | 236.5 |
| gt | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.903 [0.898, 0.909] | 0.903 | — | 236.6 |
| gt | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.866 [0.860, 0.873] | 0.866 | — | 236.7 |
| gt | cor 最高（light） | `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.734 [0.712, 0.754] | 0.734 | — | 166.3 |
| gt | cor 最高（medium） | `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.691 [0.670, 0.712] | 0.691 | — | 165.7 |
| gt | cor 最高（heavy） | `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.651 [0.629, 0.675] | 0.651 | — | 165.4 |
| base | 参照 | `box` | 0.503 [0.485, 0.521] | 0.557 | 0.390 | 0.0 |
| base | 参照 | `gen.gc_default` | 0.572 [0.547, 0.598] | 0.662 | 0.392 | 0.0 |
| base | 参照 | `genK.4` | 0.638 [0.615, 0.662] | 0.727 | 0.462 | 1.0 |
| base | spl 最高（none） | `spl.n24.raster.none` | 0.860 [0.836, 0.882] | 0.943 | 0.683 | 236.4 |
| base | spl 最高（light） | `spl.n24.raster.light` | 0.851 [0.828, 0.873] | 0.933 | 0.677 | 236.5 |
| base | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.824 [0.800, 0.846] | 0.907 | 0.647 | 236.6 |
| base | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.790 [0.767, 0.811] | 0.869 | 0.620 | 236.8 |
| base | cor 最高（light） | `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.665 [0.638, 0.691] | 0.760 | 0.468 | 164.3 |
| base | cor 最高（medium） | `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.620 [0.592, 0.646] | 0.711 | 0.434 | 163.5 |
| base | cor 最高（heavy） | `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.593 [0.566, 0.620] | 0.683 | 0.412 | 163.2 |

## 框：gt

n = 800；好框 800、坏框 0（其中解析失败 0）；框 Acc@0.5 = 1.000

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.550 [0.534, 0.566] | 0.550 | — | 0.566 | 0.0 |
| `gen.gc_default` | 0.621 [0.599, 0.644] | 0.621 | — | 0.594 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.612 [0.592, 0.632] | 0.612 | — | 0.605 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.621 [0.599, 0.645] | 0.621 | — | 0.591 | 0.0 |
| `gen.ws_pts` | 0.583 [0.567, 0.600] | 0.583 | — | 0.604 | 0.0 |
| `genK.4` | 0.700 [0.682, 0.718] | 0.700 | — | 0.705 | 1.0 |
| `spl.n12.raster.none` | 0.872 [0.865, 0.879] | 0.872 | — | 0.866 | 118.4 |
| `spl.n12.gc_trimap.none` | 0.881 [0.874, 0.888] | 0.881 | — | 0.869 | 118.4 |
| `spl.n16.raster.none` | 0.907 [0.902, 0.913] | 0.907 | — | 0.901 | 157.7 |
| `spl.n16.gc_trimap.none` | 0.900 [0.893, 0.906] | 0.900 | — | 0.887 | 157.7 |
| `spl.n24.raster.none` | 0.943 [0.938, 0.947] | 0.943 | — | 0.936 | 236.5 |
| `spl.n24.gc_trimap.none` | 0.911 [0.905, 0.916] | 0.911 | — | 0.898 | 236.5 |
| `spl.n12.raster.light` | 0.868 [0.860, 0.875] | 0.868 | — | 0.862 | 118.4 |
| `spl.n12.gc_trimap.light` | 0.880 [0.873, 0.888] | 0.880 | — | 0.868 | 118.4 |
| `spl.n16.raster.light` | 0.901 [0.895, 0.907] | 0.901 | — | 0.894 | 157.8 |
| `spl.n16.gc_trimap.light` | 0.899 [0.893, 0.905] | 0.899 | — | 0.886 | 157.8 |
| `spl.n24.raster.light` | 0.932 [0.928, 0.937] | 0.932 | — | 0.925 | 236.5 |
| `spl.n24.gc_trimap.light` | 0.910 [0.905, 0.916] | 0.910 | — | 0.898 | 236.5 |
| `spl.n12.raster.medium` | 0.847 [0.840, 0.855] | 0.847 | — | 0.840 | 118.4 |
| `spl.n12.gc_trimap.medium` | 0.870 [0.863, 0.878] | 0.870 | — | 0.858 | 118.4 |
| `spl.n16.raster.medium` | 0.873 [0.867, 0.880] | 0.873 | — | 0.864 | 157.8 |
| `spl.n16.gc_trimap.medium` | 0.891 [0.885, 0.898] | 0.891 | — | 0.878 | 157.8 |
| `spl.n24.raster.medium` | 0.892 [0.887, 0.898] | 0.892 | — | 0.884 | 236.6 |
| `spl.n24.gc_trimap.medium` | 0.903 [0.898, 0.909] | 0.903 | — | 0.891 | 236.6 |
| `spl.n12.raster.heavy` | 0.805 [0.797, 0.813] | 0.805 | — | 0.795 | 118.4 |
| `spl.n12.gc_trimap.heavy` | 0.838 [0.830, 0.846] | 0.838 | — | 0.824 | 118.4 |
| `spl.n16.raster.heavy` | 0.819 [0.811, 0.826] | 0.819 | — | 0.810 | 157.9 |
| `spl.n16.gc_trimap.heavy` | 0.854 [0.847, 0.862] | 0.854 | — | 0.842 | 157.9 |
| `spl.n24.raster.heavy` | 0.829 [0.823, 0.836] | 0.829 | — | 0.820 | 236.7 |
| `spl.n24.gc_trimap.heavy` | 0.866 [0.860, 0.873] | 0.866 | — | 0.852 | 236.7 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.680 [0.657, 0.702] | 0.680 | — | 0.653 | 55.1 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.711 [0.688, 0.732] | 0.711 | — | 0.680 | 110.6 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.732 [0.710, 0.753] | 0.732 | — | 0.697 | 166.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.680 [0.657, 0.702] | 0.680 | — | 0.653 | 55.1 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.714 [0.694, 0.735] | 0.714 | — | 0.681 | 110.6 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.734 [0.712, 0.754] | 0.734 | — | 0.698 | 166.3 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.643 [0.619, 0.666] | 0.643 | — | 0.614 | 55.0 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.673 [0.651, 0.694] | 0.673 | — | 0.640 | 110.3 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.690 [0.669, 0.710] | 0.690 | — | 0.656 | 165.7 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.643 [0.619, 0.666] | 0.643 | — | 0.614 | 55.0 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.673 [0.651, 0.694] | 0.673 | — | 0.640 | 110.3 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.691 [0.670, 0.712] | 0.691 | — | 0.657 | 165.7 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.631 [0.609, 0.654] | 0.631 | — | 0.600 | 55.0 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.640 [0.618, 0.663] | 0.640 | — | 0.611 | 110.0 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.650 [0.628, 0.673] | 0.650 | — | 0.620 | 165.4 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.631 [0.609, 0.654] | 0.631 | — | 0.600 | 55.0 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.641 [0.618, 0.663] | 0.641 | — | 0.612 | 110.0 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.651 [0.629, 0.675] | 0.651 | — | 0.622 | 165.4 |

## 框：base

n = 800；好框 516、坏框 284（其中解析失败 14）；框 Acc@0.5 = 0.895

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.503 [0.485, 0.521] | 0.557 | 0.390 | 0.514 | 0.0 |
| `gen.gc_default` | 0.572 [0.547, 0.598] | 0.662 | 0.392 | 0.550 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.559 [0.536, 0.583] | 0.650 | 0.379 | 0.554 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.573 [0.547, 0.598] | 0.662 | 0.393 | 0.548 | 0.0 |
| `gen.ws_pts` | 0.532 [0.513, 0.552] | 0.604 | 0.389 | 0.556 | 0.0 |
| `genK.4` | 0.638 [0.615, 0.662] | 0.727 | 0.462 | 0.645 | 1.0 |
| `spl.n12.raster.none` | 0.798 [0.774, 0.819] | 0.873 | 0.637 | 0.794 | 118.4 |
| `spl.n12.gc_trimap.none` | 0.803 [0.779, 0.825] | 0.884 | 0.630 | 0.795 | 118.4 |
| `spl.n16.raster.none` | 0.830 [0.806, 0.851] | 0.909 | 0.660 | 0.826 | 157.7 |
| `spl.n16.gc_trimap.none` | 0.820 [0.796, 0.841] | 0.903 | 0.643 | 0.812 | 157.7 |
| `spl.n24.raster.none` | 0.860 [0.836, 0.882] | 0.943 | 0.683 | 0.857 | 236.4 |
| `spl.n24.gc_trimap.none` | 0.829 [0.806, 0.851] | 0.913 | 0.651 | 0.821 | 236.4 |
| `spl.n12.raster.light` | 0.795 [0.771, 0.816] | 0.869 | 0.635 | 0.791 | 118.4 |
| `spl.n12.gc_trimap.light` | 0.802 [0.778, 0.823] | 0.883 | 0.629 | 0.794 | 118.4 |
| `spl.n16.raster.light` | 0.824 [0.800, 0.846] | 0.903 | 0.656 | 0.820 | 157.7 |
| `spl.n16.gc_trimap.light` | 0.819 [0.795, 0.840] | 0.902 | 0.643 | 0.811 | 157.7 |
| `spl.n24.raster.light` | 0.851 [0.828, 0.873] | 0.933 | 0.677 | 0.848 | 236.5 |
| `spl.n24.gc_trimap.light` | 0.829 [0.805, 0.850] | 0.912 | 0.651 | 0.821 | 236.5 |
| `spl.n12.raster.medium` | 0.776 [0.753, 0.796] | 0.848 | 0.620 | 0.771 | 118.4 |
| `spl.n12.gc_trimap.medium` | 0.794 [0.770, 0.816] | 0.874 | 0.621 | 0.786 | 118.4 |
| `spl.n16.raster.medium` | 0.799 [0.777, 0.820] | 0.874 | 0.638 | 0.793 | 157.8 |
| `spl.n16.gc_trimap.medium` | 0.811 [0.788, 0.833] | 0.894 | 0.635 | 0.802 | 157.8 |
| `spl.n24.raster.medium` | 0.816 [0.794, 0.837] | 0.893 | 0.652 | 0.811 | 236.6 |
| `spl.n24.gc_trimap.medium` | 0.824 [0.800, 0.846] | 0.907 | 0.647 | 0.816 | 236.6 |
| `spl.n12.raster.heavy` | 0.738 [0.718, 0.758] | 0.805 | 0.594 | 0.731 | 118.4 |
| `spl.n12.gc_trimap.heavy` | 0.766 [0.744, 0.787] | 0.842 | 0.602 | 0.757 | 118.4 |
| `spl.n16.raster.heavy` | 0.751 [0.730, 0.770] | 0.820 | 0.601 | 0.742 | 157.8 |
| `spl.n16.gc_trimap.heavy` | 0.779 [0.756, 0.799] | 0.857 | 0.611 | 0.769 | 157.8 |
| `spl.n24.raster.heavy` | 0.760 [0.739, 0.780] | 0.830 | 0.609 | 0.753 | 236.8 |
| `spl.n24.gc_trimap.heavy` | 0.790 [0.767, 0.811] | 0.869 | 0.620 | 0.781 | 236.8 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.616 [0.589, 0.646] | 0.708 | 0.430 | 0.592 | 54.2 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.644 [0.615, 0.672] | 0.734 | 0.458 | 0.618 | 109.2 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.660 [0.632, 0.688] | 0.752 | 0.469 | 0.633 | 164.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.616 [0.589, 0.646] | 0.708 | 0.430 | 0.592 | 54.2 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.645 [0.616, 0.673] | 0.736 | 0.458 | 0.619 | 109.2 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.665 [0.638, 0.691] | 0.760 | 0.468 | 0.635 | 164.3 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.590 [0.565, 0.615] | 0.679 | 0.410 | 0.562 | 54.3 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.602 [0.576, 0.629] | 0.690 | 0.422 | 0.577 | 108.8 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.618 [0.591, 0.644] | 0.708 | 0.434 | 0.590 | 163.5 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.590 [0.565, 0.615] | 0.679 | 0.410 | 0.562 | 54.3 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.603 [0.576, 0.629] | 0.692 | 0.423 | 0.578 | 108.8 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.620 [0.592, 0.646] | 0.711 | 0.434 | 0.592 | 163.5 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.566 [0.538, 0.593] | 0.653 | 0.391 | 0.546 | 54.2 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.581 [0.553, 0.607] | 0.668 | 0.404 | 0.560 | 108.6 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.592 [0.564, 0.618] | 0.681 | 0.412 | 0.572 | 163.2 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.566 [0.538, 0.593] | 0.653 | 0.391 | 0.546 | 54.2 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.583 [0.555, 0.609] | 0.670 | 0.406 | 0.561 | 108.6 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.593 [0.566, 0.620] | 0.683 | 0.412 | 0.573 | 163.2 |

## devcheck2 的 `summary.md` 原文

# X-1 上限（devcheck2，格点 noise）

- 500 句；框 IoU ≥ 0.9 记好框；CI 为逐句 bootstrap 95%（1000 次，种子 20260929）。
- token：Qwen/Qwen3.5-9B@c20223623576，格式 x1-text-v1（只数这个通道的输出，框不计）。

## 要点：每种框上的参照与各通道、各噪声档里 cIoU 最高的格点

| 框 | 行 | 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | token 均值 |
|---|---|---|---|---|---|---|
| gt | 参照 | `box` | 0.533 [0.515, 0.552] | 0.533 | — | 0.0 |
| gt | 参照 | `gen.gc_default` | 0.607 [0.579, 0.634] | 0.607 | — | 0.0 |
| gt | 参照 | `genK.4` | 0.693 [0.674, 0.713] | 0.693 | — | 1.0 |
| gt | spl 最高（none） | `spl.n24.raster.none` | 0.930 [0.917, 0.940] | 0.930 | — | 237.0 |
| gt | spl 最高（light） | `spl.n24.raster.light` | 0.919 [0.906, 0.929] | 0.919 | — | 237.0 |
| gt | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.890 [0.878, 0.901] | 0.890 | — | 237.1 |
| gt | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.853 [0.840, 0.865] | 0.853 | — | 237.3 |
| gt | cor 最高（light） | `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.739 [0.715, 0.762] | 0.739 | — | 165.9 |
| gt | cor 最高（medium） | `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.680 [0.654, 0.705] | 0.680 | — | 165.3 |
| gt | cor 最高（heavy） | `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.641 [0.613, 0.666] | 0.641 | — | 164.8 |
| base | 参照 | `box` | 0.498 [0.477, 0.520] | 0.533 | 0.421 | 0.0 |
| base | 参照 | `gen.gc_default` | 0.578 [0.549, 0.607] | 0.641 | 0.446 | 0.0 |
| base | 参照 | `genK.4` | 0.644 [0.619, 0.670] | 0.709 | 0.508 | 1.0 |
| base | spl 最高（none） | `spl.n24.raster.none` | 0.876 [0.852, 0.899] | 0.932 | 0.750 | 237.0 |
| base | spl 最高（light） | `spl.n24.raster.light` | 0.867 [0.843, 0.889] | 0.922 | 0.742 | 237.0 |
| base | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.839 [0.815, 0.860] | 0.898 | 0.705 | 237.2 |
| base | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.807 [0.783, 0.828] | 0.862 | 0.682 | 237.3 |
| base | cor 最高（light） | `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.685 [0.655, 0.713] | 0.758 | 0.524 | 165.3 |
| base | cor 最高（medium） | `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.636 [0.607, 0.665] | 0.709 | 0.480 | 164.7 |
| base | cor 最高（heavy） | `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.592 [0.561, 0.624] | 0.658 | 0.450 | 164.5 |
| w2c | 参照 | `box` | 0.508 [0.487, 0.530] | 0.538 | 0.434 | 0.0 |
| w2c | 参照 | `gen.gc_default` | 0.585 [0.553, 0.617] | 0.635 | 0.467 | 0.0 |
| w2c | 参照 | `genK.4` | 0.658 [0.631, 0.685] | 0.711 | 0.531 | 1.0 |
| w2c | 参照 | `ref.w2c_final` | 0.642 [0.616, 0.670] | 0.701 | 0.505 | 0.0 |
| w2c | spl 最高（none） | `spl.n24.raster.none` | 0.899 [0.877, 0.917] | 0.931 | 0.811 | 237.0 |
| w2c | spl 最高（light） | `spl.n24.raster.light` | 0.889 [0.868, 0.907] | 0.921 | 0.802 | 237.0 |
| w2c | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.860 [0.838, 0.877] | 0.896 | 0.762 | 237.1 |
| w2c | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.827 [0.807, 0.845] | 0.860 | 0.741 | 237.3 |
| w2c | cor 最高（light） | `cor.w2c_final.m2.nolock.grabcut.r3.light` | 0.750 [0.727, 0.774] | 0.809 | 0.605 | 167.9 |
| w2c | cor 最高（medium） | `cor.w2c_final.m2.nolock.grabcut.r3.medium` | 0.705 [0.682, 0.730] | 0.754 | 0.582 | 168.4 |
| w2c | cor 最高（heavy） | `cor.w2c_final.m2.nolock.grabcut.r3.heavy` | 0.663 [0.634, 0.693] | 0.726 | 0.512 | 167.6 |

## 框：gt

n = 500；好框 500、坏框 0（其中解析失败 0）；框 Acc@0.5 = 1.000

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.533 [0.515, 0.552] | 0.533 | — | 0.572 | 0.0 |
| `gen.gc_default` | 0.607 [0.579, 0.634] | 0.607 | — | 0.598 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.602 [0.579, 0.628] | 0.602 | — | 0.616 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.607 [0.579, 0.634] | 0.607 | — | 0.597 | 0.0 |
| `gen.ws_pts` | 0.571 [0.548, 0.598] | 0.571 | — | 0.617 | 0.0 |
| `genK.4` | 0.693 [0.674, 0.713] | 0.693 | — | 0.717 | 1.0 |
| `spl.n12.raster.none` | 0.855 [0.840, 0.867] | 0.855 | — | 0.861 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.865 [0.852, 0.876] | 0.865 | — | 0.865 | 118.7 |
| `spl.n16.raster.none` | 0.894 [0.881, 0.905] | 0.894 | — | 0.898 | 158.1 |
| `spl.n16.gc_trimap.none` | 0.887 [0.874, 0.897] | 0.887 | — | 0.885 | 158.1 |
| `spl.n24.raster.none` | 0.930 [0.917, 0.940] | 0.930 | — | 0.932 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.898 [0.886, 0.908] | 0.898 | — | 0.896 | 237.0 |
| `spl.n12.raster.light` | 0.851 [0.836, 0.863] | 0.851 | — | 0.857 | 118.7 |
| `spl.n12.gc_trimap.light` | 0.864 [0.851, 0.875] | 0.864 | — | 0.864 | 118.7 |
| `spl.n16.raster.light` | 0.888 [0.875, 0.899] | 0.888 | — | 0.892 | 158.1 |
| `spl.n16.gc_trimap.light` | 0.885 [0.873, 0.896] | 0.885 | — | 0.884 | 158.1 |
| `spl.n24.raster.light` | 0.919 [0.906, 0.929] | 0.919 | — | 0.921 | 237.0 |
| `spl.n24.gc_trimap.light` | 0.897 [0.885, 0.907] | 0.897 | — | 0.895 | 237.0 |
| `spl.n12.raster.medium` | 0.830 [0.816, 0.842] | 0.830 | — | 0.836 | 118.7 |
| `spl.n12.gc_trimap.medium` | 0.854 [0.841, 0.866] | 0.854 | — | 0.856 | 118.7 |
| `spl.n16.raster.medium` | 0.856 [0.843, 0.868] | 0.856 | — | 0.860 | 158.2 |
| `spl.n16.gc_trimap.medium` | 0.875 [0.862, 0.885] | 0.875 | — | 0.874 | 158.2 |
| `spl.n24.raster.medium` | 0.878 [0.865, 0.888] | 0.878 | — | 0.881 | 237.1 |
| `spl.n24.gc_trimap.medium` | 0.890 [0.878, 0.901] | 0.890 | — | 0.888 | 237.1 |
| `spl.n12.raster.heavy` | 0.786 [0.771, 0.799] | 0.786 | — | 0.791 | 118.7 |
| `spl.n12.gc_trimap.heavy` | 0.822 [0.808, 0.835] | 0.822 | — | 0.823 | 118.7 |
| `spl.n16.raster.heavy` | 0.802 [0.788, 0.814] | 0.802 | — | 0.806 | 158.2 |
| `spl.n16.gc_trimap.heavy` | 0.841 [0.827, 0.853] | 0.841 | — | 0.839 | 158.2 |
| `spl.n24.raster.heavy` | 0.815 [0.801, 0.828] | 0.815 | — | 0.818 | 237.3 |
| `spl.n24.gc_trimap.heavy` | 0.853 [0.840, 0.865] | 0.853 | — | 0.852 | 237.3 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.677 [0.652, 0.701] | 0.677 | — | 0.657 | 54.8 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.714 [0.690, 0.737] | 0.714 | — | 0.687 | 110.2 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.737 [0.712, 0.759] | 0.737 | — | 0.708 | 165.9 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.677 [0.652, 0.701] | 0.677 | — | 0.657 | 54.8 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.715 [0.691, 0.737] | 0.715 | — | 0.687 | 110.2 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.739 [0.715, 0.762] | 0.739 | — | 0.709 | 165.9 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.632 [0.603, 0.659] | 0.632 | — | 0.618 | 54.8 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.659 [0.633, 0.685] | 0.659 | — | 0.636 | 110.0 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.680 [0.654, 0.705] | 0.680 | — | 0.656 | 165.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.632 [0.603, 0.659] | 0.632 | — | 0.618 | 54.8 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.660 [0.634, 0.685] | 0.660 | — | 0.637 | 110.0 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.680 [0.654, 0.705] | 0.680 | — | 0.656 | 165.3 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.625 [0.596, 0.651] | 0.625 | — | 0.613 | 54.8 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.636 [0.610, 0.663] | 0.636 | — | 0.621 | 109.8 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.641 [0.613, 0.666] | 0.641 | — | 0.629 | 164.8 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.625 [0.596, 0.651] | 0.625 | — | 0.613 | 54.8 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.636 [0.610, 0.663] | 0.636 | — | 0.621 | 109.8 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.640 [0.612, 0.666] | 0.640 | — | 0.628 | 164.8 |

## 框：base

n = 500；好框 327、坏框 173（其中解析失败 4）；框 Acc@0.5 = 0.918

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.498 [0.477, 0.520] | 0.533 | 0.421 | 0.526 | 0.0 |
| `gen.gc_default` | 0.578 [0.549, 0.607] | 0.641 | 0.446 | 0.569 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.562 [0.536, 0.590] | 0.623 | 0.436 | 0.576 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.577 [0.549, 0.608] | 0.640 | 0.447 | 0.568 | 0.0 |
| `gen.ws_pts` | 0.541 [0.519, 0.565] | 0.595 | 0.426 | 0.575 | 0.0 |
| `genK.4` | 0.644 [0.619, 0.670] | 0.709 | 0.508 | 0.661 | 1.0 |
| `spl.n12.raster.none` | 0.808 [0.784, 0.830] | 0.855 | 0.701 | 0.804 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.815 [0.791, 0.836] | 0.869 | 0.693 | 0.805 | 118.7 |
| `spl.n16.raster.none` | 0.844 [0.820, 0.866] | 0.895 | 0.728 | 0.837 | 158.1 |
| `spl.n16.gc_trimap.none` | 0.834 [0.811, 0.855] | 0.892 | 0.706 | 0.822 | 158.1 |
| `spl.n24.raster.none` | 0.876 [0.852, 0.899] | 0.932 | 0.750 | 0.868 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.844 [0.820, 0.865] | 0.903 | 0.711 | 0.832 | 237.0 |
| `spl.n12.raster.light` | 0.804 [0.781, 0.827] | 0.851 | 0.699 | 0.800 | 118.7 |
| `spl.n12.gc_trimap.light` | 0.814 [0.791, 0.835] | 0.868 | 0.694 | 0.804 | 118.7 |
| `spl.n16.raster.light` | 0.838 [0.814, 0.860] | 0.889 | 0.723 | 0.831 | 158.1 |
| `spl.n16.gc_trimap.light` | 0.834 [0.810, 0.855] | 0.891 | 0.705 | 0.822 | 158.1 |
| `spl.n24.raster.light` | 0.867 [0.843, 0.889] | 0.922 | 0.742 | 0.858 | 237.0 |
| `spl.n24.gc_trimap.light` | 0.844 [0.820, 0.865] | 0.903 | 0.711 | 0.832 | 237.0 |
| `spl.n12.raster.medium` | 0.787 [0.764, 0.808] | 0.832 | 0.684 | 0.781 | 118.6 |
| `spl.n12.gc_trimap.medium` | 0.806 [0.783, 0.827] | 0.859 | 0.687 | 0.796 | 118.6 |
| `spl.n16.raster.medium` | 0.811 [0.788, 0.832] | 0.859 | 0.703 | 0.804 | 158.2 |
| `spl.n16.gc_trimap.medium` | 0.827 [0.804, 0.848] | 0.883 | 0.700 | 0.816 | 158.2 |
| `spl.n24.raster.medium` | 0.830 [0.807, 0.851] | 0.881 | 0.713 | 0.822 | 237.2 |
| `spl.n24.gc_trimap.medium` | 0.839 [0.815, 0.860] | 0.898 | 0.705 | 0.826 | 237.2 |
| `spl.n12.raster.heavy` | 0.742 [0.720, 0.762] | 0.783 | 0.647 | 0.737 | 118.6 |
| `spl.n12.gc_trimap.heavy` | 0.775 [0.751, 0.795] | 0.826 | 0.660 | 0.766 | 118.6 |
| `spl.n16.raster.heavy` | 0.759 [0.737, 0.779] | 0.800 | 0.664 | 0.752 | 158.2 |
| `spl.n16.gc_trimap.heavy` | 0.791 [0.768, 0.811] | 0.841 | 0.677 | 0.780 | 158.2 |
| `spl.n24.raster.heavy` | 0.773 [0.750, 0.793] | 0.817 | 0.670 | 0.763 | 237.3 |
| `spl.n24.gc_trimap.heavy` | 0.807 [0.783, 0.828] | 0.862 | 0.682 | 0.793 | 237.3 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.616 [0.585, 0.648] | 0.693 | 0.450 | 0.606 | 54.6 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.655 [0.624, 0.686] | 0.726 | 0.501 | 0.636 | 109.9 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.681 [0.650, 0.711] | 0.755 | 0.517 | 0.656 | 165.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.616 [0.585, 0.648] | 0.693 | 0.450 | 0.606 | 54.6 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.660 [0.629, 0.690] | 0.731 | 0.504 | 0.639 | 109.9 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.685 [0.655, 0.713] | 0.758 | 0.524 | 0.657 | 165.3 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.588 [0.559, 0.618] | 0.651 | 0.454 | 0.577 | 54.6 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.609 [0.579, 0.638] | 0.672 | 0.473 | 0.593 | 109.6 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.633 [0.603, 0.661] | 0.705 | 0.477 | 0.614 | 164.7 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.588 [0.559, 0.618] | 0.651 | 0.454 | 0.577 | 54.6 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.610 [0.581, 0.638] | 0.674 | 0.473 | 0.594 | 109.6 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.636 [0.607, 0.665] | 0.709 | 0.480 | 0.616 | 164.7 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.580 [0.548, 0.610] | 0.642 | 0.442 | 0.565 | 54.7 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.585 [0.552, 0.617] | 0.652 | 0.438 | 0.571 | 109.5 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.590 [0.559, 0.622] | 0.656 | 0.446 | 0.577 | 164.5 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.580 [0.548, 0.610] | 0.642 | 0.442 | 0.565 | 54.7 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.585 [0.552, 0.617] | 0.652 | 0.437 | 0.571 | 109.5 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.592 [0.561, 0.624] | 0.658 | 0.450 | 0.578 | 164.5 |

## 框：w2c

n = 500；好框 347、坏框 153（其中解析失败 0）；框 Acc@0.5 = 0.934

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.508 [0.487, 0.530] | 0.538 | 0.434 | 0.542 | 0.0 |
| `gen.gc_default` | 0.585 [0.553, 0.617] | 0.635 | 0.467 | 0.580 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.571 [0.543, 0.600] | 0.619 | 0.456 | 0.587 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.584 [0.552, 0.616] | 0.635 | 0.464 | 0.578 | 0.0 |
| `gen.ws_pts` | 0.552 [0.531, 0.579] | 0.593 | 0.455 | 0.591 | 0.0 |
| `genK.4` | 0.658 [0.631, 0.685] | 0.711 | 0.531 | 0.682 | 1.0 |
| `ref.w2c_final` | 0.642 [0.616, 0.670] | 0.701 | 0.505 | 0.669 | 0.0 |
| `spl.n12.raster.none` | 0.829 [0.808, 0.847] | 0.855 | 0.758 | 0.824 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.836 [0.816, 0.854] | 0.869 | 0.750 | 0.825 | 118.7 |
| `spl.n16.raster.none` | 0.865 [0.844, 0.883] | 0.895 | 0.787 | 0.858 | 158.1 |
| `spl.n16.gc_trimap.none` | 0.856 [0.835, 0.873] | 0.891 | 0.763 | 0.843 | 158.1 |
| `spl.n24.raster.none` | 0.899 [0.877, 0.917] | 0.931 | 0.811 | 0.889 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.867 [0.846, 0.884] | 0.904 | 0.769 | 0.853 | 237.0 |
| `spl.n12.raster.light` | 0.825 [0.804, 0.842] | 0.851 | 0.755 | 0.821 | 118.7 |
| `spl.n12.gc_trimap.light` | 0.835 [0.816, 0.853] | 0.868 | 0.749 | 0.824 | 118.7 |
| `spl.n16.raster.light` | 0.860 [0.839, 0.878] | 0.889 | 0.782 | 0.852 | 158.1 |
| `spl.n16.gc_trimap.light` | 0.855 [0.834, 0.872] | 0.890 | 0.762 | 0.842 | 158.1 |
| `spl.n24.raster.light` | 0.889 [0.868, 0.907] | 0.921 | 0.802 | 0.879 | 237.0 |
| `spl.n24.gc_trimap.light` | 0.867 [0.845, 0.884] | 0.904 | 0.768 | 0.853 | 237.0 |
| `spl.n12.raster.medium` | 0.803 [0.783, 0.820] | 0.829 | 0.735 | 0.799 | 118.7 |
| `spl.n12.gc_trimap.medium` | 0.825 [0.804, 0.842] | 0.858 | 0.738 | 0.815 | 118.7 |
| `spl.n16.raster.medium` | 0.831 [0.811, 0.848] | 0.859 | 0.757 | 0.824 | 158.1 |
| `spl.n16.gc_trimap.medium` | 0.846 [0.825, 0.863] | 0.881 | 0.755 | 0.834 | 158.1 |
| `spl.n24.raster.medium` | 0.849 [0.828, 0.866] | 0.879 | 0.768 | 0.841 | 237.1 |
| `spl.n24.gc_trimap.medium` | 0.860 [0.838, 0.877] | 0.896 | 0.762 | 0.847 | 237.1 |
| `spl.n12.raster.heavy` | 0.760 [0.739, 0.777] | 0.783 | 0.697 | 0.756 | 118.6 |
| `spl.n12.gc_trimap.heavy` | 0.796 [0.775, 0.813] | 0.826 | 0.715 | 0.786 | 118.6 |
| `spl.n16.raster.heavy` | 0.779 [0.760, 0.795] | 0.803 | 0.712 | 0.773 | 158.1 |
| `spl.n16.gc_trimap.heavy` | 0.815 [0.794, 0.832] | 0.846 | 0.730 | 0.803 | 158.1 |
| `spl.n24.raster.heavy` | 0.792 [0.773, 0.809] | 0.817 | 0.725 | 0.784 | 237.3 |
| `spl.n24.gc_trimap.heavy` | 0.827 [0.807, 0.845] | 0.860 | 0.741 | 0.814 | 237.3 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.629 [0.596, 0.660] | 0.686 | 0.484 | 0.619 | 54.9 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.671 [0.641, 0.699] | 0.720 | 0.547 | 0.653 | 110.4 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.691 [0.661, 0.718] | 0.745 | 0.554 | 0.669 | 166.2 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.629 [0.596, 0.660] | 0.686 | 0.484 | 0.619 | 54.9 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.674 [0.644, 0.702] | 0.724 | 0.548 | 0.656 | 110.4 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.695 [0.665, 0.721] | 0.750 | 0.555 | 0.671 | 166.2 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.607 [0.574, 0.638] | 0.667 | 0.461 | 0.597 | 55.0 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.629 [0.597, 0.660] | 0.689 | 0.482 | 0.610 | 110.2 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.646 [0.615, 0.676] | 0.703 | 0.501 | 0.624 | 165.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.607 [0.574, 0.638] | 0.667 | 0.461 | 0.597 | 55.0 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.628 [0.595, 0.659] | 0.688 | 0.481 | 0.610 | 110.2 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.646 [0.616, 0.675] | 0.703 | 0.501 | 0.625 | 165.3 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.596 [0.565, 0.626] | 0.648 | 0.468 | 0.582 | 54.9 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.603 [0.571, 0.633] | 0.653 | 0.480 | 0.587 | 110.1 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.604 [0.573, 0.635] | 0.658 | 0.471 | 0.589 | 165.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.596 [0.565, 0.626] | 0.648 | 0.468 | 0.582 | 54.9 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.603 [0.572, 0.633] | 0.654 | 0.481 | 0.588 | 110.1 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.604 [0.573, 0.635] | 0.661 | 0.462 | 0.588 | 165.3 |
| `cor.w2c_final.m2.lock.grabcut.r1.light` | 0.698 [0.675, 0.721] | 0.752 | 0.564 | 0.707 | 55.8 |
| `cor.w2c_final.m2.lock.grabcut.r2.light` | 0.733 [0.710, 0.756] | 0.790 | 0.588 | 0.732 | 111.8 |
| `cor.w2c_final.m2.lock.grabcut.r3.light` | 0.750 [0.727, 0.773] | 0.810 | 0.602 | 0.747 | 167.9 |
| `cor.w2c_final.m2.nolock.grabcut.r1.light` | 0.698 [0.675, 0.721] | 0.752 | 0.564 | 0.707 | 55.8 |
| `cor.w2c_final.m2.nolock.grabcut.r2.light` | 0.736 [0.714, 0.758] | 0.791 | 0.598 | 0.733 | 111.8 |
| `cor.w2c_final.m2.nolock.grabcut.r3.light` | 0.750 [0.727, 0.774] | 0.809 | 0.605 | 0.749 | 167.9 |
| `cor.w2c_final.m2.lock.grabcut.r1.medium` | 0.664 [0.640, 0.691] | 0.720 | 0.531 | 0.680 | 55.9 |
| `cor.w2c_final.m2.lock.grabcut.r2.medium` | 0.684 [0.659, 0.710] | 0.736 | 0.552 | 0.690 | 112.0 |
| `cor.w2c_final.m2.lock.grabcut.r3.medium` | 0.703 [0.679, 0.728] | 0.754 | 0.574 | 0.706 | 168.4 |
| `cor.w2c_final.m2.nolock.grabcut.r1.medium` | 0.664 [0.640, 0.691] | 0.720 | 0.531 | 0.680 | 55.9 |
| `cor.w2c_final.m2.nolock.grabcut.r2.medium` | 0.684 [0.659, 0.710] | 0.737 | 0.553 | 0.690 | 112.0 |
| `cor.w2c_final.m2.nolock.grabcut.r3.medium` | 0.705 [0.682, 0.730] | 0.754 | 0.582 | 0.707 | 168.4 |
| `cor.w2c_final.m2.lock.grabcut.r1.heavy` | 0.647 [0.620, 0.677] | 0.706 | 0.508 | 0.672 | 55.8 |
| `cor.w2c_final.m2.lock.grabcut.r2.heavy` | 0.654 [0.625, 0.683] | 0.715 | 0.508 | 0.675 | 111.8 |
| `cor.w2c_final.m2.lock.grabcut.r3.heavy` | 0.661 [0.632, 0.691] | 0.723 | 0.513 | 0.680 | 167.6 |
| `cor.w2c_final.m2.nolock.grabcut.r1.heavy` | 0.647 [0.620, 0.677] | 0.706 | 0.508 | 0.672 | 55.8 |
| `cor.w2c_final.m2.nolock.grabcut.r2.heavy` | 0.654 [0.626, 0.683] | 0.716 | 0.508 | 0.675 | 111.8 |
| `cor.w2c_final.m2.nolock.grabcut.r3.heavy` | 0.663 [0.634, 0.693] | 0.726 | 0.512 | 0.681 | 167.6 |
