# RUN-0077 报告：hpcgpu07（只用 CPU），探索 X-1 的上限测量，first 网格（devtune800 + devcheck2）

任务书 `prompts/server_hpc/0077-hpc-07-explore-ceiling.md`。不用 GPU，没有写暂停文件，没有调 `claim_gpu`；没有碰 RUN-0072、0073、0074。只用 devtune800 与 devcheck2。只贴数字，不解读。

## 工作树

- 经 hpcmgt 上脱离会话的 `$VI/tmp/prep0077.sh`（10-26 04:09:06Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0077（稀疏）HEAD `d62e67329bbc6ce9324b10138df4ca36295fc2d7`，status 空，`experiments/` 只有 P0；6 个 sha256 与任务书逐字相同。

## 第 3 步：探测与 bench（`srun --jobid=157297 --overlap -n1 -c 1`，任务书的公共环境）

```
nproc=1
 12:09:14 up 268 days,  3:24,  0 users,  load average: 1.00, 1.10, 1.56
              total        used        free      shared  buff/cache   available
Mem:            503          12         463          11          28         476
Swap:            31           1          30
[bench first] astronaut 0.25: 29.8 s CPU，60 个格点
[bench first] astronaut 0.4: 10.8 s CPU，60 个格点
[bench first] astronaut 0.05: 58.2 s CPU，60 个格点
[bench first] chelsea 0.25: 35.9 s CPU，60 个格点
[bench first] chelsea 0.4: 13.5 s CPU，60 个格点
[bench first] chelsea 0.05: 89.2 s CPU，60 个格点
[bench first] coffee 0.25: 34.1 s CPU，60 个格点
[bench first] coffee 0.4: 13.5 s CPU，60 个格点
[bench first] coffee 0.05: 41.1 s CPU，60 个格点
[bench first] rocket 0.25: 38.7 s CPU，60 个格点
[bench first] rocket 0.4: 13.0 s CPU，60 个格点
[bench first] rocket 0.05: 91.9 s CPU，60 个格点
{
 "machine": {
  "platform": "Linux-4.18.0-425.3.1.el8.x86_64-x86_64-with-glibc2.28",
  "processor": "x86_64",
  "cpu_count": 48,
  "python": "3.12.13",
  "opencv": "4.10.0",
  "skimage": "0.24.0",
  "numpy": "2.5.2"
 },
 "grids": {
  "first": {
   "n_jobs_measured": 12,
   "n_cells_w2c_box": 60,
   "cpu_s_per_job_mean": {
    "cor_direct": 0.12586329775001204,
    "cor_grabcut": 3.9339014519166793,
    "gen": 17.090804855916666,
    "pts": 13.002374897750002,
    "spl_gc": 4.944542538833343,
    "spl_raster": 0.028887331333322958,
    "total": 39.13876436150001
   },
   "cpu_s_per_job_median": 34.99497752400002,
   "cpu_ms_per_cell_mean": 652.3127393583334,
   "estimate": {
    "sentences": 1300,
    "jobs": 3100,
    "cpu_h": 33.70282486684723,
    "wall_h_8_cores": 4.212853108355904,
    "wall_h_16_cores": 2.106426554177952
   }
  }
 }
}
bench_rc=0
bench wall_s=476 2026-10-26T04:17:09Z
```

- 跑完后整机 `nproc --all` 48，`uptime` load average 2.76, 2.35, 2.01（04:17:10Z）。
- 每 job CPU 秒：均值 39.14 s、中位 35.00 s；开发机 M4 36 s / job，均值约 1.09 倍。估计 33.7 CPU 小时，8 核 4.21 h、16 核 2.11 h（bench 的 estimate 原文见上）。

## 第 4 步：正式运行

- `$VI/tmp/start0077.sh` 按任务书写出 `$VI/tmp/run0077.sh`（sha256 `d0af5281…`），`cat` 核对：路径都已展开为绝对路径，四个结果目录都存在：
  - `…/0070_w2_tune_r2_hpcgpu06/0070_sep_on_k4/qwen3.5-9b/refexp_box/baseline_e1/devtune800`
  - `…/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refexp_box/baseline_e1/devcheck2`
  - `…/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refexp_box/propose_w2c/devcheck2`
  - `…/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refseg/w2c/devcheck2`
- run0077.sh 原文：

```bash
set -e
cd /home/mail/chenyuma/visint/wt/wt0077
C="python scripts/explore_ceiling.py --data-root /home/mail/chenyuma/visint/b200/data --grid first --workers 12 --out /home/mail/chenyuma/visint/repo/experiments/X/0077_ceiling_hpcgpu07"
$C --splits devtune800 --box-run base=/home/mail/chenyuma/visint/repo/experiments/W2/0070_w2_tune_r2_hpcgpu06/0070_sep_on_k4/qwen3.5-9b/refexp_box/baseline_e1/devtune800 $RES
$C --splits devcheck2 --box-run base=/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refexp_box/baseline_e1/devcheck2 \
  --box-run w2c=/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refexp_box/propose_w2c/devcheck2 \
  --start-mask w2c_final=w2c:/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refseg/w2c/devcheck2 $RES
```

- 启动：04:18:14Z，`ssh -n hpcgpu07 "setsid nohup srun --jobid=157297 --overlap -n1 -c 12 env $ENVV RES= bash $VI/tmp/run0077.sh > $VI/logs/run_0077.log …"`。
- 退出码与墙钟：srun 步 `157297.106`，sacct `COMPLETED`、ExitCode `0:0`，10-26 04:18:15Z – 06:07:30Z（6,555 s，1.82 h）。
  - devtune800：`run_1.json` wall_s 3,303，1,600 job，workers 12；CPU 秒合计 28,020（每 job 均值 17.51 s）。
  - devcheck2：`run_1.json` wall_s 3,234，1,500 job，workers 12；CPU 秒合计 28,436（每 job 均值 18.96 s）。
  - 两个 `compute.json` 的 `learned_modules_in_workers` 都是 `[]`。
- **负载检查与核数**：没有降过核数（一直 12）。
  - 04:19:51Z：load 9.24 / 4.70 / 2.89（48 核）。
  - 04:27:10Z：load 6.75 / 8.62 / 5.66；12 个 worker 各 60–78% CPU；devtune800 192/1600 job。RUN-0073 当时在 `w2a_qwen3.5-2b_test_refcoco_unc_testB`，34.5 min 写出 512 句（约 0.247 句/s；testA 1000 句 3,907 s，约 0.256 句/s）。管理员 04:3xZ 回复保持 12。
  - 跑完后 06:14Z load 1.01。
- `run_0077.log` 只有进度行与两行"写出 …metrics.json、summary.md、compute.json"；grep `warn|traceback|error` 为 0。

## 输出目录 `experiments/X/0077_ceiling_hpcgpu07/`

| 文件 | 字节 |
|---|---:|
| `devcheck2/compute.json` | 1243 |
| `devcheck2/config.yaml` | 1744 |
| `devcheck2/logs/run_1.json` | 1018 |
| `devcheck2/logs/timing.jsonl` | 470424 |
| `devcheck2/meta.json` | 890 |
| `devcheck2/metrics.json` | 87593 |
| `devcheck2/predictions.jsonl` | 4474983 |
| `devcheck2/summary.md` | 16470 |
| `devtune800/compute.json` | 1242 |
| `devtune800/config.yaml` | 1211 |
| `devtune800/logs/run_1.json` | 745 |
| `devtune800/logs/timing.jsonl` | 500625 |
| `devtune800/meta.json` | 890 |
| `devtune800/metrics.json` | 50671 |
| `devtune800/predictions.jsonl` | 4315873 |
| `devtune800/summary.md` | 10200 |

- 两个 `predictions.jsonl`（4.3 MB、4.5 MB）都在推送的大文件线（20 MB）以下，没有压缩，也没有 `LARGE_FILES.sha256`。
- 推送：`ONCE=1`（TAG 0077）`2026-10-26T06:15:02Z ok`。后 squeue：157296、157297 RUNNING。

## 局部 GrabCut 退回 direct 的计数（`predictions.jsonl` 的 `info.cor.fallback_direct`）

由 `experiments/X/0077_ceiling_probe/p77_counts.py` 读出：

```
devtune800 box_kind=base: jobs 800, failure 14, 没有 info.cor 14, 有退回的 job 43, 退回 direct 的格点合计 104
devtune800 box_kind=gt: jobs 800, failure 0, 没有 info.cor 0, 有退回的 job 15, 退回 direct 的格点合计 38
devtune800 合计: jobs 1600, failure 14, 没有 info.cor 14, 有退回的 job 58, 退回 direct 的格点合计 142
devcheck2 box_kind=base: jobs 500, failure 4, 没有 info.cor 4, 有退回的 job 25, 退回 direct 的格点合计 64
devcheck2 box_kind=gt: jobs 500, failure 0, 没有 info.cor 0, 有退回的 job 10, 退回 direct 的格点合计 30
devcheck2 box_kind=w2c: jobs 500, failure 0, 没有 info.cor 0, 有退回的 job 32, 退回 direct 的格点合计 122
devcheck2 合计: jobs 1500, failure 4, 没有 info.cor 4, 有退回的 job 67, 退回 direct 的格点合计 216
```

- failure 的 job 都是 base 框（devtune800 14 个、devcheck2 4 个），这些 job 没有 `info.cor`。

## devtune800 的 `summary.md` 原文

# X-1 上限（devtune800，格点 first）

- 800 句；框 IoU ≥ 0.9 记好框；CI 为逐句 bootstrap 95%（1000 次，种子 20260929）。
- token：Qwen/Qwen3.5-9B@c20223623576，格式 x1-text-v1（只数这个通道的输出，框不计）。

## 要点：每种框上的参照与各通道、各噪声档里 cIoU 最高的格点

| 框 | 行 | 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | token 均值 |
|---|---|---|---|---|---|---|
| gt | 参照 | `box` | 0.550 [0.534, 0.566] | 0.550 | — | 0.0 |
| gt | 参照 | `gen.gc_default` | 0.621 [0.599, 0.644] | 0.621 | — | 0.0 |
| gt | 参照 | `genK.4` | 0.700 [0.682, 0.718] | 0.700 | — | 1.0 |
| gt | 参照 | `genK.8` | 0.731 [0.715, 0.748] | 0.731 | — | 1.0 |
| gt | 参照 | `genK.12` | 0.754 [0.740, 0.770] | 0.754 | — | 1.3 |
| gt | 参照 | `genK.16` | 0.763 [0.748, 0.778] | 0.763 | — | 1.5 |
| gt | pts 最高（none） | `pts.k8.fgbg.pr_fgd.none` | 0.771 [0.756, 0.788] | 0.771 | — | 165.0 |
| gt | spl 最高（none） | `spl.n24.raster.none` | 0.943 [0.938, 0.947] | 0.943 | — | 236.5 |
| gt | cor 最高（none） | `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.768 [0.748, 0.788] | 0.768 | — | 166.6 |
| base | 参照 | `box` | 0.503 [0.485, 0.521] | 0.557 | 0.390 | 0.0 |
| base | 参照 | `gen.gc_default` | 0.572 [0.547, 0.598] | 0.662 | 0.392 | 0.0 |
| base | 参照 | `genK.4` | 0.638 [0.615, 0.662] | 0.727 | 0.462 | 1.0 |
| base | 参照 | `genK.8` | 0.659 [0.635, 0.682] | 0.751 | 0.476 | 1.0 |
| base | 参照 | `genK.12` | 0.678 [0.656, 0.701] | 0.772 | 0.495 | 1.3 |
| base | 参照 | `genK.16` | 0.688 [0.665, 0.710] | 0.783 | 0.500 | 1.5 |
| base | pts 最高（none） | `pts.k8.fgbg.pr_fgd.none` | 0.705 [0.679, 0.728] | 0.801 | 0.512 | 164.4 |
| base | spl 最高（none） | `spl.n24.raster.none` | 0.860 [0.836, 0.882] | 0.943 | 0.683 | 236.4 |
| base | cor 最高（none） | `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.689 [0.659, 0.715] | 0.783 | 0.491 | 163.0 |

## 框：gt

n = 800；好框 800、坏框 0（其中解析失败 0）；框 Acc@0.5 = 1.000

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.550 [0.534, 0.566] | 0.550 | — | 0.566 | 0.0 |
| `gen.gc_default` | 0.621 [0.599, 0.644] | 0.621 | — | 0.594 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.612 [0.592, 0.632] | 0.612 | — | 0.605 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.621 [0.599, 0.645] | 0.621 | — | 0.591 | 0.0 |
| `gen.ws_pts` | 0.583 [0.567, 0.600] | 0.583 | — | 0.604 | 0.0 |
| `gen.gc_iter10` | 0.621 [0.598, 0.643] | 0.621 | — | 0.593 | 0.0 |
| `gen.gc_lab` | 0.626 [0.605, 0.649] | 0.626 | — | 0.602 | 0.0 |
| `gen.gc_ctx50` | 0.628 [0.607, 0.650] | 0.628 | — | 0.597 | 0.0 |
| `gen.slic_s50_flood` | 0.520 [0.495, 0.543] | 0.520 | — | 0.478 | 0.0 |
| `gen.felz_k100_flood` | 0.458 [0.435, 0.481] | 0.458 | — | 0.446 | 0.0 |
| `gen.gmm_k3_lab+lcc+fill` | 0.620 [0.602, 0.639] | 0.620 | — | 0.630 | 0.0 |
| `gen.ws_lab_shrink20_margin5` | 0.622 [0.606, 0.640] | 0.622 | — | 0.640 | 0.0 |
| `gen.gc_default+open3+lcc+fill` | 0.619 [0.597, 0.643] | 0.619 | — | 0.589 | 0.0 |
| `gen.gc_hsv` | 0.609 [0.587, 0.631] | 0.609 | — | 0.587 | 0.0 |
| `gen.slic_s100_color` | 0.597 [0.580, 0.614] | 0.597 | — | 0.600 | 0.0 |
| `gen.gmm_k8_rgb+lcc+fill` | 0.625 [0.607, 0.642] | 0.625 | — | 0.642 | 0.0 |
| `gen.gc_box_minus5+close5+fill` | 0.523 [0.503, 0.544] | 0.523 | — | 0.489 | 0.0 |
| `genK.4` | 0.700 [0.682, 0.718] | 0.700 | — | 0.705 | 1.0 |
| `genK.8` | 0.731 [0.715, 0.748] | 0.731 | — | 0.730 | 1.0 |
| `genK.12` | 0.754 [0.740, 0.770] | 0.754 | — | 0.754 | 1.3 |
| `genK.16` | 0.763 [0.748, 0.778] | 0.763 | — | 0.762 | 1.5 |
| `pts.k2.fg.pr_bgd.none` | 0.262 [0.237, 0.286] | 0.262 | — | 0.220 | 23.8 |
| `pts.k2.fg.pr_fgd.none` | 0.652 [0.633, 0.674] | 0.652 | — | 0.642 | 23.8 |
| `pts.k2.fgbg.pr_bgd.none` | 0.263 [0.237, 0.286] | 0.263 | — | 0.221 | 46.5 |
| `pts.k2.fgbg.pr_fgd.none` | 0.674 [0.654, 0.695] | 0.674 | — | 0.662 | 46.5 |
| `pts.k8.fg.pr_bgd.none` | 0.680 [0.660, 0.700] | 0.680 | — | 0.669 | 83.1 |
| `pts.k8.fg.pr_fgd.none` | 0.707 [0.690, 0.725] | 0.707 | — | 0.721 | 83.1 |
| `pts.k8.fgbg.pr_bgd.none` | 0.694 [0.674, 0.714] | 0.694 | — | 0.681 | 165.0 |
| `pts.k8.fgbg.pr_fgd.none` | 0.771 [0.756, 0.788] | 0.771 | — | 0.769 | 165.0 |
| `spl.n12.raster.none` | 0.872 [0.865, 0.879] | 0.872 | — | 0.866 | 118.4 |
| `spl.n12.gc_soft.none` | 0.688 [0.667, 0.709] | 0.688 | — | 0.647 | 118.4 |
| `spl.n12.gc_trimap.none` | 0.881 [0.874, 0.888] | 0.881 | — | 0.869 | 118.4 |
| `spl.n24.raster.none` | 0.943 [0.938, 0.947] | 0.943 | — | 0.936 | 236.5 |
| `spl.n24.gc_soft.none` | 0.698 [0.678, 0.718] | 0.698 | — | 0.650 | 236.5 |
| `spl.n24.gc_trimap.none` | 0.911 [0.905, 0.916] | 0.911 | — | 0.898 | 236.5 |
| `cor.gc_default.m2.lock.grabcut.r1.none` | 0.732 [0.711, 0.753] | 0.732 | — | 0.694 | 55.0 |
| `cor.gc_default.m2.lock.grabcut.r2.none` | 0.758 [0.738, 0.778] | 0.758 | — | 0.720 | 110.8 |
| `cor.gc_default.m2.lock.grabcut.r3.none` | 0.768 [0.748, 0.787] | 0.768 | — | 0.728 | 166.6 |
| `cor.gc_default.m2.lock.direct.r1.none` | 0.707 [0.689, 0.724] | 0.707 | — | 0.695 | 55.0 |
| `cor.gc_default.m2.lock.direct.r2.none` | 0.694 [0.673, 0.714] | 0.694 | — | 0.667 | 111.2 |
| `cor.gc_default.m2.lock.direct.r3.none` | 0.725 [0.708, 0.741] | 0.725 | — | 0.710 | 167.6 |
| `cor.gc_default.m2.nolock.grabcut.r1.none` | 0.732 [0.711, 0.753] | 0.732 | — | 0.694 | 55.0 |
| `cor.gc_default.m2.nolock.grabcut.r2.none` | 0.758 [0.738, 0.778] | 0.758 | — | 0.720 | 110.8 |
| `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.768 [0.748, 0.788] | 0.768 | — | 0.728 | 166.6 |
| `cor.gc_default.m2.nolock.direct.r1.none` | 0.707 [0.689, 0.724] | 0.707 | — | 0.695 | 55.0 |
| `cor.gc_default.m2.nolock.direct.r2.none` | 0.696 [0.676, 0.715] | 0.696 | — | 0.669 | 111.2 |
| `cor.gc_default.m2.nolock.direct.r3.none` | 0.728 [0.711, 0.745] | 0.728 | — | 0.716 | 167.6 |

## 框：base

n = 800；好框 516、坏框 284（其中解析失败 14）；框 Acc@0.5 = 0.895

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.503 [0.485, 0.521] | 0.557 | 0.390 | 0.514 | 0.0 |
| `gen.gc_default` | 0.572 [0.547, 0.598] | 0.662 | 0.392 | 0.550 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.559 [0.536, 0.583] | 0.650 | 0.379 | 0.554 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.573 [0.547, 0.598] | 0.662 | 0.393 | 0.548 | 0.0 |
| `gen.ws_pts` | 0.532 [0.513, 0.552] | 0.604 | 0.389 | 0.556 | 0.0 |
| `gen.gc_iter10` | 0.572 [0.547, 0.598] | 0.663 | 0.391 | 0.548 | 0.0 |
| `gen.gc_lab` | 0.575 [0.550, 0.601] | 0.664 | 0.397 | 0.555 | 0.0 |
| `gen.gc_ctx50` | 0.576 [0.549, 0.604] | 0.668 | 0.391 | 0.549 | 0.0 |
| `gen.slic_s50_flood` | 0.470 [0.445, 0.494] | 0.540 | 0.324 | 0.442 | 0.0 |
| `gen.felz_k100_flood` | 0.402 [0.377, 0.427] | 0.462 | 0.279 | 0.401 | 0.0 |
| `gen.gmm_k3_lab+lcc+fill` | 0.558 [0.533, 0.583] | 0.648 | 0.385 | 0.574 | 0.0 |
| `gen.ws_lab_shrink20_margin5` | 0.565 [0.545, 0.585] | 0.637 | 0.419 | 0.583 | 0.0 |
| `gen.gc_default+open3+lcc+fill` | 0.572 [0.546, 0.597] | 0.661 | 0.392 | 0.547 | 0.0 |
| `gen.gc_hsv` | 0.561 [0.535, 0.587] | 0.652 | 0.378 | 0.543 | 0.0 |
| `gen.slic_s100_color` | 0.537 [0.517, 0.558] | 0.607 | 0.396 | 0.544 | 0.0 |
| `gen.gmm_k8_rgb+lcc+fill` | 0.564 [0.540, 0.586] | 0.652 | 0.394 | 0.585 | 0.0 |
| `gen.gc_box_minus5+close5+fill` | 0.473 [0.448, 0.497] | 0.553 | 0.312 | 0.445 | 0.0 |
| `genK.4` | 0.638 [0.615, 0.662] | 0.727 | 0.462 | 0.645 | 1.0 |
| `genK.8` | 0.659 [0.635, 0.682] | 0.751 | 0.476 | 0.666 | 1.0 |
| `genK.12` | 0.678 [0.656, 0.701] | 0.772 | 0.495 | 0.687 | 1.3 |
| `genK.16` | 0.688 [0.665, 0.710] | 0.783 | 0.500 | 0.696 | 1.5 |
| `pts.k2.fg.pr_bgd.none` | 0.244 [0.220, 0.268] | 0.291 | 0.151 | 0.212 | 23.8 |
| `pts.k2.fg.pr_fgd.none` | 0.601 [0.575, 0.626] | 0.690 | 0.421 | 0.593 | 23.8 |
| `pts.k2.fgbg.pr_bgd.none` | 0.245 [0.220, 0.269] | 0.291 | 0.152 | 0.213 | 46.3 |
| `pts.k2.fgbg.pr_fgd.none` | 0.620 [0.594, 0.646] | 0.714 | 0.430 | 0.606 | 46.3 |
| `pts.k8.fg.pr_bgd.none` | 0.616 [0.587, 0.642] | 0.710 | 0.431 | 0.615 | 83.0 |
| `pts.k8.fg.pr_fgd.none` | 0.649 [0.626, 0.672] | 0.735 | 0.475 | 0.658 | 83.0 |
| `pts.k8.fgbg.pr_bgd.none` | 0.626 [0.596, 0.652] | 0.720 | 0.438 | 0.624 | 164.4 |
| `pts.k8.fgbg.pr_fgd.none` | 0.705 [0.679, 0.728] | 0.801 | 0.512 | 0.705 | 164.4 |
| `spl.n12.raster.none` | 0.798 [0.774, 0.819] | 0.873 | 0.637 | 0.794 | 118.4 |
| `spl.n12.gc_soft.none` | 0.632 [0.605, 0.658] | 0.722 | 0.443 | 0.592 | 118.4 |
| `spl.n12.gc_trimap.none` | 0.803 [0.779, 0.825] | 0.884 | 0.630 | 0.795 | 118.4 |
| `spl.n24.raster.none` | 0.860 [0.836, 0.882] | 0.943 | 0.683 | 0.857 | 236.4 |
| `spl.n24.gc_soft.none` | 0.635 [0.607, 0.660] | 0.724 | 0.445 | 0.591 | 236.4 |
| `spl.n24.gc_trimap.none` | 0.829 [0.806, 0.851] | 0.913 | 0.651 | 0.821 | 236.4 |
| `cor.gc_default.m2.lock.grabcut.r1.none` | 0.645 [0.617, 0.673] | 0.734 | 0.459 | 0.612 | 54.2 |
| `cor.gc_default.m2.lock.grabcut.r2.none` | 0.677 [0.648, 0.705] | 0.770 | 0.482 | 0.649 | 108.5 |
| `cor.gc_default.m2.lock.grabcut.r3.none` | 0.687 [0.658, 0.714] | 0.780 | 0.491 | 0.657 | 163.0 |
| `cor.gc_default.m2.lock.direct.r1.none` | 0.632 [0.609, 0.656] | 0.698 | 0.491 | 0.617 | 54.2 |
| `cor.gc_default.m2.lock.direct.r2.none` | 0.651 [0.629, 0.674] | 0.719 | 0.502 | 0.631 | 109.1 |
| `cor.gc_default.m2.lock.direct.r3.none` | 0.670 [0.647, 0.692] | 0.739 | 0.522 | 0.657 | 164.2 |
| `cor.gc_default.m2.nolock.grabcut.r1.none` | 0.645 [0.617, 0.673] | 0.734 | 0.459 | 0.612 | 54.2 |
| `cor.gc_default.m2.nolock.grabcut.r2.none` | 0.677 [0.648, 0.705] | 0.769 | 0.482 | 0.649 | 108.5 |
| `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.689 [0.659, 0.715] | 0.783 | 0.491 | 0.659 | 163.0 |
| `cor.gc_default.m2.nolock.direct.r1.none` | 0.632 [0.609, 0.656] | 0.698 | 0.491 | 0.617 | 54.2 |
| `cor.gc_default.m2.nolock.direct.r2.none` | 0.651 [0.629, 0.674] | 0.718 | 0.504 | 0.633 | 109.1 |
| `cor.gc_default.m2.nolock.direct.r3.none` | 0.671 [0.649, 0.693] | 0.740 | 0.526 | 0.659 | 164.2 |

## devcheck2 的 `summary.md` 原文

# X-1 上限（devcheck2，格点 first）

- 500 句；框 IoU ≥ 0.9 记好框；CI 为逐句 bootstrap 95%（1000 次，种子 20260929）。
- token：Qwen/Qwen3.5-9B@c20223623576，格式 x1-text-v1（只数这个通道的输出，框不计）。

## 要点：每种框上的参照与各通道、各噪声档里 cIoU 最高的格点

| 框 | 行 | 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | token 均值 |
|---|---|---|---|---|---|---|
| gt | 参照 | `box` | 0.533 [0.515, 0.552] | 0.533 | — | 0.0 |
| gt | 参照 | `gen.gc_default` | 0.607 [0.579, 0.634] | 0.607 | — | 0.0 |
| gt | 参照 | `genK.4` | 0.693 [0.674, 0.713] | 0.693 | — | 1.0 |
| gt | 参照 | `genK.8` | 0.716 [0.697, 0.736] | 0.716 | — | 1.0 |
| gt | 参照 | `genK.12` | 0.739 [0.722, 0.759] | 0.739 | — | 1.3 |
| gt | 参照 | `genK.16` | 0.749 [0.732, 0.768] | 0.749 | — | 1.5 |
| gt | pts 最高（none） | `pts.k8.fgbg.pr_fgd.none` | 0.768 [0.751, 0.787] | 0.768 | — | 165.4 |
| gt | spl 最高（none） | `spl.n24.raster.none` | 0.930 [0.917, 0.940] | 0.930 | — | 237.0 |
| gt | cor 最高（none） | `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.775 [0.750, 0.797] | 0.775 | — | 166.3 |
| base | 参照 | `box` | 0.498 [0.477, 0.520] | 0.533 | 0.421 | 0.0 |
| base | 参照 | `gen.gc_default` | 0.578 [0.549, 0.607] | 0.641 | 0.446 | 0.0 |
| base | 参照 | `genK.4` | 0.644 [0.619, 0.670] | 0.709 | 0.508 | 1.0 |
| base | 参照 | `genK.8` | 0.663 [0.637, 0.690] | 0.730 | 0.522 | 1.0 |
| base | 参照 | `genK.12` | 0.682 [0.656, 0.709] | 0.748 | 0.545 | 1.3 |
| base | 参照 | `genK.16` | 0.693 [0.665, 0.721] | 0.763 | 0.549 | 1.5 |
| base | pts 最高（none） | `pts.k8.fgbg.pr_fgd.none` | 0.717 [0.694, 0.741] | 0.788 | 0.567 | 165.1 |
| base | spl 最高（none） | `spl.n24.raster.none` | 0.876 [0.852, 0.899] | 0.932 | 0.750 | 237.0 |
| base | cor 最高（none） | `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.717 [0.685, 0.748] | 0.784 | 0.568 | 164.6 |
| w2c | 参照 | `box` | 0.508 [0.487, 0.530] | 0.538 | 0.434 | 0.0 |
| w2c | 参照 | `gen.gc_default` | 0.585 [0.553, 0.617] | 0.635 | 0.467 | 0.0 |
| w2c | 参照 | `genK.4` | 0.658 [0.631, 0.685] | 0.711 | 0.531 | 1.0 |
| w2c | 参照 | `genK.8` | 0.679 [0.652, 0.707] | 0.737 | 0.542 | 1.0 |
| w2c | 参照 | `genK.12` | 0.702 [0.676, 0.728] | 0.757 | 0.570 | 1.3 |
| w2c | 参照 | `genK.16` | 0.711 [0.685, 0.739] | 0.769 | 0.576 | 1.5 |
| w2c | 参照 | `ref.w2c_final` | 0.642 [0.616, 0.670] | 0.701 | 0.505 | 0.0 |
| w2c | pts 最高（none） | `pts.k8.fgbg.pr_fgd.none` | 0.731 [0.706, 0.756] | 0.789 | 0.591 | 165.1 |
| w2c | spl 最高（none） | `spl.n24.raster.none` | 0.899 [0.877, 0.917] | 0.931 | 0.811 | 237.0 |
| w2c | cor 最高（none） | `cor.w2c_final.m2.nolock.grabcut.r3.none` | 0.788 [0.766, 0.808] | 0.838 | 0.656 | 167.2 |

## 框：gt

n = 500；好框 500、坏框 0（其中解析失败 0）；框 Acc@0.5 = 1.000

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.533 [0.515, 0.552] | 0.533 | — | 0.572 | 0.0 |
| `gen.gc_default` | 0.607 [0.579, 0.634] | 0.607 | — | 0.598 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.602 [0.579, 0.628] | 0.602 | — | 0.616 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.607 [0.579, 0.634] | 0.607 | — | 0.597 | 0.0 |
| `gen.ws_pts` | 0.571 [0.548, 0.598] | 0.571 | — | 0.617 | 0.0 |
| `gen.gc_iter10` | 0.605 [0.577, 0.632] | 0.605 | — | 0.596 | 0.0 |
| `gen.gc_lab` | 0.608 [0.580, 0.635] | 0.608 | — | 0.601 | 0.0 |
| `gen.gc_ctx50` | 0.615 [0.589, 0.641] | 0.615 | — | 0.600 | 0.0 |
| `gen.slic_s50_flood` | 0.506 [0.483, 0.530] | 0.506 | — | 0.499 | 0.0 |
| `gen.felz_k100_flood` | 0.466 [0.438, 0.494] | 0.466 | — | 0.459 | 0.0 |
| `gen.gmm_k3_lab+lcc+fill` | 0.588 [0.564, 0.614] | 0.588 | — | 0.627 | 0.0 |
| `gen.ws_lab_shrink20_margin5` | 0.599 [0.579, 0.621] | 0.599 | — | 0.649 | 0.0 |
| `gen.gc_default+open3+lcc+fill` | 0.606 [0.578, 0.634] | 0.606 | — | 0.596 | 0.0 |
| `gen.gc_hsv` | 0.594 [0.566, 0.622] | 0.594 | — | 0.591 | 0.0 |
| `gen.slic_s100_color` | 0.568 [0.550, 0.589] | 0.568 | — | 0.603 | 0.0 |
| `gen.gmm_k8_rgb+lcc+fill` | 0.607 [0.584, 0.633] | 0.607 | — | 0.646 | 0.0 |
| `gen.gc_box_minus5+close5+fill` | 0.490 [0.459, 0.521] | 0.490 | — | 0.471 | 0.0 |
| `genK.4` | 0.693 [0.674, 0.713] | 0.693 | — | 0.717 | 1.0 |
| `genK.8` | 0.716 [0.697, 0.736] | 0.716 | — | 0.741 | 1.0 |
| `genK.12` | 0.739 [0.722, 0.759] | 0.739 | — | 0.765 | 1.3 |
| `genK.16` | 0.749 [0.732, 0.768] | 0.749 | — | 0.772 | 1.5 |
| `pts.k2.fg.pr_bgd.none` | 0.293 [0.261, 0.322] | 0.293 | — | 0.242 | 23.9 |
| `pts.k2.fg.pr_fgd.none` | 0.652 [0.629, 0.676] | 0.652 | — | 0.654 | 23.9 |
| `pts.k2.fgbg.pr_bgd.none` | 0.292 [0.260, 0.322] | 0.292 | — | 0.243 | 46.6 |
| `pts.k2.fgbg.pr_fgd.none` | 0.670 [0.646, 0.694] | 0.670 | — | 0.663 | 46.6 |
| `pts.k8.fg.pr_bgd.none` | 0.683 [0.663, 0.699] | 0.683 | — | 0.675 | 83.3 |
| `pts.k8.fg.pr_fgd.none` | 0.704 [0.684, 0.725] | 0.704 | — | 0.730 | 83.3 |
| `pts.k8.fgbg.pr_bgd.none` | 0.691 [0.671, 0.708] | 0.691 | — | 0.680 | 165.4 |
| `pts.k8.fgbg.pr_fgd.none` | 0.768 [0.751, 0.787] | 0.768 | — | 0.776 | 165.4 |
| `spl.n12.raster.none` | 0.855 [0.840, 0.867] | 0.855 | — | 0.861 | 118.7 |
| `spl.n12.gc_soft.none` | 0.698 [0.674, 0.720] | 0.698 | — | 0.661 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.865 [0.852, 0.876] | 0.865 | — | 0.865 | 118.7 |
| `spl.n24.raster.none` | 0.930 [0.917, 0.940] | 0.930 | — | 0.932 | 237.0 |
| `spl.n24.gc_soft.none` | 0.693 [0.670, 0.716] | 0.693 | — | 0.652 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.898 [0.886, 0.908] | 0.898 | — | 0.896 | 237.0 |
| `cor.gc_default.m2.lock.grabcut.r1.none` | 0.728 [0.700, 0.752] | 0.728 | — | 0.699 | 54.8 |
| `cor.gc_default.m2.lock.grabcut.r2.none` | 0.763 [0.738, 0.785] | 0.763 | — | 0.727 | 110.5 |
| `cor.gc_default.m2.lock.grabcut.r3.none` | 0.774 [0.749, 0.796] | 0.774 | — | 0.737 | 166.3 |
| `cor.gc_default.m2.lock.direct.r1.none` | 0.693 [0.672, 0.714] | 0.693 | — | 0.696 | 54.8 |
| `cor.gc_default.m2.lock.direct.r2.none` | 0.682 [0.661, 0.704] | 0.682 | — | 0.672 | 111.1 |
| `cor.gc_default.m2.lock.direct.r3.none` | 0.715 [0.694, 0.735] | 0.715 | — | 0.714 | 167.5 |
| `cor.gc_default.m2.nolock.grabcut.r1.none` | 0.728 [0.700, 0.752] | 0.728 | — | 0.699 | 54.8 |
| `cor.gc_default.m2.nolock.grabcut.r2.none` | 0.763 [0.738, 0.786] | 0.763 | — | 0.728 | 110.5 |
| `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.775 [0.750, 0.797] | 0.775 | — | 0.738 | 166.3 |
| `cor.gc_default.m2.nolock.direct.r1.none` | 0.693 [0.672, 0.714] | 0.693 | — | 0.696 | 54.8 |
| `cor.gc_default.m2.nolock.direct.r2.none` | 0.684 [0.663, 0.705] | 0.684 | — | 0.674 | 111.1 |
| `cor.gc_default.m2.nolock.direct.r3.none` | 0.722 [0.702, 0.741] | 0.722 | — | 0.722 | 167.4 |

## 框：base

n = 500；好框 327、坏框 173（其中解析失败 4）；框 Acc@0.5 = 0.918

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.498 [0.477, 0.520] | 0.533 | 0.421 | 0.526 | 0.0 |
| `gen.gc_default` | 0.578 [0.549, 0.607] | 0.641 | 0.446 | 0.569 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.562 [0.536, 0.590] | 0.623 | 0.436 | 0.576 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.577 [0.549, 0.608] | 0.640 | 0.447 | 0.568 | 0.0 |
| `gen.ws_pts` | 0.541 [0.519, 0.565] | 0.595 | 0.426 | 0.575 | 0.0 |
| `gen.gc_iter10` | 0.577 [0.547, 0.606] | 0.639 | 0.447 | 0.567 | 0.0 |
| `gen.gc_lab` | 0.578 [0.548, 0.609] | 0.639 | 0.452 | 0.572 | 0.0 |
| `gen.gc_ctx50` | 0.573 [0.543, 0.603] | 0.642 | 0.431 | 0.566 | 0.0 |
| `gen.slic_s50_flood` | 0.469 [0.442, 0.498] | 0.530 | 0.335 | 0.461 | 0.0 |
| `gen.felz_k100_flood` | 0.431 [0.403, 0.462] | 0.484 | 0.316 | 0.428 | 0.0 |
| `gen.gmm_k3_lab+lcc+fill` | 0.545 [0.521, 0.573] | 0.614 | 0.401 | 0.576 | 0.0 |
| `gen.ws_lab_shrink20_margin5` | 0.556 [0.532, 0.580] | 0.606 | 0.450 | 0.596 | 0.0 |
| `gen.gc_default+open3+lcc+fill` | 0.577 [0.548, 0.607] | 0.640 | 0.446 | 0.567 | 0.0 |
| `gen.gc_hsv` | 0.570 [0.544, 0.596] | 0.636 | 0.428 | 0.558 | 0.0 |
| `gen.slic_s100_color` | 0.532 [0.510, 0.557] | 0.582 | 0.428 | 0.561 | 0.0 |
| `gen.gmm_k8_rgb+lcc+fill` | 0.565 [0.540, 0.594] | 0.635 | 0.421 | 0.597 | 0.0 |
| `gen.gc_box_minus5+close5+fill` | 0.472 [0.440, 0.502] | 0.532 | 0.349 | 0.452 | 0.0 |
| `genK.4` | 0.644 [0.619, 0.670] | 0.709 | 0.508 | 0.661 | 1.0 |
| `genK.8` | 0.663 [0.637, 0.690] | 0.730 | 0.522 | 0.682 | 1.0 |
| `genK.12` | 0.682 [0.656, 0.709] | 0.748 | 0.545 | 0.704 | 1.3 |
| `genK.16` | 0.693 [0.665, 0.721] | 0.763 | 0.549 | 0.714 | 1.5 |
| `pts.k2.fg.pr_bgd.none` | 0.284 [0.254, 0.311] | 0.332 | 0.178 | 0.234 | 23.8 |
| `pts.k2.fg.pr_fgd.none` | 0.606 [0.580, 0.634] | 0.674 | 0.463 | 0.611 | 23.8 |
| `pts.k2.fgbg.pr_bgd.none` | 0.283 [0.253, 0.312] | 0.332 | 0.177 | 0.235 | 46.5 |
| `pts.k2.fgbg.pr_fgd.none` | 0.622 [0.596, 0.650] | 0.692 | 0.477 | 0.621 | 46.5 |
| `pts.k8.fg.pr_bgd.none` | 0.663 [0.640, 0.685] | 0.724 | 0.531 | 0.645 | 83.3 |
| `pts.k8.fg.pr_fgd.none` | 0.652 [0.627, 0.678] | 0.717 | 0.515 | 0.676 | 83.3 |
| `pts.k8.fgbg.pr_bgd.none` | 0.671 [0.649, 0.694] | 0.735 | 0.532 | 0.653 | 165.1 |
| `pts.k8.fgbg.pr_fgd.none` | 0.717 [0.694, 0.741] | 0.788 | 0.567 | 0.720 | 165.1 |
| `spl.n12.raster.none` | 0.808 [0.784, 0.830] | 0.855 | 0.701 | 0.804 | 118.7 |
| `spl.n12.gc_soft.none` | 0.654 [0.626, 0.682] | 0.726 | 0.499 | 0.615 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.815 [0.791, 0.836] | 0.869 | 0.693 | 0.805 | 118.7 |
| `spl.n24.raster.none` | 0.876 [0.852, 0.899] | 0.932 | 0.750 | 0.868 | 237.0 |
| `spl.n24.gc_soft.none` | 0.657 [0.629, 0.685] | 0.723 | 0.513 | 0.619 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.844 [0.820, 0.865] | 0.903 | 0.711 | 0.832 | 237.0 |
| `cor.gc_default.m2.lock.grabcut.r1.none` | 0.649 [0.616, 0.683] | 0.713 | 0.503 | 0.631 | 54.6 |
| `cor.gc_default.m2.lock.grabcut.r2.none` | 0.700 [0.668, 0.732] | 0.769 | 0.547 | 0.672 | 109.5 |
| `cor.gc_default.m2.lock.grabcut.r3.none` | 0.717 [0.684, 0.747] | 0.783 | 0.567 | 0.684 | 164.6 |
| `cor.gc_default.m2.lock.direct.r1.none` | 0.625 [0.597, 0.653] | 0.670 | 0.523 | 0.616 | 54.6 |
| `cor.gc_default.m2.lock.direct.r2.none` | 0.654 [0.630, 0.679] | 0.694 | 0.565 | 0.651 | 109.7 |
| `cor.gc_default.m2.lock.direct.r3.none` | 0.667 [0.641, 0.692] | 0.712 | 0.566 | 0.662 | 165.0 |
| `cor.gc_default.m2.nolock.grabcut.r1.none` | 0.649 [0.616, 0.683] | 0.713 | 0.503 | 0.631 | 54.6 |
| `cor.gc_default.m2.nolock.grabcut.r2.none` | 0.699 [0.667, 0.730] | 0.767 | 0.547 | 0.672 | 109.5 |
| `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.717 [0.685, 0.748] | 0.784 | 0.568 | 0.685 | 164.6 |
| `cor.gc_default.m2.nolock.direct.r1.none` | 0.625 [0.597, 0.653] | 0.670 | 0.523 | 0.616 | 54.6 |
| `cor.gc_default.m2.nolock.direct.r2.none` | 0.656 [0.632, 0.681] | 0.694 | 0.568 | 0.654 | 109.7 |
| `cor.gc_default.m2.nolock.direct.r3.none` | 0.664 [0.639, 0.690] | 0.710 | 0.561 | 0.661 | 165.0 |

## 框：w2c

n = 500；好框 347、坏框 153（其中解析失败 0）；框 Acc@0.5 = 0.934

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.508 [0.487, 0.530] | 0.538 | 0.434 | 0.542 | 0.0 |
| `gen.gc_default` | 0.585 [0.553, 0.617] | 0.635 | 0.467 | 0.580 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.571 [0.543, 0.600] | 0.619 | 0.456 | 0.587 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.584 [0.552, 0.616] | 0.635 | 0.464 | 0.578 | 0.0 |
| `gen.ws_pts` | 0.552 [0.531, 0.579] | 0.593 | 0.455 | 0.591 | 0.0 |
| `gen.gc_iter10` | 0.585 [0.554, 0.617] | 0.635 | 0.468 | 0.580 | 0.0 |
| `gen.gc_lab` | 0.587 [0.557, 0.618] | 0.638 | 0.468 | 0.584 | 0.0 |
| `gen.gc_ctx50` | 0.585 [0.554, 0.616] | 0.642 | 0.452 | 0.577 | 0.0 |
| `gen.slic_s50_flood` | 0.485 [0.456, 0.514] | 0.526 | 0.384 | 0.475 | 0.0 |
| `gen.felz_k100_flood` | 0.437 [0.408, 0.467] | 0.481 | 0.327 | 0.436 | 0.0 |
| `gen.gmm_k3_lab+lcc+fill` | 0.559 [0.535, 0.587] | 0.613 | 0.430 | 0.590 | 0.0 |
| `gen.ws_lab_shrink20_margin5` | 0.570 [0.546, 0.596] | 0.612 | 0.470 | 0.614 | 0.0 |
| `gen.gc_default+open3+lcc+fill` | 0.584 [0.552, 0.616] | 0.636 | 0.462 | 0.577 | 0.0 |
| `gen.gc_hsv` | 0.583 [0.555, 0.613] | 0.639 | 0.451 | 0.579 | 0.0 |
| `gen.slic_s100_color` | 0.541 [0.518, 0.565] | 0.579 | 0.450 | 0.575 | 0.0 |
| `gen.gmm_k8_rgb+lcc+fill` | 0.582 [0.558, 0.611] | 0.639 | 0.445 | 0.618 | 0.0 |
| `gen.gc_box_minus5+close5+fill` | 0.480 [0.448, 0.510] | 0.529 | 0.363 | 0.461 | 0.0 |
| `genK.4` | 0.658 [0.631, 0.685] | 0.711 | 0.531 | 0.682 | 1.0 |
| `genK.8` | 0.679 [0.652, 0.707] | 0.737 | 0.542 | 0.705 | 1.0 |
| `genK.12` | 0.702 [0.676, 0.728] | 0.757 | 0.570 | 0.727 | 1.3 |
| `genK.16` | 0.711 [0.685, 0.739] | 0.769 | 0.576 | 0.736 | 1.5 |
| `ref.w2c_final` | 0.642 [0.616, 0.670] | 0.701 | 0.505 | 0.669 | 0.0 |
| `pts.k2.fg.pr_bgd.none` | 0.284 [0.252, 0.313] | 0.329 | 0.171 | 0.235 | 23.9 |
| `pts.k2.fg.pr_fgd.none` | 0.615 [0.587, 0.644] | 0.676 | 0.471 | 0.620 | 23.9 |
| `pts.k2.fgbg.pr_bgd.none` | 0.285 [0.253, 0.314] | 0.329 | 0.172 | 0.236 | 46.5 |
| `pts.k2.fgbg.pr_fgd.none` | 0.636 [0.607, 0.665] | 0.699 | 0.487 | 0.635 | 46.5 |
| `pts.k8.fg.pr_bgd.none` | 0.660 [0.632, 0.685] | 0.712 | 0.531 | 0.649 | 83.3 |
| `pts.k8.fg.pr_fgd.none` | 0.663 [0.636, 0.690] | 0.720 | 0.528 | 0.689 | 83.3 |
| `pts.k8.fgbg.pr_bgd.none` | 0.673 [0.645, 0.698] | 0.723 | 0.546 | 0.659 | 165.1 |
| `pts.k8.fgbg.pr_fgd.none` | 0.731 [0.706, 0.756] | 0.789 | 0.591 | 0.737 | 165.1 |
| `spl.n12.raster.none` | 0.829 [0.808, 0.847] | 0.855 | 0.758 | 0.824 | 118.7 |
| `spl.n12.gc_soft.none` | 0.670 [0.643, 0.696] | 0.722 | 0.538 | 0.631 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.836 [0.816, 0.854] | 0.869 | 0.750 | 0.825 | 118.7 |
| `spl.n24.raster.none` | 0.899 [0.877, 0.917] | 0.931 | 0.811 | 0.889 | 237.0 |
| `spl.n24.gc_soft.none` | 0.672 [0.645, 0.699] | 0.720 | 0.547 | 0.634 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.867 [0.846, 0.884] | 0.904 | 0.769 | 0.853 | 237.0 |
| `cor.gc_default.m2.lock.grabcut.r1.none` | 0.662 [0.629, 0.693] | 0.712 | 0.525 | 0.640 | 54.8 |
| `cor.gc_default.m2.lock.grabcut.r2.none` | 0.716 [0.685, 0.744] | 0.767 | 0.584 | 0.687 | 109.8 |
| `cor.gc_default.m2.lock.grabcut.r3.none` | 0.735 [0.705, 0.762] | 0.781 | 0.610 | 0.701 | 164.8 |
| `cor.gc_default.m2.lock.direct.r1.none` | 0.646 [0.618, 0.673] | 0.684 | 0.543 | 0.637 | 54.8 |
| `cor.gc_default.m2.lock.direct.r2.none` | 0.676 [0.654, 0.699] | 0.702 | 0.608 | 0.670 | 110.2 |
| `cor.gc_default.m2.lock.direct.r3.none` | 0.691 [0.668, 0.715] | 0.724 | 0.601 | 0.683 | 165.8 |
| `cor.gc_default.m2.nolock.grabcut.r1.none` | 0.662 [0.629, 0.693] | 0.712 | 0.525 | 0.640 | 54.8 |
| `cor.gc_default.m2.nolock.grabcut.r2.none` | 0.714 [0.682, 0.741] | 0.763 | 0.585 | 0.686 | 109.8 |
| `cor.gc_default.m2.nolock.grabcut.r3.none` | 0.734 [0.703, 0.762] | 0.781 | 0.610 | 0.701 | 164.8 |
| `cor.gc_default.m2.nolock.direct.r1.none` | 0.646 [0.618, 0.673] | 0.684 | 0.543 | 0.637 | 54.8 |
| `cor.gc_default.m2.nolock.direct.r2.none` | 0.678 [0.656, 0.700] | 0.703 | 0.611 | 0.673 | 110.2 |
| `cor.gc_default.m2.nolock.direct.r3.none` | 0.688 [0.664, 0.713] | 0.720 | 0.600 | 0.681 | 165.8 |
| `cor.w2c_final.m2.lock.grabcut.r1.none` | 0.730 [0.706, 0.755] | 0.781 | 0.600 | 0.725 | 55.7 |
| `cor.w2c_final.m2.lock.grabcut.r2.none` | 0.773 [0.752, 0.794] | 0.826 | 0.638 | 0.765 | 111.4 |
| `cor.w2c_final.m2.lock.grabcut.r3.none` | 0.786 [0.765, 0.806] | 0.837 | 0.656 | 0.774 | 167.2 |
| `cor.w2c_final.m2.lock.direct.r1.none` | 0.663 [0.636, 0.692] | 0.706 | 0.548 | 0.654 | 55.7 |
| `cor.w2c_final.m2.lock.direct.r2.none` | 0.712 [0.693, 0.731] | 0.740 | 0.640 | 0.717 | 111.3 |
| `cor.w2c_final.m2.lock.direct.r3.none` | 0.718 [0.697, 0.740] | 0.754 | 0.622 | 0.711 | 166.8 |
| `cor.w2c_final.m2.nolock.grabcut.r1.none` | 0.730 [0.706, 0.755] | 0.781 | 0.600 | 0.725 | 55.7 |
| `cor.w2c_final.m2.nolock.grabcut.r2.none` | 0.775 [0.752, 0.796] | 0.826 | 0.644 | 0.767 | 111.4 |
| `cor.w2c_final.m2.nolock.grabcut.r3.none` | 0.788 [0.766, 0.808] | 0.838 | 0.656 | 0.776 | 167.2 |
| `cor.w2c_final.m2.nolock.direct.r1.none` | 0.663 [0.636, 0.692] | 0.706 | 0.548 | 0.654 | 55.7 |
| `cor.w2c_final.m2.nolock.direct.r2.none` | 0.714 [0.696, 0.733] | 0.741 | 0.645 | 0.720 | 111.3 |
| `cor.w2c_final.m2.nolock.direct.r3.none` | 0.710 [0.688, 0.732] | 0.745 | 0.615 | 0.704 | 166.8 |
