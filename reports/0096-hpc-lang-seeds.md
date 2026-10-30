# RUN-0096 报告：主方法 C1 的 seed 1、seed 2 的语言读数（07：seed 1 大题量；06：两个种子的 T 文本 + seed 2 大题量）

任务书 `prompts/server_hpc/0096-hpc-lang-seeds.md`（D153）。不碰 test 与 devcheck3。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0096.sh`（10-29 13:20:04Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0096（稀疏）HEAD `b5adf2b4638cf7a9b051c61d88ab66312b2daa75`，status 空，`experiments/` 只有 P0；7 个 sha256 与任务书逐字相同（`lang_seeds.sh` `bc97d8e4…`）；驱动用本节点暂停文件。
- 输入：`lang_battery_raw` 的 MMLU-Pro parquet 与 IFEval jsonl、0084 在 07 上的 `lb_base`、W2/0074 在 06 上的 `lb_base`、`0069_ttext` 都在；seed 1 检查点 manifest `333f446b…`，`read_manifest` 通过（4 个文件）；seed 2 检查点当时 RUN-0094 还在训练（06 的驱动等它）。
- 13:21:27Z 两节点按任务书命令启动（`-c 8`、`TINY=0`）：07 `NODE=hpcgpu07 EV_CAP_H=4`；06 `NODE=hpcgpu06 EV_CAP_H=8`。后 squeue：157296、157297 RUNNING。
- 07：13:21:29Z 第一步 `lb_x2s1`；推送循环 0096-07（PID 1765528）。
- 06：驱动等 `eval_0094.status` 以 `done 0094` 开头；21:54:34Z 开始 `ttext_x2s1`；推送循环 0096-06（PID 3299019）。

## steps.log 原文

07：

```
2026-10-29T13:21:28Z start commit b5adf2b4638cf7a9b051c61d88ab66312b2daa75 node hpcgpu07
2026-10-29T13:21:29Z x2s1 checkpoint manifest 333f446b3a3ca77a31a9fdba0968d06c02dbe89c442c41e6bf91bdb197480816
2026-10-29T15:05:02Z lb_x2s1 rc=0 wall_s=6213
2026-10-29T15:05:05Z lb_compare_x2s1 rc=0 wall_s=3
2026-10-29T15:05:05Z done; failed steps: 0
2026-10-29T15:05:05Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）C1 seeds language 0096-hpcgpu07 done (failed steps: 0)
```

06：

```
2026-10-29T21:54:34Z start commit b5adf2b4638cf7a9b051c61d88ab66312b2daa75 node hpcgpu06
2026-10-29T21:54:34Z x2s1 checkpoint manifest 333f446b3a3ca77a31a9fdba0968d06c02dbe89c442c41e6bf91bdb197480816
2026-10-29T23:42:00Z ttext_x2s1 rc=0 wall_s=6446
2026-10-29T23:42:00Z x2s2 checkpoint manifest 836e69235e33b1c97809d76d8d475eab38e53a4a4b0a12b3e0860611a683f1c8
2026-10-30T01:28:32Z ttext_x2s2 rc=0 wall_s=6392
2026-10-30T01:28:32Z x2s2 checkpoint manifest 836e69235e33b1c97809d76d8d475eab38e53a4a4b0a12b3e0860611a683f1c8
2026-10-30T03:12:45Z lb_x2s2 rc=0 wall_s=6253
2026-10-30T03:13:09Z lb_compare_x2s2 rc=0 wall_s=24
2026-10-30T03:13:09Z done; failed steps: 0
2026-10-30T03:13:12Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）C1 seeds language 0096-hpcgpu06 done (failed steps: 0)
```

- 心跳：`2026-10-29T15:05:05Z done 0096-hpcgpu07 (failed steps: 0)`；`2026-10-30T03:13:09Z done 0096-hpcgpu06 (failed steps: 0)`。两节点日志均无 Traceback / OOM。

## 大题量（各 `logs/lb_*.log` 末尾，CI95）

| 运行 | 节点 | MMLU-Pro letter acc | mean_p_gold | IFEval prompt strict | instr strict | prompt loose | instr loose | preflight | merge check |
|---|---|---|---|---|---|---|---|---|---|
| lb_x2s1（seed 1） | 07 | 51.9 [49.8, 54.1] | 46.9 [45.2, 48.6] | 83.7 [80.6, 86.7] | 88.7 [86.6, 90.7] | 87.1 [84.1, 89.8] | 90.9 [88.7, 92.8] | 2541 查、0 失败 | v1 pass True，failures merge [] |
| lb_x2s2（seed 2） | 06 | 51.9 [49.8, 54.2] | 47.7 [46.0, 49.4] | 81.9 [78.6, 85.0] | 87.4 [85.0, 89.6] | 85.2 [81.9, 88.0] | 89.8 [87.5, 91.6] | 2541 查、0 失败 | v1 pass True，failures merge [] |

### 07 `logs/lb_compare_x2s1.log` 原文（基座 = RUN-0084 在 07 上的 `lb_base`）

```
base=base  other=x2s1  (other - base, percentage points; partial_run=False)
mmlu_pro_letter  accuracy                  base=54.2 other=51.9 delta=-2.3 pp CI95=[-3.7, -0.8]  sign test 90+/135- p=0.00326  (lost 135, gained 90, n=2000)
mmlu_pro_letter  mean_p_gold               base=49.4 other=46.9 delta=-2.5 pp CI95=[-3.2, -1.9]  (n=2000)
ifeval           prompt_level_strict       base=83.7 other=83.7 delta=+0.0 pp CI95=[-2.8, +2.8]  sign test 32+/32- p=1  (lost 32, gained 32, n=541)
ifeval           instruction_level_strict  base=88.4 other=88.7 delta=+0.4 pp CI95=[-1.6, +2.5]  sign test 36+/34- p=0.905  (lost 37, gained 40, n=834)
ifeval           prompt_level_loose        base=87.6 other=87.1 delta=-0.6 pp CI95=[-3.1, +2.0]  sign test 24+/27- p=0.78  (lost 27, gained 24, n=541)
ifeval           instruction_level_loose   base=91.2 other=90.9 delta=-0.4 pp CI95=[-2.2, +1.6]  sign test 26+/29- p=0.788  (lost 31, gained 28, n=834)
biology          accuracy                  base=82.5 other=81.8 delta=-0.7 pp CI95=[-3.5, +2.1]  sign test 2+/3- p=1  (lost 3, gained 2, n=143)
business         accuracy                  base=36.4 other=29.4 delta=-7.0 pp CI95=[-13.3, -1.4]  sign test 4+/14- p=0.0309  (lost 14, gained 4, n=143)
chemistry        accuracy                  base=36.1 other=31.9 delta=-4.2 pp CI95=[-10.4, +2.1]  sign test 7+/13- p=0.263  (lost 13, gained 7, n=144)
computer science accuracy                  base=52.1 other=49.3 delta=-2.8 pp CI95=[-8.5, +2.8]  sign test 7+/11- p=0.481  (lost 11, gained 7, n=142)
economics        accuracy                  base=69.9 other=65.0 delta=-4.9 pp CI95=[-10.5, +0.0]  sign test 4+/11- p=0.118  (lost 11, gained 4, n=143)
engineering      accuracy                  base=38.0 other=29.6 delta=-8.5 pp CI95=[-16.2, -1.4]  sign test 7+/19- p=0.029  (lost 19, gained 7, n=142)
health           accuracy                  base=64.8 other=62.0 delta=-2.8 pp CI95=[-7.7, +2.1]  sign test 5+/9- p=0.424  (lost 9, gained 5, n=142)
history          accuracy                  base=68.3 other=69.0 delta=+0.7 pp CI95=[-4.2, +4.9]  sign test 6+/5- p=1  (lost 5, gained 6, n=142)
law              accuracy                  base=42.4 other=43.1 delta=+0.7 pp CI95=[-5.6, +6.3]  sign test 10+/9- p=1  (lost 9, gained 10, n=144)
math             accuracy                  base=34.0 other=32.6 delta=-1.4 pp CI95=[-6.9, +4.2]  sign test 7+/9- p=0.804  (lost 9, gained 7, n=144)
other            accuracy                  base=50.7 other=50.0 delta=-0.7 pp CI95=[-6.3, +4.9]  sign test 9+/10- p=1  (lost 10, gained 9, n=142)
philosophy       accuracy                  base=62.2 other=62.2 delta=+0.0 pp CI95=[-4.9, +4.9]  sign test 6+/6- p=1  (lost 6, gained 6, n=143)
physics          accuracy                  base=44.1 other=42.7 delta=-1.4 pp CI95=[-9.1, +6.3]  sign test 12+/14- p=0.845  (lost 14, gained 12, n=143)
psychology       accuracy                  base=76.9 other=78.3 delta=+1.4 pp CI95=[-2.1, +4.9]  sign test 4+/2- p=0.688  (lost 2, gained 4, n=143)
```

### 06 `logs/lb_compare_x2s2.log` 原文（基座 = RUN-0074 在 06 上的 `lb_base`）

```
base=base  other=x2s2  (other - base, percentage points; partial_run=False)
mmlu_pro_letter  accuracy                  base=54.2 other=51.9 delta=-2.3 pp CI95=[-3.7, -0.7]  sign test 93+/138- p=0.0037  (lost 138, gained 93, n=2000)
mmlu_pro_letter  mean_p_gold               base=49.4 other=47.7 delta=-1.8 pp CI95=[-2.4, -1.1]  (n=2000)
ifeval           prompt_level_strict       base=83.5 other=81.9 delta=-1.7 pp CI95=[-4.4, +0.9]  sign test 23+/32- p=0.281  (lost 32, gained 23, n=541)
ifeval           instruction_level_strict  base=88.2 other=87.4 delta=-0.8 pp CI95=[-2.9, +1.1]  sign test 26+/34- p=0.366  (lost 36, gained 29, n=834)
ifeval           prompt_level_loose        base=87.6 other=85.2 delta=-2.4 pp CI95=[-5.0, -0.2]  sign test 15+/28- p=0.066  (lost 28, gained 15, n=541)
ifeval           instruction_level_loose   base=91.2 other=89.8 delta=-1.4 pp CI95=[-3.3, +0.1]  sign test 17+/28- p=0.135  (lost 30, gained 18, n=834)
biology          accuracy                  base=82.5 other=81.8 delta=-0.7 pp CI95=[-3.5, +2.1]  sign test 2+/3- p=1  (lost 3, gained 2, n=143)
business         accuracy                  base=36.4 other=27.3 delta=-9.1 pp CI95=[-15.4, -3.5]  sign test 3+/16- p=0.00443  (lost 16, gained 3, n=143)
chemistry        accuracy                  base=36.1 other=34.0 delta=-2.1 pp CI95=[-7.6, +4.2]  sign test 7+/10- p=0.629  (lost 10, gained 7, n=144)
computer science accuracy                  base=52.1 other=49.3 delta=-2.8 pp CI95=[-8.5, +2.8]  sign test 6+/10- p=0.454  (lost 10, gained 6, n=142)
economics        accuracy                  base=69.9 other=69.9 delta=+0.0 pp CI95=[-4.2, +4.2]  sign test 6+/6- p=1  (lost 6, gained 6, n=143)
engineering      accuracy                  base=38.0 other=33.1 delta=-4.9 pp CI95=[-12.7, +2.8]  sign test 11+/18- p=0.265  (lost 18, gained 11, n=142)
health           accuracy                  base=64.8 other=63.4 delta=-1.4 pp CI95=[-7.0, +4.9]  sign test 8+/10- p=0.815  (lost 10, gained 8, n=142)
history          accuracy                  base=68.3 other=66.2 delta=-2.1 pp CI95=[-6.3, +1.4]  sign test 3+/6- p=0.508  (lost 6, gained 3, n=142)
law              accuracy                  base=42.4 other=40.3 delta=-2.1 pp CI95=[-7.6, +3.5]  sign test 7+/10- p=0.629  (lost 10, gained 7, n=144)
math             accuracy                  base=34.0 other=32.6 delta=-1.4 pp CI95=[-6.9, +4.2]  sign test 7+/9- p=0.804  (lost 9, gained 7, n=144)
other            accuracy                  base=50.7 other=46.5 delta=-4.2 pp CI95=[-11.3, +2.8]  sign test 10+/16- p=0.327  (lost 16, gained 10, n=142)
philosophy       accuracy                  base=62.2 other=62.9 delta=+0.7 pp CI95=[-4.2, +6.3]  sign test 9+/8- p=1  (lost 8, gained 9, n=143)
physics          accuracy                  base=44.1 other=41.3 delta=-2.8 pp CI95=[-9.8, +3.5]  sign test 10+/14- p=0.541  (lost 14, gained 10, n=143)
psychology       accuracy                  base=76.9 other=78.3 delta=+1.4 pp CI95=[-2.1, +4.9]  sign test 4+/2- p=0.688  (lost 2, gained 4, n=143)
```

## T 文本（06；`metrics.json` 的 `headline`，95% CI，1,000 次重抽）

| 标签 | refexp_box acc@0.5（n = 56,857） | 解析失败 | count accuracy（n = 442） | 解析失败 | complete | merge_check |
|---|---|---|---|---|---|---|
| `t_text/x2s1/seed1` | 0.9066 [0.9041, 0.9089] | 0 | 0.7330 [0.6900, 0.7738] | 0 | True / True | v1 pass True，failures merge [] |
| `t_text/x2s2/seed2` | 0.9063 [0.9041, 0.9087] | 1 | 0.8122 [0.7738, 0.8507] | 0 | True / True | v1 pass True，failures merge [] |

- 截断两者都是 0。计数题出自 CountBenchQA（D36-2 注明）。
- 同一 eval 子集的已有读数（照录）：seed 0（RUN-0091）box 0.9056 [0.9033, 0.9080]、count 0.8077 [0.7715, 0.8439]；RUN-0069 在 06 上的基座 box 0.8714、count 0.8937，W2-c box 0.9067、count 0.7624。

## 收尾

- 推送循环 0096-07：10-29 15:20:54Z 在睡眠中按 PID 停（1765528 + sleep 2110705），ONCE 推送 15:21:06Z ok。
- 推送循环 0096-06：10-30 03:18:09Z 按 PID 停（3299019 + sleep 4181115），ONCE 推送 03:18:46Z ok。
- 后 squeue（03:18:46Z）：157296 RUNNING 75-14:44:32 hpcgpu06；157297 RUNNING 77-12:16:41 hpcgpu07。
- 两节点现在空闲。
