# RUN-0084 报告：探索候选的大题量语言读数（07：基座、X-2、X-3；06：X-3L）

任务书 `prompts/server_hpc/0084-hpc-lb-explore.md`（D145）。MMLU-Pro 2,000 题 + IFEval 541 条，只用开发数据，不碰 test 与 devcheck3。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0084.sh`（10-28 03:06:08Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0084（稀疏）HEAD `46dff4ae14fa8c07583883b05fa4f05f5c24a4c1`，status 空，`experiments/` 只有 P0；6 个 sha256 与任务书逐字相同；驱动用本节点暂停文件。
- 输入：`lang_battery_raw` 的 MMLU-Pro parquet 与 IFEval jsonl 都在；RUN-0074 在 06 上的 `lb_base` 在；X-2（`a49cf4e5…`）、X-3（`837bb679…`）检查点的 `read_manifest` 都通过（各 4 个文件）。
- 03:07:18Z 按任务书命令两节点同时启动（`-c 8` 均被接受）：07 `EV_CAP_H=7`，06 `EV_CAP_H=3`。后 squeue：157296、157297 RUNNING。
- 07 03:07:26Z 开始 lb_base；推送循环 0084-07（PID 4187152）。
- 06 驱动等 `eval_0083.status` 以 `done 0083` 开头，03:36:30Z 开始；推送循环 0084-06（PID 147134）。

## steps.log 原文

07（`experiments/X/0084_lb_explore_hpcgpu07`）：

```
2026-10-28T03:07:26Z start commit 46dff4ae14fa8c07583883b05fa4f05f5c24a4c1 node hpcgpu07
2026-10-28T04:52:12Z lb_base rc=0 wall_s=6286
2026-10-28T04:52:13Z x2 checkpoint manifest a49cf4e5a81fd8f7f57ca5c86a879cf24cdd1449daa43cc355e6d3a074a6fd0a
2026-10-28T06:54:09Z lb_x2 rc=0 wall_s=7316
2026-10-28T06:54:10Z x3 checkpoint manifest 837bb679cfc172ecfa4bce3af1ae21c4e1774d3d9fdf0e16fc4e5d2c34ed7f5c
2026-10-28T09:12:26Z lb_x3 rc=0 wall_s=8296
2026-10-28T09:13:37Z lb_compare_x2 rc=0 wall_s=71
2026-10-28T09:13:38Z lb_compare_x3 rc=0 wall_s=1
2026-10-28T09:13:38Z done; failed steps: 0
2026-10-28T09:13:38Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）lang battery explore 0084-hpcgpu07 hpcgpu07 done (failed steps: 0)
```

06（`experiments/X/0084_lb_explore_hpcgpu06`）：

```
2026-10-28T03:36:30Z start commit 46dff4ae14fa8c07583883b05fa4f05f5c24a4c1 node hpcgpu06
2026-10-28T03:36:31Z x3l checkpoint manifest eb200b245cc620c7f28b7c8051967cb29409b0d36e7ce7de9d20235e1d628f6a
2026-10-28T05:25:56Z lb_x3l rc=0 wall_s=6565
2026-10-28T05:26:43Z lb_compare_x3l rc=0 wall_s=47
2026-10-28T05:26:43Z done; failed steps: 0
2026-10-28T05:26:43Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）lang battery explore 0084-hpcgpu06 hpcgpu06 done (failed steps: 0)
```

- 心跳：`2026-10-28T09:13:38Z done 0084-hpcgpu07 (failed steps: 0)`；`done 0084-hpcgpu06 (failed steps: 0)`（05:26:43Z）。

## 分数（各 `logs/lb_*.log` 末尾，CI95）

| 运行 | 节点 | MMLU-Pro letter acc | mean_p_gold | IFEval prompt strict | instr strict | prompt loose | instr loose |
|---|---|---|---|---|---|---|---|
| lb_base | 07 | 54.2 [52.1, 56.3] | 49.4 [47.7, 51.0] | 83.7 [80.6, 86.7] | 88.4 [86.0, 90.5] | 87.6 [84.7, 90.2] | 91.2 [89.1, 93.1] |
| lb_x2 | 07 | 54.0 [51.9, 56.0] | 48.4 [46.8, 50.1] | 80.2 [76.9, 83.4] | 86.1 [83.8, 88.5] | 83.2 [80.0, 86.1] | 88.5 [86.4, 90.4] |
| lb_x3 | 07 | 53.5 [51.4, 55.6] | 47.6 [46.0, 49.3] | 81.1 [77.8, 84.5] | 87.1 [84.7, 89.2] | 85.0 [81.9, 88.0] | 89.6 [87.4, 91.7] |
| lb_x3l | 06 | 52.8 [50.5, 54.9] | 48.8 [47.1, 50.4] | 79.5 [76.2, 83.0] | 85.4 [82.9, 87.8] | 83.4 [80.0, 86.5] | 88.2 [85.9, 90.4] |

- n：MMLU-Pro 2,000；IFEval prompt 541、instruction 834。
- 06 的 X-3L 对比的基座是 RUN-0074 在 06 上的 `lb_base`：mmlu 54.2、mean_p_gold 49.4、IFEval 83.5 / 88.2 / 87.6 / 91.2（见下方 compare 原文）。同一模型同一数据，07 本次基座的 IFEval prompt strict、instr strict 是 83.7、88.4，其余四项相同。

| 运行 | preflight | merge check | ok | 生成 tokens | call_time s | 日志 wall s |
|---|---|---|---|---|---|---|
| lb_base | 2541 查、0 失败 | —（基座） | True | 164,948 | 5892.8 | 5995.41 |
| lb_x2 | 2541 查、0 失败 | v1 pass True，failures merge [] | True | 167,484 | 5854.4 | 6986.83 |
| lb_x3 | 2541 查、0 失败 | v1 pass True，failures merge [] | True | 159,592 | 5631.1 | 8053.01 |
| lb_x3l | 2541 查、0 失败 | v1 pass True，failures merge [] | True | 166,220 | 5959.0 | 6351.21 |

- prompt tokens 四个运行都是 464,410。
- 两节点 `logs/*.log` 均无 Traceback / OOM。
- GPU 显存：本驱动与 `eval_lib.sh` 不写 `gpu_mem.csv`，本次无显存记录。

## --compare 原文

### 07 `logs/lb_compare_x2.log`

```
base=base  other=x2  (other - base, percentage points; partial_run=False)
mmlu_pro_letter  accuracy                  base=54.2 other=54.0 delta=-0.2 pp CI95=[-1.6, +1.3]  sign test 119+/122- p=0.898  (lost 122, gained 119, n=2000)
mmlu_pro_letter  mean_p_gold               base=49.4 other=48.4 delta=-1.0 pp CI95=[-1.7, -0.3]  (n=2000)
ifeval           prompt_level_strict       base=83.7 other=80.2 delta=-3.5 pp CI95=[-6.1, -0.9]  sign test 19+/38- p=0.0163  (lost 38, gained 19, n=541)
ifeval           instruction_level_strict  base=88.4 other=86.1 delta=-2.3 pp CI95=[-4.2, -0.4]  sign test 24+/44- p=0.0205  (lost 45, gained 26, n=834)
ifeval           prompt_level_loose        base=87.6 other=83.2 delta=-4.4 pp CI95=[-7.0, -2.0]  sign test 12+/36- p=0.000717  (lost 36, gained 12, n=541)
ifeval           instruction_level_loose   base=91.2 other=88.5 delta=-2.8 pp CI95=[-4.5, -1.1]  sign test 15+/39- p=0.0015  (lost 40, gained 17, n=834)
biology          accuracy                  base=82.5 other=81.8 delta=-0.7 pp CI95=[-4.2, +2.8]  sign test 3+/4- p=1  (lost 4, gained 3, n=143)
business         accuracy                  base=36.4 other=32.9 delta=-3.5 pp CI95=[-9.8, +2.8]  sign test 9+/14- p=0.405  (lost 14, gained 9, n=143)
chemistry        accuracy                  base=36.1 other=36.8 delta=+0.7 pp CI95=[-4.9, +6.3]  sign test 9+/8- p=1  (lost 8, gained 9, n=144)
computer science accuracy                  base=52.1 other=54.2 delta=+2.1 pp CI95=[-4.9, +8.5]  sign test 13+/10- p=0.678  (lost 10, gained 13, n=142)
economics        accuracy                  base=69.9 other=67.1 delta=-2.8 pp CI95=[-7.0, +0.7]  sign test 2+/6- p=0.289  (lost 6, gained 2, n=143)
engineering      accuracy                  base=38.0 other=34.5 delta=-3.5 pp CI95=[-9.9, +2.8]  sign test 8+/13- p=0.383  (lost 13, gained 8, n=142)
health           accuracy                  base=64.8 other=66.9 delta=+2.1 pp CI95=[-3.5, +7.0]  sign test 10+/7- p=0.629  (lost 7, gained 10, n=142)
history          accuracy                  base=68.3 other=66.9 delta=-1.4 pp CI95=[-5.6, +2.1]  sign test 3+/5- p=0.727  (lost 5, gained 3, n=142)
law              accuracy                  base=42.4 other=43.1 delta=+0.7 pp CI95=[-5.6, +6.9]  sign test 11+/10- p=1  (lost 10, gained 11, n=144)
math             accuracy                  base=34.0 other=36.1 delta=+2.1 pp CI95=[-3.5, +8.3]  sign test 11+/8- p=0.648  (lost 8, gained 11, n=144)
other            accuracy                  base=50.7 other=52.8 delta=+2.1 pp CI95=[-3.5, +7.7]  sign test 11+/8- p=0.648  (lost 8, gained 11, n=142)
philosophy       accuracy                  base=62.2 other=58.7 delta=-3.5 pp CI95=[-9.8, +2.8]  sign test 8+/13- p=0.383  (lost 13, gained 8, n=143)
physics          accuracy                  base=44.1 other=49.7 delta=+5.6 pp CI95=[-1.4, +12.6]  sign test 18+/10- p=0.185  (lost 10, gained 18, n=143)
psychology       accuracy                  base=76.9 other=74.8 delta=-2.1 pp CI95=[-6.3, +2.1]  sign test 3+/6- p=0.508  (lost 6, gained 3, n=143)
```

### 07 `logs/lb_compare_x3.log`

```
base=base  other=x3  (other - base, percentage points; partial_run=False)
mmlu_pro_letter  accuracy                  base=54.2 other=53.5 delta=-0.7 pp CI95=[-2.2, +0.6]  sign test 110+/124- p=0.395  (lost 124, gained 110, n=2000)
mmlu_pro_letter  mean_p_gold               base=49.4 other=47.6 delta=-1.8 pp CI95=[-2.4, -1.1]  (n=2000)
ifeval           prompt_level_strict       base=83.7 other=81.1 delta=-2.6 pp CI95=[-5.2, +0.0]  sign test 21+/35- p=0.0814  (lost 35, gained 21, n=541)
ifeval           instruction_level_strict  base=88.4 other=87.1 delta=-1.3 pp CI95=[-3.0, +0.6]  sign test 24+/36- p=0.155  (lost 41, gained 30, n=834)
ifeval           prompt_level_loose        base=87.6 other=85.0 delta=-2.6 pp CI95=[-5.0, -0.2]  sign test 18+/32- p=0.0649  (lost 32, gained 18, n=541)
ifeval           instruction_level_loose   base=91.2 other=89.6 delta=-1.7 pp CI95=[-3.2, -0.1]  sign test 20+/33- p=0.0984  (lost 36, gained 22, n=834)
biology          accuracy                  base=82.5 other=81.1 delta=-1.4 pp CI95=[-4.2, +1.4]  sign test 1+/3- p=0.625  (lost 3, gained 1, n=143)
business         accuracy                  base=36.4 other=33.6 delta=-2.8 pp CI95=[-9.1, +2.8]  sign test 7+/11- p=0.481  (lost 11, gained 7, n=143)
chemistry        accuracy                  base=36.1 other=34.7 delta=-1.4 pp CI95=[-6.9, +4.2]  sign test 8+/10- p=0.815  (lost 10, gained 8, n=144)
computer science accuracy                  base=52.1 other=51.4 delta=-0.7 pp CI95=[-6.3, +4.9]  sign test 8+/9- p=1  (lost 9, gained 8, n=142)
economics        accuracy                  base=69.9 other=69.9 delta=+0.0 pp CI95=[-4.2, +4.2]  sign test 6+/6- p=1  (lost 6, gained 6, n=143)
engineering      accuracy                  base=38.0 other=35.2 delta=-2.8 pp CI95=[-10.6, +4.2]  sign test 12+/16- p=0.572  (lost 16, gained 12, n=142)
health           accuracy                  base=64.8 other=63.4 delta=-1.4 pp CI95=[-7.0, +4.2]  sign test 7+/9- p=0.804  (lost 9, gained 7, n=142)
history          accuracy                  base=68.3 other=69.0 delta=+0.7 pp CI95=[-4.2, +5.6]  sign test 7+/6- p=1  (lost 6, gained 7, n=142)
law              accuracy                  base=42.4 other=39.6 delta=-2.8 pp CI95=[-9.7, +3.5]  sign test 10+/14- p=0.541  (lost 14, gained 10, n=144)
math             accuracy                  base=34.0 other=33.3 delta=-0.7 pp CI95=[-6.3, +4.9]  sign test 8+/9- p=1  (lost 9, gained 8, n=144)
other            accuracy                  base=50.7 other=51.4 delta=+0.7 pp CI95=[-4.9, +6.4]  sign test 10+/9- p=1  (lost 9, gained 10, n=142)
philosophy       accuracy                  base=62.2 other=63.6 delta=+1.4 pp CI95=[-2.8, +5.6]  sign test 6+/4- p=0.754  (lost 4, gained 6, n=143)
physics          accuracy                  base=44.1 other=46.2 delta=+2.1 pp CI95=[-5.6, +9.8]  sign test 16+/13- p=0.711  (lost 13, gained 16, n=143)
psychology       accuracy                  base=76.9 other=76.2 delta=-0.7 pp CI95=[-4.9, +3.5]  sign test 4+/5- p=1  (lost 5, gained 4, n=143)
```

### 06 `logs/lb_compare_x3l.log`（基座 = RUN-0074 在 06 上的 lb_base）

```
base=base  other=x3l  (other - base, percentage points; partial_run=False)
mmlu_pro_letter  accuracy                  base=54.2 other=52.8 delta=-1.4 pp CI95=[-2.9, +0.1]  sign test 106+/134- p=0.0811  (lost 134, gained 106, n=2000)
mmlu_pro_letter  mean_p_gold               base=49.4 other=48.8 delta=-0.7 pp CI95=[-1.3, +0.0]  (n=2000)
ifeval           prompt_level_strict       base=83.5 other=79.5 delta=-4.1 pp CI95=[-7.2, -1.1]  sign test 23+/45- p=0.0103  (lost 45, gained 23, n=541)
ifeval           instruction_level_strict  base=88.2 other=85.4 delta=-2.9 pp CI95=[-5.0, -0.8]  sign test 25+/50- p=0.00523  (lost 55, gained 31, n=834)
ifeval           prompt_level_loose        base=87.6 other=83.4 delta=-4.3 pp CI95=[-7.2, -1.3]  sign test 18+/41- p=0.00379  (lost 41, gained 18, n=541)
ifeval           instruction_level_loose   base=91.2 other=88.2 delta=-3.0 pp CI95=[-4.9, -1.1]  sign test 19+/44- p=0.00223  (lost 47, gained 22, n=834)
biology          accuracy                  base=82.5 other=82.5 delta=+0.0 pp CI95=[-3.5, +3.5]  sign test 3+/3- p=1  (lost 3, gained 3, n=143)
business         accuracy                  base=36.4 other=28.7 delta=-7.7 pp CI95=[-13.3, -2.1]  sign test 3+/14- p=0.0127  (lost 14, gained 3, n=143)
chemistry        accuracy                  base=36.1 other=34.7 delta=-1.4 pp CI95=[-7.6, +4.9]  sign test 8+/10- p=0.815  (lost 10, gained 8, n=144)
computer science accuracy                  base=52.1 other=52.8 delta=+0.7 pp CI95=[-6.3, +7.7]  sign test 13+/12- p=1  (lost 12, gained 13, n=142)
economics        accuracy                  base=69.9 other=67.8 delta=-2.1 pp CI95=[-7.0, +2.8]  sign test 5+/8- p=0.581  (lost 8, gained 5, n=143)
engineering      accuracy                  base=38.0 other=33.8 delta=-4.2 pp CI95=[-11.3, +2.8]  sign test 9+/15- p=0.307  (lost 15, gained 9, n=142)
health           accuracy                  base=64.8 other=67.6 delta=+2.8 pp CI95=[-2.8, +8.5]  sign test 11+/7- p=0.481  (lost 7, gained 11, n=142)
history          accuracy                  base=68.3 other=66.2 delta=-2.1 pp CI95=[-7.0, +2.1]  sign test 4+/7- p=0.549  (lost 7, gained 4, n=142)
law              accuracy                  base=42.4 other=41.0 delta=-1.4 pp CI95=[-7.6, +4.2]  sign test 9+/11- p=0.824  (lost 11, gained 9, n=144)
math             accuracy                  base=34.0 other=34.7 delta=+0.7 pp CI95=[-4.9, +6.3]  sign test 10+/9- p=1  (lost 9, gained 10, n=144)
other            accuracy                  base=50.7 other=49.3 delta=-1.4 pp CI95=[-7.7, +4.9]  sign test 10+/12- p=0.832  (lost 12, gained 10, n=142)
philosophy       accuracy                  base=62.2 other=61.5 delta=-0.7 pp CI95=[-4.9, +3.5]  sign test 5+/6- p=1  (lost 6, gained 5, n=143)
physics          accuracy                  base=44.1 other=44.8 delta=+0.7 pp CI95=[-6.3, +7.0]  sign test 13+/12- p=1  (lost 12, gained 13, n=143)
psychology       accuracy                  base=76.9 other=73.4 delta=-3.5 pp CI95=[-7.7, +0.7]  sign test 3+/8- p=0.227  (lost 8, gained 3, n=143)
```

## 收尾

- 推送循环 0084-06：05:54:50Z 在睡眠中按 PID 停（147134 + sleep 491640），ONCE 推送 05:54:58Z ok。第一次停循环的脚本上传时成了空文件（嵌套 ssh 未转发 stdin，后台命令同行），重新上传并核对 sha256 后再执行。
- 推送循环 0084-07：09:21:04Z 按 PID 停（4187152 + sleep 1310531），ONCE 推送 09:21:15Z ok。
- 后 squeue（09:21:15Z）：157296 RUNNING 73-20:47:01 hpcgpu06；157297 RUNNING 75-18:19:10 hpcgpu07。
- 两节点现在空闲。
