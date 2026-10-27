# RUN-0076 B 段报告：hpcgpu06，W2-c 加回放的评测（test 5 划分、健康检查、T 文本、大题量语言读数）

任务书 `prompts/server_hpc/0076-hpc-w2c-replay.md`（B 段）+ `0076-addendum-1.md`（env 加 `TINY=0`）+ `0076-addendum-2.md`（test 重跑，D140）+ `0076-addendum-3.md`（T 文本重跑，D141）。只贴数字，不解读。

## B 段的启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0076b.sh`（10-27 02:01:57Z 起）。前 squeue：157296、157297 RUNNING。
- wt0076 HEAD `237fbf453a6936381d5da699cec52fb7f6d53895`，status 空，15/15 sha256 再核对一次都相同。
- 回放检查点 `experiments/W2/0076_w2c_replay_hpcgpu07/0076_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`：manifest sha256 在 hpcmgt 与 hpcgpu06 上读到的都是 `ecad7667b4588f4840ba13f79b71b8c52d475e26a99c79a0e6d3999c99dcecfd`；`read_manifest`（wt0076，CPU）通过，4 个文件。
- 02:02:26Z 按管理员的命令启动：`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 NODE=hpcgpu06 STAGES=eval WAIT_GLOB=eval_0078.status EV_CAP_H=9 bash …/replay.sh > $VI/logs/eval_0076b.driver.log`；`-c 8` 被接受；后 squeue 两条 RUNNING。
- 等 RUN-0078 期间心跳 "waiting for gpu (running: eval_0078.status …)"；09:24:46Z 拿到 GPU；推送循环 TAG 0076b（PID 3500395）。

## B 段 steps.log 原文

```
2026-10-27T09:24:46Z start commit 237fbf453a6936381d5da699cec52fb7f6d53895 node hpcgpu06 model qwen3.5-9b stages [eval] replay cfg a775861041c6bf4ebb8c05a7d02505ef50407727b6c2c91e072e885e76b0f920
2026-10-27T09:24:46Z checkpoint /home/mail/chenyuma/visint/repo/experiments/W2/0076_w2c_replay_hpcgpu07/0076_train/w2c/qwen3.5-9b/seed0/ckpt/final/model manifest ecad7667b4588f4840ba13f79b71b8c52d475e26a99c79a0e6d3999c99dcecfd
2026-10-27T09:27:11Z w2c_replay_test_refcoco_unc_testA rc=2 wall_s=145
2026-10-27T09:27:15Z w2c_replay_test_refcoco_unc_testB rc=2 wall_s=4
2026-10-27T09:27:19Z w2c_replay_test_refcoco+_unc_testA rc=2 wall_s=4
2026-10-27T09:27:23Z w2c_replay_test_refcoco+_unc_testB rc=2 wall_s=4
2026-10-27T09:27:27Z w2c_replay_test_refcocog_umd_test rc=2 wall_s=4
2026-10-27T09:53:17Z health_w2c_replay rc=0 wall_s=1550
2026-10-27T09:53:19Z health_compare rc=0 wall_s=2
2026-10-27T09:53:21Z ttext_w2c_replay rc=2 wall_s=2
2026-10-27T11:38:50Z lb_w2c_replay rc=0 wall_s=6329
2026-10-27T11:39:07Z lb_compare rc=0 wall_s=17
2026-10-27T11:39:07Z done; failed steps: 6
2026-10-27T11:39:07Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2-c replay 0076 [eval] done (failed steps: 6)
```

- 心跳 `2026-10-27T11:39:07Z done 0076 [eval] (failed steps: 6)`。
- 失败的 6 步都是 `--out` 的目录名没有运行编号（驱动的错误，D140、D141），在参数解析时就退出，没有写结果：
  - 5 个 test 划分（`logs/w2c_replay_test_*.log` 最后一行相同）：`run_w2.py: error: --out 的目录名必须带运行编号，形如 0063_w2_tune，收到 '/home/mail/chenyuma/visint/repo/experiments/W2/0076_w2c_replay_eval_hpcgpu06/test'`
  - T 文本（`logs/ttext_w2c_replay.log` 最后一行）：`run_e1.py: error: --out 的目录名必须带运行编号，形如 0011_e1_main（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 '/home/mail/chenyuma/visint/repo/experiments/W2/0076_w2c_replay_eval_hpcgpu06/ttext'`
- 两者由补充 2、补充 3 重跑，见下。

## 健康检查（与 RUN-0069 的 health_base 配对）

- health_w2c_replay：ok True，headline {'mmlu_pro': 52.0, 'gsm8k': 92.0, 'ifeval': 88.0}，wall_time_s 1547；合并核对 v1，pass True，failures {"merge": []}。

`logs/health_compare.log` 原文：

```
protocol: base v3, merged v3
mmlu_pro_letter accuracy                  base=53.0 merged=52.0 delta=-1.0 pp  CI95=[-6.0, +5.0]  sign test 4+/5- p=1  (lost 5, gained 4, n=100)
mmlu_pro_letter mean_p_gold               base=49.0 merged=47.7 delta=-1.3 pp  CI95=[-3.6, +1.2]  (n=100)  [诊断，不标红]
mmlu_pro_free256 accuracy                  base=53.0 merged=52.0 delta=-1.0 pp  CI95=[-6.0, +5.0]  sign test 4+/5- p=1  (lost 5, gained 4, n=100)  [诊断，不标红]
mmlu_pro_fmt32  accuracy                  base=53.0 merged=52.0 delta=-1.0 pp  CI95=[-6.0, +5.0]  sign test 4+/5- p=1  (lost 5, gained 4, n=100)  [诊断，不标红]
gsm8k           flexible                  base=86.0 merged=92.0 delta=+6.0 pp  CI95=[+0.0, +12.0]  sign test 3+/0- p=0.25  (lost 0, gained 3, n=50)
gsm8k           strict                    base=86.0 merged=92.0 delta=+6.0 pp  CI95=[+0.0, +12.0]  sign test 3+/0- p=0.25  (lost 0, gained 3, n=50)
ifeval          prompt_level_strict       base=94.0 merged=88.0 delta=-6.0 pp  CI95=[-14.0, +0.0]  sign test 0+/3- p=0.25  (lost 3, gained 0, n=50)  <-- 标红
ifeval          instruction_level_strict  base=96.1 merged=89.6 delta=-6.5 pp  CI95=[-11.8, -1.4]  sign test 0+/5- p=0.0625  (lost 5, gained 0, n=77)
ifeval          prompt_level_loose        base=98.0 merged=90.0 delta=-8.0 pp  CI95=[-16.0, -2.0]  sign test 0+/4- p=0.125  (lost 4, gained 0, n=50)
ifeval          instruction_level_loose   base=98.7 merged=92.2 delta=-6.5 pp  CI95=[-11.5, -1.4]  sign test 0+/5- p=0.0625  (lost 5, gained 0, n=77)
truncation mmlu_pro_free256 max_new_tokens=256   base=0.0% merged=0.0% delta=+0.0 pp (n=100)
truncation mmlu_pro_fmt32  max_new_tokens=32    base=0.0% merged=0.0% delta=+0.0 pp (n=100)
truncation gsm8k           max_new_tokens=512   base=12.0% merged=4.0% delta=-8.0 pp (n=50)
truncation ifeval          max_new_tokens=1024  base=2.0% merged=2.0% delta=+0.0 pp (n=50)
any_flag=True  flagged=['ifeval']  truncation_notices=[]  incomparable=[]  unscored=[]  partial_run=False
```

## 大题量语言读数（与 RUN-0074 的 lb_base 配对）

- lb_w2c_replay：MMLU-Pro 字母作答 acc 52.2 [50.1, 54.3]，mean_p_gold 46.7 [45.1, 48.3]；IFEval prompt strict 80.0 [76.5, 83.4]，instr strict 86.1 [83.7, 88.3]，prompt loose 83.2 [79.9, 86.1]，instr loose 88.2 [86.0, 90.2]；preflight 失败 0；wall_time_s 6287；合并核对 v1，pass True，failures {"merge": []}。

`logs/lb_compare.log` 原文：

```
base=base  other=w2c_replay  (other - base, percentage points; partial_run=False)
mmlu_pro_letter  accuracy                  base=54.2 other=52.2 delta=-2.0 pp CI95=[-3.6, -0.3]  sign test 111+/151- p=0.0158  (lost 151, gained 111, n=2000)
mmlu_pro_letter  mean_p_gold               base=49.4 other=46.7 delta=-2.7 pp CI95=[-3.4, -2.0]  (n=2000)
ifeval           prompt_level_strict       base=83.5 other=80.0 delta=-3.5 pp CI95=[-6.5, -0.6]  sign test 21+/40- p=0.0204  (lost 40, gained 21, n=541)
ifeval           instruction_level_strict  base=88.2 other=86.1 delta=-2.2 pp CI95=[-4.1, -0.1]  sign test 25+/44- p=0.0295  (lost 44, gained 26, n=834)
ifeval           prompt_level_loose        base=87.6 other=83.2 delta=-4.4 pp CI95=[-7.2, -1.8]  sign test 12+/36- p=0.000717  (lost 36, gained 12, n=541)
ifeval           instruction_level_loose   base=91.2 other=88.2 delta=-3.0 pp CI95=[-4.8, -1.2]  sign test 15+/40- p=0.00102  (lost 40, gained 15, n=834)
biology          accuracy                  base=82.5 other=78.3 delta=-4.2 pp CI95=[-9.1, +0.0]  sign test 2+/8- p=0.109  (lost 8, gained 2, n=143)
business         accuracy                  base=36.4 other=32.9 delta=-3.5 pp CI95=[-9.8, +3.5]  sign test 9+/14- p=0.405  (lost 14, gained 9, n=143)
chemistry        accuracy                  base=36.1 other=34.0 delta=-2.1 pp CI95=[-7.6, +3.5]  sign test 7+/10- p=0.629  (lost 10, gained 7, n=144)
computer science accuracy                  base=52.1 other=51.4 delta=-0.7 pp CI95=[-7.7, +5.6]  sign test 11+/12- p=1  (lost 12, gained 11, n=142)
economics        accuracy                  base=69.9 other=63.6 delta=-6.3 pp CI95=[-11.2, -1.4]  sign test 2+/11- p=0.0225  (lost 11, gained 2, n=143)
engineering      accuracy                  base=38.0 other=35.9 delta=-2.1 pp CI95=[-9.2, +4.9]  sign test 12+/15- p=0.701  (lost 15, gained 12, n=142)
health           accuracy                  base=64.8 other=62.7 delta=-2.1 pp CI95=[-7.0, +3.5]  sign test 7+/10- p=0.629  (lost 10, gained 7, n=142)
history          accuracy                  base=68.3 other=64.1 delta=-4.2 pp CI95=[-9.2, +0.0]  sign test 3+/9- p=0.146  (lost 9, gained 3, n=142)
law              accuracy                  base=42.4 other=41.0 delta=-1.4 pp CI95=[-8.3, +4.9]  sign test 11+/13- p=0.839  (lost 13, gained 11, n=144)
math             accuracy                  base=34.0 other=32.6 delta=-1.4 pp CI95=[-6.9, +4.9]  sign test 8+/10- p=0.815  (lost 10, gained 8, n=144)
other            accuracy                  base=50.7 other=51.4 delta=+0.7 pp CI95=[-6.3, +7.7]  sign test 13+/12- p=1  (lost 12, gained 13, n=142)
philosophy       accuracy                  base=62.2 other=62.2 delta=+0.0 pp CI95=[-5.6, +5.6]  sign test 8+/8- p=1  (lost 8, gained 8, n=143)
physics          accuracy                  base=44.1 other=46.9 delta=+2.8 pp CI95=[-4.2, +9.8]  sign test 15+/11- p=0.557  (lost 11, gained 15, n=143)
psychology       accuracy                  base=76.9 other=73.4 delta=-3.5 pp CI95=[-8.4, +0.7]  sign test 3+/8- p=0.227  (lost 8, gained 3, n=143)
```

## test 重跑（补充 2）

- `$VI/tmp/start0076t.sh`（10-27 09:45:38Z 起）：wt0076t（稀疏）HEAD `e11efee0d5c3088905447dbeb43cbe418baf3442`，`test_replay.sh` sha256 `dbd89144806a1b096eb17eb0d2989c512a47a7d3a01314c677bd3ce6743881ae` 相同；CODE 用 wt0072（HEAD `d38a5205…`），其 12 个文件与 RUN-0072 任务书相同；检查点 manifest `ecad7667…`，`read_manifest` 通过。
- 09:46:27Z 启动：`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=4 bash …/test_replay.sh > $VI/logs/eval_0076t.driver.log`；`-c 8` 被接受；前后 squeue 两条 RUNNING；等 B 段期间心跳 "waiting for eval_0076.status to start with 'done 0076 [eval]' …"。
- 输出 `experiments/W2/0076_w2c_replay_test_hpcgpu06/`。steps.log 原文：

```
2026-10-27T11:39:34Z start code d38a520525c6208a3f5ddb1062ca09a9646e85b6 node hpcgpu06 checkpoint manifest ecad7667b4588f4840ba13f79b71b8c52d475e26a99c79a0e6d3999c99dcecfd
2026-10-27T12:01:29Z w2c_replay_test_refcoco_unc_testA rc=0 wall_s=1315
2026-10-27T12:14:28Z w2c_replay_test_refcoco_unc_testB rc=0 wall_s=779
2026-10-27T12:27:12Z w2c_replay_test_refcoco+_unc_testA rc=0 wall_s=764
2026-10-27T12:40:03Z w2c_replay_test_refcoco+_unc_testB rc=0 wall_s=771
2026-10-27T12:53:35Z w2c_replay_test_refcocog_umd_test rc=0 wall_s=812
2026-10-27T12:53:35Z done; failed steps: 0
2026-10-27T12:53:35Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2-c replay test 0076t done (failed steps: 0)
```

- 心跳 `2026-10-27T12:53:35Z done 0076t (failed steps: 0)`。

| 划分 | cIoU [95% CI] | gIoU [95% CI] | n | 挑中最优 | 挑 1 号 | 都不对（none） | 解析失败 |
|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 0.6413 [0.6250, 0.6562] | 0.6506 [0.6387, 0.6616] | 1000 | 0.669（669/1000） | 0.088（88/1000） | 0.000（0/1000） | 0.000（0/1000） |
| test_refcoco_unc_testB | 0.6573 [0.6359, 0.6797] | 0.6808 [0.6647, 0.6966] | 1000 | 0.640（640/1000） | 0.081（81/1000） | 0.000（0/1000） | 0.000（0/1000） |
| test_refcoco+_unc_testA | 0.6173 [0.5986, 0.6357] | 0.6390 [0.6260, 0.6512] | 1000 | 0.639（639/1000） | 0.121（121/1000） | 0.000（0/1000） | 0.000（0/1000） |
| test_refcoco+_unc_testB | 0.6050 [0.5802, 0.6275] | 0.6261 [0.6070, 0.6449] | 1000 | 0.662（662/1000） | 0.128（128/1000） | 0.000（0/1000） | 0.000（0/1000） |
| test_refcocog_umd_test | 0.6413 [0.6207, 0.6611] | 0.6390 [0.6238, 0.6537] | 1000 | 0.664（664/1000） | 0.136（136/1000） | 0.000（0/1000） | 0.000（0/1000） |

- 推送循环 TAG 0076t（PID 557551），13:06:02Z 按 PID 停掉（当时在 sleep），随后 `ONCE=1` 推一轮：`2026-10-27T13:06:22Z ok`。

## T 文本重跑（补充 3）

- `$VI/tmp/start0076x.sh`（10-27 10:16:57Z 起）：wt0076x（稀疏）HEAD `e569342d167bbe2cbcab8b43af96be3c72a64287`，`ttext_replay.sh` sha256 `89f6d9ef278573abcacd9eedef4a0ae1ded4311bace954318479473f6e005434` 相同；CODE 用 wt0072，12/12 sha256 相同；检查点 manifest `ecad7667…`，`read_manifest` 通过。
- 10:18:33Z 启动：`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=4 bash …/ttext_replay.sh > $VI/logs/eval_0076x.driver.log`；`-c 8` 被接受；前后 squeue 两条 RUNNING；等补充 2 期间心跳 "waiting for eval_0076t.status to start with 'done 0076t' …"。
- 输出 `experiments/W2/0076_w2c_replay_ttext_hpcgpu06/`（label `t_text/w2c_replay/seed0`）。steps.log 原文：

```
2026-10-27T12:53:40Z start code d38a520525c6208a3f5ddb1062ca09a9646e85b6 node hpcgpu06 checkpoint manifest ecad7667b4588f4840ba13f79b71b8c52d475e26a99c79a0e6d3999c99dcecfd
2026-10-27T14:37:07Z ttext_w2c_replay rc=0 wall_s=6207
2026-10-27T14:37:07Z done; failed steps: 0
2026-10-27T14:37:08Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2-c replay T-text 0076x done (failed steps: 0)
```

- refexp_box：acc@0.5 0.9073 [0.9049, 0.9096]，n 56857；parse n_ok 56856、n_failed 1（{'malformed': 1}）；complete True
- count：accuracy 0.8032 [0.7670, 0.8394]，n 442；parse n_ok 442、n_failed 0（{}）；complete True
- merge_check.json：v1，pass True，failures {"merge": []}，error None
- 心跳 `2026-10-27T14:37:07Z done 0076x (failed steps: 0)`。推送循环 TAG 0076x（PID 1950005），15:14:02Z 按 PID 停掉（当时在 sleep），随后 `ONCE=1` 推一轮：`2026-10-27T15:14:10Z ok`。

## 显存与 traceback

- 显存峰值：B 段的 `gpu_mem.csv` 8050 行，最大 31459 MiB；补充 2、补充 3 的驱动不采样显存。
- 三个结果目录 `logs/` 下的步日志 grep `Traceback`、`out of memory`：合计 0。

## 收尾

- B 段推送循环 0076b（PID 3500395）12:00:23Z 一轮推送成功（已含 done）；12:19:08Z 按 PID 停掉，随后 `ONCE=1` 推一轮：`2026-10-27T12:19:17Z ok`。
- 15:14Z：squeue 157296、157297 都 RUNNING；06 无 GPU 进程、无暂停文件，空着。
