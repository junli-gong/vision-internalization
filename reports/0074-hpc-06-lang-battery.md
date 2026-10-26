# RUN-0074 报告：hpcgpu06，9B 的语言代价读数（大题量 lang_battery：基座 / W2-c / E3b+C3；E3b+C3 健康检查；基座可复现核对）

任务书 `prompts/server_hpc/0074-hpc-06-lang-battery.md`。只报告、不判定。没有碰 RUN-0072 与 07。只贴数字，不解读。

## 准备（hpcmgt 上脱离会话的 `$VI/tmp/prep0074.sh`，10-26 03:29:56Z – 03:30:26Z）

- 前 squeue：157296、157297 RUNNING。fetch 第 1 次 rc=0；wt0074（稀疏）HEAD `06557387cc6711b4cfa4e154a4308861b3e6633d`（含 76e71e0d），status 空，`experiments/` 只有 P0。
- 9 个 sha256 与任务书逐字相同（含 D127 之后的 `eval_lib.sh` `5af04263…`）；battery.sh 用本节点暂停文件。

### 数据下载（经 hf-mirror）

- 命令：`cd $VI/wt/wt0074 && env -u HF_HUB_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=$VI/hf $VI/env/bin/python scripts/lang_battery.py --fetch --raw-dir $VI/data/lang_battery_raw`。rc=0，03:30:12Z – 03:30:24Z，没有报错。
- 来源与文件（都与脚本的固定 sha256 相同）：

| 来源 | 文件 | sha256 |
|---|---|---|
| TIGER-Lab/MMLU-Pro | `TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet` | `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8` |
| google/IFEval | `google__IFEval/ifeval_input_data.jsonl` | `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2` |

- 另有 huggingface 下载留下的 `.cache/` 小文件（锁、元数据）。数据在 `$VI/data/lang_battery_raw`，不在仓库、不在结果目录。

### 两个检查点的核对

- W2-c（RUN-0069）：manifest `c605408f…93b7`，由驱动开头再核对一次（RUN-0072 开跑前恢复，见 RUN-0072 报告的"检查点恢复"）。
- E3b+C3（`$VI/ckpt/hpcgpu06/0056/e3b_c3/qwen3.5-9b/seed0/ckpt/final/model`）：
  - `manifest.json` sha256 `02f416c93689a72da0f7c941fbfaf3d2320e6afe4448e38248710630fc9de3f6`，与 RUN-0056 `ckpt_ledger.tsv` 的 e3b_c3_s0 一行相同。
  - 4 个文件的 sha256 与字节数都与 manifest 一致，字节数与训练叶目录 `train_summary.json` 的 `final_checkpoint_files` 一致：adapter_config 1,291；adapter_model 346,302,176；token_rows 1,097,920；tokens.json 16,268。没有多出的文件。
  - 作为 `E3B_MANIFEST` 传给驱动。

## 启动

- `$VI/tmp/start0074.sh`：03:32:07Z，`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… EV_CAP_H=8 E3B_MANIFEST=02f416c9… bash …/battery.sh`。`-c 8` 被接受。前后 squeue 两条 RUNNING。
- 等 RUN-0072 期间心跳 "waiting for gpu (running: eval_0072.status …)"；13:18:40Z 开始第一步，13:19:11Z 起推送循环 TAG 0074（PID 2827678）。

## steps.log 原文

```
2026-10-26T13:18:40Z start commit 06557387cc6711b4cfa4e154a4308861b3e6633d node hpcgpu06 model qwen3.5-9b
2026-10-26T14:51:42Z lb_base rc=0 wall_s=5582
2026-10-26T16:13:24Z lb_w2c rc=0 wall_s=4902
2026-10-26T18:19:15Z lb_e3b_c3 rc=0 wall_s=7551
2026-10-26T18:19:38Z lb_compare_w2c rc=0 wall_s=23
2026-10-26T18:19:38Z lb_compare_e3b_c3 rc=0 wall_s=0
2026-10-26T18:43:05Z health_base rc=0 wall_s=1407
2026-10-26T19:04:30Z health_e3b_c3 rc=0 wall_s=1285
2026-10-26T19:04:32Z health_compare_e3b_c3 rc=0 wall_s=2
2026-10-26T19:04:34Z health_compare_base_0069 rc=0 wall_s=2
2026-10-26T19:04:34Z done; failed steps: 0
2026-10-26T19:04:34Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）lang battery 0074 done (failed steps: 0)
```

- 心跳 `2026-10-26T19:04:34Z done 0074 (failed steps: 0)`。从第一步起的墙钟 13:18:40Z – 19:04:34Z（20,754 s，5.77 h），上限 8 h。
- `lb_compare_self` 没有出现在 steps.log 里：battery.sh 只在 TINY 分支里跑它（彩排的自比），服务器上不跑，管理员确认。

## lang_battery 读数（各 `lang_battery.json` 的 scores；95% CI 为按题 bootstrap）

| 模型 | MMLU-Pro 字母作答 acc（n 2,000） | mean_p_gold | IFEval prompt strict（n 541） | instr strict（n 834） | prompt loose | instr loose | 合并核对 |
|---|---|---|---|---|---|---|---|
| base | 54.2 [52.1, 56.3] | 49.4 [47.7, 51.0] | 83.5 [80.4, 86.5] | 88.2 [85.9, 90.3] | 87.6 [84.7, 90.4] | 91.2 [89.1, 93.0] | — |
| w2c | 45.2 [42.9, 47.2] | 38.2 [36.6, 39.6] | 78.9 [75.4, 82.3] | 85.0 [82.5, 87.3] | 81.9 [78.2, 84.8] | 87.3 [84.8, 89.4] | v1 pass True，failures {"merge": []} |
| e3b_c3 | 42.0 [39.8, 44.1] | 36.3 [34.8, 37.8] | 59.3 [55.1, 63.6] | 68.6 [65.3, 71.6] | 61.4 [57.3, 65.4] | 70.9 [67.7, 73.9] | v1 pass True，failures {"merge": []} |

- 三个 `lang_battery.json` 的 preflight：各 2,541 题核对，失败 0。

## lb `--compare` 原文

### 基座 对 W2-c（`logs/lb_compare_w2c.log`）

```
base=base  other=w2c  (other - base, percentage points; partial_run=False)
mmlu_pro_letter  accuracy                  base=54.2 other=45.2 delta=-9.0 pp CI95=[-10.8, -7.1]  sign test 105+/285- p=2.59e-20  (lost 285, gained 105, n=2000)
mmlu_pro_letter  mean_p_gold               base=49.4 other=38.2 delta=-11.3 pp CI95=[-12.4, -10.2]  (n=2000)
ifeval           prompt_level_strict       base=83.5 other=78.9 delta=-4.6 pp CI95=[-8.0, -1.7]  sign test 29+/54- p=0.00804  (lost 54, gained 29, n=541)
ifeval           instruction_level_strict  base=88.2 other=85.0 delta=-3.2 pp CI95=[-5.7, -0.9]  sign test 33+/60- p=0.0067  (lost 64, gained 37, n=834)
ifeval           prompt_level_loose        base=87.6 other=81.9 delta=-5.7 pp CI95=[-9.1, -2.6]  sign test 22+/53- p=0.00045  (lost 53, gained 22, n=541)
ifeval           instruction_level_loose   base=91.2 other=87.3 delta=-4.0 pp CI95=[-6.2, -1.7]  sign test 25+/56- p=0.000752  (lost 61, gained 28, n=834)
biology          accuracy                  base=82.5 other=75.5 delta=-7.0 pp CI95=[-12.6, -1.4]  sign test 4+/14- p=0.0309  (lost 14, gained 4, n=143)
business         accuracy                  base=36.4 other=22.4 delta=-14.0 pp CI95=[-21.7, -7.0]  sign test 6+/26- p=0.000535  (lost 26, gained 6, n=143)
chemistry        accuracy                  base=36.1 other=30.6 delta=-5.6 pp CI95=[-13.9, +2.1]  sign test 11+/19- p=0.2  (lost 19, gained 11, n=144)
computer science accuracy                  base=52.1 other=39.4 delta=-12.7 pp CI95=[-21.8, -4.2]  sign test 13+/31- p=0.00956  (lost 31, gained 13, n=142)
economics        accuracy                  base=69.9 other=60.1 delta=-9.8 pp CI95=[-15.4, -4.2]  sign test 3+/17- p=0.00258  (lost 17, gained 3, n=143)
engineering      accuracy                  base=38.0 other=25.4 delta=-12.7 pp CI95=[-20.4, -4.9]  sign test 7+/25- p=0.0021  (lost 25, gained 7, n=142)
health           accuracy                  base=64.8 other=59.2 delta=-5.6 pp CI95=[-12.0, +0.0]  sign test 7+/15- p=0.134  (lost 15, gained 7, n=142)
history          accuracy                  base=68.3 other=54.9 delta=-13.4 pp CI95=[-20.4, -6.3]  sign test 5+/24- p=0.000546  (lost 24, gained 5, n=142)
law              accuracy                  base=42.4 other=32.6 delta=-9.7 pp CI95=[-18.1, -1.4]  sign test 12+/26- p=0.0336  (lost 26, gained 12, n=144)
math             accuracy                  base=34.0 other=26.4 delta=-7.6 pp CI95=[-14.6, -0.7]  sign test 8+/19- p=0.0522  (lost 19, gained 8, n=144)
other            accuracy                  base=50.7 other=43.7 delta=-7.0 pp CI95=[-14.8, +0.0]  sign test 10+/20- p=0.0987  (lost 20, gained 10, n=142)
philosophy       accuracy                  base=62.2 other=58.7 delta=-3.5 pp CI95=[-9.8, +2.8]  sign test 8+/13- p=0.383  (lost 13, gained 8, n=143)
physics          accuracy                  base=44.1 other=34.3 delta=-9.8 pp CI95=[-18.2, -2.1]  sign test 10+/24- p=0.0243  (lost 24, gained 10, n=143)
psychology       accuracy                  base=76.9 other=69.2 delta=-7.7 pp CI95=[-13.3, -2.8]  sign test 1+/12- p=0.00342  (lost 12, gained 1, n=143)
```

### 基座 对 E3b+C3（`logs/lb_compare_e3b_c3.log`）

```
base=base  other=e3b_c3  (other - base, percentage points; partial_run=False)
mmlu_pro_letter  accuracy                  base=54.2 other=42.0 delta=-12.2 pp CI95=[-14.3, -10.2]  sign test 124+/368- p=4.15e-29  (lost 368, gained 124, n=2000)
mmlu_pro_letter  mean_p_gold               base=49.4 other=36.3 delta=-13.2 pp CI95=[-14.3, -12.0]  (n=2000)
ifeval           prompt_level_strict       base=83.5 other=59.3 delta=-24.2 pp CI95=[-28.5, -19.8]  sign test 26+/157- p=4.99e-24  (lost 157, gained 26, n=541)
ifeval           instruction_level_strict  base=88.2 other=68.6 delta=-19.7 pp CI95=[-23.2, -16.1]  sign test 31+/170- p=2.02e-24  (lost 205, gained 41, n=834)
ifeval           prompt_level_loose        base=87.6 other=61.4 delta=-26.2 pp CI95=[-30.7, -22.0]  sign test 20+/162- p=8.22e-29  (lost 162, gained 20, n=541)
ifeval           instruction_level_loose   base=91.2 other=70.9 delta=-20.4 pp CI95=[-23.7, -17.2]  sign test 23+/170- p=6.71e-29  (lost 200, gained 30, n=834)
biology          accuracy                  base=82.5 other=72.7 delta=-9.8 pp CI95=[-16.1, -3.5]  sign test 5+/19- p=0.00661  (lost 19, gained 5, n=143)
business         accuracy                  base=36.4 other=22.4 delta=-14.0 pp CI95=[-21.7, -6.3]  sign test 7+/27- p=0.000821  (lost 27, gained 7, n=143)
chemistry        accuracy                  base=36.1 other=27.1 delta=-9.0 pp CI95=[-16.7, -1.4]  sign test 10+/23- p=0.0351  (lost 23, gained 10, n=144)
computer science accuracy                  base=52.1 other=39.4 delta=-12.7 pp CI95=[-21.1, -5.6]  sign test 10+/28- p=0.0051  (lost 28, gained 10, n=142)
economics        accuracy                  base=69.9 other=59.4 delta=-10.5 pp CI95=[-16.8, -4.2]  sign test 6+/21- p=0.00592  (lost 21, gained 6, n=143)
engineering      accuracy                  base=38.0 other=25.4 delta=-12.7 pp CI95=[-21.8, -4.2]  sign test 13+/31- p=0.00956  (lost 31, gained 13, n=142)
health           accuracy                  base=64.8 other=51.4 delta=-13.4 pp CI95=[-20.4, -6.3]  sign test 4+/23- p=0.000311  (lost 23, gained 4, n=142)
history          accuracy                  base=68.3 other=47.2 delta=-21.1 pp CI95=[-29.6, -12.7]  sign test 8+/38- p=9.25e-06  (lost 38, gained 8, n=142)
law              accuracy                  base=42.4 other=31.9 delta=-10.4 pp CI95=[-19.4, -1.4]  sign test 14+/29- p=0.0315  (lost 29, gained 14, n=144)
math             accuracy                  base=34.0 other=26.4 delta=-7.6 pp CI95=[-16.7, +0.7]  sign test 16+/27- p=0.126  (lost 27, gained 16, n=144)
other            accuracy                  base=50.7 other=40.8 delta=-9.9 pp CI95=[-19.7, -0.7]  sign test 15+/29- p=0.0488  (lost 29, gained 15, n=142)
philosophy       accuracy                  base=62.2 other=46.9 delta=-15.4 pp CI95=[-23.1, -9.1]  sign test 4+/26- p=5.95e-05  (lost 26, gained 4, n=143)
physics          accuracy                  base=44.1 other=32.2 delta=-11.9 pp CI95=[-19.6, -4.2]  sign test 8+/25- p=0.00455  (lost 25, gained 8, n=143)
psychology       accuracy                  base=76.9 other=64.3 delta=-12.6 pp CI95=[-19.6, -6.3]  sign test 4+/22- p=0.000534  (lost 22, gained 4, n=143)
```

## 健康检查

- health_base：headline mmlu_pro 53.0、gsm8k 86.0、ifeval 94.0。health_e3b_c3：45.0、86.0、74.0；合并核对 v1，pass True，failures merge []。

### 基座 对 E3b+C3（`logs/health_compare_e3b_c3.log`）

```
protocol: base v3, merged v3
mmlu_pro_letter accuracy                  base=53.0 merged=45.0 delta=-8.0 pp  CI95=[-17.0, +1.0]  sign test 7+/15- p=0.134  (lost 15, gained 7, n=100)  <-- 标红
mmlu_pro_letter mean_p_gold               base=49.0 merged=40.8 delta=-8.3 pp  CI95=[-13.7, -3.3]  (n=100)  [诊断，不标红]
mmlu_pro_free256 accuracy                  base=53.0 merged=45.0 delta=-8.0 pp  CI95=[-17.0, +1.0]  sign test 7+/15- p=0.134  (lost 15, gained 7, n=100)  [诊断，不标红]
mmlu_pro_fmt32  accuracy                  base=53.0 merged=45.0 delta=-8.0 pp  CI95=[-17.0, +1.0]  sign test 7+/15- p=0.134  (lost 15, gained 7, n=100)  [诊断，不标红]
gsm8k           flexible                  base=86.0 merged=86.0 delta=+0.0 pp  CI95=[-10.0, +10.0]  sign test 3+/3- p=1  (lost 3, gained 3, n=50)
gsm8k           strict                    base=86.0 merged=86.0 delta=+0.0 pp  CI95=[-10.0, +10.0]  sign test 3+/3- p=1  (lost 3, gained 3, n=50)
ifeval          prompt_level_strict       base=94.0 merged=74.0 delta=-20.0 pp  CI95=[-32.0, -10.0]  sign test 0+/10- p=0.00195  (lost 10, gained 0, n=50)  <-- 标红
ifeval          instruction_level_strict  base=96.1 merged=77.9 delta=-18.2 pp  CI95=[-26.3, -9.9]  sign test 0+/12- p=0.000488  (lost 14, gained 0, n=77)
ifeval          prompt_level_loose        base=98.0 merged=74.0 delta=-24.0 pp  CI95=[-36.0, -12.0]  sign test 0+/12- p=0.000488  (lost 12, gained 0, n=50)
ifeval          instruction_level_loose   base=98.7 merged=79.2 delta=-19.5 pp  CI95=[-27.6, -11.5]  sign test 0+/13- p=0.000244  (lost 15, gained 0, n=77)
truncation mmlu_pro_free256 max_new_tokens=256   base=0.0% merged=0.0% delta=+0.0 pp (n=100)
truncation mmlu_pro_fmt32  max_new_tokens=32    base=0.0% merged=1.0% delta=+1.0 pp (n=100)
truncation gsm8k           max_new_tokens=512   base=12.0% merged=2.0% delta=-10.0 pp (n=50)
truncation ifeval          max_new_tokens=1024  base=2.0% merged=16.0% delta=+14.0 pp (n=50)  <-- 截断率提示
any_flag=True  flagged=['mmlu_pro_letter', 'ifeval']  truncation_notices=['ifeval']  incomparable=[]  unscored=[]  partial_run=False
```

### 基座可复现核对：本次基座 对 RUN-0069 的 `health/health_base.json`（`logs/health_compare_base_0069.log`）

```
protocol: base v3, merged v3
mmlu_pro_letter accuracy                  base=53.0 merged=53.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=100)
mmlu_pro_letter mean_p_gold               base=49.0 merged=49.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  (n=100)  [诊断，不标红]
mmlu_pro_free256 accuracy                  base=53.0 merged=53.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=100)  [诊断，不标红]
mmlu_pro_fmt32  accuracy                  base=53.0 merged=53.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=100)  [诊断，不标红]
gsm8k           flexible                  base=86.0 merged=86.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=50)
gsm8k           strict                    base=86.0 merged=86.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=50)
ifeval          prompt_level_strict       base=94.0 merged=94.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=50)
ifeval          instruction_level_strict  base=96.1 merged=96.1 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=77)
ifeval          prompt_level_loose        base=98.0 merged=98.0 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=50)
ifeval          instruction_level_loose   base=98.7 merged=98.7 delta=+0.0 pp  CI95=[+0.0, +0.0]  sign test 0+/0- p=1  (lost 0, gained 0, n=77)
truncation mmlu_pro_free256 max_new_tokens=256   base=0.0% merged=0.0% delta=+0.0 pp (n=100)
truncation mmlu_pro_fmt32  max_new_tokens=32    base=0.0% merged=0.0% delta=+0.0 pp (n=100)
truncation gsm8k           max_new_tokens=512   base=12.0% merged=12.0% delta=+0.0 pp (n=50)
truncation ifeval          max_new_tokens=1024  base=2.0% merged=2.0% delta=+0.0 pp (n=50)
any_flag=False  flagged=[]  truncation_notices=[]  incomparable=[]  unscored=[]  partial_run=False
```

- 各读数差值都是 0（CI 也是 [0, 0]），any_flag=False。

## 显存与 traceback

- 显存峰值：battery.sh 不采样显存，结果目录里没有 gpu_mem.csv，日志里也没有。
- `logs/` 下 9 个步日志 grep `Traceback`、`out of memory`：都是 0。
- `ExternalCallForbidden` 之类的外部调用：本任务不涉及外部 API。

## 收尾

- 推送循环 18:51:58Z 一轮推送成功。19:08:40Z 在 git.lock 下按 PID 停掉循环 2827678（当时在 sleep），随后 `ONCE=1` 推一轮：`2026-10-26T19:09:03Z ok`。
- 后 squeue：157296、157297 RUNNING。06 接着跑 RUN-0072 补充 1（19:04:45Z 起 testA 的 w2c 补跑）。
