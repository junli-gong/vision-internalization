# RUN-0078 报告：hpcgpu06，外部基线 LLMSeg（官方代码 + 本地 Qwen3.5-9B 零样本），W2 的 test 子集 5 × 300 句

任务书 `prompts/server_hpc/0078-hpc-06-llmseg.md`；D138（停止线改为外推超过 18 小时）、D139（compare 由服务器 CC 手动补跑一次）。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0078.sh`（10-26 05:44:45Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0078（稀疏，含 `third_party/`）HEAD `5c79a423c67db54e529a73ce54cbd912fce3ff7d`，status 空，`experiments/` 只有 P0；`third_party/llmseg`：evals、geometry.py、LICENSE、llm.py、pipeline.py、prompts.py、PROVENANCE.md、tests。
- 12 个 sha256 与任务书逐字相同；驱动用本节点暂停文件；hpcgpu06 上 srun 环境里没有 API key 变量（API_KEY、OPENAI、ANTHROPIC、GEMINI、GOOGLE_API、OPENROUTER），没有设任何 key。
- 05:44:59Z 按任务书命令启动（`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=12`），`-c 8` 被接受；心跳 "waiting for eval_0072r.status to be done"；10-26 19:18:29Z RUN-0072 补跑完成后开始（steps.log 的 start 行带 subsets `dfd53ce8…`、provenance `6be8e738…`）。
- 推送循环 TAG 0078（PID 4011425，1800 s），`work/` 由 .gitignore 排除。

## steps.log 原文

```
2026-10-26T19:18:29Z start commit 5c79a423c67db54e529a73ce54cbd912fce3ff7d node hpcgpu06 model qwen3.5-9b subsets dfd53ce8e9a89d85ba828de6c0b3e331f83bfcf7d476cfc7ed1751e64c7497ae provenance 6be8e7382736e8f0a31ad7fdd1fe0ac7105e07d647169c7c237d301ed266240d
2026-10-27T09:24:23Z llmseg rc=0 wall_s=50754
2026-10-27T09:24:24Z STOP before compare: cap 43200s reached
2026-10-27T09:24:29Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）stopped before compare: cap 43200s reached
2026-10-27T09:27:23Z compare rc=0 wall_s=28 （D139：驱动在 EV_CAP 处停在 compare 之前，compare 由服务器 CC 按驱动的命令手动补跑一次，2026-10-27T09:26:55Z 起，07 上 srun -c 1、只用 CPU）
```

- llmseg 一步 rc=0，墙钟 50,754 s（14.10 h）。
- **cap 停止原文**：`2026-10-27T09:24:24Z STOP before compare: cap 43200s reached`；心跳 `2026-10-27T09:24:24Z blocked: cap 43200s reached`；驱动日志 `srun: error: hpcgpu06: task 0: Exited with exit code 3`。eval_lib 在每条命令之前查 `EV_CAP_S`（12 h），llmseg 一步开始时在线内，compare 一步开始时已超出。
- D138 只改了停止线（外推 > 18 h 才停），没有改 `EV_CAP_H`；管理员回复后（D139）compare 由服务器 CC 按驱动的命令手动补跑一次，见下。

## 第一组 150 句之后的吞吐与显存（D138 之前的报告时点）

- 第一个划分的 `logs/gpu_summary.json` 写在划分的叶目录下（不是运行根目录的 `logs/`），监视脚本按根目录找，所以这一时点的报告推迟到 testA、testB 都跑完之后（10-27 01:39Z）才发。
- testA 外推 12.22 h（超过 12 h）；管理员回复 D138：不停，停止线改为外推超过 18 h，或 OOM、traceback、ExternalCallForbidden。之后各划分的外推都在 18 h 以下（见下表）。

## 各划分的读数与计算记账（由 `experiments/W2/0078_llmseg_probe/p78_tables.py` 读出）

### 读数（n 300 / 划分；rec = 修正后，标题；seg = 修正前）

| 划分 | seg cIoU | seg gIoU | rec cIoU | rec gIoU | ok / failed | complete |
|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 0.3871 [0.3602, 0.4149] | 0.3817 [0.3580, 0.4049] | 0.2463 [0.2143, 0.2774] | 0.2354 [0.2106, 0.2603] | 290 / 10 | True |
| test_refcoco_unc_testB | 0.4528 [0.4176, 0.4871] | 0.4489 [0.4202, 0.4757] | 0.3201 [0.2804, 0.3620] | 0.2940 [0.2649, 0.3240] | 294 / 6 | True |
| test_refcoco+_unc_testA | 0.3664 [0.3341, 0.4006] | 0.3520 [0.3284, 0.3758] | 0.2340 [0.2021, 0.2712] | 0.2173 [0.1940, 0.2400] | 289 / 11 | True |
| test_refcoco+_unc_testB | 0.3926 [0.3566, 0.4283] | 0.3992 [0.3693, 0.4264] | 0.2942 [0.2559, 0.3318] | 0.2696 [0.2415, 0.2969] | 294 / 6 | True |
| test_refcocog_umd_test | 0.3954 [0.3585, 0.4304] | 0.3755 [0.3483, 0.4034] | 0.2745 [0.2304, 0.3149] | 0.2317 [0.2043, 0.2575] | 284 / 16 | True |

### 计算记账

| 划分 | 每图调用 均值 / p50 / max | 各阶段调用（detect / segment / focus_boxes / focus_splines / box_adjust） | 每图 prompt / 生成 token 均值 | 截断（全部） | 格式失败（全部） | 坐标 > 1000（值 / 调用） | 未解析调用 | max_abs |
|---|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 13.93 / 16 / 18 | 300 / 286 / 1838 / 1754 / 0 | 19783 / 3531 | 72/4178 = 0.0172 | 75/4178 = 0.0180 | 319/105704 / 3 | 1 | 7450 |
| test_refcoco_unc_testB | 13.98 / 16 / 18 | 300 / 293 / 1858 / 1744 / 0 | 20083 / 3750 | 64/4195 = 0.0153 | 78/4195 = 0.0186 | 115/110054 / 5 | 6 | 5350 |
| test_refcoco+_unc_testA | 14.00 / 16 / 18 | 300 / 288 / 1847 / 1765 / 0 | 20008 / 3747 | 82/4200 = 0.0195 | 89/4200 = 0.0212 | 270/109373 / 3 | 3 | 3600 |
| test_refcoco+_unc_testB | 13.47 / 14 / 18 | 300 / 291 / 1778 / 1671 / 0 | 19392 / 3545 | 52/4040 = 0.0129 | 81/4040 = 0.0200 | 226/97378 / 7 | 2 | 3250 |
| test_refcocog_umd_test | 13.52 / 16 / 18 | 300 / 286 / 1782 / 1688 / 0 | 19572 / 3637 | 78/4056 = 0.0192 | 93/4056 = 0.0229 | 112/103294 / 5 | 8 | 8070 |

### 各阶段的截断率与格式失败率

| 划分 | 截断 detect / segment / focus_boxes / focus_splines | 格式失败 detect / segment / focus_boxes / focus_splines |
|---|---|---|
| test_refcoco_unc_testA | 0.0000 / 0.0350 / 0.0076 / 0.0274 | 0.0067 / 0.0350 / 0.0076 / 0.0279 |
| test_refcoco_unc_testB | 0.0100 / 0.0239 / 0.0081 / 0.0224 | 0.0567 / 0.0239 / 0.0075 / 0.0229 |
| test_refcoco+_unc_testA | 0.0000 / 0.0382 / 0.0049 / 0.0351 | 0.0200 / 0.0451 / 0.0049 / 0.0346 |
| test_refcoco+_unc_testB | 0.0000 / 0.0275 / 0.0090 / 0.0168 | 0.0867 / 0.0378 / 0.0090 / 0.0168 |
| test_refcocog_umd_test | 0.0000 / 0.0559 / 0.0101 / 0.0261 | 0.0467 / 0.0559 / 0.0101 / 0.0267 |

### gpu_summary.json（每个划分）

| 划分 | 子批 | generate_wall_s | s / 条 | 外推 1,500 条 GPU 小时 | 阶段墙钟 detect / segment / focus_boxes / focus_splines（s） |
|---|---|---|---|---|---|
| test_refcoco_unc_testA | 295 | 8,798 | 29.33 | 12.22 | 166 / 1,484 / 2,490 / 4,658 |
| test_refcoco_unc_testB | 314 | 11,593 | 38.64 | 16.10 | 404 / 2,084 / 3,187 / 5,918 |
| test_refcoco+_unc_testA | 314 | 9,688 | 32.29 | 13.46 | 214 / 1,428 / 2,842 / 5,204 |
| test_refcoco+_unc_testB | 358 | 10,685 | 35.62 | 14.84 | 395 / 1,622 / 3,490 / 5,178 |
| test_refcocog_umd_test | 284 | 9,049 | 30.16 | 12.57 | 184 / 1,506 / 2,740 / 4,620 |

### 其他

- test_refcoco_unc_testA：pipeline_events {'detect malformed reply': 2, 'refine pass unparseable': 31, 'refine truncated reply': 32, 'segment error': 10}；refine converged 143/276，iterations p50 8；n_empty_reply 0；n_input_rejected 0；生成 token / 图 p50 3264、max 11819
- test_refcoco_unc_testB：pipeline_events {'detect malformed reply': 17, 'refine pass unparseable': 33, 'refine truncated reply': 21, 'segment error': 6, 'segment truncated reply': 1}；refine converged 167/287，iterations p50 7；n_empty_reply 0；n_input_rejected 0；生成 token / 图 p50 3206、max 49764
- test_refcoco+_unc_testA：pipeline_events {'detect malformed reply': 6, 'refine pass unparseable': 45, 'refine truncated reply': 25, 'segment error': 11, 'segment truncated reply': 2}；refine converged 152/277，iterations p50 8；n_empty_reply 0；n_input_rejected 0；生成 token / 图 p50 3466、max 18209
- test_refcoco+_unc_testB：pipeline_events {'detect malformed reply': 26, 'refine pass unparseable': 37, 'refine truncated reply': 7, 'segment error': 6, 'segment truncated reply': 5}；refine converged 172/285，iterations p50 7；n_empty_reply 0；n_input_rejected 0；生成 token / 图 p50 3072、max 23792
- test_refcocog_umd_test：pipeline_events {'detect malformed reply': 14, 'refine pass unparseable': 43, 'refine truncated reply': 20, 'segment error': 16}；refine converged 147/270，iterations p50 8；n_empty_reply 0；n_input_rejected 0；生成 token / 图 p50 3353、max 17914

## compare（D139，手动补跑）

- `$VI/tmp/compare0078.sh`：从 wt0078（12/12 sha256 核对，`compare_llmseg.py` `f697bb75…`），在 07 上 `srun --jobid=157297 --overlap -n1 -c 1`，env `PATH=$VI/env/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0 HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp`：
  `python scripts/compare_llmseg.py --llmseg $OUT/qwen3.5-9b/refseg/llmseg --w2c $REPO/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c --out $OUT/compare_w2c`
- 09:26:55Z 开始，rc=0，墙钟 28 s；脚本只用 numpy / yaml / PIL，不加载模型、不用 GPU。前后 squeue 两条 RUNNING。
- 追加到 steps.log 的一行见上面的 steps.log 原文（最后一行）。
- 输出：`compare_w2c/compare.json`（9,371 字节）、`compare_w2c/compare.md`（2,693 字节）。

### `compare_w2c/compare.md` 原文

# LLMSeg 与 W2-c 的配对比较

LLMSeg − W2-c 的配对差（a = LLMSeg，b = W2-c）；rec = LLMSeg 修正后（标题），seg = LLMSeg 修正前；W2-c 是它的标题读数

| 划分 | LLMSeg 读数 | 指标 | n | LLMSeg | W2-c | 差 | 95% CI |
|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | rec | cIoU | 300 | 0.2463 | 0.6645 | -0.4182 | [-0.4590, -0.3779] |
| test_refcoco_unc_testA | rec | gIoU | 300 | 0.2354 | 0.6633 | -0.4279 | [-0.4580, -0.3990] |
| test_refcoco_unc_testA | seg | cIoU | 300 | 0.3871 | 0.6645 | -0.2774 | [-0.3087, -0.2446] |
| test_refcoco_unc_testA | seg | gIoU | 300 | 0.3817 | 0.6633 | -0.2816 | [-0.3085, -0.2543] |
| test_refcoco_unc_testB | rec | cIoU | 300 | 0.3201 | 0.6918 | -0.3718 | [-0.4166, -0.3246] |
| test_refcoco_unc_testB | rec | gIoU | 300 | 0.2940 | 0.6986 | -0.4046 | [-0.4392, -0.3728] |
| test_refcoco_unc_testB | seg | cIoU | 300 | 0.4528 | 0.6918 | -0.2390 | [-0.2752, -0.2013] |
| test_refcoco_unc_testB | seg | gIoU | 300 | 0.4489 | 0.6986 | -0.2497 | [-0.2786, -0.2200] |
| test_refcoco+_unc_testA | rec | cIoU | 300 | 0.2340 | 0.6126 | -0.3786 | [-0.4196, -0.3345] |
| test_refcoco+_unc_testA | rec | gIoU | 300 | 0.2173 | 0.6326 | -0.4153 | [-0.4468, -0.3852] |
| test_refcoco+_unc_testA | seg | cIoU | 300 | 0.3664 | 0.6126 | -0.2462 | [-0.2872, -0.2031] |
| test_refcoco+_unc_testA | seg | gIoU | 300 | 0.3520 | 0.6326 | -0.2806 | [-0.3077, -0.2524] |
| test_refcoco+_unc_testB | rec | cIoU | 300 | 0.2942 | 0.6071 | -0.3129 | [-0.3633, -0.2657] |
| test_refcoco+_unc_testB | rec | gIoU | 300 | 0.2696 | 0.6323 | -0.3628 | [-0.3985, -0.3270] |
| test_refcoco+_unc_testB | seg | cIoU | 300 | 0.3926 | 0.6071 | -0.2145 | [-0.2586, -0.1749] |
| test_refcoco+_unc_testB | seg | gIoU | 300 | 0.3992 | 0.6323 | -0.2332 | [-0.2679, -0.2000] |
| test_refcocog_umd_test | rec | cIoU | 300 | 0.2745 | 0.6483 | -0.3739 | [-0.4181, -0.3302] |
| test_refcocog_umd_test | rec | gIoU | 300 | 0.2317 | 0.6364 | -0.4047 | [-0.4366, -0.3757] |
| test_refcocog_umd_test | seg | cIoU | 300 | 0.3954 | 0.6483 | -0.2529 | [-0.2856, -0.2179] |
| test_refcocog_umd_test | seg | gIoU | 300 | 0.3755 | 0.6364 | -0.2609 | [-0.2878, -0.2324] |
| 合并 | rec | cIoU | 1500 | 0.2749 | 0.6436 | -0.3687 | [-0.3895, -0.3472] |
| 合并 | rec | gIoU | 1500 | 0.2496 | 0.6526 | -0.4030 | [-0.4176, -0.3882] |
| 合并 | seg | cIoU | 1500 | 0.3983 | 0.6436 | -0.2453 | [-0.2626, -0.2277] |
| 合并 | seg | gIoU | 1500 | 0.3914 | 0.6526 | -0.2612 | [-0.2741, -0.2479] |

- {'rule': 'D125-4：LLMSeg（9B）不低于 W2-c 就立即报告；这里给点估计与 CI 两种口径，不作判定', 'point_estimate_ge_0': False, 'ci_high_ge_0': False}

- `compare.json` 里的 `flag_llmseg_not_lower`（单一一项，不分 seg / rec）：`{"rule": "D125-4：LLMSeg（9B）不低于 W2-c 就立即报告；这里给点估计与 CI 两种口径，不作判定", "point_estimate_ge_0": false, "ci_high_ge_0": false}`。

## 显存、traceback 与外部调用

- 显存峰值：`gpu_mem.csv` 50746 行，最大 29569 MiB。
- `logs/llmseg.log` grep `Traceback`、`out of memory`：0；grep `ExternalCallForbidden`：0。全程没有任何外部 API 调用的记录。

## 收尾

- 推送循环 09:01:01Z 一轮推送成功；09:27:23Z 在 git.lock 下按 PID 停掉循环 4011425（当时在 sleep），随后 `ONCE=1` 推一轮（含 compare）：`2026-10-27T09:28:16Z ok`。
- 后 squeue：157296、157297 RUNNING。06 接着跑 RUN-0076 B 段（09:24:46Z 起）。
