# RUN-0114 报告：**hpcgpu06**，X-M 的定位与计数 T 文本读数，以及大题量 --compare（D182 第 3 点；补充 1 / D183）

任务书 `prompts/server_hpc/0114-hpc-06-xm-ttext.md`（含补充 1）。**运行节点：hpcgpu06**（`--jobid=157296`），与 RUN-0069 在 06 上的 T 文本基座配对；06 上不跑 ttext_base。只贴数字，不解读。

## 启动

- 11-05 21:34:10Z 首个启动器 `$VI/tmp/run0114.sh`：wt0114（稀疏）@ `e4420e058`，4/4 sha256 相同；X-M 检查点 manifest `29e26667…`；`lb_base/lang_battery.json`（RUN-0084）与 `0107_lb_xm/lang_battery.json` 都在。21:34:57Z 06 被占（PID 1059981，35,190 MiB，VLLM::EngineCore），未启动；约 21:36Z 按补充 1 按 PID 停掉。
- 21:37:40Z `$VI/tmp/run0114b.sh`：wt0114 移到 `128eba5d2e0da9fd04a16cd32cac36ab63cb5cd3`，status 空，4/4 sha256 相同（驱动 `8456a9d3…`）。一次判定时 06、07 都被占（07：PID 398986，35,188 MiB，VLLM::EngineCore），未启动，报告管理员。
- 管理员选 (a)：21:40:14Z `$VI/tmp/run0114c.sh`（sha256 `d46628a7…`），每分钟查 06、07，谁先没有计算进程用谁，只定一次；ssh 失败不算空。前 squeue：157296 RUNNING 82-09:06:00 hpcgpu06；157297 RUNNING 84-06:38:09 hpcgpu07。
- **11-06 02:05:36Z 06 没有计算进程 → NODE=hpcgpu06**（07 仍被占）。按任务书命令加 `NODE=hpcgpu06` 启动：`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 NODE=hpcgpu06 EV_CAP_H=6 bash …/xm_ttext.sh`。后 squeue（约 02:06Z）：157296 RUNNING 82-13:32:23 hpcgpu06；157297 RUNNING 84-11:04:32 hpcgpu07。
- 推送循环 TAG 0114（PID 2377300，02:06:37Z 起，路径 `experiments/XM/0114_xm_ttext_hpcgpu06`）。

## steps.log 原文

```
2026-11-06T02:05:37Z start commit 128eba5d2e0da9fd04a16cd32cac36ab63cb5cd3 node hpcgpu06
2026-11-06T02:05:40Z xm checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e
2026-11-06T04:26:29Z ttext_xm rc=0 wall_s=8449
2026-11-06T04:27:04Z lb_compare_xm rc=0 wall_s=35
2026-11-06T04:27:04Z done; failed steps: 0
2026-11-06T04:27:05Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M T-text 0114 done (failed steps: 0)
```

- 心跳 `2026-11-06T04:27:04Z done 0114 (failed steps: 0)`。`logs/` 下日志无 Traceback / OutOfMemory。用时合计 8,484 s（2.36 h），上限 EV_CAP_H=6。

## 定位与计数（`t_text/xm/seed0/qwen3.5-9b/<任务>/metrics.json` 的 `headline`；95% CI，1,000 次重抽）

| 标签 | refexp_box acc@0.5（n = 56,857） | 解析失败 | count accuracy（n = 442） | 解析失败 | 截断 | complete |
|---|---|---|---|---|---|---|
| `t_text/xm/seed0` | 0.9068 [0.9044, 0.9093] | 1 / 56,857（malformed） | 0.6878 [0.6425, 0.7285] | 0 | 0 / 0 | True / True |

- refexp_box `valid`：n_valid 56,856，n_refusal 0。count `valid`：n_valid 442，n_refusal 0。
- 计数题出自 CountBenchQA（PaliGemma，big_vision 仓库；D36-2 注明）。
- 配对参照（RUN-0069 在 06 上的 `0069_ttext/` 基座，同一 eval 子集；照录 RUN-0091 报告里的数字）：refexp_box acc@0.5 0.8714 [0.8687, 0.8740]，count accuracy 0.8937 [0.8597, 0.9231]。

## 合并核对（`t_text/xm/seed0/qwen3.5-9b/merge_check.json`）

- pass True，failures merge []（commit `128eba5d2`，dirty false；`--n 200`，`--data-root $VI/b200/data`）。

## 大题量 `lb_compare_xm.json`（base = RUN-0084 `lb_base`，other = RUN-0107 `lb_xm`；other − base，百分点；配对 bootstrap 1,000 次，95%）

- identity：battery_version 1，mmlu_pro_n 2000，partial_run false；other 的检查点 manifest `29e26667…`。

| reading | metric | headline | n | base % | X-M % | delta pp | CI95 pp | lost / gained | 符号检验 p |
|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro_letter | accuracy | True | 2000 | 54.2 | 52.9 | -1.3 | [-2.7, +0.1] | 127 / 101 | 0.098 |
| mmlu_pro_letter | mean_p_gold | False | 2000 | 49.4 | 47.3 | -2.1 | [-2.7, -1.5] | | |
| ifeval | prompt_level_strict | True | 541 | 83.7 | 80.6 | -3.1 | [-6.1, -0.4] | 39 / 22 | 0.040 |
| ifeval | instruction_level_strict | False | 834 | 88.4 | 86.2 | -2.2 | [-4.3, +0.0] | 48 / 30 | 0.032 |
| ifeval | prompt_level_loose | False | 541 | 87.6 | 83.0 | -4.6 | [-7.2, -2.2] | 39 / 14 | 0.0008 |
| ifeval | instruction_level_loose | False | 834 | 91.2 | 88.1 | -3.1 | [-5.1, -1.3] | 46 / 20 | 0.0015 |

## 收尾

- 推送循环 0114：04:27:43Z 在睡眠中按 PID 停（2377300，sleep 3208169），ONCE 推送 04:27:59Z ok。
- 后 squeue（04:27:59Z）：157296 RUNNING 82-15:53:45 hpcgpu06；157297 RUNNING 84-13:25:54 hpcgpu07。
