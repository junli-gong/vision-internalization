# RUN-0071 报告：hpcgpu07，N37(b) E3 结构强制网格评测按 e5-v2 判据补评（2B E3b 种子 1、4B E3b 种子 2，连同机内配对的 E3b+C3）

任务书 `prompts/server_hpc/0071-hpc-07-e3-reeval-e5v2.md`。没有碰 06（RUN-0069）。只贴数字，不解读；**网格的标题读数不进本报告**，在结果目录 `experiments/E3/0071_e3_reeval_e5v2_hpcgpu07/eval/` 下。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0071.sh`（10-24 19:06:23Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0071（稀疏）HEAD `470229938eb09314f89a4c0bc540784ad935b3b5`，status 空，`experiments/` 只有 P0。
- 8 个 sha256 与任务书逐字相同；reeval.sh 用本节点暂停文件；07 无暂停文件、GPU 空。
- 19:07:34Z 按任务书命令启动（`srun --jobid=157297 --overlap -n1`，没有 `-c`；`EV_CAP_H=14`）。
- reeval.sh 开头按台账核对 4 个检查点 `manifest.json` 的 sha256，都对上（steps.log 没有 ABORT，接着写了 start 行）：
  - 2B E3b s1 `7c3d2938…`、2B E3b+C3 s1 `7c72a500…`、4B E3b s2 `dca97e75…`、4B E3b+C3 s2 `1b9fc106…`。
- 推送循环 TAG 0071（PID 2967405，1800 s）。启动后 squeue 两条 RUNNING。

## steps.log 原文

```
2026-10-24T19:07:36Z start commit 470229938eb09314f89a4c0bc540784ad935b3b5 node hpcgpu07
2026-10-24T20:56:16Z grid_e3b_qwen3.5-2b_s1_refseg rc=0 wall_s=6519
2026-10-24T21:03:26Z grid_e3b_qwen3.5-2b_s1_depth rc=0 wall_s=430
2026-10-24T22:43:16Z grid_e3b_c3_qwen3.5-2b_s1_refseg rc=0 wall_s=5990
2026-10-24T22:49:27Z grid_e3b_c3_qwen3.5-2b_s1_depth rc=0 wall_s=371
2026-10-25T01:50:06Z grid_e3b_qwen3.5-4b_s2_refseg rc=0 wall_s=10839
2026-10-25T01:58:21Z grid_e3b_qwen3.5-4b_s2_depth rc=0 wall_s=495
2026-10-25T04:58:28Z grid_e3b_c3_qwen3.5-4b_s2_refseg rc=0 wall_s=10807
2026-10-25T05:06:46Z grid_e3b_c3_qwen3.5-4b_s2_depth rc=0 wall_s=498
2026-10-25T05:06:46Z done; failed steps: 0
2026-10-25T05:06:46Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）N37(b) E3 re-eval e5-v2 done (failed steps: 0)
```

- 心跳 `2026-10-25T05:06:46Z done 0071 (failed steps: 0)`。总墙钟 10-24 19:07:36Z – 10-25 05:06:46Z（35,950 s，9.99 h），上限 14 h。
- 每步墙钟：

| 步 | wall_s |
|---|---:|
| grid_e3b_qwen3.5-2b_s1_refseg | 6,519 |
| grid_e3b_qwen3.5-2b_s1_depth | 430 |
| grid_e3b_c3_qwen3.5-2b_s1_refseg | 5,990 |
| grid_e3b_c3_qwen3.5-2b_s1_depth | 371 |
| grid_e3b_qwen3.5-4b_s2_refseg | 10,839 |
| grid_e3b_qwen3.5-4b_s2_depth | 495 |
| grid_e3b_c3_qwen3.5-4b_s2_refseg | 10,807 |
| grid_e3b_c3_qwen3.5-4b_s2_depth | 498 |

- 两个种子的 E3b 都按 e5-v2 通过，所以两个 E3b+C3 都跑了；steps.log 没有 skip 行。

## 4 个臂的合并核对（`eval/<variant>/<model>/seed<k>/merge_check.json`）

| 臂 | 写于 | 判据 | pass | failures | v1_report_only.pass | v1_report_only.failures | timing_s.total_s |
|---|---|---|---|---|---|---|---:|
| e3b qwen3.5-2b s1 | 2026-10-24T19:23:06Z | e5-v2 | True | `{"merge": [], "restricted": []}` | False | `{"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9400 < 0.95"], "merge": []}` | 588 |
| e3b_c3 qwen3.5-2b s1 | 2026-10-24T21:10:09Z | e5-v2 | True | `{"merge": [], "restricted": []}` | True | `{"restricted": [], "merge": []}` | 398 |
| e3b qwen3.5-4b s2 | 2026-10-24T23:02:08Z | e5-v2 | True | `{"merge": [], "restricted": []}` | False | `{"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9300 < 0.95"], "merge": []}` | 756 |
| e3b_c3 qwen3.5-4b s2 | 2026-10-25T02:10:18Z | e5-v2 | True | `{"merge": [], "restricted": []}` | True | `{"restricted": [], "merge": []}` | 713 |

- 4 个 merge_check 的 error 都是 null。
- 中途已报管理员：2B s1 E3b（10-24 21:08Z）、4B s2 E3b（10-25 02:30Z），只报 pass / failures，没有贴读数。

## 显存与 traceback

- 显存峰值：reeval.sh 不采样显存（结果目录没有 gpu_mem.csv），评测结果文件里也没有记录显存的字段，日志里没有。
- `logs/` 下 8 个步日志 grep `Traceback`、`out of memory`：都是 0。

## 收尾

- 推送循环 10-25 05:12:08Z 一轮推送成功（已含 done）。05:14:30Z 在 git.lock 下按 PID 停掉循环 2967405（当时在 sleep），随后 `ONCE=1` 推一轮：`2026-10-25T05:14:38Z ok`。
- 之后 07 无暂停文件、GPU 空；squeue 157296、157297 都 RUNNING。07 空着等下一份任务书。
