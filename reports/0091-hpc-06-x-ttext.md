# RUN-0091 报告：hpcgpu06，X-2（C1）与 X-3（C2）的定位与计数 T 文本读数（D147-4）

任务书 `prompts/server_hpc/0091-hpc-06-x-ttext.md`。用 E1 的 eval 子集，不碰 devcheck3 与 test。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0091.sh`（10-28 22:02:51Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0091（稀疏）HEAD `4cf0850659aa2a1498a5a511d687ea1bc8d9d7ec`（含 d11bf74a），status 空，`experiments/` 只有 P0；3 个 sha256 与任务书逐字相同（`x_ttext.sh` `1c4a80fe…`）；驱动用本节点暂停文件。
- 检查点 manifest：x2 `a49cf4e5…`、x3 `837bb679…`，与任务书一致，`read_manifest` 都通过（各 4 个文件）。
- 22:03:06Z 按任务书命令启动（`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=5 bash …/x_ttext.sh`），`-c 8` 被接受。后 squeue：157296、157297 RUNNING。
- 22:03:09Z 第一步 `ttext_x2`；推送循环 TAG 0091（PID 428233）。

## steps.log 原文

```
2026-10-28T22:03:09Z start commit 4cf0850659aa2a1498a5a511d687ea1bc8d9d7ec node hpcgpu06
2026-10-28T22:03:09Z x2 checkpoint manifest a49cf4e5a81fd8f7f57ca5c86a879cf24cdd1449daa43cc355e6d3a074a6fd0a
2026-10-28T23:50:49Z ttext_x2 rc=0 wall_s=6460
2026-10-28T23:50:49Z x3 checkpoint manifest 837bb679cfc172ecfa4bce3af1ae21c4e1774d3d9fdf0e16fc4e5d2c34ed7f5c
2026-10-29T01:34:34Z ttext_x3 rc=0 wall_s=6225
2026-10-29T01:34:34Z done; failed steps: 0
2026-10-29T01:34:36Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-2/X-3 T-text 0091 done (failed steps: 0)
```

- 心跳 `2026-10-29T01:34:34Z done 0091 (failed steps: 0)`。两个日志无 Traceback / OOM。

## 标题读数（`metrics.json` 的 `headline`，95% CI，1,000 次重抽）

| 标签 | refexp_box acc@0.5（n = 56,857） | 解析失败 | count accuracy（n = 442） | 解析失败 | complete |
|---|---|---|---|---|---|
| `t_text/x2/seed0` | 0.9056 [0.9033, 0.9080] | 1 / 56,857（malformed） | 0.8077 [0.7715, 0.8439] | 0 | True / True |
| `t_text/x3/seed0` | 0.9057 [0.9034, 0.9082] | 1 / 56,857（malformed） | 0.7195 [0.6787, 0.7602] | 0 | True / True |

- 截断率两者都是 0。refexp_box 的 conditional 列：x2 0.9056 [0.9033, 0.9081]（n_valid 56,856），x3 0.9057 [0.9033, 0.9082]。
- 计数题出自 CountBenchQA（PaliGemma，big_vision 仓库；D36-2 注明）。

配对参照（RUN-0069 在 06 上的 `0069_ttext/`，同一 eval 子集；照录文件中的数字）：

| 标签 | refexp_box acc@0.5 | 解析失败 | count accuracy |
|---|---|---|---|
| 基座 `qwen3.5-9b` | 0.8714 [0.8687, 0.8740] | 1,063 / 56,857（multiple_boxes 989、truncated 69、malformed 4、no_box 1） | 0.8937 [0.8597, 0.9231] |
| W2-c `t_text/w2c/seed0` | 0.9067 [0.9044, 0.9092] | 1 / 56,857 | 0.7624 [0.7217, 0.8009] |

## 合并核对（`merge_check.json`）

| 标签 | pass | failures |
|---|---|---|
| x2 | True | merge [] |
| x3 | True | merge [] |

## 收尾

- 推送循环 0091：01:46:27Z 在睡眠中按 PID 停（428233 + sleep 3459877），ONCE 推送 01:46:35Z ok。
- 后 squeue（01:46:35Z）：157296 RUNNING 74-13:12:21 hpcgpu06；157297 RUNNING 76-10:44:30 hpcgpu07。
- 06 上 RUN-0092 已按驱动自动接上（01:34:35Z running `x2_test_refcoco_unc_testA`）。
