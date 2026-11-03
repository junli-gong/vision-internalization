# RUN-0105 报告：hpcgpu07，基座零样本在 X-M 调参集上，以及 X-2 在 ReasonSeg val 上（D165 第六节，D168）

任务书 `prompts/server_hpc/0105-hpc-07-xm-tune.md`。不碰判定集 `xm_decision`。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0105.sh`（11-03 07:08:54Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0105（稀疏）HEAD `e0113b4b8eea8dcc08981cb4034f25adbf11ada4`（含 61abf1e3），status 空；12 个 sha256 与任务书逐字相同（`xm_tune.sh` `1c2e44a6…`，`configs/xm_splits.json` `995345f4…`）；驱动用本节点暂停文件。
- X-2 seed 0 检查点 manifest `a49cf4e5…`（同 RUN-0080）；COCO val2017 标注与图、`reasonseg/val`、`b200/data` 都在。
- 07:11:55Z 按任务书命令启动（`srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=4 bash …/xm_tune.sh`），与 RUN-0103 第 2 步（`-c 14`，只用 CPU）同节点。后 squeue：157296、157297 RUNNING。推送循环 TAG 0105。

## steps.log 原文

```
2026-11-03T07:11:57Z start commit e0113b4b8eea8dcc08981cb4034f25adbf11ada4 node hpcgpu07 model qwen3.5-9b
2026-11-03T08:02:29Z base_tune rc=0 wall_s=3031
2026-11-03T08:02:29Z x2 checkpoint manifest a49cf4e5a81fd8f7f57ca5c86a879cf24cdd1449daa43cc355e6d3a074a6fd0a
2026-11-03T09:54:55Z reasonseg_x2 rc=0 wall_s=6746
2026-11-03T09:54:56Z done; failed steps: 0
2026-11-03T09:55:23Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M tune/ReasonSeg 0105 done (failed steps: 0)
```

- 心跳 `done 0105 (failed steps: 0)`。`logs/` 下日志无 Traceback / OOM。

## 第一组的每对用时（`base_trace/xm_tune/logs/timing.jsonl`，前 128 行，chunk 0–1）

- `model_s.list`（每批一次调用，同批共享）：中位 1.805 s，最大 1.991 s。
- `tool_s.instances`（GrabCut，n = 118）：中位 0.930 s，均值 1.678 s，p90 3.363 s，最大 15.939 s。
- 加载权重约 53 s；transformers 提示 `causal_conv1d` 未装、用参考实现。

## 基座零样本，`xm_tune`（`0105_base_tune/qwen3.5-9b/xm/<设置>/xm_tune/metrics.json`；95% CI，1,000 次重抽）

n = 923 对、300 张图、78 类、2,263 个实例；complete True；每对 1 次模型调用。

| 读数 | base_det（框） | base_box_grabcut（mask） |
|---|---|---|
| f1_rexomni（主读数）[CI] | 0.6955 [0.6704, 0.7461] | 0.4094 [0.3811, 0.4587] |
| P / R | 0.7028 / 0.6884 | 0.4172 / 0.4020 |
| F1@mIoU（f1_rexomni_miou）[CI] | 0.5203 [0.4951, 0.5604] | 0.2578 [0.2322, 0.2915] |
| 其 P / R | 0.5279 / 0.5130 | 0.2640 / 0.2518 |
| 微平均 F1 [CI] | 0.6049 [0.5651, 0.6493] | 0.3121 [0.2783, 0.3515] |
| 其 P / R | 0.5967 / 0.6133 | 0.3026 / 0.3221 |
| 配上的对的平均 IoU [CI] | 0.8313 [0.8230, 0.8411] | 0.7567 [0.7467, 0.7668] |
| IoU 0.5 计数 | tp 1388、fp 938、fn 875、ignored 95 | tp 729、fp 1680、fn 1534、ignored 12 |

- `base_box_grabcut` 的检测读数与 `base_det` 相同（同一次调用 1 的框，NMS 之后）。
- 诊断（不进判定，没有 CI）：`f1_rexomni_exact` 0.7251（P 0.7551、R 0.6974）；`f1_rexomni_exact_miou` 0.5653。
- 失败与计数（两个设置相同）：`list_failure` ok 883、out_of_range 2、truncated 38；`list_counts` n_raw 2,628、n_capped 97、n_clipped 0、n_inverted 0；`nms_suppressed` 110；每对预测框数中位 1、p95 9、最大 30（0 个的 51 对，30 个的 7 对）。`base_box_grabcut` 另有 fallback ok 2,421、`tool_not_ok` 9。

## X-2 seed 0 在 ReasonSeg val（`0105_reasonseg_x2/qwen3.5-9b/`）

- 合并核对 `w2c_merge_check.json`：pass True。n = 196；`failures`、`fallback` 都是 `{"ok": 196}`；complete True。

| x2_spline_trimap | gIoU | cIoU |
|---|---|---|
| W2 格式（`refseg/x2_spline_trimap/reasonseg_val/metrics.json` 的 headline，不扣忽略区域）[95% CI] | 0.5396 [0.4963, 0.5830] | 0.5145 [0.4399, 0.5895] |
| `reasonseg_readout/reasonseg_val.json`，扣掉忽略区域（`meta.ignore_rle` 的像素不进交与并，同 LISA 的 ignore_index；真值空且预测空记 1） | 0.5749 | 0.5840 |

- 196 条里有忽略区域的 76 条。

## 收尾

- 推送循环 0105：10:02:58Z 在睡眠中按 PID 停（1752734），ONCE 推送 10:05:20Z ok。
- 后 squeue（10:05:20Z）：157296 RUNNING 79-21:31:06 hpcgpu06；157297 RUNNING 81-19:03:15 hpcgpu07。
- 07 上随后由 RUN-0104（补充 2，D175）接上。
