# RUN-0107 报告：hpcgpu07，X-M 在 2,300 句开发数据上的指代分割与大题量语言（XM-4b、XM-5、XM-5a）

任务书 `prompts/server_hpc/0107-hpc-07-xm-refseg-lb.md`。**这些读数在门报告出来之前不看、不报**：本报告只有 complete、失败数与用时。

## 启动

- 启动器 `start0106_0107.sh`：wt0107 `b3e96ee4…`，8 个 sha256 相同；驱动用本节点暂停文件；输入（val2017、MMLU-Pro 原始数据、RUN-0084 的 lb_base、b200/data）都在。
- 06 被 lly-mle 占着，按任务书 RUN-0106 先在 07 上跑；本任务 00:17:36Z 启动（`srun --jobid=157297 --overlap -n1 -c 12 env … TINY=0 EV_CAP_H=8 WAIT_DONE='eval_0106.status:done 0106' bash …/xm_refseg_lb.sh`），等 RUN-0106 首次运行写出 `done 0106` 后 01:04:37Z 开始。前后 squeue：157296、157297 RUNNING。推送循环 TAG 0107（PID 3220588）。

## steps.log 原文

```
2026-11-04T01:04:37Z start commit b3e96ee4a090167fa2c9345719d586ca08cbf861 node hpcgpu07 model qwen3.5-9b x-m checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e
2026-11-04T01:34:18Z xm_refseg_explore_val_refcoco_unc_val rc=0 wall_s=1781
2026-11-04T01:58:12Z xm_refseg_explore_val_refcoco+_unc_val rc=0 wall_s=1434
2026-11-04T02:10:45Z xm_refseg_explore_val_refcocog_umd_val rc=0 wall_s=753
2026-11-04T02:58:55Z xm_refseg_devtune800 rc=0 wall_s=2890
2026-11-04T03:30:40Z xm_refseg_devcheck2 rc=0 wall_s=1905
2026-11-04T05:13:42Z lb_xm rc=0 wall_s=6182
2026-11-04T05:13:42Z done; failed steps: 0
2026-11-04T05:13:42Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M refseg + battery 0107 done (failed steps: 0)
```

- 心跳 `2026-11-04T05:13:42Z done 0107 (failed steps: 0)`。`logs/` 下无 Traceback / OOM。

## 指代分割（`0107_refseg/qwen3.5-9b/refseg/x2_spline_trimap/<划分>/metrics.json`；只报 complete 与失败数）

| 划分 | n | complete | failures | fallback |
|---|---|---|---|---|
| explore_val_refcoco_unc_val | 400 | True | ok 400 | ok 399、parse_failure 1 |
| explore_val_refcoco+_unc_val | 400 | True | ok 400 | ok 398、parse_failure 2 |
| explore_val_refcocog_umd_val | 200 | True | ok 200 | ok 200 |
| devtune800 | 800 | True | ok 800 | ok 800 |
| devcheck2 | 500 | True | ok 500 | ok 500 |

- 五个 `x2_summary/<划分>.json` 各 20 个设置。合并核对 `w2c_merge_check.json`：pass True，failures merge []。

## 大题量（`0107_lb_xm/`）

- ok True；preflight 2,541 查、0 失败；合并核对 v1 pass True，failures merge []。
- 驱动没有 `--compare` 一步；与 RUN-0084 在 07 上的 `lb_base` 的对比由管理员做。

## 收尾

- 推送循环 0107：08:07:47Z 按 PID 停，ONCE 推送 ok。后 squeue：157296 RUNNING 80-19:33:33 hpcgpu06；157297 RUNNING 82-17:05:42 hpcgpu07。没碰 lly-mle 的进程。
