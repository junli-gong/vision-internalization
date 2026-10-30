# RUN-0100 报告：hpcgpu06，统一主表的两只"手"在 W2 test 的 1,500 句子集上（D154、D159；预注册 U-2）

任务书 `prompts/server_hpc/0100-hpc-06-u-hands.md`。E1 结构强制多边形（9B 零样本）与 9B 的 E3b 像素 token 网格。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0100.sh`（10-30 04:06:27Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0100（稀疏）HEAD `813e523362fa6e290a863187cf91cab64d949f34`（含 b61a5ef6），status 空，`src/unified/`（`hands.py`、`__init__.py`）在；10 个 sha256 与任务书逐字相同（`u_hands.sh` `c6829504…`）；驱动用本节点暂停文件。
- E3b 检查点 `$VI/ckpt/hpcgpu06/0056/e3b/qwen3.5-9b/seed0/ckpt/final/model` 的 manifest sha256 `316d25c7bf0b21edd6355a55bc0201249078b7c59305563dd3e25082c7c088db`，与任务书一致（文件：lora、manifest.json、token_rows.safetensors、tokens.json）；`configs/w2.json` frozen True；RUN-0097 已 done。
- 04:06:36Z 按任务书命令启动（`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=3 bash …/u_hands.sh`），`-c 8` 被接受。后 squeue：157296、157297 RUNNING。
- 04:06:37Z 第一步 `u_e1poly`；推送循环 TAG 0100（PID 240993）。

## steps.log 原文

```
2026-10-30T04:06:37Z start commit 813e523362fa6e290a863187cf91cab64d949f34 node hpcgpu06 model qwen3.5-9b
2026-10-30T04:24:53Z u_e1poly rc=0 wall_s=1096
2026-10-30T04:24:53Z e3b checkpoint /home/mail/chenyuma/visint/ckpt/hpcgpu06/0056/e3b/qwen3.5-9b/seed0/ckpt/final/model manifest 316d25c7bf0b21edd6355a55bc0201249078b7c59305563dd3e25082c7c088db
2026-10-30T05:13:33Z u_e3bgrid rc=0 wall_s=2920
2026-10-30T05:13:33Z done; failed steps: 0
2026-10-30T05:13:33Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）U hands 0100 done (failed steps: 0)
```

- 心跳 `2026-10-30T05:13:33Z done 0100 (failed steps: 0)`。`logs/` 下日志无 Traceback / OOM。

## E1 结构强制多边形（`0100_u_e1poly/qwen3.5-9b/refseg/u_e1_poly_structured/<划分>/metrics.json`；95% CI，1,000 次重抽）

| 划分 | n | cIoU [95% CI] | gIoU [95% CI] | complete | failures | empty_mask.n |
|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 300 | 0.3532 [0.3235, 0.3835] | 0.3372 [0.3150, 0.3591] | True | {"ok": 300} | 17 |
| test_refcoco_unc_testB | 300 | 0.4166 [0.3817, 0.4530] | 0.4270 [0.3990, 0.4576] | True | {"ok": 300} | 15 |
| test_refcoco+_unc_testA | 300 | 0.2950 [0.2610, 0.3368] | 0.2789 [0.2580, 0.2994] | True | {"ok": 300} | 22 |
| test_refcoco+_unc_testB | 300 | 0.3420 [0.3050, 0.3797] | 0.3471 [0.3184, 0.3742] | True | {"ok": 300} | 13 |
| test_refcocog_umd_test | 300 | 0.3454 [0.3132, 0.3755] | 0.3355 [0.3092, 0.3595] | True | {"ok": 300} | 18 |

## E3b 像素 token 网格（`0100_u_e3bgrid/qwen3.5-9b/refseg/u_e3b_grid/<划分>/metrics.json`；同上）

| 划分 | n | cIoU [95% CI] | gIoU [95% CI] | complete | failures | empty_mask.n |
|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 300 | 0.6973 [0.6570, 0.7372] | 0.6881 [0.6550, 0.7181] | True | {"ok": 300} | 30 |
| test_refcoco_unc_testB | 300 | 0.5760 [0.5165, 0.6335] | 0.5604 [0.5165, 0.6072] | True | {"ok": 300} | 63 |
| test_refcoco+_unc_testA | 300 | 0.6284 [0.5788, 0.6750] | 0.6372 [0.5994, 0.6719] | True | {"ok": 300} | 27 |
| test_refcoco+_unc_testB | 300 | 0.4367 [0.3796, 0.4936] | 0.4190 [0.3749, 0.4648] | True | {"ok": 300} | 93 |
| test_refcocog_umd_test | 300 | 0.6160 [0.5651, 0.6654] | 0.5672 [0.5254, 0.6077] | True | {"ok": 300} | 59 |

- 合并核对（06 上重做，`0100_u_e3bgrid/qwen3.5-9b/e3b_merge_check.json`；`check_training_interfaces --checks merge restricted --n 200`）：criteria v1，**pass True**，failures `{'restricted': [], 'merge': []}`，error None；检查点 manifest `316d25c7…`；用时 merge_before 213.8 s、merge_after 123.3 s、restricted 500.3 s、restricted_across_merge 56.6 s，合计 894.8 s。

## 收尾

- 推送循环 0100：05:15:31Z 在睡眠中按 PID 停（240993 + sleep 453868），ONCE 推送 05:15:44Z ok。
- 后 squeue（05:15:44Z）：157296 RUNNING 75-16:41:30 hpcgpu06；157297 RUNNING 77-14:13:39 hpcgpu07。
- 06 现在空闲；07 上 RUN-0098 在跑。
