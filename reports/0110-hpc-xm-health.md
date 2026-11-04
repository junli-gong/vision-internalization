# RUN-0110 报告：hpcgpu06，X-M 合并后模型的健康检查（简报 §8.11；G-XM 门报告第 5 项）

任务书 `prompts/server_hpc/0110-hpc-xm-health.md`。对照 RUN-0074 在 06 上的 `health_base.json`（协议 v3）。只贴数字，不解读。

## 运行

- 经 hpcmgt 上脱离会话的 `$VI/tmp/run0110.sh`（11-04 08:37:47Z 起）。前 squeue：157296 RUNNING 80-20:03:33 hpcgpu06；157297 RUNNING 82-17:35:42 hpcgpu07。
- 启动前 06 的 nvidia-smi：没有计算进程，显存 0 MiB（lly-mle 已自行结束）。
- fetch 第 1 次 rc=0；wt0110（稀疏）HEAD `2c2bb42c48f50c3bd36a23a02bf24cd50985b918`，status 空，`tests/fixtures/health` 在；10 个工作树 sha256 与任务书相同（`xm_health.sh` `e6fb7b7f…`），REPO 里的 `health_base.json` `bb66634f…` 也相同（11/11）。
- 08:37:59Z 按任务书命令启动（`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 NODE=hpcgpu06 EV_CAP_H=2 bash …/xm_health.sh`）。
- 一次 ONCE 推送 `experiments/XM/0110_xm_health_hpcgpu06`，09:05:10Z ok。后 squeue（09:05:10Z）：157296 RUNNING 80-20:30:56 hpcgpu06；157297 RUNNING 82-18:03:05 hpcgpu07。

## steps.log 原文

```
2026-11-04T08:38:00Z start commit 2c2bb42c48f50c3bd36a23a02bf24cd50985b918 node hpcgpu06 model qwen3.5-9b x-m checkpoint manifest 29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e health_base bb66634f24752fdb30ca4fc2a428e02574f8fe0b6ee1bb421bc07ce92d34bfcd
2026-11-04T09:04:30Z health_xm rc=0 wall_s=1590
2026-11-04T09:04:32Z health_compare rc=0 wall_s=2
2026-11-04T09:04:32Z done; failed steps: 0
2026-11-04T09:04:32Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）X-M health 0110 done (failed steps: 0)
```

- 心跳 `2026-11-04T09:04:32Z done 0110 (failed steps: 0)`。

## 合并核对与读数

- `health_xm_merge_check.json`：v1 pass True，failures merge []。
- `health_xm.json`：ok True，标题 mmlu_pro 54.0、gsm8k 90.0、ifeval 90.0。

## `health_compare.json` 的 rows

| reading | metric | headline | diagnostic | n | base % | merged % | delta pp | CI95 pp | lost/gained | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro_letter | accuracy | True | False | 100 | 53.0 | 54.0 | +1.0 | [-4.0, +7.0] | 4/5 | False |
| mmlu_pro_letter | mean_p_gold | False | False | 100 | 49.0 | 48.6 | -0.4 | [-2.8, +2.2] |  | None |
| mmlu_pro_free256 | accuracy | False | True | 100 | 53.0 | 54.0 | +1.0 | [-4.0, +7.0] | 4/5 | None |
| mmlu_pro_fmt32 | accuracy | False | True | 100 | 53.0 | 54.0 | +1.0 | [-4.0, +7.0] | 4/5 | None |
| gsm8k | flexible | True | False | 50 | 86.0 | 90.0 | +4.0 | [-4.0, +12.0] | 1/3 | False |
| gsm8k | strict | False | False | 50 | 86.0 | 90.0 | +4.0 | [-4.0, +12.0] | 1/3 | None |
| ifeval | prompt_level_strict | True | False | 50 | 94.0 | 90.0 | -4.0 | [-10.0, +0.0] | 2/0 | False |
| ifeval | instruction_level_strict | False | False | 77 | 96.1 | 90.9 | -5.2 | [-10.3, -1.3] | 4/0 | None |
| ifeval | prompt_level_loose | False | False | 50 | 98.0 | 92.0 | -6.0 | [-12.0, +0.0] | 3/0 | None |
| ifeval | instruction_level_loose | False | False | 77 | 98.7 | 93.5 | -5.2 | [-10.3, -1.3] | 4/0 | None |

any_flag = False; flagged = []; threshold_pp = 5; partial_run = False

## `logs/health_compare.log` 原文

```
protocol: base v3, merged v3
mmlu_pro_letter accuracy                  base=53.0 merged=54.0 delta=+1.0 pp  CI95=[-4.0, +7.0]  sign test 5+/4- p=1  (lost 4, gained 5, n=100)
mmlu_pro_letter mean_p_gold               base=49.0 merged=48.6 delta=-0.4 pp  CI95=[-2.8, +2.2]  (n=100)  [诊断，不标红]
mmlu_pro_free256 accuracy                  base=53.0 merged=54.0 delta=+1.0 pp  CI95=[-4.0, +7.0]  sign test 5+/4- p=1  (lost 4, gained 5, n=100)  [诊断，不标红]
mmlu_pro_fmt32  accuracy                  base=53.0 merged=54.0 delta=+1.0 pp  CI95=[-4.0, +7.0]  sign test 5+/4- p=1  (lost 4, gained 5, n=100)  [诊断，不标红]
gsm8k           flexible                  base=86.0 merged=90.0 delta=+4.0 pp  CI95=[-4.0, +12.0]  sign test 3+/1- p=0.625  (lost 1, gained 3, n=50)
gsm8k           strict                    base=86.0 merged=90.0 delta=+4.0 pp  CI95=[-4.0, +12.0]  sign test 3+/1- p=0.625  (lost 1, gained 3, n=50)
ifeval          prompt_level_strict       base=94.0 merged=90.0 delta=-4.0 pp  CI95=[-10.0, +0.0]  sign test 0+/2- p=0.5  (lost 2, gained 0, n=50)
ifeval          instruction_level_strict  base=96.1 merged=90.9 delta=-5.2 pp  CI95=[-10.3, -1.3]  sign test 0+/4- p=0.125  (lost 4, gained 0, n=77)
ifeval          prompt_level_loose        base=98.0 merged=92.0 delta=-6.0 pp  CI95=[-12.0, +0.0]  sign test 0+/3- p=0.25  (lost 3, gained 0, n=50)
ifeval          instruction_level_loose   base=98.7 merged=93.5 delta=-5.2 pp  CI95=[-10.3, -1.3]  sign test 0+/4- p=0.125  (lost 4, gained 0, n=77)
truncation mmlu_pro_free256 max_new_tokens=256   base=0.0% merged=0.0% delta=+0.0 pp (n=100)
truncation mmlu_pro_fmt32  max_new_tokens=32    base=0.0% merged=0.0% delta=+0.0 pp (n=100)
truncation gsm8k           max_new_tokens=512   base=12.0% merged=6.0% delta=-6.0 pp (n=50)
truncation ifeval          max_new_tokens=1024  base=2.0% merged=4.0% delta=+2.0 pp (n=50)
any_flag=False  flagged=[]  truncation_notices=[]  incomparable=[]  unscored=[]  partial_run=False
```
