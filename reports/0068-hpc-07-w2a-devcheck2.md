# RUN-0068 报告：hpcgpu07，W2-a 在 devcheck2 上的去留判定（qwen3.5-9b，500 句，冻结配置 D122）

任务书 `prompts/server_hpc/0068-hpc-07-w2a-devcheck2.md`。没有碰 06（RUN-0069）。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0068.sh`（10-24 13:12:08Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0068（稀疏）HEAD `7f53f240c616d416f808813ba8c67149483bb647`，status 空，`experiments/` 只有 P0。
- 12 个 sha256 与任务书逐字相同；`configs/w2.json` frozen True；驱动用本节点暂停文件；07 无暂停文件、GPU 空。
- 13:12:23Z 按任务书命令启动（`srun --jobid=157297 --overlap -n1 -c 8`，`EV_CAP_H=4 NODE=hpcgpu07 TQ_STATUS=nonexistent_queue.status`）。`-c 8` 被接受，没有用 CW=1 的退路。
- 推送循环 TAG 0068（PID 1582904，1800 s）。启动后 squeue 两条 RUNNING。

## steps.log 原文

```
2026-10-24T13:12:24Z start commit 7f53f240c616d416f808813ba8c67149483bb647 node hpcgpu07 model qwen3.5-9b w2.json c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29
2026-10-24T13:14:13Z e1_box rc=0 wall_s=109
2026-10-24T13:17:26Z w1f rc=0 wall_s=193
2026-10-24T14:08:27Z w2a rc=0 wall_s=3061
2026-10-24T14:08:31Z gate rc=0 wall_s=4
2026-10-24T14:08:35Z ownbox rc=0 wall_s=4
2026-10-24T14:08:41Z random_pick rc=0 wall_s=6
2026-10-24T14:08:48Z oracle_pick rc=0 wall_s=7
2026-10-24T14:08:48Z done; failed steps: 0
2026-10-24T14:08:48Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2-a devcheck2 gate done (failed steps: 0)
```

- 心跳 `2026-10-24T14:08:48Z done 0068 (failed steps: 0)`。总墙钟 13:12:24Z – 14:08:48Z（3,384 s，0.94 h），上限 4 h。

## gate.md 原文

```markdown
# W2 去留判定：W2-a（D113 第六条，规则写死）

- 模型 qwen3.5-9b@c20223623576…，划分 devcheck2，n = 500；判定点核对：{'model_is_qwen3.5-9b': True, 'split_is_devcheck2': True, 'n_is_500': True, 'not_tiny': True, 'no_limit': True, 'w2_config_frozen': True, 'same_w2_config': True}

| 读数 | 值 |
|---|---|
| W2-a（w2a）cIoU | 0.5958 |
| W1-F（w1f）cIoU | 0.5777 |

- 主配置 − W1-F 的 cIoU 差（Σinter / Σunion），配对 bootstrap 97.5% CI（重抽样 10,000 次，种子 src/constants.BOOTSTRAP_SEED），下界 > 0 才算通过：差 0.0181（8247163605477/455781889946864），97.5% CI [-0.0001, 0.0369]（10000 次，种子 20260929）→ **不通过**

**结论：不通过**

两个主配置（W2-a、W2-c）各判一次；"至少一个通过"由管理员在门报告里合起来写。
```

- `gate.json`：diff 0.018094539926626907（`8247163605477/455781889946864`），ci_low -8.34870277306345e-05，ci_high 0.036863071076801215，level 0.975，n_resamples 10000，seed 20260929，pass false，is_gate_point true。
- gate.md 在 14:30Z 发给管理员（gate 一步 14:08:31Z 跑完；中继延迟，监视脚本 14:30Z 才看到）。

## 标题读数与挑选统计

由 `experiments/W2/0068_w2a_devcheck2_probe/p68_tables.py` 从各 `metrics.json` 读出。下表与 RUN-0066 的选臂表对应："挑中最优" = pick.final_is_best，"挑 1 号" = pick.final_is_1，"都不对" = pick.none，"解析失败" = pick.parse_failure；分母 = 有候选的 496 句。

### 各设置（devcheck2）

- refexp_box/baseline_e1：acc@0.5 0.9180 [0.8960, 0.9420]，n 500

| 设置 | cIoU [95% CI] | gIoU [95% CI] | 挑中最优 | 挑 1 号 | 都不对（none） | 解析失败 | prompt token 中位/最大 | 生成 token 中位/最大 | 出框失败 | 模型调用均值 |
|---|---|---|---|---|---|---|---|---|---|---|
| w1f | 0.5777 [0.5486, 0.6067] | 0.5688 [0.5438, 0.5936] | 0.175（87/496） | 1.000（496/496） | 0.000（0/496） | 0.000（0/496） | 342 / 370 | 45 / 221 | {'multiple_boxes': 4} | 1.0 |
| w2a | 0.5958 [0.5690, 0.6222] | 0.6157 [0.5959, 0.6366] | 0.347（172/496） | 0.512（254/496） | 0.006（3/496） | 0.026（13/496） | 2016 / 2133 | 2094 / 2172 | {'multiple_boxes': 4} | 1.992 |
| ownbox | 0.4976 [0.4769, 0.5196] | 0.5265 [0.5093, 0.5441] | — | — | — | — | — | — | {} | — |
| random_pick | 0.5587 [0.5327, 0.5855] | 0.5726 [0.5501, 0.5961] | 0.282（140/496） | 0.262（130/496） | 0.000（0/496） | 0.000（0/496） | 342 / 370 | 45 / 221 | {'multiple_boxes': 4} | 1.0 |
| oracle_pick | 0.6445 [0.6188, 0.6702] | 0.6614 [0.6425, 0.6821] | 1.000（496/496） | 0.175（87/496） | 0.000（0/496） | 0.000（0/496） | 342 / 370 | 45 / 221 | {'multiple_boxes': 4} | 1.0 |

### w2a 的挑选（metrics.json 的 pick 与 select）

- 挑中最优 0.347（172/496）；挑 1 号 0.512（254/496）；none 0.006（3/496）；解析失败 0.026（13/496）
- answer {'1': 238, '2': 37, '3': 32, '4': 173, 'none': 3, 'parse_failure': 13}；parse_failure {'out_of_range': 1, 'truncated': 12}；stopped {'False': 12, 'True': 484}；thinking_end {'forced': 410, 'natural': 86}
- final_index {'1': 254, '2': 37, '3': 32, '4': 173}；best_index {'1': 87, '2': 101, '3': 143, '4': 165}；oracle cIoU（有候选） 0.6524；n_with_candidates 496；config sha256 c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29
- pick 的规则：分母 = 有候选的样本（提出失败的没有候选）；final_is_best = 交的候选的 IoU 等于这条样本候选里的最高 IoU（并列都算）；none、parse_failure 都交 1 号候选

### 候选

- k 4，n_with_candidates 496，names ['gc_default', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts']，duplicate_rate 0.0645（128），not_ok_by_index {}

### 显存（gpu_mem.csv：3384 行，13:12:24Z – 14:08:47Z，全程最大 27129 MiB）

- e1_box：wall_s 109，窗口内最大 27129 MiB（109 行）
- w1f：wall_s 193，窗口内最大 4 MiB（193 行）
- w2a：wall_s 3061，窗口内最大 25625 MiB（3061 行）
- gate：wall_s 4，窗口内最大 0 MiB（4 行）
- ownbox：wall_s 4，窗口内最大 4 MiB（4 行）
- random_pick：wall_s 6，窗口内最大 4 MiB（6 行）
- oracle_pick：wall_s 7，窗口内最大 4 MiB（7 行）

（显存窗口按 steps.log 每步结束时间往前 wall_s；gpu_mem.csv 时间戳为节点本地 +08:00，已换成 UTC。）

## traceback

- `logs/` 下 7 个步日志 grep `Traceback`、`out of memory`：都是 0。

## 收尾

- 推送循环 14:14:42Z 一轮推送成功（origin `20300fc2`，已含 done）。14:31:28Z 在 git.lock 下按 PID 停掉循环 1582904（当时在 sleep），随后 `ONCE=1` 推一轮：`2026-10-24T14:31:42Z ok`（没有新改动）。
- 之后 07 无暂停文件、GPU 空；squeue 157296、157297 都 RUNNING。07 空着等下一份任务书。
