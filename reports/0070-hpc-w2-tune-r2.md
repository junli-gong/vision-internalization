# RUN-0070 报告：hpcgpu06 + hpcgpu07，W2-a 调参第 2 步重做（以 sep_on 为底，K = 4、6；D121）

任务书 `prompts/server_hpc/0070-hpc-w2-tune-r2.md`。只用 devtune800。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0070.sh`（10-24 11:09:47Z 起）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0070（稀疏）HEAD `af5a8a6a1acd9c704be3ff00f6ee01ee9ec8e41d`（含 1b6c12ee），status 空，`experiments/` 只有 P0。
- 9 个 sha256 与任务书逐字相同；arm.sh 用本节点暂停文件；两个节点都没有暂停文件、GPU 都空。
- 11:10:12Z 按任务书命令同时启动：06 K=4（`srun --jobid=157296 --overlap -n1 -c 8`），07 K=6（`srun --jobid=157297 --overlap -n1 -c 8`），`EV_CAP_H=3`。`-c 8` 都被接受，没有用 CW=1 的退路。

## steps.log 原文

06（K = 4）：

```
2026-10-24T11:10:13Z start commit af5a8a6a1acd9c704be3ff00f6ee01ee9ec8e41d node hpcgpu06 model qwen3.5-9b arm sep_on_k4
2026-10-24T11:11:50Z arm_cfg_sep_on_k4 rc=0 wall_s=97
2026-10-24T12:39:04Z arm_sep_on_k4 rc=0 wall_s=5233
2026-10-24T12:39:04Z done; failed steps: 0
2026-10-24T12:39:04Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2 tuning round-2 arm sep_on_k4 done (failed steps: 0)
```

07（K = 6）：

```
2026-10-24T11:10:14Z start commit af5a8a6a1acd9c704be3ff00f6ee01ee9ec8e41d node hpcgpu07 model qwen3.5-9b arm sep_on_k6
2026-10-24T11:11:50Z arm_cfg_sep_on_k6 rc=0 wall_s=96
2026-10-24T12:50:12Z arm_sep_on_k6 rc=0 wall_s=5901
2026-10-24T12:50:12Z done; failed steps: 0
2026-10-24T12:50:12Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2 tuning round-2 arm sep_on_k6 done (failed steps: 0)
```

- 心跳：`2026-10-24T12:39:04Z done 0070_k4 (failed steps: 0)`；`2026-10-24T12:50:12Z done 0070_k6 (failed steps: 0)`。
- 墙钟：06 共 5,331 s（11:10:13Z – 12:39:04Z），07 共 5,998 s（11:10:14Z – 12:50:12Z）；上限各 3 h。

## 两个臂的标题读数

### 分割臂（refseg，devtune800）

| 运行目录 | 叶 | cIoU [95% CI] | gIoU | 挑中最优 | 挑 1 号 | none | 解析失败 | prompt token 中位/最大 | 生成 token 中位/最大 | 出框失败 | 模型调用均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0070_sep_on_k4 | w2a | 0.5957 [0.5729, 0.6190] | 0.5986 | 0.392（308/786） | 0.444（349/786） | 0.008（6/786） | 0.027（21/786） | 2003 / 2133 | 2094 / 2172 | {'multiple_boxes': 13, 'truncated': 1} | 1.9825 |

### 选择阶段（metrics.json 的 select，只有 w2a 臂有）

- 0070_sep_on_k4：answer {'1': 322, '2': 64, '3': 51, '4': 322, 'none': 6, 'parse_failure': 21}；parse_failure {'no_choice': 2, 'truncated': 19}；stopped {'False': 19, 'True': 767}；thinking_end {'forced': 632, 'natural': 154}；final_index {'1': 349, '2': 64, '3': 51, '4': 322}；best_index {'1': 152, '2': 167, '3': 185, '4': 282}；oracle cIoU（有候选） 0.6520；config sha256 68fbaa696ce9…

### 候选（candidates*/metrics.json）

- 0070_sep_on_k4/candidates：k 4，n_with_candidates 786，names ['gc_default', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts']，duplicate_rate 0.0741（233），not_ok_by_index {}

### 出框（refexp_box/baseline_e1）与 box_points 出点

- 0070_sep_on_k4：acc@0.5 0.8950 [0.8738, 0.9150]，n 800


### 分割臂（refseg，devtune800）

| 运行目录 | 叶 | cIoU [95% CI] | gIoU | 挑中最优 | 挑 1 号 | none | 解析失败 | prompt token 中位/最大 | 生成 token 中位/最大 | 出框失败 | 模型调用均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0070_sep_on_k6 | w2a | 0.5677 [0.5448, 0.5916] | 0.5826 | 0.271（213/786） | 0.399（314/786） | 0.004（3/786） | 0.032（25/786） | 2640 / 2809 | 2094 / 2164 | {'multiple_boxes': 13, 'truncated': 1} | 1.9825 |

### 选择阶段（metrics.json 的 select，只有 w2a 臂有）

- 0070_sep_on_k6：answer {'1': 286, '2': 32, '3': 31, '4': 21, '5': 175, '6': 213, 'none': 3, 'parse_failure': 25}；parse_failure {'no_choice': 1, 'truncated': 24}；stopped {'False': 24, 'True': 762}；thinking_end {'forced': 659, 'natural': 127}；final_index {'1': 314, '2': 32, '3': 31, '4': 21, '5': 175, '6': 213}；best_index {'1': 134, '2': 52, '3': 124, '4': 172, '5': 209, '6': 95}；oracle cIoU（有候选） 0.6585；config sha256 f69f7cd7fa16…

### 候选（candidates*/metrics.json）

- 0070_sep_on_k6/candidates：k 6，n_with_candidates 786，names ['gc_default', 'gc_box_plus10', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts', 'box']，duplicate_rate 0.0575（271），not_ok_by_index {'2': 1}

### 出框（refexp_box/baseline_e1）与 box_points 出点

- 0070_sep_on_k6：acc@0.5 0.8950 [0.8738, 0.9150]，n 800


- 配置文件 sha256：`cfg/sep_on_k4.json` `68fbaa696ce984ece6933c03c648ff6b528bcb9c15355fadce577c280eb88737`（与 metrics.json 的 w2_config_sha256 相同）；`cfg/sep_on_k6.json` `f69f7cd7fa16fb7eb16801d84864215fa2756b4c717640c128d2fe805d75ee95`（同）。
- 表由 `experiments/W2/0070_w2_tune_r2_probe/p70_tables.py` 读出（p66 的副本，改了目录）。

## 选臂（hpcmgt 的 wt0070，按任务书第 4 步的命令原样）

- 12:51:14Z 两个都 done 后开始：step1r rc=0，输出 `sep_on`；12:51:15Z step2r rc=0，输出 `sep_on_k4`。
- `step2r.json`：max_arm `sep_on_k4`，chosen `sep_on_k4`。按任务书第 6 步，最终臂的配置是 `0070_w2_tune_r2_hpcgpu06/cfg/sep_on_k4.json`。

### step1r.md 原文

```markdown
# W2 调参：第 1 步（D121 重选） 选臂

- 规则：W2 预注册 §5：取减 W1-F 的配对 cIoU 差最大的臂；与最大者相差不到 0.005 的取最简单的（给出的顺序，越前越简单）；只在仪器有效的臂里选（挑选解析失败 ≤ 0.1、出框失败 ≤ 0.1；§5 补记 2，D121）
- n = 800；参照 cIoU：w1f 0.5724，random_pick 0.5274，oracle_pick 0.6470

| 臂（简单→复杂） | cIoU | 减 W1-F | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 | 出框失败 | 参选 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| grid_off | 0.5193 | -0.0530 | 0.131 | 0.095 | 0.055 | 0.004 | 0.018 | 是 |
| sep_off | 0.5641 | -0.0083 | 0.165 | 0.926 | 0.011 | 0.911 | 0.018 | 否（parse_failure 0.911 > 0.1） |
| grid_on | 0.5476 | -0.0248 | 0.169 | 0.342 | 0.061 | 0.048 | 0.018 | 是 |
| sep_on | 0.5610 | -0.0113 | 0.228 | 0.345 | 0.008 | 0.018 | 0.018 | 是 |

**差最大**：sep_on；**选中**：sep_on
```

### step2r.md 原文

```markdown
# W2 调参：第 2 步（D121 重做） 选臂

- 规则：W2 预注册 §5：取减 W1-F 的配对 cIoU 差最大的臂；与最大者相差不到 0.005 的取最简单的（给出的顺序，越前越简单）；只在仪器有效的臂里选（挑选解析失败 ≤ 0.1、出框失败 ≤ 0.1；§5 补记 2，D121）
- n = 800；参照 cIoU：w1f 0.5724，random_pick 0.5274，oracle_pick 0.6470

| 臂（简单→复杂） | cIoU | 减 W1-F | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 | 出框失败 | 参选 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sep_on_k4 | 0.5957 | +0.0233 | 0.392 | 0.444 | 0.008 | 0.027 | 0.018 | 是 |
| sep_on_k6 | 0.5677 | -0.0047 | 0.271 | 0.399 | 0.004 | 0.032 | 0.018 | 是 |
| sep_on | 0.5610 | -0.0113 | 0.228 | 0.345 | 0.008 | 0.018 | 0.018 | 是 |

**差最大**：sep_on_k4；**选中**：sep_on_k4
```

## 显存与 traceback

- 显存峰值：arm.sh 不采样显存，两个结果目录里都没有 gpu_mem.csv，日志里也没有。
- 4 个步日志（`logs/arm_cfg_sep_on_k{4,6}.log`、`logs/arm_sep_on_k{4,6}.log`）grep `Traceback`、`out of memory`：都是 0。

## 收尾

- 推送（`ONCE=1`，TAG 0070）：`0070_w2_tune_r2_hpcgpu06` 第 1 次 `2026-10-24T12:51:28Z ok`；`0070_w2_tune_r2_hpcgpu07` 第 1 次 `2026-10-24T12:51:38Z ok`（origin `8890849b`）。没有起常驻循环。
- 后 squeue：157296、157297 RUNNING。两张卡空着等管理员冻结。
