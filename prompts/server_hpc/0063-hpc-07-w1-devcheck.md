[RUN-TASK 0063] hpcgpu07：W1 去留判定的开发-检查集运行（D103 §三；D111 冻结的协议；W1 预算内，约 6 GPU 小时）。**在 RUN-0057 的段边界让路**，做法同 RUN-0061、0062。上限 8 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0056。
- **RUN-0057 不停、不杀**：只靠 `$VI/logs/train.pause.hpcgpu07` 让它在段边界或步之间自己等着。由脚本的 `claim_gpu` 写入，退出时（含出错）自动删除。
- 不写全局的 `$VI/logs/train.pause`。

## 为什么

- D103 §三的去留判定，规则写死在 `scripts/w1_gate.py` 里，在开发-检查集 500 句上看 Qwen3.5-9B 的完整工作台（W1-1）：
  1. W1-1 的 cIoU 显著高于同一模型"自己的框"：配对 bootstrap 95% CI 下界 > 0；
  2. W1-1 的 cIoU 至少比同一模型的 E1 结构强制多边形高 0.2。
- 两条基线都在这 500 句上，用同一个模型、同一套协议补跑。
- 回路协议已在开发-调参集上选定并冻结（D111）：`configs/w1.json` 的 `frozen: true`。未冻结的配置碰开发-检查集会被 `run_w1.py` 拒绝。
- 另外按 W1 预注册 §10b 第 5 条，用冻结的协议在开发-调参集上跑定位 64 条、计数 49 条，只报告。

## 文件（commit 见发出时的启动语；工作树 wt0063 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0063-hpc-w1-devcheck/devcheck.sh` | `739e91d06f0db33aeb71f0cdc87b6976fee55b4c96526a1a7e3955df7436bdeb` |
| `configs/w1.json` | `e4cc841907284009b0d5ea51c075b3d52699eacb3c624207b084bc9e4952f928`（冻结，D111） |
| `configs/w1_splits.json` | `a1371851555e854fac6f9d0e267f19a23602ee9cf31d5c303ef08432d1f7f0f7` |
| `scripts/run_w1.py` | `23770d71a72ff38ae80388ab906d8ad2b04f022a32c6617ca27b6eae5a6904b4` |
| `scripts/w1_gate.py` | `0826f7822886173913fad93db86be7901f752c384a2eae1b98d58ce24c83c208` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |
| `src/w1/loop.py` | `7f493d6ae2e089b3b7e6d651d5b4c010ee9b27163f5f5d05feb7dbbeb7101190` |
| `src/w1/config.py` | `0de5b6eebd56a62f6b46ce907c312411cbc1cd18146c0a2f262136224b31ed1d` |
| `src/w1/runner.py` | `a7f4cd4cfd674fca5fd36cffa39e4f29668246d658bfd42298d4966fb51d6033` |
| `src/w1/workbench.py` | `775e0dca6901662ddd2e1880c01dbf848efaad73127f6d594308fcf797d46591` |
| `src/w1/baselines.py` | `5159164561cb482232afa1209dd1f76552b94514035404256075fcf65dd84a94` |
| `src/w1/scoring.py` | `add4295866e8fb21d707fe13a998cd19dd732c75ad779d79c887fe1a880e7413` |

管理员在开发机上核对过：
- tiny 彩排 8 步都 rc=0。
- 非 tiny 的参数检查：冻结的 `configs/w1.json` 能过冻结保护，未冻结的 `configs/w1_tune/U2.json` 被拒；目录名 `0063_w1_devcheck_hpcgpu07` 合规。

## 内容（全部在 devcheck.sh 里，顺序固定）

| 步 | 内容 |
|---|---|
| `baseline_e1_refexp_box_devcheck` | W1-0 定位，500 句（自己的框、W1-F 要用） |
| `baseline_ownbox_devcheck` | 自己的框光栅化，只用 CPU |
| `baseline_poly_structured_devcheck` | E1 结构强制多边形 |
| `w1f_devcheck` | 自己的框交给 GrabCut，只用 CPU |
| `full_refseg_devcheck` | **W1-1**：9B 完整工作台，指代分割，500 句，不加 `--limit` |
| `gate` | `w1_gate.py`：`gate.json`、`gate.md` |
| `full_refexp_box_devtune64`、`full_count_devtune_countbench49` | 冻结协议在开发-调参集上的定位与计数，只报告 |

- 结果目录：`experiments/W1/0063_w1_devcheck_hpcgpu07/`。
- 预计：
  - 基线约 25 分钟；
  - W1-1 约 4.5 小时（RUN-0062 的 U2 跑 96 句用了 3,028 s）；
  - 定位与计数约 40 分钟。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0063（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试，不在旧的 origin/main 上建。核对上表 12 个文件的 sha256。
3. 启动（方案 B），现在就可以发：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=8 NODE=hpcgpu07 TQ_STATUS=train_0057-hpcgpu07.status \
     bash /home/mail/chenyuma/visint/wt/wt0063/prompts/server_hpc/0063-hpc-w1-devcheck/devcheck.sh \
     > /home/mail/chenyuma/visint/logs/eval_0063.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0063.status`。
   - 另起一个 v3 推送循环，TAG 0063，路径 `experiments/W1/0063_w1_devcheck_hpcgpu07`。跑完、推完后按 PID 停掉。
4. 拿到 GPU 时报一次，带上 RUN-0057 的心跳原文。
5. 基线四步跑完报一次：各自的 rc 与标题读数。
6. `gate` 一步跑完，**立即**把 `gate.md` 原文发给管理员，不等定位与计数。
7. 全部跑完：
   - 核对暂停文件已删、RUN-0057 已恢复；
   - 写报告；
   - 再跑一次 `squeue`。

## 报告

推送 `reports/0063-hpc-07-w1-devcheck.md`，内容：
- `steps.log` 原文；
- 四条基线与 W1-1 的标题读数（cIoU、gIoU 与 CI）；
- W1-1 的表，格式同 RUN-0061：提交数、失败码、每轮调用个数的分布、被拒原因、生成 token 的中位与最大；
- `gate.md` 原文；
- 定位与计数的标题读数；
- 显存峰值、每步墙钟、任何 traceback。

只贴数字，不解读。
