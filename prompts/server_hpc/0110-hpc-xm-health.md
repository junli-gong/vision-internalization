[RUN-TASK 0110] X-M 合并后模型的最低限度健康检查（简报 §8.11；G-XM 门报告第 5 项）。**在 06 上跑**，对同节点基座 RUN-0074。约 0.4 GPU 小时，上限 2 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰别的项目的进程。
- 只用 06 自己的暂停文件。

## 为什么

简报 §8.11 规定，每个门都对合并后的模型跑约 200 题的固定小集合，即 MMLU-Pro 100、GSM8K 50、IFEval 50，与基座比较，任何一项降超过 5 个点就标红。G-XM 还缺这一项。

- 06 的 GPU 已空出（lly-mle 自己结束了），07 在跑 RUN-0109，所以本任务在 06 上跑。
- 对照用 RUN-0074 在 06 上的 `health_base.json`，协议 v3。健康检查是确定性的固定协议，同节点重测与已有读数逐题相同；RUN-0074 核对过，G-EX 也是这样做的。
- 命令行照抄 RUN-0080：`--model`、`--checkpoint`、合并核对的 `--data-root $VI/b200/data`。

## 文件（工作树 wt0110 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0110-hpc-xm-health/xm_health.sh` | `e6fb7b7f386b0ee402721b33c6bb7f2d851569a89dfcf48e20b7f756a9f061b7` |
| `scripts/health_check.py` | `4728a0ca26433f5c02b8b3c064b637a437a32de4c1a1847056614aee2ec5c926` |
| `src/health/__init__.py` | `0537e59956503a5e932394072e8219902ac2330666dab2868344b291f473f9eb` |
| `src/health/items.py` | `0ad9ec23fe8e80baa1bb1faa2bfbdb419f95987b7b3753f935f9f43f93ef5224` |
| `src/health/parquet_lite.py` | `f470f95468379135247a45f50d9e6890e5a232ec4e6a2ee745a0a8b1ad52c503` |
| `src/health/protocol.py` | `60c6bc53edd5455c27becf2ab71174144bd523788965904b70e56b5ad84605bf` |
| `src/health/run.py` | `ca4a995857d69fa080ba6a01aac4225c076e69ef0b4150c6f84cf3a5dfad1437` |
| `src/health/scoring.py` | `05d11f230fe66b67abcabdff2064bc1cb72bb8401cbdc042e0515378871111be` |
| `configs/e3b_train.json` | `b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |
| `experiments/W2/0074_lang_battery_hpcgpu06/health/health_base.json`（REPO 里） | `bb66634f24752fdb30ca4fc2a428e02574f8fe0b6ee1bb421bc07ce92d34bfcd` |

## 做法

1. hpcmgt 上 `squeue`。启动前看一次 06 的 nvidia-smi：没有别的计算进程才启动；有，就停下告诉管理员。
2. 建工作树 wt0110（稀疏，含 `src/`、`scripts/`、`configs/`、`prompts/`、`tests/fixtures/health`），核对上表的 sha256。
3. 启动：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 NODE=hpcgpu06 EV_CAP_H=2 \
     bash /home/mail/chenyuma/visint/wt/wt0110/prompts/server_hpc/0110-hpc-xm-health/xm_health.sh \
     > /home/mail/chenyuma/visint/logs/eval_0110.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0110.status`。
   - 推送：一次 ONCE 推送 `experiments/XM/0110_xm_health_hpcgpu06` 即可。
4. 报告 `reports/0110-hpc-xm-health.md`：
   - steps.log；
   - 合并核对的结果；
   - `health_compare.json` 的 rows 表：每项的 base、merged、delta、CI、flag；
   - any_flag。
5. 再跑一次 `squeue`。

管理员在开发机上用 tiny 做了彩排：rc=0，先跑 tiny 基座，再跑 X-M 与比较，any_flag 为 False。NODE 写错时拒绝。正式分支的参数逐条对过 health_check.py 的限制：`--checkpoint` 要 `--model`；`--data-root` 只配 `--checkpoint`；`--compare` 不与跑模型的参数同给。
