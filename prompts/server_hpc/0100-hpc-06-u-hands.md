[RUN-TASK 0100] hpcgpu06：统一主表的两只"手"在 W2 test 的 1,500 句子集上：E1 结构强制多边形（9B 零样本）、9B 的 E3b 像素 token 网格（D154、D159；预注册 U-2）。约 1.2 GPU 小时，上限 3 小时；驱动排在 RUN-0097 之后，现在就发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上在跑的 RUN-0097，驱动自己等它；不碰 07 的 RUN-0098。
- 只用 06 自己的暂停文件。

## 为什么

- 负责人的统一主表（D154）要这两只"手"与其余系统在同样的 1,500 句上，即 `configs/w2_llmseg_subsets.json`。
- 运行器 `scripts/run_u_hands.py`（DEV-0074）原样调用 E1 与 E3 的推理，用 W2 的计分。必须给批准编号 D154 才放行。
- E3b 用 RUN-0056 的 seed 0 检查点。驱动先核对它的 manifest sha256 为 `316d25c7…`，再在 06 上重做一次合并核对。

## 文件（工作树 wt0100 建在 origin/main；稀疏检出要含 `src/unified/`）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0100-hpc-06-u-hands/u_hands.sh` | `c6829504693093249b7474c44207377fb6a48fa4354812b511a213d3ec85cb38` |
| `scripts/run_u_hands.py` | `808a352f2476552f37739810b9064ad37b41697f6798511b95891122c2906b49` |
| `src/unified/hands.py` | `9d6fe1938c920a011af2ce657e7ac7ba0a18db420bd6041d71bbd02ac5b7d11d` |
| `src/e1/posthoc_runner.py` | `e9af5916a409c9d2bdf30388960a740486b75920612a5ab6f4e59c0a04c75d1c` |
| `src/e3/evaluate.py` | `80ccd48b920f0b7694417fd66562d09671186d5f53637d37b22716d90a4115cb` |
| `scripts/eval_e3b.py` | `9820ab447eac5d58f6a601a7ca2e9935569aee29fc3fbc81b3a646c8e076ecbf` |
| `configs/e3b_train.json` | `b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375` |
| `configs/w2_llmseg_subsets.json` | `dfd53ce8e9a89d85ba828de6c0b3e331f83bfcf7d476cfc7ed1751e64c7497ae` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0100（稀疏），核对上表 10 个 sha256。
   - 确认检查点在 `$VI/ckpt/hpcgpu06/0056/e3b/qwen3.5-9b/seed0/ckpt/final/model`。
2. **启动**（驱动等 `eval_0097.status` 以 `done 0097` 开头）：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=3 \
     bash /home/mail/chenyuma/visint/wt/wt0100/prompts/server_hpc/0100-hpc-06-u-hands/u_hands.sh \
     > /home/mail/chenyuma/visint/logs/eval_0100.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0100.status`。推送循环 TAG 0100，路径 `experiments/W2/0100_u_hands_hpcgpu06`。
   - 两个系统的 `--out` 是它下面的 `0100_u_e1poly`、`0100_u_e3bgrid`。
3. 每个系统跑完，贴每个划分的 cIoU、gIoU、failures、`empty_mask.n`。E3b 另贴合并核对的结论。
4. 跑完写 `reports/0100-hpc-06-u-hands.md`，再跑一次 `squeue`。

管理员在开发机上用 tiny 彩排过：E1 用 tiny 模型，E3b 用 `train_e3b --tiny` 的检查点。首次运行与驱动重启两种情形都是 rc=0；目录名不带运行编号时拒绝。
