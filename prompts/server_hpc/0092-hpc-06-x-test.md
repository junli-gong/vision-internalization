[RUN-TASK 0092] hpcgpu06：通过 G-EX 的两个候选在 W2 的 test 上运行（预注册 X-6、X-6a；D147、D148）。约 9–12 GPU 小时，上限 14 小时。现在就发，驱动自动排在 RUN-0091 之后

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 RUN-0091。
- 只用 06 自己的暂停文件。

## 为什么

- G-EX（RUN-0090）：两个候选都通过（`experiments/X/0090_gex_hpcgpu07/gate_x2/`、`gate_x3/`）。
- 负责人批准（D147）：通过的都跑与 W2-c 相同的 test，C1（X-2）为主方法，C2（X-3）作消融。
- 同一次运行导出能力归属的消融 `x2_ellipse_trimap`。
- 与 RUN-0072 在 06 上的 W2-c 逐句配对。

## 文件（工作树 wt0092 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0092-hpc-06-x-test/x_test.sh` | `8eeec087757ae34a79d24ba6f58f6bcdef7a35d95cd5779e049b95e2fc3b512c` |
| `scripts/run_x2.py` | `08f9962bea2ae6b60a70ea0940a9bbf53581ead9bf2ef76e6b9c24042d577bcf` |
| `src/explore/x2_runner.py` | `1ef0f768e11bbde99b569634ef47fc1d386f84a3fe64e3da62788d54ca64eff7` |
| `src/w2/runner.py` | `35b0b0a7370aa39752a2a790825630856b9b8a76234a5368661ee27a229b08b7` |
| `src/explore/x2_prompts.py` | `80097f709bbae6e80a60d6d2b0c7412430ded1e2750cd72e3a650696dee84d3b` |
| `src/explore/x2_tools.py` | `1055192ce9e2e2b259533639d7f7dab39b01c5d30875bbb06a0564127d9e343b` |
| `src/explore/guidance.py` | `40c52ebc0cd6eabe0e6d51c9f6906530fb8fb4a0c5a63b8cef5d912de7ee5ef5` |
| `configs/x2_run.json` | `c92ad838a9e698236e8a34d17f6ce187ee185ffaa7df96df5ee99a04f47dc061` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

## 候选（`$R = $VI/repo/experiments/X`）

| 名 | 检查点（manifest） | gate_explore.json |
|---|---|---|
| x2（C1） | `$R/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`（`a49cf4e5…`） | `$R/0090_gex_hpcgpu07/gate_x2/gate_explore.json` |
| x3（C2） | `$R/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`（`837bb679…`） | `$R/0090_gex_hpcgpu07/gate_x3/gate_explore.json` |

- `run_x2` 自己核对 gate：结论通过、非彩排、判定点全真、检查点与设置一致；还要批准编号 D147。不合格就拒跑。
- 输出：`$R/0092_x2_test_hpcgpu06`、`$R/0092_x3_test_hpcgpu06`。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0092（稀疏），建在 origin/main，核对上表 11 个 sha256 与两个 manifest。
2. **启动**：把 CANDS 写进 `$VI/tmp/start0092.sh`，`cat` 核对路径都存在：
   ```bash
   CANDS="x2|$R/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model|$R/0090_gex_hpcgpu07/gate_x2/gate_explore.json x3|$R/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model|$R/0090_gex_hpcgpu07/gate_x3/gate_explore.json"
   ssh -n hpcgpu06 "setsid nohup srun --jobid=157296 --overlap -n1 -c 8 env PATH=$VI/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=14 \
     CANDS='$CANDS' bash $VI/wt/wt0092/prompts/server_hpc/0092-hpc-06-x-test/x_test.sh \
     > $VI/logs/eval_0092.driver.log 2>&1 < /dev/null &"
   ```
   - 驱动先等 `eval_0091.status` 以 `done 0091` 开头，再 `claim_gpu`。
   - 心跳在 `eval_0092.status`。推送循环 TAG 0092：两个输出目录加 `0092_x_test_hpcgpu06`。
3. 每个划分跑完，报两个设置（`x2_spline_trimap`、`x2_ellipse_trimap`）的 cIoU、gIoU 与 CI，以及解析失败率。
4. 跑完写 `reports/0092-hpc-06-x-test.md`，再跑一次 `squeue`。只贴数字，不解读。
