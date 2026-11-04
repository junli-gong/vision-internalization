[RUN-TASK 0111] hpcgpu06：Rex-Omni 外部参照（只做检测），以及 X-M 在 RefCOCO 的 W2 test 子集上（D165 第四、五节；D177–D179）。约 8–9 GPU 小时，上限 14 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰别的项目的进程。
- 只用 06 自己的暂停文件。

## 为什么

**A. Rex-Omni（D178）。** D165 第四节点名的外部参照，许可证允许（IDEA License 1.0，非商业研究）。它是学习型检测器，单独运行，输出不进入我们的任何数据或训练。两个设置：
- `rexomni_det_joint`：它自己的协议，每张图问一次，列出全部正类别。这是标题读数。
- `rexomni_det`：与 X-M 同协议，每个（图, 类别）问一次。只作诊断。

顺序：
1. `xm_tune` 冒烟：joint 的 `f1_rexomni_exact` 低于 0.60 时，run_rexomni 退出码 3，驱动跳过后两步。
2. `xm_final`。
3. `lvis_sub`：每图最多 3 类。与 RUN-0112 的 X-M 必须是同一个 commit，即本任务书所在的 commit。

**B. X-M 在 W2 test 的 5 个划分上（D165 第五节、D179）。**
- `run_x2 --xm-gate <RUN-0108 的 gate_xm.json> --test-approval D165`，设置同 RUN-0092：x2_spline_trimap、x2_ellipse_trimap。
- 与 RUN-0092 的 X-2 同在 06 上，逐句配对。test 只用这一次。

## 文件（工作树 wt0111 建在本任务书所在的 commit，稀疏签出 `src/`、`scripts/`、`configs/`、`prompts/`）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0111-hpc-06-rexomni-xmtest/rexomni_xmtest.sh` | `3216cf290d63e1e2726d1de4946efe6c86b6ceb6a8a240c1fc03cd6569dbf507` |
| `scripts/run_rexomni.py` | `26ccd7dd1a86e54836f76748e4718ee2cad08655049f570730b73f6204a27a74` |
| `src/rexomni/__init__.py` | `b72d89ed246a137ccc29da4c7278197353e6f9292d29853fe26fadb2f8acc0e9` |
| `src/rexomni/protocol.py` | `ccb8bea4fa9498a68ad78c91ec47f7afd4367d80ecb99c21df9a738ff538cc60` |
| `src/rexomni/model.py` | `5982ec85b3332145daba0ec329b5e70c4475993f27156a32bb7ceaa39df57d7a` |
| `src/rexomni/runner.py` | `25426ebd816d58c6a0ab9ca035e81760beae976274288d6149f1e19bbdb2ee85` |
| `configs/rexomni_weights.json` | `a4c77e863d74010ea02e1feec50965ca237648decb7735ec6c710e3634752aae` |
| `scripts/run_x2.py` | `1e76c41f40a799387f064de33c696b802c5a6d6b30cfd3c97bb38a9d05c6e86e` |
| `src/xm/gate.py` | `dc9bef5506f8247980a76572f05d95c35f4df883685938beba69df982fc54b45` |
| `src/xm/data.py` | `62416435e38e92f5319e79ec629843c37bd67671bb3278038b8a4c18ef651e1d` |
| `src/xm/metrics.py` | `448ebb26139d0df2f3555d67108b0316254cda77021ce526105e852b5bd85df9` |
| `configs/xm_splits.json` | `995345f470111e3ac73c9681754dd46eca66da7d9ca0b648873cc1429219db8d` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |
| `configs/x2_run.json` | `c92ad838a9e698236e8a34d17f6ce187ee185ffaa7df96df5ee99a04f47dc061` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |
| `experiments/XM/0108_gate_xm_hpcgpu07/0108_gate/gate_xm.json`（REPO 里） | `d2f32d22fcf11acf75e312fd93d7d2390f28c12376fe8b98a05c124ff7dfc8f0` |

## 做法

1. hpcmgt 上 `squeue`。
2. Rex-Omni 的权重已按先前的消息下载，`weights --verify` 通过。没下载完就先做完那一步。
3. 启动前看一次 06 的 nvidia-smi：有别的计算进程就停下，告诉管理员。
4. 建工作树 wt0111，核对上表 17 个 sha256。
5. 启动：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=14 \
     bash /home/mail/chenyuma/visint/wt/wt0111/prompts/server_hpc/0111-hpc-06-rexomni-xmtest/rexomni_xmtest.sh \
     > /home/mail/chenyuma/visint/logs/eval_0111.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0111.status`。
   - 推送循环 TAG 0111，路径 `experiments/XM/0111_rexomni_xmtest_hpcgpu06`。
   - 重跑某一部分用 `STAGES=rexomni` 或 `STAGES=xmtest`，会自动续跑。
6. 报告 `reports/0111-hpc-06-rexomni-xmtest.md`：
   - **Rex-Omni**：每个划分、每个设置的检测 `f1_rexomni`（含 CI）与 `f1_rexomni_exact`；冒烟线的结果；解析失败、截断、未问类名的框数；每对的 token 数与用时；权重核对。
   - **X-M 的 test**：每个划分的 cIoU、gIoU，两个设置；合并核对；放行记录 `w2.test_unlock`。
   - 只贴，不解读。
7. 再跑一次 `squeue`。

管理员在开发机上做了 tiny 彩排：首次与重启都 rc=0。Rex-Omni 跑了 tiny 的 xm_tune；xm_final、lvis_sub 与 test 只写出命令行，逐条核对过与 DEV-0082、DEV-0083 的总结及 RUN-0092 的写法相同：
- run_x2：`--data-root $VI/b200/data`，CW、TW 为 8，按叶目录续跑；
- run_rexomni：xm_tune 不带 gate，其余带 `--xm-gate`，目录名含 `_rexomni`。
