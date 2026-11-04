[RUN-TASK 0112] hpcgpu07：X-M 与基座零样本在 LVIS 子集上，描述性（D165 第五节；XM-2b，D179）。约 4–5 GPU 小时，上限 8 小时；**排在 RUN-0109 之后**

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 只用 07 自己的暂停文件。

## 为什么

D165 第五节规定，通过后跑 LVIS 子集，只作描述。
- 1,000 张 val2017 图，载入时每图最多取 3 个类别（XM-2b）。上限前 3,134 对，计数写进 config.yaml 的 `xm.lvis_cap`。
- 跑 X-M 的 4 个设置与基座的 2 个设置，同一节点，同 RUN-0109 的写法。
- 与 RUN-0111 的 Rex-Omni lvis_sub 必须是同一个 commit，即本任务书所在的 commit，这样两边是同一批对。

## 文件（工作树 wt0112 建在本任务书所在的 commit）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0112-hpc-07-xm-lvis/xm_lvis.sh` | `5b9a934a7110b0e6388c705980d7bf849570cfa4c68e967c0f74b588709f940d` |
| `scripts/run_xm.py` | `fad0c30c64c72e21051340ea91a601cfd17c1868241aceed42a0059f67c6e65a` |
| `src/xm/runner.py` | `7fdaf81dcd19cf9ff32aca8c1c0e88a7475cff2a7d758c6fbea918f6e25a3180` |
| `src/xm/prompts.py` | `89208d257aa277ac910f23dfb4b40538b5c8f940ef3dc57be464372eb752cb7a` |
| `src/xm/post.py` | `799f3f2cef9f69707d49a1d4a8b8a903af706cc63af5681c4bb16668f757e721` |
| `src/xm/metrics.py` | `448ebb26139d0df2f3555d67108b0316254cda77021ce526105e852b5bd85df9` |
| `src/xm/gate.py` | `dc9bef5506f8247980a76572f05d95c35f4df883685938beba69df982fc54b45` |
| `src/xm/data.py` | `62416435e38e92f5319e79ec629843c37bd67671bb3278038b8a4c18ef651e1d` |
| `configs/xm_splits.json` | `995345f470111e3ac73c9681754dd46eca66da7d9ca0b648873cc1429219db8d` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |
| `experiments/XM/0108_gate_xm_hpcgpu07/0108_gate/gate_xm.json`（REPO 里） | `d2f32d22fcf11acf75e312fd93d7d2390f28c12376fe8b98a05c124ff7dfc8f0` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0112，核对上表 12 个 sha256。
2. 现在就启动，驱动会等 `eval_0109.status` 以 `done 0109` 开头：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=8 \
     bash /home/mail/chenyuma/visint/wt/wt0112/prompts/server_hpc/0112-hpc-07-xm-lvis/xm_lvis.sh \
     > /home/mail/chenyuma/visint/logs/eval_0112.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0112.status`。
   - 推送循环 TAG 0112，路径 `experiments/XM/0112_xm_lvis_hpcgpu07`。
3. 报告 `reports/0112-hpc-07-xm-lvis.md`：
   - 每个设置的 f1_rexomni，检测与实例分割，含 CI；
   - `xm.lvis_cap` 的计数：`n_pairs_before_cap` 必须是 3,134，以及上限后的对数与被去掉的对数；
   - 合并核对、失败数、用时；
   - 只贴，不解读。
4. 再跑一次 `squeue`。

管理员做了 tiny 彩排：首次与重启都 rc=0，重启时续跑。正式分支的命令行与 RUN-0109 逐行相同，只把划分换成 lvis_sub，已与 run_xm 的限制对过：lvis_sub 在 GATED 里，xm_* 与 base_* 都要 `--xm-gate`。
