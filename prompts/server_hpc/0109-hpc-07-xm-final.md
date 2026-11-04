[RUN-TASK 0109] hpcgpu07：G-XM 通过后的报告运行，跑三项（D165 第五节；D177）：X-M 在最终报告集上、基座零样本在最终报告集上、X-M 在 ReasonSeg val 上。约 7 GPU 小时，上限 10 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上 lly-mle 的进程。
- 只用 07 自己的暂停文件。

## 为什么

G-XM 通过（RUN-0108）。D165 第五节规定，四条全部满足后，要跑最终报告集、LVIS 子集、ReasonSeg 和 RefCOCO 的 test 子集。本任务跑其中不需要新代码的两项，LVIS 与 test 子集要先改代码。推理设置一律不改（XM-4b）。

**最终报告集**（`xm_final`，1,000 张图、2,940 对）：
- X-M 的 4 个设置：xm_spline_trimap、xm_box_grabcut、xm_ellipse_trimap、xm_spline_raster；
- 基座零样本 base_det、base_box_grabcut。

两者都要 `--xm-gate` 放行，用的是 RUN-0108 的 `gate_xm.json`。X-M 的设置另外核对检查点的 manifest。

**ReasonSeg val**：
- X-M 的检查点，走 X-2 的协议（`--system xm`），缺省设置 x2_spline_trimap；
- 与 RUN-0105 的 X-2 同节点，只作描述。

## 文件（工作树 wt0109 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0109-hpc-07-xm-final/xm_final.sh` | `a96ee54a52cb4aa93d45520f22814445d859e64b03ddeb2edc39eb140774b419` |
| `scripts/run_xm.py` | `9a8cb1047db3e133b3970404d1f9c85b905818fd4755e85c6f885ea836632802` |
| `src/xm/runner.py` | `5551d9cf430f74d3046b5aa81366db53fd96ae6bb2f06af3abb5644b6691f8d9` |
| `src/xm/prompts.py` | `89208d257aa277ac910f23dfb4b40538b5c8f940ef3dc57be464372eb752cb7a` |
| `src/xm/post.py` | `799f3f2cef9f69707d49a1d4a8b8a903af706cc63af5681c4bb16668f757e721` |
| `src/xm/metrics.py` | `448ebb26139d0df2f3555d67108b0316254cda77021ce526105e852b5bd85df9` |
| `src/xm/gate.py` | `dc9bef5506f8247980a76572f05d95c35f4df883685938beba69df982fc54b45` |
| `src/xm/data.py` | `f51759702f8dc5b67c4775859df97d64334fb0b9df03e5c7a430c01ba7a73870` |
| `configs/xm_splits.json` | `995345f470111e3ac73c9681754dd46eca66da7d9ca0b648873cc1429219db8d` |
| `configs/w2.json` | `c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29` |
| `configs/x2_run.json` | `c92ad838a9e698236e8a34d17f6ce187ee185ffaa7df96df5ee99a04f47dc061` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |
| `experiments/XM/0108_gate_xm_hpcgpu07/0108_gate/gate_xm.json`（REPO 里，不在工作树里） | `d2f32d22fcf11acf75e312fd93d7d2390f28c12376fe8b98a05c124ff7dfc8f0` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0109（稀疏，含 `src/`、`scripts/`、`configs/`、`prompts/`），核对上表 13 个 sha256。
2. 启动：
   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=10 \
     bash /home/mail/chenyuma/visint/wt/wt0109/prompts/server_hpc/0109-hpc-07-xm-final/xm_final.sh \
     > /home/mail/chenyuma/visint/logs/eval_0109.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `eval_0109.status`。
   - 推送循环 TAG 0109，路径 `experiments/XM/0109_xm_final_hpcgpu07`。
   - 某一步失败时，用 `STAGES` 只重跑那几步，可选值 `xm`、`base`、`reasonseg`；已完成的组会自动续跑。
3. 最终报告集与 ReasonSeg 都已揭盲，读数可以报：
   - 每个设置的 f1_rexomni，检测与实例分割，含 CI；
   - ReasonSeg 的 gIoU、cIoU；
   - 合并核对的结果、失败数、用时。
4. 跑完写 `reports/0109-hpc-07-xm-final.md`，再跑一次 `squeue`。

管理员在开发机上用 tiny 做了彩排：首次与重启都 rc=0，重启时续跑。tiny 只有 xm_tune 的合成对，不带 gate；ReasonSeg 只写出命令行，核对过与 RUN-0105 那一行相同，只把 `--system` 换成 xm，并加上 `--xm-gate`。

正式分支的参数逐条对过 run_xm.py 的限制：
- `--split xm_final` 与 `reasonseg_val --system xm` 都要 `--xm-gate`（src/xm/gate.needs_xm_gate）；
- 不能给 `--gate`，那只配密封划分；
- xm_* 带检查点与 `--merge-check-data-root`（D176）；
- base_* 不带检查点，gate 文件不比 manifest；
- ReasonSeg 不给 `--setting`，取缺省值。
