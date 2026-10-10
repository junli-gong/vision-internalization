[RUN-TASK 0044] ITSC HPC 两张 A100：按批准的方案做开训前准备——停 MiniMax、建环境、deploy key、数据与权重落地并核对、跨机器一致性、吞吐实测

## 为什么做

- 用户批准了 RUN-0043 的方案，全部按建议（D89）。原文见 `DECISIONS.md` 的 D88、D89，方案细节见 `reports/0043-hpc-bootstrap-plan.md` 第 3 节。
- 本任务把那一节的方案执行到"可以开训"为止。**不开正式训练**；正式训练由下一份任务书发。

## 0. 铁律（最高优先级）

`~/a100-hpc-guide.md` 第 0 节原样适用：

- **任何情况下都不能取消或停止 Slurm 任务 157296 和 157297**，实验跑完也不关；
- 不 `scancel`，不 `scontrol requeue/hold/suspend/update`，不杀 `slurm_script`；
- 不自己申请资源。
- 每一步前后都 `squeue -h -j 157296,157297 -o "%i %T %M %N"`，两行都必须 RUNNING。不是就立即停手，问用户。
- 只能停**我们自己的**进程。

## 1. 先读

- `~/a100-hpc-guide.md` 全文；`reports/0043-hpc-bootstrap-plan.md` 全文。
- `DECISIONS.md` 的 D72-4、D78-4（一致性判据）、D85、D86、D88、D89。
- `reports/0034-b200-e3-prep.md`：B200 的同一套准备；`experiments/E3/0034_b200_prep/probe_scripts/`：比对脚本要原样复用。
- `scripts/compare_train_logs.py --help`、`scripts/data/land_train.py --help`。
- `.claude/skills/exp-triad-flow/references/lessons.md`：环境照抄已验证的那份；两个 `conda run` 不能同时跑；正在运行的 bash 脚本不要改；长驱动与 CC 脱钩。

## 2. 目录与环境（RUN-0043 §3.2、§3.4）

- 按 §3.2 建 `VI=/home/mail/chenyuma/visint`，以及它下面的子目录。`.ssh/` 设成 700。
- **Lustre 份额上限 80 GB**（D89-7）：每个阶段之后用 `lfs quota -hu chenyuma /home` 记一次；不要对大目录 `du`。
- 按 §3.4 建 `$VI/env`：py3.12、torch 2.13.0+cu130 等，版本照 B200 逐个钉住；可选加速包与 B200 一致（不装 flash-attn、causal-conv1d）。
  - `pip freeze` 与 B200 的对照差异列进报告。
- **全量测试**：在 hpcgpu06 上经 `srun --jobid=157296 --overlap … < /dev/null` 跑普通与 server-parity 两套。
  - 各自 `--basetemp`，不用 `conda run`，直接调用 env 里的 python；
  - 对照 D87 的 2743 passed + 18 skipped。
  - 计时类测试（`tests/test_synth_*::test_speed_*`）在共享节点上可能失败；只有这类失败时，单独重跑确认，并写进报告。

## 3. git 与 deploy key（RUN-0043 §3.6，D89-4）

- 在 hpcmgt 上生成 `$VI/.ssh/id_ed25519_github`，文件 600。
- **公钥加到 GitHub**：用户已同意。在本机 Mac 上执行：

  ```
  gh repo deploy-key add <公钥文件> --repo junli-gong/vision-internalization --allow-write --title "itsc-hpc-a100 (chenyuma@hpcmgt)"
  ```

  公钥文件可以先从 hpcmgt 读回本机的临时目录。只加这一把，别的设置不动。
- 在 hpcmgt 上用 ssh 地址 clone 到 `$VI/repo`。仓库级 `core.sshCommand` 与 `user.name hpc-a100-server-cc`，不碰 `~/.gitconfig`、`~/.ssh`。
- 试推一次：在 `experiments/E3/0044_hpc_bootstrap/` 里写一个 README 推上来，确认通路。
- 推送一律 `flock $VI/logs/git.lock`，先 `pull --rebase` 再 push。

## 4. 训练数据与权重（RUN-0043 §3.5，D89-2、D89-3）

- **nyu_depth_train**：在 hpcgpu06 上经 srun，用 `land_train.py nyu_depth_train --hf-base https://hf-mirror.com --root $VI/data/train_data …` 原生落地。
- **refcoco_train、ade20k、nyu_normals 与 `dedupe/`**：从 B200 的 `/dev/shm/visint/train_data/` 经 Mac 中转。命令照 §3.5：`ssh b200 tar c … | ssh a100 'ssh gpu "tar x -C …"'`。
  - **对 B200 只读**，不在 B200 上写任何东西；
  - **不用 `ssh -A`**。
  - 中转期间并行做第 2、3 节。
- **核对**（原样复用 RUN-0034 的 `p33_manifest_compare.py`、`p34_exclusions_compare.py`）：
  - 四个源的 MANIFEST 逐文件 sha256，与 B200（`experiments/E3/0034_b200_prep/land_manifests/`）、A800 两份清单比，给出"相同 / 不同的文件数"。中转来的三个源，从盘上的文件重新算 sha256，不采信传过来的 MANIFEST。
  - `dedupe/train_exclusions.json` 与 B200、A800 的 sha256 与逐条记录相同。
  - **任何一项不同就停下报告**，不进第 6 节。
- E1 的 refcoco 与 nyuv2 不需要：这里只训练，不评测。
- **权重**：0.8B、2B、4B 经 hf-mirror，revision 钉 `docs/model_notes.md` 的值，下载后逐文件核 LFS sha256。只在一个节点上下载，不并发。

## 5. MiniMax 停车（RUN-0043 §3.3 的逐步方案，D89-1）

**在第 2–4 节完成、要用 GPU 之前做。**

- 严格按 §3.3 的第 1–7 步做：
  - 先只读核对脚本的 sha256 与 mtime 没变，并记下 PID 与进程树；
  - 备份 `.orig`，核对 sha256 相同；
  - 写新文件，`bash -n` 检查语法；
  - mv 原子替换，核对 inode；
  - 先 06：touch 开关文件，按 PID kill uvicorn、comfy，确认 slurm_script 还在，等 300 s 停在 `sleep infinity`；
  - 再 07。
- **每一小步前后都 squeue。**
- 任何一步与预期不同（PID 变了、slurm_script 不见了、squeue 不是 RUNNING）：立即停手，问用户。
- 两台停好之后，nvidia-smi 上我们那张卡没有任何进程。恢复方法照 §3.3 第 8 步写进报告。

## 6. 启动方式的测试（D89-6）

- 方案 B：`ssh -n hpcgpu06` 后，在节点上 `setsid nohup srun --jobid=157296 --overlap -n1 sleep 60 < /dev/null &`。
  - 确认它在任务的资源范围里：`CUDA_VISIBLE_DEVICES`、可用核数；
  - 确认它自己结束后任务仍是 RUNNING。
- 测通就用方案 B 启动后面的长运行；不通就用方案 A（hpcmgt 上的 setsid nohup srun），并把风险写进报告。

## 7. 跨机器一致性（D72-4、D88）

- **在固定于 `6e77ab0` 的 worktree 上跑**：这是 RUN-0034 用的 commit。之后 E3b、E3a 的配置版本变了，`compare_train_logs.py` 会先核对运行身份，版本不同就拒比。
- 两台节点各跑一遍，命令与 RUN-0034 §5 的一致性两条相同。只换 `--data-root $VI/data/train_data`，`--out` 用 `$VI/repo/experiments/E3/0044_hpc_consistency_<node>`，带运行编号：

  ```
  python scripts/train_e3b.py --model qwen3.5-2b --supervision grid --seed 0 --stop-after-steps 50 --resume …
  python scripts/train_e3a.py --model qwen3.5-2b --variant full --seed 0 --stop-after-steps 50 --resume …
  ```

- 环境块照 RUN-0043 §3.7，一行一个 export，含 `OMP_NUM_THREADS=16`、`PYTHONHASHSEED=0`。
- 比对：用 `scripts/compare_train_logs.py`（当前 commit 的版本），把每台的 `train_log.jsonl` 与 B200 的 `experiments/E3/0034_b200_prep/0034_e3_consistency/` 对应文件比；06 与 07 也互比。
- 判据是 D78-4 的暂定 v1：前 50 步每步相对差 ≤ 2%、中位数 ≤ 0.5%，最后 10 步平均 ≤ 1%，没有 NaN、Inf。
- **不通过就停下报告**，不跑第 8 节，也不开训。

## 8. 吞吐实测（当前 commit 的代码）

- hpcgpu07：0.8B 的 `--supervision text`（C3）与 `--supervision grid+text`（E3b+C3），各 `--stop-after-steps 20`。
- hpcgpu06：4B 的同样两条。
- 另做一组 A/B（RUN-0043 §3.2）：同一条 20 步命令，`--data-root` 分别指向 Lustre 与 `/dev/shm/visint-chenyuma/` 的副本，看 Lustre 是否拖慢。
  - 只有 Lustre 明显慢，才在正式训练里改用 `/dev/shm`。
  - 用完删掉 `/dev/shm` 的副本。
- 每条记：每步中位耗时，按 `logs/timing_run_*.jsonl` 的 `step_s`、`build_inputs_s`、`fwd_s`、`bwd_s`；`accounting.json` 的显存峰值；同期的节点负载。
- 检查点写在单独的目录，跑完删掉。

## 9. 停止线

- 铁律第 0 节。
- 节点 MemAvailable < 100 GB，或我们的进程 RSS 合计 > 150 GB：停自己的进程，写 `blocked`。
- Lustre 份额 > 80 GB：停。
- **预计合计约半天**：中转 4–5 小时与环境、权重并行，一致性与吞吐每台约 1 小时。超过 1 天还没完成，先推送进度报告，说明卡在哪里。

## 10. 报告

`reports/0044-hpc-bootstrap.md`，只贴数字与事实：

- 每次 squeue 的结果；
- 第 2–8 节各项的原始输出或摘要，原始文件放 `experiments/E3/0044_hpc_bootstrap/`；
- 数据与去重的核对表：每个源"相同 / 不同的文件数"；
- 全量测试结果；
- MiniMax 停车前后的进程树与恢复方法；
- 启动方式的测试结论；
- **一致性比对的判定**：每台对 B200，06 对 07；
- **吞吐表与正式训练的逐项预估**（D81-2）：0.8B、4B 的 C3 与 E3b+C3，各 3 个种子；
- Lustre 份额；
- 遇到的与方案不符之处。

推送后，在对话里告诉用户要点。
