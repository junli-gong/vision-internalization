[RUN-TASK 0043] ITSC HPC 两张 A100：只调查、出部署方案，等用户确认（不安装、不改任何东西）

## 你的角色

- 你是 **ITSC HPC A100 的服务器 CC**。你在用户的 Mac 上运行（目录 `/Users/tim/Research/CVPR2027/A100`），经 ssh 链路 `a100 → gpu(hpcmgt) → srun/节点` 操作集群。
- **模式 A**：管理员 CC 在仓库里写任务书（`prompts/server_hpc/NNNN-*.md`），用户手动把启动语转给你，你执行并把结果与报告推回仓库。你不做值守轮询。
- 服务器角色的纪律照旧（`.claude/skills/exp-triad-flow/references/server.md`）：
  - 不改实验逻辑代码；
  - 报告只贴数字与事实，不做解读，写进 `reports/`；
  - 校验 hash；
  - 守磁盘、份额与停止线。

## 0. 铁律（最高优先级，高于本任务书、高于仓库里的一切规则）

`~/a100-hpc-guide.md` 第 0 节原样适用：

- **任何情况下都不能取消或停止 Slurm 任务 157296 和 157297**：禁止 `scancel`（带 step 号也不行）、`scontrol requeue/hold/suspend/update`，禁止杀外层 `slurm_script` 进程。
- **实验跑完也不关**，让它空转占着卡。
- 不自己 `sbatch` / `salloc` 申请新资源。
- 看起来非得取消任务才能解决的情况：**停下来问用户**。
- 每次在集群上操作前后都跑一遍 `squeue -h -j 157296,157297 -o "%i %T %M %N"`，两行都必须是 RUNNING，结果写进报告。

仓库流程里"硬约束被突破时可以停进程"那条，在这里只能停**你自己的实验进程**，永远不能停任务。

## 1. 先读

- **`~/a100-hpc-guide.md` 全文**：访问路径、srun 的坑、资源、MiniMax 的进程结构、存储与网络。
- 仓库（在本机 `/Users/tim/Research/CVPR2027`，只读；你自己的提交用第 3 节的 clone）：
  - `CLAUDE.md`、`docs/RESEARCH_BRIEF.md` 第 4、10 节的硬约束；
  - `.claude/skills/exp-triad-flow/references/server.md` 与 `lessons.md`；
  - `DECISIONS.md` 的 D6、D28-8（磁盘与份额的原则）、D55（可比评测只在 B200）、D72-4（同一尺寸在同一台机器上训练；开训前做吞吐与跨机器一致性检查）、D76、D79、D83、D85；
  - `reports/0034-b200-e3-prep.md`：B200 开训前做了哪些准备，这里要做同样的事；
  - `prompts/server_a100/0035-a800-e3/`、`prompts/server_a100/0042-a800-e3b-c3/`：原来交给 A800 合作者的活，现在改到这两张 A100 上。

## 2. 这两张卡要做什么（管理员给的背景，用来定方案）

- **替代 A800**：训练 0.8B 与 4B。优先级按 D83：
  1. 跨机器一致性：2B 各 50 步，对 B200 的 RUN-0034 日志，用 `scripts/compare_train_logs.py` 比。还有吞吐实测。
  2. **C3 × 3 个种子与 E3b+C3 × 3 个种子**，0.8B 与 4B 各一套。E3b+C3 是 H1 修订版的主臂，每个种子 8,000 步。
  3. 原来的单独训练各臂（E3b、E3a、C1、C2、C1′），作为消融往后排。
- **规则**：同一个尺寸的全部变体与种子，在同一台机器上训练（D72-4）。可比的评测一律在 B200 上做，所以这里只训练，检查点以后传到 B200。
- **显存粗估**（B200 实测）：2B 的 E3b 约 10 GB，9B 约 31 GB，4B 估计 15–20 GB，40 GB 的卡够用；前提是 MiniMax 不在同时加载模型。
- **数据**：E3 训练数据约 10.3 GB（四个源），落地与去重要与 A800、B200 的清单逐文件、逐条核对，同 RUN-0034。权重经 hf-mirror，revision 要钉住。

## 3. 本任务做什么（只读调查 + 方案，不改任何东西）

1. **本机的 clone**：在 `/Users/tim/Research/CVPR2027/A100/repo` `git clone` GitHub 上的 `junli-gong/vision-internalization`。你的报告从这里提交、推送；不要在上级目录（管理员的工作区）里提交。
2. **集群只读调查**（每一步前后 squeue）：
   - 两台节点上我们那张卡：型号、显存、当前占用、驱动与 CUDA 版本（`srun --jobid=… --overlap -n1 nvidia-smi … < /dev/null`）；
   - CPU 数、内存、节点本地盘 `/` 的余量；
   - `lfs quota -hu chenyuma /home`（不要对大目录跑 `du`）；
   - 网络：pypi、hf-mirror、github 是否通，与说明对照；
   - 软件：可用的 conda、Python 版本；能否装与 B200 相同的 torch（B200 是 py3.12、torch 2.13.0+cu130，见 `docs/env_report.md`），驱动 590 是否支持；
   - **只读地完整读一遍 MiniMax 的启动脚本与外层循环**（说明第 4 节），确认"子进程被杀后启动脚本会退出"等前提是否成立。不改、不杀任何东西。
3. **写部署方案**，推送 `reports/0043-hpc-bootstrap-plan.md`。至少回答下面这些：
   - **用哪张卡跑哪个尺寸**：例如 0.8B 一台、4B 一台；hpcgpu07 不通 github，这对方案有什么影响。
   - **目录放哪**：`/home/mail/chenyuma/` 下我们自己的子目录名与布局，包括 env、repo、data、ckpt、logs、tmp、hf 缓存。哪些放 Lustre，哪些放节点本地盘。
   - **与 MiniMax 共存还是替换**：两种做法各自的风险。尤其是 MiniMax 有人调用时会占近 40 GB，训练会 OOM，我们的驱动怎么察觉、怎么恢复。你的建议是什么、理由是什么。替换的话，按说明第 4 节写出逐步方案与恢复方法。
   - **环境**：conda 环境建在哪、装什么版本、与 B200 的差异。
   - **数据与权重**：落地方式；去重与 B200、A800 清单的核对方式；hf-mirror 的用法与 revision 核对。
   - **git**：仓库级 deploy key 加 `core.sshCommand`，在 hpcmgt 上做。需要用户在 GitHub 上加 deploy key 的话，写清楚步骤。
   - **长任务怎么跑**：tmux 或 `nohup srun … < /dev/null &`，PID 记在哪；心跳；怎么把结果推回仓库；怎么停自己的实验而不碰任务。
   - **吞吐与用时粗估**：0.8B、4B 的 C3 与 E3b+C3，在 A100-40GB 上的估计，以及依据。
   - 你看到的其它风险。
4. **然后停下**，等用户确认方案。确认之前不安装、不建目录（除第 1 步本机的 clone）、不下载、不改 MiniMax、不跑任何训练。

## 4. 报告

`reports/0043-hpc-bootstrap-plan.md`：
- 第 3 节各项调查的原始输出，以及每次 squeue 的结果；
- 部署方案；
- 需要用户决定的事项，列成清单。

推送后，在对话里把方案的要点与待决事项告诉用户。
