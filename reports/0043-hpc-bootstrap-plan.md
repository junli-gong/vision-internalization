# RUN-0043 — ITSC HPC 两张 A100-40GB：只读调查与部署方案（等用户确认）

服务器 CC（HPC），在用户的 Mac 上经 `ssh a100 → gpu(hpcmgt) → srun --overlap` 操作。2026-10-10 05:20–06:00 UTC。
调查部分只贴数字与事实；§3 起是方案，方案是任务书要求的提议，不是对实验结果的解读。

## 0. 铁律与改动范围

- **没有** `scancel`、`scontrol`（只用了 `scontrol show`）、`sbatch`、`salloc`，没有杀任何进程，没有改 MiniMax 的任何文件。
- 每一批操作前后都跑了 `squeue -h -j 157296,157297 -o "%i %T %M %N"`，**全部是两行 RUNNING**，见 §1.1。
- 集群上没有建目录、没有安装、没有下载落盘。留下的痕迹只有：
  - 若干个 `srun --overlap` 步骤（step 号 114–120 与 64–69 之类，结束即消失）；
  - hpcmgt 的 `/tmp` 里写过一个 1 KB 的网络测试脚本，同一条命令里已删除；
  - 测速用的 `curl -r` 都写到 `/dev/null`。
- 本机只做了一件事：在 `A100/repo` clone 仓库（任务书 §3.1）。

## 1. 调查的原始输出

原始输出全文在 `experiments/E3/0043_hpc_survey/raw/`。下面摘录要用的部分，数字照原文。

| 文件 | 内容 |
|---|---|
| `s1_nodes.txt` | 两张卡的 nvidia-smi、CPU、内存、磁盘、OS（经 srun --overlap） |
| `s2_login.txt` | 配额、conda、pip 配置、Slurm 分区与抢占设置、节点上的其他任务。chenyuma 家目录的列表已删去 |
| `s3_minimax.txt` | MiniMax 的 sbatch、启动脚本、env、run/ 目录、守护脚本、crontab、slurm 日志尾部，原样 |
| `r_net.txt` | 三处（hpcmgt、hpcgpu06、hpcgpu07）的网络可达性、torch 轮子、hf-mirror 测速、数据源主机测速 |
| `r_ps.txt` | 两个节点上 MiniMax 的进程树、API 与 ComfyUI 日志、步骤的 cgroup 限制 |
| `r_relay.txt` | 中转链路测速（B200 → Mac → lab5 → hpcmgt），B200 上训练数据的体积 |
| `b200_pip_versions.txt` | B200 venv 的包版本（只读 `pip freeze`），作为本机环境的目标 |

### 1.1 squeue（每次操作前后）

所有记录都是下面这个形式，任务的运行时长单调增长，没有出现非 RUNNING：

```
## squeue 2026-10-10T05:20:28Z [before]
157296 RUNNING 55-16:46:14 hpcgpu06
157297 RUNNING 57-14:18:23 hpcgpu07
...
157296 RUNNING 55-17:12:56 hpcgpu06
157297 RUNNING 57-14:45:05 hpcgpu07
```

逐次记录在各个 raw 文件的开头与结尾。

### 1.2 卡、CPU、内存、磁盘（`s1_nodes.txt`）

| | hpcgpu06（157296） | hpcgpu07（157297） |
|---|---|---|
| 步骤里看到的卡 | 1 张，`CUDA_VISIBLE_DEVICES=0` | 1 张，`CUDA_VISIBLE_DEVICES=0` |
| 型号 / 显存 | NVIDIA A100-PCIE-40GB，40960 MiB | 同左 |
| PCI / compute cap | 00000000:65:00.0 / 8.0 | 同左 |
| 当前占用 | 423 MiB，GPU-Util 0%，46 W / 250 W | 423 MiB，0%，48 W / 250 W |
| 卡上进程 | comfy-env python，416 MiB | 同左 |
| 驱动 / CUDA | 590.48.01 / 13.1 | 同左 |
| 步骤可用 CPU | `nproc` = 16（cpuset 0,2,6,8,…,44），整机 48 核 Xeon Gold 6342，2 路 | 16（cpuset 0,2,4,…,44），同型号 |
| 内存 | 整机 503 GiB，available 482 | 整机 503 GiB，available 478 |
| 内存的 cgroup 限制 | **没有**：memory 控制器挂在 `system.slice/slurmd.service`，limit = 9223372036854771712 | 同左 |
| 本地盘 `/` | 414G，可用 283G | 414G，可用 287G |
| `/dev/shm` | 252G，可用 245G | 252G，可用 245G |
| `/home`（Lustre） | 100T，已用 73T，可用 23T | 同一个文件系统 |
| OS | Rocky Linux 8.7，kernel 4.18.0-425，glibc 2.28 | 同左 |
| 负载 | load average 0.06 | load average 2.20 |

- hpcmgt 也是 Rocky Linux 8.7；`uptime` 是 14 天 17 小时。**两个任务都已运行 55、57 天，所以 hpcmgt 在任务运行期间重启过。**
- `lfs quota -hu chenyuma /home`：已用 6.575T，quota 与 limit 都是 0k（未设限）；组 domain users 已用 72.07T，同样未设限。

### 1.3 Slurm（`s2_login.txt`）

- 分区 gpu：`PreemptMode=OFF`，`PreemptType=preempt/none`，MaxTime UNLIMITED。
- `JobRequeue=1`；两个任务都是 `Requeue=1`（sbatch 里写了 `--requeue`）。
- `ProctrackType=proctrack/cgroup`，`TaskPlugin=task/cgroup`，`SelectTypeParameters=CR_CPU_MEMORY`。
- 两个任务的步骤：`157296.batch/extern`；`157297.batch/extern` 以及 **`157297.34`（名为 env，已运行 45 天）**。
  - 它在 hpcgpu07 上是 `python3 -u -`，PID 3102924，PWD 为 `/home/mail/chenyuma`，没有打开 nvidia 设备文件（fd 计数 0）。
  - 不是我们的，没有碰。
- 同一节点上的其他用户任务：zhanchen 两个（各 1 张 A100，45 天）、u430202701 一个（2 卡）、r130201615 两个（各 1 卡）。
  - 两个节点的 GPU 都已全部分配（AllocTRES gres/gpu=4）。

### 1.4 网络（`r_net.txt` 与本节的第一遍测试）

| 目标 | hpcmgt | hpcgpu06 | hpcgpu07 |
|---|---|---|---|
| pypi.org、清华 pypi 镜像 | 200 | 200 | 200 |
| download.pytorch.org/whl/cu130 | 200 | 200 | 200 |
| hf-mirror.com API | 第一遍 200，第二、三遍 15 s 超时 | 200（第三遍用了 10.4 s） | 200 |
| huggingface.co | 000 | 000 | 000 |
| github.com（https） | 200（1.3 s / 9.8 s / 0.8 s） | 第一遍超时（15 s）；第三遍 200，用了 15.0 s | 200（15.0 s / 7.5 s） |
| github.com:22（ssh） | 能连上，`Permission denied (publickey)` | 同左 | 同左 |
| Google / Drive / web.archive.org | 000 | 000 | 000 |
| bvisionweb1.cs.unc.edu（refer 官方） | — | 000 | 000 |
| raw.githubusercontent.com（BTS 清单） | — | 第一遍 200，第三遍 20 s 超时 | 200 |
| images.cocodataset.org | — | 200，Range 下载 **约 28–30 KB/s** | 同左 |
| data.csail.mit.edu（ADE20K） | — | 200，**约 21 KB/s** | 200 |
| horatio.cs.nyu.edu（NYU labeled .mat） | — | 200，**约 7 KB/s** | 200 |
| hf-mirror 大文件（Range 200 MB，两遍） | 2.36 / 2.86 MB/s | 3.90 / 5.39 MB/s | 2.72 / 2.65 MB/s |

- hf-mirror 对 `Qwen/Qwen3.5-0.8B` main 返回 `x-repo-commit: 2fc06364715b967f1860aea9cf38778875588b17`，与 `docs/model_notes.md` 钉住的 revision 相同。
- 第一遍的数字见本节；第二遍（第三次测）原文在 `r_net.txt`，表里的斜杠分开各遍。github 的 https 在三处都不稳定，ssh 的 22 端口三处都能连上。

### 1.5 中转链路（`r_relay.txt`）

| 段 | 测法 | 结果 |
|---|---|---|
| B200 → Mac | 100 MB，两遍 | 15.7 s / 9.7 s，约 6.7 / 10.4 MB/s |
| Mac → lab5 | 50 MB，两遍 | 119 s / 135 s，**约 0.42 / 0.37 MB/s** |
| Mac → lab5，两路并行 | 2 × 25 MB | 78.7 s，合计约 0.64 MB/s |
| lab5 → hpcmgt | 50 MB，两遍 | 1.2 s / 1.2 s，约 40 MB/s |
| Mac → hpcmgt（两跳） | 100 MB | 222 s，约 0.45 MB/s |
| lab5 自己的外网 | COCO 24 / 87 KB/s；hf-mirror 3.2 MB/s / 第二遍 000；Drive 000 | |
| B200 → pi.machenyu.cn:8889（lab5 的入口） | TCP 连接 | open |

B200 上的训练数据（`/dev/shm/visint/train_data`，du apparent）：refcoco_train 4.0G、nyu_depth_train 3.8G、nyu_normals 964M、ade20k 952M、dedupe 212K；`refcoco`、`nyuv2` 是指向 `$PROJ/data` 的链接（训练不读，评测读）。

### 1.6 软件（`s2_login.txt`、`r_net.txt`、`b200_pip_versions.txt`）

- **conda：** `/home/mail/chenyuma/soft/miniconda3`，conda 24.11.3，base 的 Python 是 3.11.8。
  - channels：defaults，经 `~/.condarc` 指到北外镜像（mirrors.bfsu.edu.cn）。
  - `pkgs_dirs` 的第一项是 base 的 `miniconda3/pkgs`。
- **pip：** `~/.pip/pip.conf` 的 index-url 是清华镜像。
- **其他工具：** git 2.50.1、tmux 2.7；系统 python3 就是 conda base 的 3.11.8。
- **torch：** download.pytorch.org 上有 `torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl`。
  - 清华镜像上 torch 2.13.0 的 cp312 轮子是不带 `+cu130` 的那一版。
  - 驱动 590.48.01 报 CUDA 13.1，不低于 cu130 要求的 13.0；glibc 2.28 满足 manylinux_2_28。
  - B200 的 torch 2.13.0+cu130 的 arch list 含 sm_80（`docs/env_report.md` §1）。
  - 以上只是"能装"的条件核对，**没有实际安装与运行**。
- **B200 venv 的目标版本**（只读）：
  - Python 3.12.13；
  - torch 2.13.0+cu130、torchvision 0.28.0+cu130、triton 3.7.1；
  - transformers 5.17.0、peft 0.21.1、accelerate 1.15.0；
  - flash-linear-attention 0.5.2、fla-core 0.5.2；
  - numpy 2.5.2、scipy 1.18.0、pillow 12.3.0、safetensors 0.8.0、tokenizers 0.23.2；
  - huggingface_hub 1.33.0、hf-xet 1.6.0；
  - pymunk 7.3.0、shapely 2.1.2、pycocotools 2.0.11、h5py 3.16.0、pyarrow 25.0.1、einops 0.8.2。
  - nvidia-* 运行时与 cuDNN 9.20.0.48 随 torch 轮子装。

### 1.7 MiniMax 的进程结构与启动脚本（`s3_minimax.txt`、`r_ps.txt`，只读）

两台节点相同，以 hpcgpu06 为例：

```
3065231 bash /var/spool/slurm/d/job157296/slurm_script                    (外层循环，绝不能杀)
3065235  \_ bash /home/mail/chenyuma/minimax-h3-fp8/start_minimax_h3_a100_1gpu_fp16.sh
3065243      \_ comfy-env/bin/python main.py --port 18606 ...              (RSS 81 MB)
3066512      \_ comfy-env/bin/uvicorn api:create_app --port 8406           (RSS 2.5 MB)
```

hpcgpu07 是 4010090 / 4010094 / 4012165（comfy，RSS 823 MB）/ 4013622（uvicorn，端口 8407）。

**外层循环**（两份 sbatch 相同，只差节点名）：

```bash
while true; do
  "$BASE/start_minimax_h3_a100_1gpu_fp16.sh"
  status=$?
  echo "... MiniMax H3 exited status=$status; allocation remains active; retrying in 300 seconds" >&2
  touch "$BASE/run/h3-fp16-a100-06-startup-failed"
  sleep 300
done
```

**启动脚本里与"替换"有关的事实**（原文见 `s3_minimax.txt`）：

- 第 2 行是 `set -euo pipefail`。
- 设了 `trap cleanup_children EXIT INT TERM`：先 kill api，再 kill comfy，然后 `wait`。
- comfy 与 uvicorn 都在后台启动；脚本最后一行是 `wait "$api_pid"`。
- 按路径调用：外层循环每一轮都执行 `$BASE/start_minimax_h3_a100_1gpu_fp16.sh`，两台节点共用这一个文件，mtime 是 2026-08-12 02:55:55 +0800，权限 755。
- 卡的用法：`MINIMAX_H3_DIT_COMPUTE_DEVICE=cuda:0`、`MINIMAX_H3_DIT_DONOR_DEVICE=cpu`、`MINIMAX_H3_VAE_DEVICE=cuda:0`、`MINIMAX_H3_API_CONCURRENCY=1`。
- 两次启动时 nvidia-smi 打印的空闲显存都是 40439 MiB。
- **前提核对（只读推导，未实测）：**
  - 杀掉 uvicorn 之后，`wait "$api_pid"` 返回非零，`set -e` 让脚本退出。
  - 退出时 EXIT trap 再去 kill comfy，并 `wait` 它。
  - 所以"子进程被杀后启动脚本会退出"成立，**条件是 comfy 收到 SIGTERM 后确实结束**；不结束的话，trap 里的 `wait` 会一直挂着。
  - 只杀 comfy、不杀 uvicorn，脚本不会退出：它在 `wait "$api_pid"` 上。
- **使用痕迹：**
  - `api-a100-1gpu-hpcgpu0{6,7}.log` 的最后修改时间都是 2026-08-16 23:51 +0800，全文只有启动信息与两条 `GET /health`，没有 POST。
  - comfy 日志里 "Prompt executed" 的计数是 0。
  - 两个 API 端口上各有 9–10 条 `CLOSE-WAIT` 连接，Recv-Q 是 78–122：本机 127.0.0.1 发起的，以及来自 172.31.10.50 的。
- **crontab：** 只有两行，都被注释掉了（`occupy_gpus.sh`、`h3_guard.sh`）。
  - `~/h3_guard.sh` 与 `schedule_a100_h3.sh` 里有 `scancel`/`sbatch` 逻辑，但没有被调度。`gpu_park.log` 的最后一条是 2026-09-23。

## 2. 与任务书或说明不符的发现

- **hpcgpu06 的 github：** 说明写的是 ✓，这次 https 第一遍 15 s 超时；hpcgpu07 说明写的是 ✗，这次第一遍返回了 200（15.0 s）。三处都不稳定。
- **数据源大多到不了，或者极慢**（§1.4）：
  - COCO、ADE20K、NYU 官方在节点上只有 7–30 KB/s；Google Drive、web.archive.org 不通。
  - RUN-0034 的做法（`land_train.py all` 直接下载）在这里只有 nyu_depth_train 可行，它走 HF（`--hf-base https://hf-mirror.com`）。
- **中转慢：** 瓶颈在 Mac → lab5（约 0.37–0.42 MB/s），不在集群内部。
- **内存不受 cgroup 限制**（§1.2）：180G 是申请值，不是硬限，要靠我们自己守。
- **hpcmgt 在任务期间重启过**（uptime 14 天）。挂在 hpcmgt 上的 `srun` 客户端，会随 hpcmgt 重启而断开。
- **任务是 `Requeue=1`：** 节点故障时 Slurm 会把任务重新排队。这不是我们能控制的，但一旦发生，卡就可能拿不回来，本地盘与 `/dev/shm` 上的东西也会丢。

## 3. 部署方案（提议，等确认）

### 3.1 用哪张卡跑哪个尺寸

| 节点 | 任务 | 尺寸 | 理由 |
|---|---|---|---|
| **hpcgpu06** | 157296 | **4B**：C3 × 3 → E3b+C3 × 3，之后是 4B 的消融 | 4B 的用时最长，hpcgpu06 的负载低（0.06 对 2.20） |
| **hpcgpu07** | 157297 | **0.8B**：C3 × 3 → E3b+C3 × 3，之后是 0.8B 的消融 | 0.8B 先完，接着做 0.8B 的消融，仍然同一台机器（D72-4） |

- **github 不通的影响：** 很小。git 一律在 hpcmgt 上做（说明 §5），代码与数据在 Lustre 上，两台节点看到的是同一份。节点自己不需要连 github。
- **一致性与吞吐（D83 优先级 1）：**
  - 两台节点各跑一遍 2B 的 E3b 与 E3a 50 步，命令与 RUN-0034 / RUN-0035 第 1 步相同，`--out` 带运行编号；
  - 用 `scripts/compare_train_logs.py` 对 B200 的 RUN-0034 日志比，也让 06 与 07 互比；
  - 吞吐：0.8B 在 07、4B 在 06，各跑 C3 与 E3b+C3 20 步。显存峰值从 `accounting.json` 读。
  - 2B 只用于一致性，不在这里训练。

### 3.2 目录

```
VI=/home/mail/chenyuma/visint          # Lustre，两台节点与 hpcmgt 共用；现在不存在（已核对）
├── env/                               # conda -p 建的 py3.12 环境（§3.4）
├── conda-pkgs/                        # CONDA_PKGS_DIRS，不往 base 的 pkgs 写
├── repo/                              # git clone（只在 hpcmgt 上 pull/push）
├── wt/wtNNNN/                         # 每个运行固定 commit 的 worktree（lessons：长任务从固定 commit 跑）
├── data/train_data/                   # 四个源 + dedupe/，一份，两台节点共用
├── hf/                                # HF_HOME；hf/hub = HF_HUB_CACHE（0.8B、2B、4B，约 15.6 GB）
├── ckpt/<node>/                       # --ckpt-root：检查点与续跑状态。放 Lustre，节点重启不丢
├── logs/                              # 驱动日志、心跳 <run>.status、PID 文件、git.lock
├── cache/<node>/{triton,inductor,xdg}/ # TRITON_CACHE_DIR 等，按节点分开
├── tmp/                               # TMPDIR、PIP_CACHE_DIR
└── .ssh/                              # 仓库级 deploy key 与 known_hosts（不碰 ~/.ssh）
```

- **结果**写在 `$VI/repo/experiments/E3/<NNNN_slug>/`，由 hpcmgt 推送，与 B200 的做法相同。
- **节点本地盘与 `/dev/shm`**：默认不用。
  - 吞吐实测时加一组 A/B：同一条 20 步命令，`--data-root` 分别指向 Lustre 与拷到 `/dev/shm/visint-chenyuma/` 的副本。
  - Lustre 明显慢时，才把训练数据（10.3 GB）在每台节点的 `/dev/shm` 各放一份。启动脚本先核对清单 sha256，节点重启后自动重拷。
  - 两台节点的 `/dev/shm` 都有 245G 可用，整机内存 503G。
- **份额：** 说明没有给 Lustre 份额。估计总量约 40 GB：数据 10.3、权重 15.6、环境约 8、检查点 4B 约 10、0.8B 约 4，加结果。请用户给一个上限（§5）。

### 3.3 与 MiniMax 共存还是替换

| | 共存（不碰 MiniMax） | 替换（按说明 §4 停在 sleep 上） |
|---|---|---|
| 显存 | 现在只占 0.4 GB；4B 估计 15–20 GB | 整张 40 GB |
| 有人调用 MiniMax 时 | 它把 DiT 加载到 cuda:0，接近 40 GB。**我们 OOM，或者它 OOM**，看谁先分配。它 OOM 的话请求失败，可能还会让启动脚本退出，5 分钟后重启 | 端口 8406/8407 拒绝连接，调用方直接报错 |
| 要改的东西 | 无 | chenyuma 共享目录里的一个文件（mv 原子替换，留 `.orig`）、两个 run/park 开关文件；杀两个子进程 |
| 恢复 | 无需 | 删开关文件，杀 sleep 进程，5 分钟后 MiniMax 自己重启（启动约 10 分钟，见 slurm 日志） |
| 风险 | 训练被打断，损失最多 500 步（`ckpt_every` = 500）；MiniMax 那一侧的请求也可能失败 | 文件改错会让外层循环每 5 分钟报一次错，但不会结束任务；要按步骤做、逐步 squeue |

**共存时，驱动怎样察觉与恢复：**

1. 每条训练开始前，`nvidia-smi --query-compute-apps` 核对卡上只有 comfy（≤ 1 GB）与我们的进程，并且 `memory.free` ≥ 35 GB；不满足就写 `waiting-gpu`，每 5 分钟查一次。
2. rc ≠ 0 并且日志里有 `OutOfMemoryError` 或 `CUDA out of memory`：
   - 状态写 `oom-wait`；
   - 等 `memory.free` 连续两次 ≥ 35 GB；
   - 用同一条命令加 `--resume` 重跑。训练脚本本来就支持从最新检查点续跑。
3. 同一个运行 OOM 三次：写 `blocked oom`，不再重试，等用户决定。
4. 后台另开一个采样器，每 60 s 记一次显存与卡上进程，事后能看出是谁占了显存。

**我的建议：两台都替换**，或者至少替换 hpcgpu06（4B）。理由：

- **只陈述事实：** API 日志自 2026-08-16 起没有任何请求记录，comfy 没有执行过一次任务（§1.7）。
- 4B 单个种子的 E3b+C3 估计要 14–17 小时（§3.8），三个种子加 C3 要 2.5–3 天。共存时被打断一次就损失最多 500 步，并且 MiniMax 那一侧的调用也会失败，对两边都不好。
- 管理员说过"直接替换任务"。但文件是 chenyuma 的服务，所以要你点头。

**替换的逐步方案**（按说明 §4；每一步前后 squeue，两行都必须 RUNNING，否则停下问）：

1. **只读核对：**
   - 再读一遍启动脚本，核对 sha256 与 mtime 没变；
   - 记下两台节点上 slurm_script、启动脚本、comfy、uvicorn 的 PID 与它们的父子关系（`ps --forest`）；
   - 记下 `squeue -s`。
2. **备份：** `cp -p $BASE/start_minimax_h3_a100_1gpu_fp16.sh $BASE/start_minimax_h3_a100_1gpu_fp16.sh.orig`，并核对两份的 sha256 相同。
3. **写新文件**：先写到 `$BASE/.start_minimax_h3_a100_1gpu_fp16.sh.new`，`chmod 755`，再 `bash -n` 检查语法。内容：

   ```bash
   #!/usr/bin/env bash
   # visint 停车开关（2026-10-xx，RUN-00NN）：存在 run/park-<节点> 就占着卡什么也不做；否则照旧启动 MiniMax。恢复：删开关、杀 sleep。
   BASE="${MINIMAX_H3_BASE:-/home/mail/chenyuma/minimax-h3-fp8}"
   if [ -e "$BASE/run/park-$(hostname -s)" ]; then
     echo "[$(date -u +%FT%TZ)] parked on $(hostname -s) by visint (run/park-$(hostname -s))" >&2
     exec sleep infinity
   fi
   exec "$BASE/start_minimax_h3_a100_1gpu_fp16.sh.orig" "$@"
   ```

4. **原子替换：** `mv -f $BASE/.start_….new $BASE/start_minimax_h3_a100_1gpu_fp16.sh`。
   - 正在运行的两个旧启动脚本持有的是旧 inode，不受影响。
   - 替换后 `ls -li` 核对 inode 已变、权限是 755。
5. **先停 hpcgpu06：**
   - `touch $BASE/run/park-hpcgpu06`；
   - 经 `srun --jobid=157296 --overlap` 按 PID `kill` uvicorn（3066512），再 `kill` comfy（3065243）。只按 PID，不用 pkill。
   - 10 秒后核对：两个 PID 都不在；启动脚本 3065235 已退出；**slurm_script 3065231 还在**；squeue RUNNING。
   - comfy 30 秒内还没退出：只对这个 PID 发 `kill -9`。
6. **等 300 s：** 外层循环调用新脚本，进程树里应该出现 `slurm_script → sleep infinity`，nvidia-smi 显示卡上没有进程。
7. **再按同样的步骤停 hpcgpu07**（PID 4013622、4012165、4010094，外层 4010090）。
8. **恢复**（任一节点）：
   - `rm $BASE/run/park-<节点>`，按 PID kill 那个 `sleep infinity`；
   - 外层循环记一次 "exited"，300 s 后调用新脚本，没有开关文件，就 exec `.orig`，MiniMax 约 10 分钟后就绪。
   - 要完全复原：先确认两台都不在停车状态，再 `mv .orig` 回原名。
9. **注意：**
   - 外层循环每次 "exited" 都会 `touch run/h3-fp16-a100-0X-startup-failed`，这是它本来的行为。
   - 停车期间 MiniMax 的日志不再更新。

### 3.4 环境

- **建在** `$VI/env`：
  - 用 chenyuma 的 conda 执行 `conda create -p $VI/env python=3.12`，并设 `CONDA_PKGS_DIRS=$VI/conda-pkgs`，不往 base 的 pkgs 写；
  - 不改 `~/.condarc`。
  - 在 hpcmgt 上装，hpcmgt 与节点是同一个 OS、同一个 Lustre。
- **pip：** 全部用 `$VI/env/bin/python -m pip`，`PIP_CACHE_DIR=$VI/tmp/pip` 或 `PIP_NO_CACHE_DIR=1`。
  1. `torch==2.13.0+cu130`、`torchvision==0.28.0+cu130`，从 `--index-url https://download.pytorch.org/whl/cu130` 装，triton 3.7.1 随 torch 装；
  2. 然后 `-r requirements.txt`；
  3. 然后把 §1.6 的版本逐个钉住，包括 flash-linear-attention / fla-core 0.5.2、numpy 2.5.2、safetensors、tokenizers、huggingface_hub、hf-xet。
     - B200 装了 fla，没装 flash-attn、causal-conv1d、kernels，这里照样不装（lessons：可选加速包两边要一致）。
- **装完核对：**
  - `pip freeze` 与 B200 逐项对照，差异列表进报告；
  - `torch_backends.json` 与 RUN-0035 第 1 步同一段代码；
  - 在 hpcgpu06 上经 srun 跑一遍**全量测试**（普通与 server-parity 两套，各自 `--basetemp`），与 D87 的 2743 passed + 18 skipped 对照。
- **与 B200 的差异：**
  - 卡：sm_80 对 sm_100；驱动 590 对 580；PCIe 250 W；
  - 显存 40 GB 对 178 GB；
  - CPU：16 核 Xeon 6342；
  - sdpa 与 fla 的 Triton 内核会在 A100 上各自选实现、各自 autotune，**数值不会与 B200 逐位相同**，这正是一致性检查要量的东西。

### 3.5 数据与权重

**训练数据**（10.276 GB，四个源）：

| 源 | 做法 | 依据 | 估计用时 |
|---|---|---|---|
| nyu_depth_train（3.97 GB） | 在 hpcgpu06 上 `land_train.py nyu_depth_train --hf-base https://hf-mirror.com`，代码自己核对中央目录 sha256 与成员 CRC32 | hf-mirror 2.6–5.4 MB/s | 约 15–40 分钟 |
| refcoco_train（4.26 GB） | 从 B200 的 `/dev/shm/visint/train_data/refcoco_train` 经 Mac 中转：`ssh b200 tar c … \| ssh a100 'ssh gpu "tar x -C $VI/data/train_data"'`，途中不落盘 | COCO 只有 28 KB/s；web.archive 不通 | 约 2.8 小时 |
| ade20k（0.98 GB） | 同上中转 | csail 21 KB/s | 约 40 分钟 |
| nyu_normals（1.01 GB） | 同上中转 | Drive 不通 | 约 40 分钟 |
| dedupe/（212 KB） | 同上中转 | | 秒级 |

- 中转合计约 6.25 GB，按 0.37–0.42 MB/s 约 **4.1–4.7 小时**。
- **落地后的核对**，同 RUN-0034 的 `land_compare`：每个源的 `MANIFEST.json` 逐文件 sha256 与 B200、A800 两份清单比，B200 的清单已在仓库里（`experiments/E3/0034_b200_prep/land_manifests/`）。
  - 用 RUN-0034 的 `probe_scripts/p33_manifest_compare.py` 原样再跑，不另写逻辑；
  - 中转来的三个源，再从盘上的文件重新算一遍 sha256，不采信 tar 传过去的 MANIFEST。
- **去重：** 默认做法是中转 B200 的 `dedupe/`，核对 `train_exclusions.json` 与 B200、A800 两份的 sha256 相同，用 `p34_exclusions_compare.py`。
  - 要像 RUN-0034 那样在本机重跑去重，得先中转 E1 的 refcoco eval 与 nyuv2 test（约 2.3 GB，再加约 1.5 小时）。见 §5。
- **中转更快的办法**（要用户同意，§5）：
  - 从 Mac `ssh -A b200` 转发 agent，让 B200 直接 `ssh -p 8889 root@pi.machenyu.cn` 推到 lab5；B200 到 lab5 的入口端口已经能连上，速度未测。
  - 代价是转发期间，B200 上其他 root 用户能用这个 agent。

**权重**：

- 下载：在 hpcgpu06 上 `HF_ENDPOINT=https://hf-mirror.com`，`snapshot_download(repo, revision=<model_notes 钉住的 sha>)`，进 `$VI/hf/hub`。
  - Qwen3.5-0.8B 1.75 GB、2B 4.55 GB、4B 9.32 GB，合计约 15.6 GB；按 2.4–3.9 MB/s 约 70–110 分钟。
- 核对：
  - 下载前用 API 核对 revision 与钉住值相同，0.8B 已核对；
  - 下载后逐文件核 LFS sha256，同 B200 的"权重核对 13/13、16/16"。
- 训练时 `HF_HUB_OFFLINE=1`。

### 3.6 git

- **在 hpcmgt 上**：`$VI/repo` 用 ssh 地址 clone。
  - `git config core.sshCommand "ssh -i $VI/.ssh/id_ed25519_github -o IdentitiesOnly=yes -o UserKnownHostsFile=$VI/.ssh/known_hosts"`；
  - 身份只设在仓库级：`user.name hpc-a100-server-cc`。
  - 不碰 `~/.gitconfig` 与 `~/.ssh`。
- **要用户做的事**（我生成密钥后，把公钥贴给你）：
  1. 打开 GitHub `junli-gong/vision-internalization` → Settings → Deploy keys → Add deploy key；
  2. Title 写 `itsc-hpc-a100 (chenyuma@hpcmgt)`，Key 粘贴公钥；
  3. 勾 **Allow write access**，驱动要推结果；
  4. 保存。
- 推送一律 `flock $VI/logs/git.lock`，先 `pull --rebase` 再 push。
  - 每个阶段跑完单独 `[results] NNNN: …` 推送；
  - 失败只记日志，事后补推。
- **备选：** 不给 deploy key。结果由我从集群拉回 Mac，在 `A100/repo` 里推送。
  - 代码则要从 Mac 往集群传 git bundle：仓库 243 MB，按 0.45 MB/s 约 9 分钟，之后是增量。
  - 缺点：我的会话不在时，结果推不回去。

### 3.7 长任务怎么跑

- **启动：** 每个运行从固定 commit 的 worktree `$VI/wt/wtNNNN` 启动。驱动是一个 bash 队列脚本，仿 RUN-0036/0040 的 `queue.sh`：已完成的跳过，没完成的 `--resume`。
- **环境块**照 RUN-0034 / A800 的 common.sh 逐条抄，一行一个 export：
  - `HF_HOME`、`HF_HUB_CACHE`、`HF_HUB_OFFLINE=1`；
  - `TMPDIR`、`PIP_NO_CACHE_DIR=1`；
  - `OMP_NUM_THREADS=16`、`MKL_NUM_THREADS=16`；
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
  - `TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`XDG_CACHE_HOME`；
  - `PYTHONHASHSEED=0`（D86-2）。
- **启动方式（方案 A，说明已实测）：** 在 hpcmgt 上

  ```bash
  setsid nohup srun --jobid=157296 --overlap -n1 bash $VI/wt/wtNNNN/…/queue.sh > $VI/logs/NNNN_06.srun.log 2>&1 < /dev/null &
  echo $! > $VI/logs/NNNN_06.srun.pid
  ```

  驱动在节点上把自己的 PID、当前 python 的 PID 写进 `$VI/logs/NNNN_06.pids`。
- **方案 A 的风险：** srun 客户端活在 hpcmgt 上。hpcmgt 重启过（§2），重启会断开这个步骤，训练停在最近的检查点，最多损失 500 步。
- **方案 B（未实测，建议在 bootstrap 里用一条 `sleep 60` 测一次）：** `ssh -n hpcgpu06` 后，在节点上 `setsid nohup srun --jobid=157296 --overlap …`。
  - 步骤仍然受任务的 cgroup 限制；srun 客户端活在节点上，不受 hpcmgt 重启影响。
  - 前提是节点上能运行 srun；batch 脚本本来就在节点上，这一点要先测。
- **心跳：**
  - 驱动每完成一段就写 `$VI/logs/NNNN_<node>.status`：时间、运行名、步数；
  - GPU 采样器每 60 s 写一行 `gpu_<node>.tsv`。
  - 检查时同时看 status 的时间，以及步骤与 PID 是否还在（`squeue -s` 加 `kill -0`），不只看心跳文件（lessons）。
- **停自己的实验：**
  - 按 PID 文件，在 hpcmgt 上 kill 那个 srun 客户端（步骤结束，任务不受影响，说明 §3 已实测）；
  - 或者经 `srun --overlap` 按 PID kill 节点上的 python。
  - **不用 scancel，不碰 slurm_script。**
- **守线：**
  - 每条命令开始前检查：两个任务 RUNNING；Lustre 剩余空间与份额；节点 MemAvailable ≥ 100 GB；
  - 内存没有 cgroup 硬限，我们自己守 ≤ 180 GB。
  - 任何一项不满足就停自己的进程，并写 `blocked`。
- **结果回传：** 驱动每个运行结束后，在 hpcmgt 一侧的推送脚本里 `flock` + commit + push。
  - 只推日志、accounting、train_log；检查点不进 git。
  - 检查点以后传 B200（D55）：走 Mac 中转，B200 下行快，瓶颈仍是 Mac 上行；或者走 §5 第 2 项的直连。0.8B 每个约 0.1–0.3 GB，4B 稍大，只传 `final/`。

### 3.8 吞吐与用时粗估（未实测，第一项实测会替换）

**依据：**

- **(a) A800 的估计**（RUN-0042 包的 README，管理员估，不是实测；A800 第 1 步的吞吐还没回来）：
  - 0.8B 的 E3b+C3 每个种子约 6 小时，4B 约 13 小时。
- **(b) A100-PCIe-40GB 相对 A800-SXM：** 同一颗 GA100，功耗 250 对 400 W，HBM 1.56 对 2.04 TB/s。按规格取 **慢 1.1–1.3 倍**，未实测。
- **(c) C3 相对 E3b+C3：** B200 上 2B 的 C3 每步 1.05–1.14 s × 4,000 步，E3b+C3 约 3.6 h × 8,000 步（RUN-0040 §2）。C3 的用时约是 E3b+C3 的 0.33–0.35。

| 节点 | 内容 | 单个 | 合计 |
|---|---|---|---|
| hpcgpu07 | 0.8B C3 × 3 | 2.2–2.7 h | 6.6–8.2 h |
| hpcgpu07 | 0.8B E3b+C3 × 3 | 6.6–7.8 h | 20–23 h |
| hpcgpu07 | **0.8B 优先级 2 合计** | | **约 26–32 h** |
| hpcgpu06 | 4B C3 × 3 | 4.7–5.7 h | 14–17 h |
| hpcgpu06 | 4B E3b+C3 × 3 | 14–17 h | 43–51 h |
| hpcgpu06 | **4B 优先级 2 合计** | | **约 58–68 h（2.4–2.8 天）** |
| 两台 | 一致性（2B，50 步 × 2）与吞吐（20 步 × 2 × 2） | | 每台约 0.5–1 h |

- 4B 的单个 E3b+C3 超过 12 小时报批线，已在 D83/D85 批准的范围内（D85-3）。
- 开跑前按实测吞吐逐条复核，每条开始前再判一次"跑完会不会超"（lessons：批了上限的运行）。
- **显存：**
  - 4B 的 E3b+C3 按管理员的估计是 15–20 GB；B200 实测 2B 的 E3b 是 9.9 GB、9B 是 30.5 GB。40 GB 够用。
  - 前提是 MiniMax 不同时加载。
- **前置准备的墙钟：** 中转 4.1–4.7 h 与权重 1–2 h 可以并行；环境与测试约 1–2 h。从确认到开训约半天。

### 3.9 其他风险

- **节点故障与重新排队：** `Requeue=1`，节点重启后任务会重新排队，卡可能拿不回来。
  - 检查点放 Lustre，不放本地；
  - 这种情况不能自行处理，停下报告。
- **Lustre：**
  - 小文件很多（训练数据约 11.7 万个），元数据可能慢，所以要做 §3.2 的 A/B；
  - HF 缓存的文件锁在 Lustre 上的行为没验证，下载时只在一个节点上进行，不并发。
- **hpcgpu07 上另有一个 45 天的步骤**（157297.34，`python3 -u -`），不是我们的，不碰。
  - 它没占卡，但在同一个任务的 CPU 份额里，会和训练抢 16 核里的 CPU。只陈述事实，影响未量。
- **共享节点：** 两台节点的其余 3 张卡都被别人占着（hpcgpu07 load 2.2），CPU 与 PCIe 有争用的可能。吞吐实测时记录节点负载。
- **hf-mirror、github 都不稳：** 下载要可续传、可重试；github 只在 hpcmgt 上用，推送失败不中断训练。

## 4. 生成的文件

- `reports/0043-hpc-bootstrap-plan.md`（本文件）
- `experiments/E3/0043_hpc_survey/raw/`：§1 表里列的 7 个原始输出文件
- `experiments/E3/0043_hpc_survey/probe_scripts/`：本次用的只读探针脚本（经 `ssh a100 'ssh gpu bash -s'` 或 `srun --overlap … bash -s` 从 stdin 送入执行）

## 5. 需要用户决定的事项

1. **MiniMax：共存还是替换？**
   - 建议两台都按 §3.3 替换；最低限度替换 hpcgpu06（4B）。
   - 替换要改 chenyuma 共享目录里的启动脚本（留 `.orig`，可恢复），并杀两个子进程。
2. **数据怎么传？**
   - 默认：三个源经 Mac 中转，约 4.1–4.7 h；nyu_depth_train 走 hf-mirror 原生落地。
   - 可选：`ssh -A` 让 B200 直推 lab5，可能更快，但转发期间 agent 暴露在多人共用的 B200 上。
3. **去重：**
   - 默认：搬 B200 的剔除清单，并对 B200、A800 核 sha256。
   - 或者：像 RUN-0034 那样本机重跑，多传 2.3 GB，多约 1.5 h。
4. **git：**
   - 默认：你在 GitHub 上加一个可写的 deploy key（步骤见 §3.6）。
   - 或者：不加，结果由 Mac 中转推送。
5. **卡的分配：** 4B → hpcgpu06、0.8B → hpcgpu07，是否同意？
   - 另：是否允许 4B 的种子分到两台节点上，以缩短墙钟？这不符合 D72-4 的字面意思，我的建议是不分。
6. **启动方式：** 先用方案 A（srun 挂在 hpcmgt，已实测）；同意的话，我在 bootstrap 里用 `sleep 60` 测一次方案 B（srun 从节点发起）。
7. **Lustre 份额：** 本项目在 `/home/mail/chenyuma/visint` 最多占多少？我估计约 40 GB，建议上限 80 GB。目录名 `visint` 是否可以？
