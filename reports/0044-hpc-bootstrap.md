# RUN-0044 — ITSC HPC A100 开训前准备（进度报告：停在第 5 节，等用户）

服务器 CC（HPC），在用户的 Mac 上经 `ssh a100 → gpu(hpcmgt)` 操作。2026-10-10 06:06–18:15 UTC。只贴数字与事实，不做解读。

## 0. 一句话状态

- **第 2、3、4、6 节完成。**
  - 环境、全量测试、deploy key、数据（四个源）与去重、权重（0.8B、2B、4B）都已到位并核对；
  - 从节点发起 srun 测通。
- **第 5 节（MiniMax 停车）没有执行**：
  - 第 2–4 步的命令（备份 `.orig`、写新文件、mv 原子替换）被 Claude Code 的 auto mode 权限分类器拦下："judged this action dangerous"。
  - 命令一条都没有运行，MiniMax 的文件与进程**完全没动**。
  - 第 1 步（只读核对）已做，见 §5。
- **第 7、8 节（一致性、吞吐）没有执行**：任务书要求停车之后才用 GPU。需要用户决定，见 §11。
- **铁律：** 每一步前后 squeue，两个任务始终都是 RUNNING（§1）。没有 scancel、scontrol 改动、sbatch、salloc，没有碰 slurm_script。
  - 停过的只有我们自己的步骤：weights v1、v2 两个 srun 步骤，以及一个慢的 git clone，见 §9。
- **Lustre 用量约 40 GB**（上限 80 GB）。

## 1. squeue

- 原始记录在 `experiments/E3/0044_hpc_bootstrap/raw/` 各文件的开头与结尾（`dirs.txt`、`park_step1_readonly.txt`、`methodB_test.txt`、`cluster_logs/logs/*`）。
- 全程每一条都是两行 RUNNING，运行时长单调增长。例如：
  - 首条：`2026-10-10T06:06:58Z  157296 RUNNING 55-17:32:44 hpcgpu06 / 157297 RUNNING 57-15:04:53 hpcgpu07`；
  - 停车第 1 步之后：`08:57:05Z  157296 RUNNING 55-20:22:51 / 157297 RUNNING 57-17:55:00`；
  - 最后：`18:00Z` 前后，两行 RUNNING。
- 停我们自己的 weights v1 步骤（157296.127）时，前后各一次 squeue，都是 RUNNING：

  ```
  slurmstepd: error: *** STEP 157296.127 ON hpcgpu06 CANCELLED AT 2026-10-10T14:51:25 ***
  157296 RUNNING 55-18:17:16 hpcgpu06
  ```

  这是步骤结束的标准信息，任务本身不受影响，与说明 §3 一致。

## 2. 目录与环境（任务书 §2）

- **目录：** `VI=/home/mail/chenyuma/visint`。子目录按 RUN-0043 §3.2 建立；`.ssh` 是 700（`raw/dirs.txt`）。
- **conda 环境：** `$VI/env`。
  - `conda create --solver classic python=3.12.13`，并设 `CONDA_PKGS_DIRS=$VI/conda-pkgs`。
  - 第一次建成的是 3.12.15，与 B200 不同，于是删掉、按 3.12.13 重建。
- **torch：** 从 `download.pytorch.org/whl/cu130` 装，额外加清华 PyPI 作 extra-index。
  - 第一次 rc=1：B200 的版本约束里钉了 `setuptools==83.0.0`，pytorch 的源上没有这个版本。
  - 原始日志：`raw/cluster_logs/logs/0044_env_install{,_v1_failed}.log`。

  | 步骤 | rc | 用时 |
  |---|---|---|
  | conda create | 0 | 37 s |
  | torch 2.13.0+cu130、torchvision 0.28.0+cu130 | 0 | 1,069 s |
  | requirements + fla 0.5.2 + einops（以 B200 的 freeze 作 `-c` 约束） | 0 | 139 s |

- **在 hpcgpu06 上的核对**（经 srun）：
  - torch 2.13.0+cu130，cuda 13.0，cuDNN 92000，arch list 含 sm_80；
  - `NVIDIA A100-PCIE-40GB`，cap (8, 0)，bf16 matmul 正常；
  - `cudnn.allow_tf32=True`、`cudnn.conv.fp32_precision=tf32`、`cuda.matmul.fp32_precision=none`。
- **pip freeze 对 B200**（`raw/pip_freeze_vs_b200.md`）：
  - 两边都有的 80 个包，版本全部相同；
  - 只在 B200 上的 20 个：annotated-types、contourpy、cycler、fonttools、kiwisolver、matplotlib、ml-dtypes、ninja、onnx、onnx-ir、onnxscript、pandas、protobuf、pybind11、pydantic、pydantic-core、pyparsing、python-dateutil、transformer-engine、typing-inspection。都来自 B200 的基础环境；
  - 只在 HPC 上的：0。
- **后加 git：** 计算节点上没有 `git`（`bash: git: command not found`）。
  - 于是 `conda install -p $VI/env --freeze-installed git`，得到 git 2.55.0；
  - 前后 pip freeze 逐行相同，Python 仍是 3.12.13。
  - 理由：`tests/test_runlog.py` 要用 git；另外训练脚本的 `runlog.git_info` 在没有 git 时把 commit 记成 null（代码本来就容许，不会崩）。

### 2.1 全量测试（hpcgpu06，经 srun；GPU 隐藏，`CUDA_VISIBLE_DEVICES=`；两套并行，各自 `--basetemp`；直接用 env 的 python）

| 轮 | 前提 | 普通 | server-parity |
|---|---|---|---|
| 1 | 只有 Qwen 的缓存 | 162 failed, 2208 passed, 13 skipped, 378 errors（3,079 s） | 163 failed, 2207 passed, 13 skipped, 378 errors |
| 2 | 加：从开发机拷来 Gemma-4-E2B、InternVL3.5-2B、4B 的 config 与 tokenizer 缓存（固定 revision，无权重）；env 里装了 git | 103 failed, 2477 passed, 16 skipped, 165 errors | 103 failed, 2477 passed, 16 skipped, 165 errors |
| **3** | 加：那三个 `tokenizer.json` 指向开发机 HF 缓存的共享 `hub/blobs/`，补拷两个共享 blob | **3 failed, 2739 passed, 19 skipped**（6,327 s） | **3 failed, 2739 passed, 19 skipped**（6,238 s） |

- **第 1、2 轮失败的原因：**
  - 第 1 轮：`OSError: We couldn't connect to 'https://huggingface.co'…`、`GatedRepoError`、`LocalEntryNotFoundError`。离线时缺 Gemma 与 InternVL 的文件，Gemma 是 gated。
  - 第 2 轮：224 处 `ValueError: config.image_token_id=258880 与 tokenizer 的 <|image|> id 16 不一致`。tokenizer.json 是悬空链接，加载出来的 tokenizer 只有 5 个词。
- **拷来的文件：**
  - 开发机缓存里的 revision 正是钉住的值：gemma `3e22461f`、InternVL-2B `3f301ffc`、InternVL-4B `6bd44874`。
  - 32 个普通文件的 sha256 两边逐个相同。
  - 两个共享 blob 的内容 sha256，等于各自 per-model 链接名里的 LFS sha256：`cc8d3a0c…` 是 Gemma 的 tokenizer.json，`7b9d1866…` 是 InternVL 的。
  - macOS tar 带过来的 `._*` 文件（AppleDouble）已删除。
  - 拷贝之后没有悬空链接。
- **第 3 轮总数：** 2739 + 3 + 19 = 2,761 条，`--co` 也是 2,761，与 D87 的 2,743 passed + 18 skipped 总数相同。
  - skipped 多 1 条，是哪一条还在查：带 `-rs` 的普通套件 18:07 起在 hpcgpu06 上跑，日志 `$VI/logs/0044_skips.log`。
- **第 3 轮的 3 条失败**，两套相同。各条单独重跑 4 次（`raw/cluster_logs/logs/0044_rerun3.log`、`0044_rerun4.txt`）：

  | 测试 | 全量时 | 单独重跑 4 次 |
  |---|---|---|
  | `test_synth_3d_tasks.py::test_speed_floor_single_core[viewpoint]`（计时） | 18.63/s < 20 | 4 次都通过 |
  | `test_synth_occlusion.py::test_speed_at_least_50_per_second_single_core`（计时） | 失败 | 4 次都失败（第 4 次 40.21/s < 50）；节点负载 1.9–5.6 |
  | **`test_check_training_interfaces.py::test_grid_checkpoint_split_across_merge_fails_restricted`**（非计时） | `assert (0 == 1)`：病态检查点的 restricted 核对返回 0（通过），测试期望 1 | **4 次都失败，结果确定** |

  - 第三条不属于"计时类"，按任务书单独列出，不解读。
  - 它是 `--model qwen3.5-0.8b`、tiny、CPU fp32 下的"合并前后网格 logits 分家"检测（`tests/test_check_training_interfaces.py:465`）。
  - 实验代码与测试都没有改。

## 3. git 与 deploy key（§3）

- **生成：** 在 hpcmgt 上生成 `$VI/.ssh/id_ed25519_github`（权限 600，目录 700）。
- **GitHub：** 在 Mac 上执行 `gh repo deploy-key add … --allow-write --title "itsc-hpc-a100 (chenyuma@hpcmgt)"`，key id 165983660，read-write。仓库里原有的 B200 key 不动。
- **known_hosts：** 用 `ssh-keyscan` 取得，三个指纹与 GitHub 公布的一致：ED25519 `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`、ECDSA `p2QAMX…`、RSA `uNiVzt…`。
- **仓库级配置：** `core.sshCommand`（`-i $VI/.ssh/…`、`IdentitiesOnly`、`UserKnownHostsFile`、`StrictHostKeyChecking=yes`）；身份 `hpc-a100-server-cc <hpc-a100-server-cc@users.noreply.github.com>`。`~/.gitconfig`、`~/.ssh` 都没动。
- **clone：** 从 hpcmgt 直连 GitHub 约 27–110 KB/s，仓库打包后 236 MB。
  - 第一次 clone 30 分钟超时；第二次也很慢，我按 PID 停掉了。
  - 改用开发机做的 git bundle：249 MB，经 Mac 传了 1,133 s，两端 sha256 相同（`560dfd84…`）。
  - 从 bundle 取 `refs/remotes/origin/main` 之后，`git fetch origin` 只用了 5 s，main 跟上 origin（`511b136`）。
- **试推：** `c103fee [results] 0044: HPC deploy key trial push (README)`，作者 hpc-a100-server-cc，开发机上能 pull 到。
- **推送脚本：** `probe_scripts/push.sh`，`flock $VI/logs/git.lock`，先 `pull --rebase` 再 push，失败重试 3 次。
- **worktree：**
  - `$VI/wt/wt0044` 固定在 `ea49e15`（本任务书的 commit；与 `511b136` 相比 src/、configs/、训练脚本都没有变化）。e3b 配置 `b265cd87…`、e3a 配置 `1dc34430…`。
  - `$VI/wt/wt0044_6e77ab0` 固定在 `6e77ab0`，配置 `35e69b56…`、`28be66e5…`，与 RUN-0034 相同。
  - 两个都没有未提交的改动。

## 4. 训练数据与权重（§4）

### 4.1 落地

- **中转**（B200 `/dev/shm/visint/train_data` → Mac → lab5 → hpcmgt → Lustre；B200 上只读，没用 `ssh -A`）：
  - 第 1 轮：06:07:25 起，三路并行（refcoco_train 的单号、双号文件各一路，ade20k+nyu_normals+dedupe 一路）。
  - **07:26:58 lab5 重启**（`uptime` 显示 up 0 min），经它的 ssh 全部断开（`Connection to pi.machenyu.cn closed by remote host`）。集群上的任务不受影响。
  - 第 2 轮：按"B200 上的文件与字节数"对照 HPC，只重传缺失或字节数不符的 11,615 个文件（其中 3 个字节数不符），合计 2.474 GB；文件清单从 stdin 交给 B200 的 tar。
    - 07:28:35–08:25:35，三路都是 rc=0。
  - 中转完成后，四个目录（含 dedupe）共 71,740 个文件、6,282,219,710 B，与 B200 的文件与字节数逐个相同。
  - 日志：`raw/relay.log`、`raw/relay_*.err.txt`。
- **nyu_depth_train：** 在 hpcgpu06 上用 `land_train.py nyu_depth_train --hf-base https://hf-mirror.com --workers 8` 落地，rc=0，3,035 s。
  - `[完成] nyu_depth_train：45698 个文件，3.970 GB；失败 0；核对不通过 0`
  - 原始日志：`raw/cluster_logs/logs/0044_land_nyu.log`。

### 4.2 核对（hpcgpu07，经 srun；`raw/cluster_logs/logs/0044_verify/`、`0044_verify.log`）

**盘上重算 sha256**，对照各源自己的 MANIFEST.json（`p44_disk_sha.py`；中转来的源不采信传过来的 MANIFEST），321 s：

| 源 | MANIFEST 文件数 | 盘上文件数 | sha256 与字节都相同 | 不同 | 缺 | 多 |
|---|---|---|---|---|---|---|
| refcoco_train | 24411 | 24411 | 24411 | 0 | 0 | 0 |
| nyu_depth_train | 45698 | 45698 | 45698 | 0 | 0 | 0 |
| nyu_normals | 2899 | 2899 | 2899 | 0 | 0 | 0 |
| ade20k | 44424 | 44424 | 44424 | 0 | 0 | 0 |

**MANIFEST 对 A800**（RUN-0034 的 `p33_manifest_compare.py`，原样复用，所以表头写的是"B200"，那一列实际是 HPC）与**对 B200**（`p44_manifest_vs_b200.py`），两份对照都是：

| 源 | HPC 文件数 | 对方文件数 | 总字节（两边相同） | 失败成员 | 两边都有且 sha256 相同 | 同路径 sha256 不同 | 只在 HPC | 只在对方 |
|---|---|---|---|---|---|---|---|---|
| refcoco_train | 24411 | 24411 | 4264204008 | 0 / 0 | 24411 | 0 | 0 | 0 |
| nyu_depth_train | 45698 | 45698 | 3969712007 | 0 / 0 | 45698 | 0 | 0 | 0 |
| nyu_normals | 2899 | 2899 | 1009158299 | 0 / 0 | 2899 | 0 | 0 | 0 |
| ade20k | 44424 | 44424 | 975810735 | 0 / 0 | 44424 | 0 | 0 | 0 |

**去重**（RUN-0034 的 `p34_exclusions_compare.py`，原样复用，HPC 的 `train_exclusions.json` 分别对 B200、A800）：

- 两份都是：max_hamming 8 / 8，eval_sets 相同（13 / 13），**完全相同：True**。
- 逐条剔除数：ade20k 26、nyu_depth_train 132、nyu_normals 50、refcoco_train 10，两边都剔除、整条记录相同，只在一边的都是 0。
- 文件 sha256：
  - HPC `40f214fb…` = B200 `40f214fb…`；
  - A800 是 `4ddb8fc2…`，逐条记录相同、字节不同。RUN-0034 的 B200 对 A800 也是这样。

### 4.3 权重（hpcgpu06，hf-mirror，按 `weights_manifest.json` 钉 revision，RUN-0009 的 `p12_verify_weights.py` 逐文件核对）

| 模型 | revision | 核对 | 用时 |
|---|---|---|---|
| qwen3.5-0.8b | 2fc06364… | 13/13 | 544 s |
| qwen3.5-2b | 15852e8c… | 13/13 | 1,133 s |
| qwen3.5-4b | 851bf6e8… | 14/14 | 1,393 s |

- **v1 失败（rc=1）：** hf-mirror 会把请求重定向到 Xet 的 CAS，返回 `401 Unauthorized`（`cas-server.xethub.hf.co`）。改设 `HF_HUB_DISABLE_XET=1` 后正常。
- **v2 慢：** 一条长连接降到约 87–117 KB/s；同时新开的连接是 2.7–3.3 MB/s。我停掉 v2 步骤并重启，之后约 3.4 MB/s。
- **清理：** 停掉的两次留下了 `.incomplete` 文件（0.8B 1.41 GB、2B 0.53 GB），确认是我们的残留后已删除。

## 5. MiniMax 停车（§5）：只做了第 1 步

**第 1 步（只读，08:56:35–08:57:05，前后 squeue 都 RUNNING；`raw/park_step1_readonly.txt`）：**

- **启动脚本没变：** 3996 B，mtime `2026-08-12 02:55:55 +0800`，755，sha256 `e42bfde15b8d62f0b32f5251338ecab520252d75098c6b5b4d3bcf321f1afae9`。这个值与 RUN-0043 原始输出里那份脚本全文算出的 sha256 相同。
- `run/` 里没有 park 或 startup-failed 文件。
- **进程树与 RUN-0043 相同：**
  - hpcgpu06：slurm_script 3065231 → 启动脚本 3065235 → comfy 3065243（卡上 416 MiB）、uvicorn 3066512；
  - hpcgpu07：4010090 → 4010094 → comfy 4012165（416 MiB）、uvicorn 4013622。
- **准备好的新启动脚本**（`probe_scripts/park_start.sh`，sha256 `6a9acbf7…`）：内容即 RUN-0043 §3.3 第 3 步，日期写成 2026-10-10、编号 RUN-0044。
  - `bash -n` 通过；
  - 在 hpcmgt 上用假的 BASE（`$VI/tmp/parktest`，测完删除）测过两种情况：没有开关文件时 exec `.orig`（输出 `ORIG_RAN x`）；有开关文件时 exec `sleep infinity`（`timeout 3` 返回 124）。

**第 2–4 步被拦下：**

- 要运行的脚本：被拦下的那条命令本身就包含"写出脚本"这一步，所以脚本文件没有生成，仓库里也没有存档。逐条是：
  1. 核对 sha256；
  2. `cp -p` 成 `.orig` 并核对；
  3. 把 `park_start.sh` 拷成 `.start_…new`，`chmod 755`，`bash -n`；
  4. `mv -f` 原子替换，核对 inode。
  - 每一步前后 squeue，任何一项不符就停。
- 结果：被 auto mode 的权限分类器拒绝，**一条命令都没有运行**。
- 我没有尝试绕过。MiniMax 的文件、开关文件、进程都是原样。

**恢复方法**（停车做完之后适用，RUN-0043 §3.3 第 8 步）：

1. 删掉 `run/park-<节点>`；
2. 按 PID kill 那个 `sleep infinity`；
3. 外层循环 300 s 后调用新脚本，没有开关文件，就 exec `.orig`，MiniMax 约 10 分钟后就绪；
4. 要完全复原：确认两台都没有停车之后，`mv .orig` 回原名。

### 5.1 hpcgpu07 的卡（管理员 18:10 的消息里提到"我们那张卡上有别的用户的进程"）

18:12 UTC 的两种视图，只读（`raw/gpu07_node_view_1812Z.txt`、`raw/gpu_in_allocation_1812Z.txt`）：

- **在任务分配里**（`srun --jobid=157297 --overlap`）：只有一张卡，00000000:65:00.0（GPU-447d1acc…）。
  - memory.used 423 MiB、memory.free 40,019 MiB、利用率 0%；
  - 卡上进程只有 comfy 4012165（416 MiB）。
- **整个节点**（直接 ssh，不受分配限制）：r130201615 的两个 `…/recon/bin/python` 在别的卡上，不在我们这张：

  | 卡 | 进程 | 显存 |
  |---|---|---|
  | 00000000:CA:00.0 | 144839 | 19,668 MiB |
  | 00000000:E3:00.0 | 4051039 | 32,894 MiB |
  | 00000000:17:00.0 | zhanchen 的 4010292 | 19,908 MiB |

- 08:57 的只读记录里列出的那个 23 GB 进程，是用直接 ssh 的 `nvidia-smi` 查到的。那条命令列出整机所有卡的进程，没有带总线号。
- 没有碰任何别人的进程。
- hpcgpu06 同一时刻：我们的卡 00000000:65:00.0（GPU-43e409cb…）used 423 MiB、free 40,019 MiB，只有 comfy 3065243。

## 6. 启动方式（§6，方案 B）测通

**测试（06:51:53–06:53:17；`raw/methodB_test.txt`）：** 从 hpcmgt `ssh -n hpcgpu06`，在节点上 `setsid nohup srun --jobid=157296 --overlap -n1 … sleep 60`。

- 节点上有 `/usr/bin/srun`、`squeue`、`sinfo`。
- 步骤 157296.131：`host=hpcgpu06 CVD=0 nproc=16`；cgroup 是 `devices:/slurm/uid_19023/job_157296/step_131/task_0`、`cpuset:/slurm/…/step_131`；`nvidia-smi -L` 只看到我们那一张（GPU-43e409cb…）。
- `end 2026-10-10T06:52:55Z`，正常结束。之后 squeue 两个任务都是 RUNNING。

**之后的长运行都用方案 B 启动**，经 `probe_scripts/launch.sh`（节点上执行）：

- 每次调用只起一个 `setsid nohup srun --jobid=<157296|157297> --overlap -n1 …`，只接受这两个任务号，然后立即返回；
- 记录在 `$VI/logs/0044_launched.tsv`。
- 后续都用它启动：nyu 落地、测试、核对、权重。
- 原来的写法是把两个 srun 用 `&` 放在同一条 ssh 命令里，第二个没有起来，ssh 也一直挂着；用 `launch.sh` 一次只起一个就没有这个问题。

## 7. 跨机器一致性（§7）：未执行

- 卡在第 5 节。
- **驱动已写好**：`probe_scripts/driver_0044.sh`。命令行与 RUN-0034 §5 的两条相同，只换 `--data-root` 与 `--out 0044_hpc_consistency_<node>`。
  - 环境块一行一个 export，含 `OMP_NUM_THREADS=16`、`PYTHONHASHSEED=0`；
  - 带内存与 RSS 的停止线（MemAvailable < 100 GB，或我们的 RSS > 150 GB）。
- **彩排：** 在开发机上 `TINY=1` 原样跑过（6e77ab0 的工作树 consistency 两条、ea49e15 的工作树 throughput 两条，都是 rc=0）。
- `compare_train_logs.py` 对 tiny 日志自比，判定"一致"，rc=0。

## 8. 吞吐（§8）：未执行

同上，驱动的 throughput 阶段已彩排，含 `/dev/shm` 的 A/B 组。

## 9. 与方案或任务书不符之处、我自己造成的问题

- **计算节点没有 git**：RUN-0043 没查到。已在我们的 env 里装 git（§2）。
- **hpcmgt 到 GitHub 很慢**，clone 不可行，改用 bundle（§3）。RUN-0043 测的是 https 首页的连通，没有测吞吐。
- **lab5 在 07:27 重启**，中转断了一次，靠只重传缺的部分续上（§4.1）。
- **hf-mirror 不支持 Xet**，要设 `HF_HUB_DISABLE_XET=1`；长连接会降速（§4.3）。
- **测试要的 HF 缓存**（Gemma、InternVL 的 config 与 tokenizer）在 HPC 上原来没有；Gemma 是 gated，hf-mirror 拿不到。用的是开发机缓存里同 revision 的副本，sha256 都核对过（§2.1）。
- **我自己的失误：**
  - **v1/v2 权重步骤重叠：** 重启权重时，v1 的步骤还在跑，有约 2 分钟是 v1 与 v2 两个步骤同时下载到同一个缓存。之后我按 PID 停掉 v1（`STEP 157296.127 … CANCELLED`，任务 RUNNING）。0.8B 下载完整，13/13 核对。
  - **一次后台 ssh 挂着没退出**，后面要写的 `bundle_send.sh` 因此没生成，我晚了几分钟才发现。
  - **macOS tar 带进了 `._*` 文件**，已全部删除（HF 缓存与 probe 目录，共 81 个）。
  - **第 3 轮测试的完成通知晚到：** 测试约 11:47 跑完，我在约 18:00 才收到后台通知，中间没有新动作。集群任务不受影响，只是墙钟多耗了约 6 小时。
- **不在 Lustre 上对大目录 du：** 为份额，我对我们自己的子目录跑过一次 `du -s --apparent-size`，几十秒内完成。份额主要看 `lfs quota` 的差值。

## 10. Lustre 份额（`raw/quota_after_bootstrap.txt`）

- `lfs quota -u chenyuma`（KB）：开始 7,059,403,936，现在 7,098,882,948，**差 39,479,012 KB ≈ 40.4 GB**，上限 80 GB。
- 按目录（apparent bytes）：

  | 目录 | 大小 |
  |---|---|
  | env | 6.06 GB |
  | hf | 15.69 GB |
  | train_data | 10.28 GB（refcoco 4.28、nyu_depth 4.00、nyu_normals 1.01、ade20k 1.00） |
  | wt | 3.65 GB（两个工作树） |
  | repo | 2.39 GB |
  | tmp | 1.29 GB（含 bundle 0.25 GB 与 pytest 临时目录） |
  | conda-pkgs | 0.85 GB |
  | ckpt | 0 |

- 节点本地盘与 `/dev/shm` 都没有用。
- `hf/hub` 里除 Qwen 外，还有 Gemma-4-E2B-it 与 InternVL3.5-2B/4B-HF 的 config 与 tokenizer，没有权重。
  - 来历：开发机 HF 缓存里同一钉住 revision 的副本，只为全量测试拷来（§2.1）；
  - 大小见 `raw/hf_small_sizes.txt`，合计约 52.7 MB（Gemma 0.05、InternVL 各 4.5、共享 blobs 43.6 MB）；
  - 不删。

## 11. 需要用户决定

1. **MiniMax 停车：** 怎么做？
   - (a) 你在 Claude Code 里放行这一类操作（加一条 Bash 权限规则，或在提示时手动同意），我按 §5 的步骤继续，先 06 后 07；
   - (b) 你自己执行 §5 的四步，我只做核对；
   - (c) 改成与 MiniMax 共存，按 RUN-0043 §3.3 的 OOM 察觉与恢复做法来跑。这与 D89-1 不同，要新的裁定。
2. **非计时的测试失败：** `test_grid_checkpoint_split_across_merge_fails_restricted` 在 HPC 上 4/4 确定失败（§2.1）。是否先处理（管理员与开发），再开一致性与吞吐？它不涉及训练脚本本身。

之后的顺序不变：停车 → 第 7 节一致性（两台节点各 2 条，约 1 小时）→ 判定通过 → 第 8 节吞吐 → 正式训练的逐项预估。
