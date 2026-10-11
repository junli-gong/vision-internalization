# RUN-0044 — ITSC HPC A100 开训前准备（完成）

服务器 CC（HPC），在用户的 Mac 上经 `ssh a100 → gpu(hpcmgt)` 操作。2026-10-10 06:06 – 2026-10-11 05:10 UTC。只贴数字与事实，不做解读。

## 0. 一句话状态

- **第 2–8 节全部完成。**
  - 环境、全量测试、deploy key、数据（四个源）与去重、权重（0.8B、2B、4B）都已到位并核对；
  - MiniMax 两台都已停车；从节点发起 srun 测通。
- **一致性：6 组比对全部判"一致"**（§7）：每台对 B200 RUN-0034 各 2 组，06 对 07 共 2 组。
  - e3a 对 B200 的两组用了 `--ignore-key class_names_check.file`：这是唯一不同的运行身份键，是本机路径，同一处的 sha256 相同。
- **吞吐与预估**（§8）：
  - hpcgpu06（4B）：C3 × 3 + E3b+C3 × 3 约 **76 h**（+10 % 余量约 84 h）；
  - hpcgpu07（0.8B）：约 **27 h**（约 30 h）。
- **停车经过**（§5）：
  - 第 2–4 步第一次被 Claude Code 的权限分类器拦下，用户本人放行后执行。
  - 执行中发现两台的 uvicorn 与 comfy 早已处于 SIGSTOP（State T），SIGTERM 只会挂起；经用户选择，改为 SIGTERM 加 SIGCONT，进程 2 秒内正常退出。
- **铁律：** 每一步前后 squeue，两个任务始终都是 RUNNING（§1）。没有 scancel、scontrol 改动、sbatch、salloc，没有碰 slurm_script。
  - 停过的只有我们自己的步骤与进程：weights v1、v2，一个慢的 git clone，见 §9；以及 MiniMax 的子进程，见 §5。
- **Lustre 用量约 40 GB**（上限 80 GB）。

## 1. squeue

- 原始记录在 `experiments/E3/0044_hpc_bootstrap/raw/` 各文件的开头与结尾（`dirs.txt`、`park_step1_readonly.txt`、`methodB_test.txt`、`cluster_logs/logs/*`）。
- 全程每一条都是两行 RUNNING，运行时长单调增长。例如：
  - 首条：`2026-10-10T06:06:58Z  157296 RUNNING 55-17:32:44 hpcgpu06 / 157297 RUNNING 57-15:04:53 hpcgpu07`；
  - 停车第 1 步之后：`08:57:05Z  157296 RUNNING 55-20:22:51 / 157297 RUNNING 57-17:55:00`；
  - 停车与一致性前后，以及最后（2026-10-11 05:00Z 前后）：两行 RUNNING，例如 `04:58Z  157296 RUNNING 56-16:25:12 / 157297 RUNNING 58-13:57:21`。
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
  - **skip 清单**（`raw/skips_rs.txt`，`-rs` 重跑的普通套件，结果同样是 3 failed、2739 passed、19 skipped）：
    - `test_health_ifeval.py` 10 条（没有 punkt_tab）；
    - `test_e3_eval.py` 4 条与 `test_e5_train.py` 2 条（"只在一个家族上核对"）；
    - `test_e2_bilinear.py:298`（没有 CUDA）；
    - `test_e5_train_data.py:220`（规范平台）；
    - **`test_metrics_seg.py:129`：could not import 'cv2'。** 开发机装了 cv2 5.0.0，HPC 的环境里没有（B200 的 freeze 里也没有 opencv）。
  - 开发机的 skip 清单我手上没有，所以"多出的 1 条"能确定的只有：cv2 这一条在 HPC 上 skip，在开发机上不会。
- **第 3 轮的 3 条失败**，两套相同。各条单独重跑 4 次（`raw/cluster_logs/logs/0044_rerun3.log`、`0044_rerun4.txt`）：

  | 测试 | 全量时 | 单独重跑 4 次 |
  |---|---|---|
  | `test_synth_3d_tasks.py::test_speed_floor_single_core[viewpoint]`（计时） | 18.63/s < 20 | 4 次都通过 |
  | `test_synth_occlusion.py::test_speed_at_least_50_per_second_single_core`（计时） | 失败 | 4 次都失败（第 4 次 40.21/s < 50）；节点负载 1.9–5.6 |
  | **`test_check_training_interfaces.py::test_grid_checkpoint_split_across_merge_fails_restricted`**（非计时） | `assert (0 == 1)`：病态检查点的 restricted 核对返回 0（通过），测试期望 1 | **4 次都失败，结果确定** |

  - 第三条不属于"计时类"，按任务书单独列出，不解读。
  - 它是 `--model qwen3.5-0.8b`、tiny、CPU fp32 下的"合并前后网格 logits 分家"检测（`tests/test_check_training_interfaces.py:465`）。
  - 带 `-l` 的完整失败输出：`raw/failtest_grid_checkpoint_split_locals.txt`；工具写出的 `check.json`：`raw/failtest_grid_checkpoint_split_check.json`。
    - 其中 `pass: true`；restricted 的 `across_merge.by_thinking.thinking_off.teacher_forcing`：n_tokens 3171、argmax_agreement 1.0、n_disagreements 0、max_abs_logit 8.574。
    - 即病态检查点在这台机器上，合并前后的受限 logits 没有出现 argmax 分家。
  - 实验代码与测试都没有改。管理员已裁定这条不挡本任务与 A100 训练（评测前的合并核对在 B200 上做），另开开发任务查。

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

## 5. MiniMax 停车（§5）：完成

**第 1 步（只读，08:56:35–08:57:05，前后 squeue 都 RUNNING；`raw/park_step1_readonly.txt`）：**

- **启动脚本没变：** 3996 B，mtime `2026-08-12 02:55:55 +0800`，755，sha256 `e42bfde15b8d62f0b32f5251338ecab520252d75098c6b5b4d3bcf321f1afae9`。这个值与 RUN-0043 原始输出里那份脚本全文算出的 sha256 相同。
- `run/` 里没有 park 或 startup-failed 文件。
- **进程树与 RUN-0043 相同：**
  - hpcgpu06：slurm_script 3065231 → 启动脚本 3065235 → comfy 3065243（卡上 416 MiB）、uvicorn 3066512；
  - hpcgpu07：4010090 → 4010094 → comfy 4012165（416 MiB）、uvicorn 4013622。
- **新启动脚本**（`probe_scripts/park_start.sh`，sha256 `6a9acbf7…`）：内容即 RUN-0043 §3.3 第 3 步。
  - `bash -n` 通过；
  - 在 hpcmgt 上用假的 BASE 测过两种情况：没有开关文件时 exec `.orig`；有开关文件时 exec `sleep infinity`。

**第 2–4 步：**

- 2026-10-10 第一次执行时，被 Claude Code 的 auto mode 权限分类器拒绝，一条命令都没有运行，我没有绕过。用户本人放行后，18:18:44 执行（`probe_scripts/park_steps2to4.sh`，输出 `raw/pk2.out`）：
  - `cp -p` 成 `.orig`，sha256 仍是 `e42bfde1…`；
  - 新文件 `.start_…new`：755，`bash -n` 通过，sha256 `6a9acbf7…`；
  - `mv -f` 原子替换：inode 144115351133661484 → 144115375678745635；`.orig` 完好。
  - 前后 squeue 都是 RUNNING。

**第 5 步（hpcgpu06）：**

- 前两次在我自己的 PID 核对处停下，开关文件没写，什么都没动：
  - 第一次是匹配模式写错：PID 后面还有 PPID 等列；
  - 第二次是 `cut -c1-150` 把 uvicorn 那一行截成了 "uvico"。
- 第三次（`probe_scripts/park_step5_hpcgpu06.sh`，`raw/pk5_06.out`）：PID 与记录相同 → `touch run/park-hpcgpu06` → 经 `srun --overlap` 按 PID 发 SIGTERM：uvicorn 3066512、comfy 3065243，两条都是 rc=0。
- **与预期不同：** 10 s 后两个进程都还在。只读查 `/proc`（`raw/procstat_06.txt`、`raw/procstat_07.txt`）：
  - uvicorn 与 comfy 都是 **`State: T (stopped)`**，ShdPnd `0x4000`，即 SIGTERM 挂起；
  - 启动脚本与 slurm_script 是 S；freezer 是 THAWED。
  - hpcgpu07 的 uvicorn 与 comfy 同样是 T（当时我还没动 07）。
  - 只陈述事实：这与 API 日志自 08-16 起没有记录、端口上 CLOSE-WAIT 连接里有未读数据、comfy 从没执行过任务都对得上。是谁、何时暂停的，查不到。
- 停下问用户，用户选择"SIGCONT 让它们处理 SIGTERM"。
  - 18:24:54 发 SIGCONT（`probe_scripts/park_step5b_sigcont_hpcgpu06.sh`，`raw/pk5b_06.out`），两个进程 2 s 内退出。
  - uvicorn 日志：`Shutting down … Application shutdown complete. Finished server process [3066512]`。
  - 启动脚本 3065235 随之退出；slurm_script 3065231 还在；外层循环记 `MiniMax H3 exited status=143; … retrying in 300 seconds`。
  - 没有用到 kill -9。

**第 6 步（hpcgpu06，`raw/pk6_06.out`）：**

- 18:29:54：`parked on hpcgpu06 by visint (run/park-hpcgpu06)`。
- 进程树：`3065231 slurm_script → 215571 sleep infinity`。
- 我们的卡：memory.used 0 MiB、free 40,442 MiB，没有进程。squeue RUNNING。

**第 7 步（hpcgpu07，`probe_scripts/park_step5_hpcgpu07.sh`，`raw/pk5_07.out`）：**

- 18:31:03：PID 与记录相同 → `touch run/park-hpcgpu07` → SIGTERM uvicorn 4013622、comfy 4012165 → SIGCONT 两者。
- 回传的输出在 `SIGTERM uvicorn rc=0` 之后断了：ssh 流断开，我的会话随后中断了约 8 小时。事后只读核对：
  - slurm 日志 `[2026-10-10T18:31:07Z] MiniMax H3 exited status=143`、`[18:36:07Z] parked on hpcgpu07`；
  - uvicorn 日志 `Finished server process [4013622]`；
  - 进程树 `4010090 slurm_script → 202178 sleep infinity`。
- 两台的 `run/` 里都有开关文件；启动脚本是 `6a9acbf7…`，`.orig` 是 `e42bfde1…`。
- 一致性开跑前（02:56Z），两台我们的卡都是 0 MiB 占用、40,442 MiB 可用、没有进程（`raw/s7s8/logs/0044_*/gpu_snapshots.txt`）。

**恢复方法**（RUN-0043 §3.3 第 8 步）：

1. 删掉 `run/park-<节点>`；
2. 按 PID kill 那个 `sleep infinity`（06：215571；07：202178）；
3. 外层循环 300 s 后调用新脚本，没有开关文件，就 exec `.orig`，MiniMax 约 10 分钟后就绪（会是正常运行，不是暂停状态）；
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

## 7. 跨机器一致性（§7）

- **启动：** 02:56:15Z 用方案 B 在两台节点上启动 `run_both.sh`（先 consistency 再 throughput），步骤 157296.150、157297.78。
- **驱动：** `probe_scripts/driver_0044.sh`（sha256 `0733a382…`）。
  - 命令行与 RUN-0034 §5 的两条相同，只换 `--data-root $VI/data/train_data` 与 `--out …/0044_hpc_consistency_<node>`；
  - 工作树 `$VI/wt/wt0044_6e77ab0`，`6e77ab064da0…`，没有本地改动。
  - 每条运行前查闸门（我们的卡上其他进程 ≤ 1 GB 且 memory.free ≥ 35,000 MiB）并记 nvidia-smi，运行后再记一次；遇到 OOM 就停（管理员的规则）。没有触发。
  - 内存与 RSS 停止线没有触发：MemAvailable 最低约 498 GB。
- **运行**（`raw/s7s8/logs/0044_<node>/driver_consistency.log`）：

  | 节点 | e3b 50 步 | e3a 50 步 |
  |---|---|---|
  | hpcgpu06 | rc=0，611 s | rc=0，251 s |
  | hpcgpu07 | rc=0，612 s | rc=0，227 s |

- **比对**：`scripts/compare_train_logs.py`（ea49e15 版本），判据 v1-provisional；结果在 `experiments/E3/0044_hpc_consistency_compare/`。

  | 比对 | 判定 | loss 相对差：最大 | 中位数 | 最后 10 步平均 | grad_norm（只报）最大 / 中位数 | 忽略的机器字段 |
  |---|---|---|---|---|---|---|
  | e3b：06 对 B200 | 一致 | 0.00727 | 0.000366 | 0.00191 | 0.0604 / 0.00271 | 无 |
  | e3b：07 对 B200 | 一致 | 0.00506 | 0.000731 | 0.00229 | 0.364 / 0.00515 | 无 |
  | e3b：06 对 07 | 一致 | 0.00818 | 0.00143 | 0.00419 | 0.384 / 0.00362 | 无 |
  | e3a：06 对 B200 | 一致 | 0.00572 | 0.000637 | 0.000947 | 0.0444 / 0.00586 | `class_names_check.file` |
  | e3a：07 对 B200 | 一致 | 0.00445 | 0.000856 | 0.00150 | 0.0525 / 0.00709 | `class_names_check.file` |
  | e3a：06 对 07 | 一致 | 0.00686 | 0.000514 | 0.000549 | 0.0513 / 0.00616 | 无 |

  - 六组都是"数据与日程逐步相同：True（不同的步 0）"，没有 NaN 或 Inf。
- **e3a 对 B200 第一遍 rc=2（不可比）：**
  - 报错原文：`不可比：两边的运行身份（config.yaml）除机器字段外不同：['class_names_check.file']`。
  - 两边 config.yaml 的唯一差别（diff）：`file: /dev/shm/visint/train_data/ade20k/objectInfo150.txt` 对 `file: /home/mail/chenyuma/visint/data/train_data/ade20k/objectInfo150.txt`；同一处的 `sha256: c368be0c…` 两边相同。
  - 按工具文档的 `--ignore-key` 把它当作机器字段重比（`probe_scripts/compare_e3a.sh`），报告里照记。是否接受由管理员定。
  - e3b 的配置里没有这个键，06 对 07 两边路径相同，所以只有这两组受影响。

## 8. 吞吐（§8）与正式训练的逐项预估

- **运行：** 工作树 `$VI/wt/wt0044`（ea49e15）。
  - hpcgpu07：0.8B 的 C3 与 E3b+C3；hpcgpu06：4B 的同样两条；各 20 步。
  - A/B 组：同一条 E3b+C3 20 步，`--data-root` 改为拷到 `/dev/shm/visint-chenyuma/train_data` 的副本。拷贝用时：06 2,964 s、07 2,638 s，各 10,276,337,019 B；用完已删。
  - 所有运行 rc=0，没有 OOM，闸门没有触发。
- **表**由 `probe_scripts/p44_tput_table.py` 从 `accounting.json` 与 `logs/timing_run_*.jsonl` 生成（`experiments/E3/0044_hpc_bootstrap_tput_table.{md,json}`），原样：

| 节点 | 运行 | 模型 | rc | 命令墙钟 s（driver，含加载） | accounting wall_s | 步数（计时 / 记录） | 每步中位 s | 输入构建中位 s | 前向中位 s | 反向中位 s | 每步平均 s | 显存峰值 GB | 目录 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hpcgpu06 | consistency_e3b_qwen3.5-2b | qwen3.5-2b | 0 | 611 | 386.5 | 50 / 50 | 5.874 | 0.281 | 1.742 | 3.805 | 7.698 | 9.83 | `0044_hpc_consistency_hpcgpu06/e3b/qwen3.5-2b/seed0` |
| hpcgpu06 | consistency_e3a_qwen3.5-2b | qwen3.5-2b | 0 | 251 | 188.3 | 50 / 50 | 3.314 | 0.547 | 1.139 | 1.577 | 3.734 | 7.07 | `0044_hpc_consistency_hpcgpu06/e3a/full/qwen3.5-2b/seed0` |
| hpcgpu06 | tput_c3_qwen3.5-4b | qwen3.5-4b | 0 | 247 | 200.0 | 20 / 20 | 5.562 | 0.211 | 1.723 | 3.609 | 9.910 | 16.53 | `0044_hpc_throughput_hpcgpu06/c3/qwen3.5-4b/seed0` |
| hpcgpu06 | tput_e3b_c3_qwen3.5-4b | qwen3.5-4b | 0 | 224 | 192.4 | 20 / 20 | 8.636 | 0.213 | 2.559 | 5.683 | 9.525 | 19.21 | `0044_hpc_throughput_hpcgpu06/e3b_c3/qwen3.5-4b/seed0` |
| hpcgpu06 | tput_shm_e3b_c3_qwen3.5-4b | qwen3.5-4b | 0 | 265 | 180.9 | 20 / 20 | 8.586 | 0.217 | 2.554 | 5.701 | 8.949 | 19.21 | `0044_hpc_throughput_shm_hpcgpu06/e3b_c3/qwen3.5-4b/seed0` |
| hpcgpu07 | consistency_e3b_qwen3.5-2b | qwen3.5-2b | 0 | 612 | 402.1 | 50 / 50 | 5.812 | 0.119 | 1.766 | 3.838 | 7.997 | 9.83 | `0044_hpc_consistency_hpcgpu07/e3b/qwen3.5-2b/seed0` |
| hpcgpu07 | consistency_e3a_qwen3.5-2b | qwen3.5-2b | 0 | 227 | 191.4 | 50 / 50 | 3.315 | 0.543 | 1.138 | 1.592 | 3.798 | 7.07 | `0044_hpc_consistency_hpcgpu07/e3a/full/qwen3.5-2b/seed0` |
| hpcgpu07 | tput_c3_qwen3.5-0.8b | qwen3.5-0.8b | 0 | 75 | 44.5 | 20 / 20 | 2.102 | 0.186 | 0.622 | 1.248 | 2.172 | 6.40 | `0044_hpc_throughput_hpcgpu07/c3/qwen3.5-0.8b/seed0` |
| hpcgpu07 | tput_e3b_c3_qwen3.5-0.8b | qwen3.5-0.8b | 0 | 90 | 63.7 | 20 / 20 | 2.986 | 0.211 | 0.834 | 1.912 | 3.137 | 6.43 | `0044_hpc_throughput_hpcgpu07/e3b_c3/qwen3.5-0.8b/seed0` |
| hpcgpu07 | tput_shm_e3b_c3_qwen3.5-0.8b | qwen3.5-0.8b | 0 | 513 | 115.6 | 20 / 20 | 2.971 | 0.217 | 0.832 | 1.872 | 5.609 | 6.43 | `0044_hpc_throughput_shm_hpcgpu07/e3b_c3/qwen3.5-0.8b/seed0` |

### 正式训练的逐项预估（每步中位 × 配置步数；3 个种子；另列 +10 % 的加载、检查点与分段重启余量）

步数：C3 = e3b_train.json `steps` = 4000；E3b+C3 = `mixed.steps` = 8000。

| 节点 | 模型 | 变体 | 每步中位 s（Lustre） | 单个种子 h | 3 个种子 h | 3 个种子 h（+10 %） |
|---|---|---|---|---|---|---|
| hpcgpu06 | qwen3.5-4b | c3 | 5.562 | 6.18 | 18.54 | 20.39 |
| hpcgpu06 | qwen3.5-4b | e3b_c3 | 8.636 | 19.19 | 57.58 | 63.33 |
| hpcgpu07 | qwen3.5-0.8b | c3 | 2.102 | 2.34 | 7.01 | 7.71 |
| hpcgpu07 | qwen3.5-0.8b | e3b_c3 | 2.986 | 6.64 | 19.91 | 21.90 |
| hpcgpu06 | 合计（C3 × 3 + E3b+C3 × 3） | | | | 76.12 | 83.73 |
| hpcgpu07 | 合计（C3 × 3 + E3b+C3 × 3） | | | | 26.91 | 29.60 |

- **nvidia-smi 采样**（每 10 s，`raw/s7s8/logs/0044_<node>/gpu_*.csv`）：

  | 节点 | 阶段 | memory.used 最大 | 卡总显存 | 利用率最大 | 节点 load1 范围 | MemAvailable 最低 |
  |---|---|---|---|---|---|---|
  | 06 | 一致性 | 11,355 MiB | 40,960 MiB | 100 % | 0.06–11.95 | 500.6 GB |
  | 06 | 吞吐 | 21,779 MiB | 40,960 MiB | 100 % | 0.00–3.69 | 475.0 GB |
  | 07 | 一致性 | 11,355 MiB | 40,960 MiB | 100 % | 0.92–7.71 | 498.3 GB |
  | 07 | 吞吐 | 7,213 MiB | 40,960 MiB | 100 % | 0.01–5.57 | 475.9 GB |

- **hpcgpu07 的可用显存**（管理员要求）：每条运行之前都是 40,442 MiB free，卡上没有进程（`gpu_snapshots.txt`）。
- **0.8B 的显存峰值**：`accounting.json` 的 peak_mem 是 C3 6.40 GB、E3b+C3 6.43 GB；nvidia-smi 看到的最大 memory.used 是 7,213 MiB。
- **Lustre 对 /dev/shm（只陈述，不解读）：**
  - 每步中位：06 是 8.636 s 对 8.586 s；07 是 2.986 s 对 2.971 s。输入构建中位：06 0.213 对 0.217 s，07 0.211 对 0.217 s。
  - 但 07 的 `/dev/shm` 那条，命令墙钟 513 s（Lustre 那条 90 s），accounting wall_s 115.6 对 63.7，每步平均 5.609 对 3.137 s。
  - 06 的命令墙钟 265 对 224 s，accounting wall_s 180.9 对 192.4。
  - 管理员已定：正式训练的数据仍放 Lustre。
- **预估的读法：** 每步中位 × 配置步数（C3 4,000、E3b+C3 8,000），不含加载与检查点；另列 +10 %。
  - 4B 的 C3 每步平均（9.910 s）比中位（5.562 s）高得多，原因是前几步慢（`timing_run_001.jsonl`）。20 步太短，平均值不代表稳态。
- **与 RUN-0043 §3.8 的估计对照（只陈述）：** 当时估 0.8B 26–32 h、4B 58–68 h；按本次中位算是 26.9 h 与 76.1 h。4B 的 E3b+C3 单个种子 19.2 h，高于当时估的 14–17 h。

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
  - **停车第 5 步的 PID 核对写错两次**（匹配模式、行截断），都在动手之前就停下了，没有副作用（§5）。
  - **会话中断约 8 小时**（18:31 → 02:55 UTC）：07 的停车已经在 18:36 完成，一致性因此晚开了约 8 小时。
- **MiniMax 的进程处于 SIGSTOP**：RUN-0043 的调查没查进程状态，方案按"空闲在运行"来写，实际是暂停。改用 SIGTERM 加 SIGCONT，由用户选定（§5）。
- **e3a 一致性的 `class_names_check.file`**：比对工具的机器字段列表里没有这个本机路径，要 `--ignore-key`（§7）。
- **hpcgpu07 上 `/dev/shm` 的 A/B 墙钟比 Lustre 长**（§8），与"内存盘更快"的预期相反，照实列出。
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

## 11. 待管理员确认

1. e3a 对 B200 两组用了 `--ignore-key class_names_check.file`（§7），是否接受。
2. `test_grid_checkpoint_split_across_merge_fails_restricted` 在 HPC 上 4/4 确定失败（§2.1），管理员已定另开开发任务，不挡训练。
3. 正式训练的预估（§8）：4B 约 76 h（+10 % 约 84 h），比 RUN-0043 的估计长。

本任务到此结束，没有开正式训练。正式训练按 RUN-0045 另行执行。
