# RUN-0025 — E1 验收 (i) 的诊断：qwen3.5-2b 定位同配置重跑那 1 条不同（GPU 上的非确定性）

服务器 CC，b200，2026-10-02 13:34–13:41 UTC，值守中接单。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- **没有复现。**
  - 第 4 批（64 条，含 `refcoco+_unc/val/7909`）在 S1、S3、S4 各 20 遍、S2 三个新进程各 3 遍里，每一遍都与本进程第 1 遍逐位相同。
  - 6 个进程第 1 遍的 64 条 token 序列也彼此相同，sha256 前 16 位都是 `5a220fe5b20a3d19`。
- **完整开发子集也没有复现。** S1、S3 下各跑一次 `run_e1.py --acceptance --task refexp_box`（600 条）：A1 与 A2 逐字节相同（0 条不同），`all_pass=True`。
- **本次所有运行都与 RUN-0024 的 A2 相同，与 RUN-0024 的 A1 差 1 条。**
  - 两次完整运行的 A1 文件，sha256 都等于 RUN-0024 A2 的 `320edc1d2664…`，与 RUN-0024 A1 只差 `refcoco+_unc/val/7909` 这 1 条。
  - 这一条在本次所有设置里都是 `[508, 496, 631, 988]`，与 RUN-0024 A2 相同；RUN-0024 A1 是 `[506, 496, 631, 988]`。
- **确定性设置的代价：**
  - 第 4 批每遍 generate 墙钟的中位数：S1 1.391 s，S3 1.553 s（+11.6%），S4 1.551 s。
  - 完整开发子集 A1 遍的墙钟：S1 14.8 s，S3 16.2 s（+9.5%）。
- **warn_only：** S3、S4、full_S3 期间没有任何与确定性相关的 warning。去重后各 8 条，都是 `torch.jit.script_method` 的弃用提示，以及 fla 的 "Flash Attention is not installed"。
- **fla 0.5.2 源码：**
  - GDN 的 chunk 前向路径（预填）上的内核都用 `fla_cache_autotune`（Triton autotune），这些文件里没有 `atomic`。
  - 解码用的 `fused_recurrent_gated_delta_rule_fwd_kernel` 没有 autotune，也没有 atomic。
  - 唯一与固定内核配置有关的环境变量是 `FLA_CACHE_MODE`。但本机没有 `fla/configs/NVIDIA_B200/`，找不到配置文件时它回退到 Triton autotune，所以 S4 与 S3 的内核选择路径相同。见 §3.6。
- **用时：** 累计 329 s（0.09 h）。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `317a09e`。本任务书所在的 `6b348c3` 在 HEAD 的历史里。
- **环境：** `pip install --dry-run -r requirements.txt` 没有任何 "Would install"。环境变量同 RUN-0024。
- **运行用的代码：** 全部在固定于 `317a09e` 的 worktree 里运行，跑完已删除。worktree 里未跟踪的只有两个探针的副本，删除前核对过与本目录的版本逐字节相同。
- **改动范围：**
  - 没有修改 `src/`、`scripts/`、`tests/`、`configs/`，也没有安装包。
  - 探针只调用现有接口，放在 `probe_scripts/` 下：`rerun_batch.py`、`run_e1_det.py`、`p27_det_summary.py`、`run_probe_25.sh`。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。
- **模型与数据：** 只用了 qwen3.5-2b，权重 13/13 核对一致。只用了开发子集。

## 2. 执行记录

| 步骤 | 进程 pid | 墙钟（`time` real） | rc |
|---|---|---|---|
| S1（20 遍） | 1009874 | 39.5 s | 0 |
| S2a / S2b / S2c（各 3 遍） | 1010271 / 1010551 / 1010748 | 15 / 14 / 19 s | 0 / 0 / 0 |
| S3（20 遍） | 1010943 | 43 s | 0 |
| S4（20 遍） | 1011190 | 42 s | 0 |
| full_S1（A1、A2、B，600 条） | 1011447 | 1m07.7s | 0 |
| full_S3（同上） | 1011875 | 1m12.5s | 0 |

- **GPU：** NVIDIA B200，bf16。
  - nvidia-smi 采样期间 GPU 上依次出现这 8 个 python 进程，时间上互不重叠，没有别的进程。
  - 按进程的显存峰值是 9,406 MiB。
- **版本：** torch 2.13.0+cu130，CUDA 13.0，cuDNN 92000，fla 0.5.2，triton 3.7.1。
  - S3、S4 中 `torch.are_deterministic_algorithms_enabled()` 为 True，S1 为 False。
  - 所有设置下 `torch.backends.cudnn.deterministic=False`、`cudnn.benchmark=False`。
- **探针与 run_e1 的一处不同：** 探针给 `generate_batch` 多传了一个只读的 logits 处理器，记录每一步的前两名、原样返回 scores。两次完整开发子集运行走的是 run_e1 原来的路径，没有这个处理器。

## 3. 结果表（原样）

由 `probe_scripts/p27_det_summary.py` 生成，与 `det_summary.md` 相同。

#### 每种设置：同一进程里与第 1 遍不同的条数（第 4 批，64 条；每遍一列）

| 设置 | pid | 环境 | deterministic | 遍数 | 每遍与第 1 遍不同的条数 | 不同的位置 | 每遍 generate 墙钟 s：中位 / 最小 / 最大 | 模型加载 s |
|---|---|---|---|---|---|---|---|---|
| S1 | 1009874 | （同 RUN-0024） | False | 20 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 无 | 1.391 / 1.382 / 1.568 | 5.8 |
| S2a | 1010271 | （同 RUN-0024） | False | 3 | [0, 0, 0] | 无 | 1.414 / 1.411 / 1.597 | 5.22 |
| S2b | 1010551 | （同 RUN-0024） | False | 3 | [0, 0, 0] | 无 | 1.381 / 1.379 / 1.568 | 5.24 |
| S2c | 1010748 | （同 RUN-0024） | False | 3 | [0, 0, 0] | 无 | 1.387 / 1.386 / 1.573 | 10.34 |
| S3 | 1010943 | CUBLAS_WORKSPACE_CONFIG=:4096:8 | True | 20 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 无 | 1.553 / 1.549 / 1.734 | 5.1 |
| S4 | 1011190 | CUBLAS_WORKSPACE_CONFIG=:4096:8, FLA_CACHE_MODE=default | True | 20 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 无 | 1.551 / 1.545 / 1.731 | 4.67 |

#### 跨进程与跨设置：各进程第 1 遍的 64 条 token 序列（与 S1 第 1 遍比较；sha256 是 64 行 token id 的 JSON）

| 设置 | 第 1 遍 sha256（前 16 位） | 与 S1 第 1 遍不同的条数 | 不同的 id |
|---|---|---|---|
| S1 | 5a220fe5b20a3d19 | 0 | 无 |
| S2a | 5a220fe5b20a3d19 | 0 | 无 |
| S2b | 5a220fe5b20a3d19 | 0 | 无 |
| S2c | 5a220fe5b20a3d19 | 0 | 无 |
| S3 | 5a220fe5b20a3d19 | 0 | 无 |
| S4 | 5a220fe5b20a3d19 | 0 | 无 |

#### 与 RUN-0024 的对照：第 4 批（第 192–255 条）的 raw_text

| 比较 | 不同的条数（/64） | 不同的 id |
|---|---|---|
| S1 第 1 遍 vs RUN-0024 A1 | 1 | refcoco+_unc/val/7909 |
| S1 第 1 遍 vs RUN-0024 A2 | 0 | 无 |
| S1 第 1 遍 vs RUN-0024 B | 41 | refcoco_unc/val/136108, refcoco_unc/val/137635, refcoco_unc/val/137938, refcoco_unc/val/139675, refcoco_unc/val/140960, refcoco+_unc/val/1162, refcoco+_unc/val/2207, refcoco+_unc/val/2718, refcoco+_unc/val/6472, refcoco+_unc/val/7186, refcoco+_unc/val/7543, refcoco+_unc/val/7909, refcoco+_unc/val/8452, refcoco+_unc/val/12233, refcoco+_unc/val/17563, refcoco+_unc/val/17675, refcoco+_unc/val/18811, refcoco+_unc/val/19023, refcoco+_unc/val/19471, refcoco+_unc/val/19848, refcoco+_unc/val/21625, refcoco+_unc/val/22732, refcoco+_unc/val/24209, refcoco+_unc/val/24441, refcoco+_unc/val/25737, refcoco+_unc/val/26021, refcoco+_unc/val/26884, refcoco+_unc/val/26949, refcoco+_unc/val/27165, refcoco+_unc/val/27984, refcoco+_unc/val/30576, refcoco+_unc/val/31296, refcoco+_unc/val/31298, refcoco+_unc/val/31688, refcoco+_unc/val/32538, refcoco+_unc/val/34159, refcoco+_unc/val/34919, refcoco+_unc/val/35493, refcoco+_unc/val/35767, refcoco+_unc/val/37793, refcoco+_unc/val/37798 |

`refcoco+_unc/val/7909` 的 raw_text：

| 来源 | raw_text |
|---|---|
| RUN-0024 A1 | ````json\n[\n\t{\"bbox_2d\": [506, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| RUN-0024 A2 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| RUN-0024 B（同 id） | ````json\n[\n\t{\"bbox_2d\": [508, 495, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S1 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S2a 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S2b 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S2c 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S3 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S4 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |

#### 完整开发子集（run_e1.py --acceptance --task refexp_box，600 条；A1 与 A2，另有 B）

| 设置 | A1 vs A2 文件逐字节相同 | A1 vs A2 不同的条数 | A1 sha256（前 16 位） | A1 = RUN-0024 A1 | A1 = RUN-0024 A2 | 与 RUN-0024 A1 不同的条数 | 墙钟 s：A1 / A2 / B（compute.json） | all_pass |
|---|---|---|---|---|---|---|---|---|
| S1 | True | 0 | 320edc1d26647052 | False | True | 1 | 14.8 / 14.1 / 16.2 | True |
| S3 | True | 0 | 320edc1d26647052 | False | True | 1 | 16.2 / 15.9 / 18.1 | True |

#### 第 1 个不同位置的前两名 token 与 logit 差

本次所有设置里都没有出现与第 1 遍不同的条，没有可记录的分歧位置。

#### warn_only 列出的非确定性算子（torch.use_deterministic_algorithms(True, warn_only=True) 期间的 warnings，去重）

- S3（0 条与确定性相关；全部 warnings 去重后 8 条）：无
- S4（0 条与确定性相关；全部 warnings 去重后 8 条）：无
- full_S3（0 条与确定性相关；全部去重后 8 条；deterministic_algorithms=True，rc=0）：无

### 3.6 fla 0.5.2 源码的事实（`$PROJ/venv/lib/python3.12/site-packages/fla/`）

| 文件 | 在 GDN 前向里的角色 | `@…autotune(` 的行号 | `atomic` |
|---|---|---|---|
| `ops/gated_delta_rule/chunk.py` | chunk 入口（预填） | 无 | 无 |
| `ops/gated_delta_rule/chunk_fwd.py` | chunk 内部的前向 | 30 | 无 |
| `ops/gated_delta_rule/wy_fast.py` | `recompute_w_u_fwd` 等 | 29、109 | 无 |
| `ops/gated_delta_rule/gate.py` | `gdn_gate_chunk_cumsum` 等 | 54、112、233 | 无 |
| `ops/common/chunk_delta_h.py` | `chunk_gated_delta_rule_fwd_h` | 39、348 | 无 |
| `ops/common/chunk_o.py` | `chunk_fwd_o` | 33、149、337、442 | 无 |
| `ops/common/chunk_scaled_dot_kkt.py` | `chunk_scaled_dot_kkt_fwd` | 22 | 无 |
| `ops/utils/solve_tril.py` | `solve_tril` | 28、97、187 | 无 |
| `ops/utils/cumsum.py` | `chunk_local_cumsum` | 24、80、140、202 | 无 |
| `modules/l2norm.py` | `l2norm_fwd` | 21、49、78、112 | 无 |
| `ops/gated_delta_rule/fused_recurrent.py` | 解码（recurrent） | 无（第 19 行 `@triton.heuristics`，第 29 行 `@triton.jit`） | 无 |

- **autotune 装饰器：** 上表的 autotune 都是 `fla_cache_autotune`（`ops/utils/cache.py` 第 405 行起），它是 `triton.runtime.autotuner.Autotuner` 的子类 `CachedAutotuner`。
  - `autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS}`（`utils/_compat.py` 第 29 行）。
  - `FLA_CACHE_RESULTS` 默认 `'1'`（`utils/_config.py` 第 11 行）。
- **`FLA_CACHE_MODE`**（`ops/utils/cache.py` 第 27–62 行）：
  - 可取 disabled（默认）/ strict / fuzzy / full / default / always。
  - `load_cached_config` 从 `get_fla_config_dir()/<kernel_name>.json` 读配置，文件不存在就返回 None（第 299–300 行），这时回退到 Triton autotune。
  - 本机 `get_fla_config_dir()` 是 `fla/configs/NVIDIA_B200`，不存在（fla 安装目录下也没有 `configs/`）。
- **其他 `FLA_*` 环境变量**（grep `os.environ`/`getenv`）：
  - `FLA_TRIL_PRECISION`（默认 `ieee`，`ops/utils/solve_tril.py` 第 19 行）；
  - `FLA_USE_FAST_OPS`（`ops/utils/op.py` 第 16 行）；
  - `FLA_DISABLE_BACKEND_DISPATCH`（`ops/backends/__init__.py` 第 28 行）；
  - `FLA_CONV_BACKEND`、`FLA_USE_TMA`、`FLA_DISABLE_TENSOR_CACHE`、`FLA_TENSOR_CACHE_SIZE`、`FLA_CI_ENV`、`FLA_INTRACARD_MAX_SPLITS`、`FLA_USE_COMPILE`、`FLA_GPU_NAME`、`FLA_CONFIG_DIR`。
  - 其中没有名为"deterministic"或关掉 autotune 的开关。
  - `utils/_device.py` 第 152 行把 `TRITON_F32_DEFAULT` 设为 `ieee`。

## 4. 与任务书假设不符的发现

- **没有复现，也就没有"第 1 个不同位置的前两名 token 与 logit 差"可记。** 记录器在每一遍都在跑，见 `probe_out/*.json`，但没有分歧可读。
- **S4 的"fla 确定性开关"：** fla 0.5.2 里没有这样的开关。最接近的 `FLA_CACHE_MODE=default` 在本机找不到配置文件，回退到 autotune，所以 S4 的内核选择与 S3 相同。照任务书跑了，结果见上。
- **RUN-0024 里 A1 是先跑的一遍**（同一进程里的顺序是 A1 → A2 → B）。本次 full_S1 按同样的顺序跑，A1 与 RUN-0024 的 A2 相同。
- **用时：** 任务书预计 1–1.5 GPU 小时，实际 329 s。

## 5. 生成的文件

`experiments/E1/0025_determinism_probe/`（6.8 MB，83 个文件）：

- `probe_out/`（6 份探针 JSON 与 full_S3 的 warnings）；
- `0025_full_S1/`、`0025_full_S3/`；
- `det_summary.md/json`、`README.md`、`disk_ledger.tsv`、`pip_dryrun.txt`；
- `logs/`、`probe_scripts/`。

一次推回。

## 6. 磁盘与内存盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 13:35:02 | 0025 start | 38.015 | 3.329 | 4.699 |
| 13:35:02 | before download qwen3.5-2b | 38.015 | 3.329 | 4.699 |
| 13:35:18 | after download qwen3.5-2b | 38.015 | 7.888 | 4.699 |
| 13:40:30 | after probes | 38.007 | 7.889 | 4.705 |
| 13:40:31 | after delete qwen3.5-2b | 38.007 | 3.331 | 4.705 |
| 13:41:47 | 0025 end (worktree removed) | 38.008 | 3.081 | 4.704 |

- **根盘：** 剩 38.0 GB，远高于 3 GB 红线。
- **`$PROJ`：** 4.704 GB，在 15 GB 份额以内。
- **内存盘：** 峰值 7.889 GB，低于 25 GB 上限。qwen3.5-2b 跑完已删除（释放 4.558 GB），hf-hub 只剩 Qwen3.5-0.8B；worktree 已删除。
