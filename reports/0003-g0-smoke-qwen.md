# RUN-0003 — G0 冒烟（Qwen3.5）：服务器全量测试 + 0.8B 与 4B 适配器冒烟

服务器 CC，b200，2026-09-30 07:00–07:25 UTC（值守接单）。只贴数字与事实，不做解读。

**状态：按任务书 §3 停在全量测试。** 服务器上有 16 个测试失败，而开发机是 312 passed。按任务书，这属于"开发机绿、算力机红"：停下，推回 traceback 与环境差异，不改代码。**§4 的适配器冒烟没有运行**；任务书 §4 里的删除 Gemma、下载 4B 也都没有执行。

## 0. 一句话观察（只陈述，不解读）

- 服务器全量测试：`16 failed, 295 passed, 1 skipped, 14 warnings in 37.64s`。
- 16 个失败全在 `tests/test_adapter_qwen35.py`：
  - 15 个是 `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`，抛出位置在 `fla` 的 Triton 内核（`fla/ops/gated_delta_rule/chunk.py` → `fla/modules/l2norm.py` → `triton/backends/nvidia/driver.py:328`）；
  - 1 个是 `test_smoke_script_tiny`：`--tiny` 子进程返回码 1，检查项 1–8 全部为 false。
- 诊断：同一套测试，只在该进程树里让 `import fla` 失败（仓库代码与已装包都不动），结果是 `311 passed, 1 skipped in 29.35s`。

## 1. 前置校验

- `git pull --ff-only` 后 HEAD 为 `a55ca61`；`a4a60e8`（DEV-0003 的 merge）在 HEAD 历史中（`git merge-base --is-ancestor` 成立）。
- `pip install --dry-run -r requirements.txt`（venv）：所有依赖均为 "Requirement already satisfied"，没有 "Would install"，不会改动任何已装包；transformers==5.17.0 已装（`pip_dryrun_requirements.txt`）。
- 本次没有修改任何仓库代码、config 或已装包。

## 2. 执行记录

| 步骤 | 命令 | 墙钟 | 结果 |
|---|---|---|---|
| 依赖 dry-run | `pip install --dry-run -r requirements.txt` | 数秒 | 不改动任何包 |
| 全量测试 | `cd $REPO && python -m pytest tests/ -q`（venv；环境来自 env.sh，HF_HUB_CACHE=/dev/shm/visint/hf-hub） | 39.3 s（pytest 自报 37.64 s） | **16 failed, 295 passed, 1 skipped**，rc=1 |
| 诊断（不是修复） | `env PYTHONPATH=$PROJ/tmp/diag_nofla python -m pytest tests/ -q`；该目录下的 `sitecustomize.py` 只有一行 `sys.modules['fla'] = None`（副本 `diag_nofla_sitecustomize.py`） | 30.4 s | **311 passed, 1 skipped**，rc=0 |
| 跳过项原因 | 同上加 `-rs` | 29.6 s | `SKIPPED [1] tests/test_metrics_seg.py:129: could not import 'cv2': No module named 'cv2'` |
| §4 冒烟 | — | — | **未运行**（按 §3 的规定停下） |

GPU：测试路径不用 GPU（小型随机模型在 CPU 上）；机器上 B200 可见（`torch.cuda.is_available()`=True）。

## 3. 结果表（原样）

失败清单（`pytest.txt` 为完整原始输出，含每个失败的全部 traceback）：

- `tests/test_adapter_qwen35.py::test_empty_instruction_and_zero_images`
- `tests/test_adapter_qwen35.py::test_image_first_bitwise_equal_instruction_first_differs`
- `tests/test_adapter_qwen35.py::test_layer_numbering_matches_decoder_outputs`
- `tests/test_adapter_qwen35.py::test_hidden_states_shapes_and_order`
- `tests/test_adapter_qwen35.py::test_own_positions_and_merge_equal_model_path[instruction_first]`
- `tests/test_adapter_qwen35.py::test_own_positions_and_merge_equal_model_path[image_first]`
- `tests/test_adapter_qwen35.py::test_vision_features_equal_layer0_image_positions`
- `tests/test_adapter_qwen35.py::test_query_causal_prefix_bitwise_equal_across_modes`
- `tests/test_adapter_qwen35.py::test_query_embeds_replace_placeholder_and_carry_grad`
- `tests/test_adapter_qwen35.py::test_add_250_grows_to_exact_rows[True]`
- `tests/test_adapter_qwen35.py::test_add_250_grows_to_exact_rows[False]`
- `tests/test_adapter_qwen35.py::test_allowed_token_ids_restrict_generation`
- `tests/test_adapter_qwen35.py::test_generation_result_fields[False]`
- `tests/test_adapter_qwen35.py::test_generation_result_fields[True]`
- `tests/test_adapter_qwen35.py::test_bf16_model`
- `tests/test_adapter_qwen35.py::test_smoke_script_tiny`

错误类型计数（`pytest.txt` 里以 `E ` 开头的行）：`ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)` ×15；`AssertionError`（test_smoke_script_tiny）×1。

第一个失败（`test_empty_instruction_and_zero_images`）的调用栈摘要（完整栈见 `pytest.txt`）：

```
tests/test_adapter_qwen35.py:328
src/adapters/qwen35.py:590  in hidden_states
src/adapters/qwen35.py:931  in _run
transformers/models/qwen3_5/modeling_qwen3_5.py:1855 in forward      (Qwen3_5ForConditionalGeneration)
transformers/models/qwen3_5/modeling_qwen3_5.py:1648 in forward      (Qwen3_5Model)
transformers/models/qwen3_5/modeling_qwen3_5.py:1292 in forward      (Qwen3_5TextModel)
transformers/models/qwen3_5/modeling_qwen3_5.py:889  in forward      (Qwen3_5DecoderLayer)
transformers/models/qwen3_5/modeling_qwen3_5.py:639  in forward      (Qwen3_5GatedDeltaNet -> torch_chunk_gated_delta_rule)
transformers/integrations/hub_kernels.py:885 in wrapped              (implementation = fla chunk_gated_delta_rule)
fla/ops/gated_delta_rule/chunk.py:567 in chunk_gated_delta_rule
fla/ops/gated_delta_rule/chunk.py:282 in forward
fla/modules/l2norm.py:178 in l2norm_fwd
triton/runtime/autotuner.py ... triton/runtime/jit.py:761 in run
triton/backends/nvidia/driver.py:328: ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
```

`test_smoke_script_tiny`（`tests/test_adapter_qwen35.py:879`）：

```
AssertionError: {"0_facts": true, "1_build_generate": false, "2_task_modulation": false, "3_grid_tokens": false, "4_query_tokens": false, "5_position_ids": false, "6_vision_features": false, "7_thinking": false, "8_device_audit": false}
assert (1 == 0)   # 子进程 returncode
```

诊断运行：`311 passed, 1 skipped in 29.35s`。transformers 解析到的 GDN 实现，默认环境下为 `fla.ops.gated_delta_rule.chunk`；屏蔽 fla 后为 `transformers.models.qwen3_5.modeling_qwen3_5`（torch 回退）。两次都直接用 `inspect.getclosurevars` 读取，见下节。

## 4. 与任务书假设不符的发现

1. **服务器与开发机的环境差异**（服务器实测见 `server_env.txt`；开发机我看不到，需管理员补）：
   ```
   torch 2.13.0+cu130 cuda_available True NVIDIA B200
   torchvision 0.28.0+cu130
   flash-linear-attention 0.5.2
   fla-core 0.5.2
   triton 3.7.1
   opencv-python not installed
   opencv-python-headless not installed
   causal-conv1d not installed
   flash-attn not installed
   transformers 5.17.0
   numpy 2.5.2
   ```
   - 服务器的 venv 装了 **flash-linear-attention 0.5.2 / fla-core 0.5.2**（RUN-0001 §3 批次 2）。transformers 5.17 在 **import 时**就把 `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` 绑定到 fla 的 Triton 内核（RUN-0001 model_notes §5.6.3），与张量在哪个设备上无关。测试里的小型随机 Qwen3.5 在 CPU 上跑，于是 CPU 张量进了 Triton 内核。
   - 本次直接读取的绑定结果：默认环境为 `fla.ops.gated_delta_rule.chunk`；`sys.modules['fla']=None` 时为 `transformers.models.qwen3_5.modeling_qwen3_5`。
   - 屏蔽 fla 后 311 passed。唯一的跳过项是 opencv（cv2）未装：服务器没有 cv2，D10 允许 opencv 只在开发机 visint 环境里用于测试对照。311 + 1 = 312，与开发机的 312 passed 条数相同。
   - 其余差异（只列事实）：服务器 torch 2.13.0+cu130，CUDA 可用；torchvision 0.28.0+cu130 已装；triton 3.7.1；causal-conv1d 与 flash-attn 未装。
2. **任务书 §4 的冒烟命令用了 `--dtype bfloat16`**，但 `scripts/smoke_adapter.py` 的 argparse 是 `choices=sorted(DTYPES)`，即 `['bf16', 'fp32']`（脚本开头的说明也写 `--dtype bf16`）。照抄会被 argparse 拒绝。本次 §4 没有运行；继续时按脚本的写法用 `--dtype bf16`，除非另有指示。
3. pytest 的 `tmp_path` 落在 `$PROJ/tmp/pytest-of-root/`（根盘，因为 TMPDIR=$PROJ/tmp），体积为 KB 级。

## 5. 自己造成的错误或浪费

无。诊断运行多花了约 60 s，属于为报告取证，不是修复。

## 6. 生成文件清单与磁盘收支

**仓库内**：`experiments/P0/0003_g0_smoke_qwen/`：
- `pytest.txt`（全量测试原始输出，含全部 traceback）
- `pytest_diag_fla_blocked.txt`、`pytest_diag_skip_reason.txt`、`diag_nofla_sitecustomize.py`
- `server_env.txt`、`pip_dryrun_requirements.txt`、`disk_ledger.tsv`、`README.md`

另有本 report。两份冒烟 JSON **不存在**，因为 §4 没有运行。

**仓库外**：`$PROJ/logs/0003/`（原件）；`$PROJ/tmp/diag_nofla/`（1 个 3 行文件）；`$PROJ/tmp/pytest-of-root/`（pytest 临时目录）。

**内存盘**：本任务没有变动，仍为 16.765 GB（Qwen3.5-0.8B、Gemma 4 E2B-it、InternVL3.5-2B-HF）。§4 要求的"删 Gemma、下载并删除 4B"没有执行。

**磁盘收支**（`disk_ledger.tsv`；GB = 1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 07:20:47 | 0003 start | 5.905 | 16.765 | 0.482 |
| 07:21:26 | after pytest | 5.902 | 16.765 | 0.485 |
| 07:25:01 | after diag pytest; stop at §3 | 5.901 | 16.765 | 0.485 |

根盘最低 5.901 GB（红线 3 GB）。没有删除、移动或修改任何非本项目文件。
