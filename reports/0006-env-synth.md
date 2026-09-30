# RUN-0006 — 服务器环境更新（pymunk）+ 全量测试 + E5 生成器的跨机器复现核对与生成速度

服务器 CC，b200，2026-09-30 11:13–11:19 UTC（值守接单）。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- pip 只新装了 cffi 2.1.1、pycparser 3.0、pymunk 7.3.0（Chipmunk 2.0.1-ade7ed7），没有升降任何已装包。
- 全量测试：`2 failed, 849 passed, 1 skipped, 23 warnings in 117.99s (0:01:57)`。两个失败都是钉住摘要的核对，任务书规定这不是停下的条件；没有其他失败。
- 10 个钉住的 case 里，**8 个与开发机的摘要相同，2 个不同**：`occlusion/test/2`（c）与 `physics/test/2`（d）。
  - 这两个 case 的 scene_seed 相同（562949953421314）。
  - `imagery/test/2` 用的也是同一个 scene_seed，它的摘要相同。
- 诊断：关掉 numpy 的 AVX2/AVX-512 分派（`NPY_DISABLE_CPU_FEATURES`，已确认生效）后，这两个 case 的摘要不变。
- 生成速度（单进程，各 200 个样本）：c 96.8、d 195.0、e 303.7 样本/s（墙钟）。

## 1. 前置校验

- `git pull --ff-only` 后 HEAD 为 `7733cda`；`5b59be7` 在 HEAD 历史中。
- 本次没有修改 `src/`、`scripts/`、`tests/`、`configs/`，也没有改摘要。新写的只有 `experiments/P0/0006_env_synth/probe_scripts/p10_synth_probe.py`，它只调用 `src.synth` 的公开接口（`generate`、`TASK_BINS`、`Sample` 的公开字段），钉住的摘要直接从 `tests/test_synth_repro.py` 导入。

## 2. 执行记录

| 步骤 | 命令 | 墙钟 | 结果 |
|---|---|---|---|
| 依赖 dry-run | `pip install --dry-run -r requirements.txt` | 数秒 | `Would install cffi-2.1.1 pycparser-3.0 pymunk-7.3.0`，没有 uninstall（`pip_dryrun.txt`） |
| 安装 | `PIP_NO_CACHE_DIR=1 pip install -r requirements.txt` | 0.6 s | `Successfully installed cffi-2.1.1 pycparser-3.0 pymunk-7.3.0` |
| 全量测试 | `python -m pytest tests/ -q` | 约 2 min（pytest 自报 117.99 s） | 2 failed, 849 passed, 1 skipped, 23 warnings in 117.99s (0:01:57) |
| §4.1–4.2 摘要与分量子摘要 | `p10_synth_probe.py digests synth_digests.json` | 0.4 s | 8/10 相同 |
| 诊断 | `NPY_DISABLE_CPU_FEATURES=... p10_synth_probe.py digests_only`（三种设置） | 各约 2 s | 三个 test-2 case 的摘要在三种设置下都一样 |
| §4.3 速度 | `p10_synth_probe.py speed speed.json` | 4.0 s | 见 §3 |

- 平台：x86_64 Linux，AMD EPYC 9575F，Python 3.12.13，numpy 2.5.2，pymunk 7.3.0，Chipmunk 2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3。
- 测试 warnings 23 条（原文见 `pytest.txt`）。

## 3. 结果表（原样）

### 3.1 全量测试

`2 failed, 849 passed, 1 skipped, 23 warnings in 117.99s (0:01:57)`，失败项：

- `tests/test_synth_repro.py::test_pinned_digests_occlusion_and_imagery`
- `tests/test_synth_repro.py::test_pinned_digests_physics`

`tests/test_synth_repro.py::test_pinned_digests_occlusion_and_imagery`：
```
E       AssertionError: assert {('occlusion'...47d9ad8', ...} == {('occlusion'...47d9ad8', ...}
E         {('occlusion', 'test', 2): 'd2ef5d62b5258d4abb36a33d4d7d652777c9dc9fee01c4a72996a74693327c1f'} != {('occlusion', 'test', 2): '019931825e2b2a7957c0d69dd3d46fc9931c115695fdfff4da3abb974f5cc246'}
E       AssertionError: assert {('physics', ...6930a4c0f77c'} == {('physics', ...6930a4c0f77c'}
```
`tests/test_synth_repro.py::test_pinned_digests_physics`：
```
E       AssertionError: assert {('physics', ...6930a4c0f77c'} == {('physics', ...6930a4c0f77c'}
E         {('physics', 'test', 2): 'f4a84d111dd2b7d40320f2594c3555b8c0429e2552f4a604b7bf322468725f10'} != {('physics', 'test', 2): '7628065e65af781b85e2fbf1fa18019fc6f15020e36a5f7d9935cb456444b148'}
```

跳过项与 RUN-0004 相同（`tests/test_metrics_seg.py:129`，cv2 未装）。

### 3.2 摘要比对表（`synth_digests.json`）

| case | scene_seed | 难度值 | 服务器摘要 | 钉住的摘要（开发机） | 相同 |
|---|---|---|---|---|---|
| imagery/test/2 | 562949953421314 | 0.18255995754372764 | `19303b5b3aab28b1f15962950aeb31ed5874c3ba3a9d2b92613dba29c4617f6a` | `19303b5b3aab28b1f15962950aeb31ed5874c3ba3a9d2b92613dba29c4617f6a` | **True** |
| imagery/train/0 | 0 | 0.04253260431912642 | `d61cc558399a54c8eb9fc01f1324c0cde38628184a5094987226caba447d9ad8` | `d61cc558399a54c8eb9fc01f1324c0cde38628184a5094987226caba447d9ad8` | **True** |
| imagery/val/1 | 281474976710657 | 0.05384898489270067 | `770c74a1b120b5b3d5902959d19ca1c42df425839a32aa1035dbfa4509f76b42` | `770c74a1b120b5b3d5902959d19ca1c42df425839a32aa1035dbfa4509f76b42` | **True** |
| occlusion/test/2 | 562949953421314 | 0.5668023116643299 | `d2ef5d62b5258d4abb36a33d4d7d652777c9dc9fee01c4a72996a74693327c1f` | `019931825e2b2a7957c0d69dd3d46fc9931c115695fdfff4da3abb974f5cc246` | **False** |
| occlusion/train/0 | 0 | 0.24626262626262627 | `f8336d559de7d67752d1881a3840dbaad7f2687058f87e24709b8f64d548a0a0` | `f8336d559de7d67752d1881a3840dbaad7f2687058f87e24709b8f64d548a0a0` | **True** |
| occlusion/val/1 | 281474976710657 | 0.42929350511154185 | `20f7be078a60955872ba3fc08e0f2fbb7c72f0d925cfc5cff2124316c13aa05c` | `20f7be078a60955872ba3fc08e0f2fbb7c72f0d925cfc5cff2124316c13aa05c` | **True** |
| physics/test/2 | 562949953421314 | 20 | `f4a84d111dd2b7d40320f2594c3555b8c0429e2552f4a604b7bf322468725f10` | `7628065e65af781b85e2fbf1fa18019fc6f15020e36a5f7d9935cb456444b148` | **False** |
| physics/train/0 | 0 | 5 | `ad7f37cf80412f8b7bdf3b605de2d096e25250c7236b013e3bf3eb214f82bdf7` | `ad7f37cf80412f8b7bdf3b605de2d096e25250c7236b013e3bf3eb214f82bdf7` | **True** |
| physics/train/3 | 3 | 40 | `6b3df26c0da8a164b71824c3ed46a34b26564434ffa91631071b6930a4c0f77c` | `6b3df26c0da8a164b71824c3ed46a34b26564434ffa91631071b6930a4c0f77c` | **True** |
| physics/val/1 | 281474976710657 | 10 | `f5fd8fb9896b1c6f9ee873dd2305919adeedb697ed7c189cef4f79d88caf0aa1` | `f5fd8fb9896b1c6f9ee873dd2305919adeedb697ed7c189cef4f79d88caf0aa1` | **True** |

不一致 case 的各分量子摘要（sha256 前 16 位，由 `p10_synth_probe.py` 的 `component_digests` 算出。开发机上对同一 case 跑同一函数即可逐分量对照）：

- `occlusion/test/2`：`{"instruction": "3378b2c7ded31dcf", "image": "7a9c9e5b790e4671", "target_mask": "87175d4a2ce1634a", "target_grid": "e356d29320d9e480", "object_masks": "6849cb41062ae40d", "baseline:convex_hull:mask": "90b73e05f27a864c", "baseline:convex_hull:grid": "a85b42330b997ad6", "baseline:visible:mask": "c49c142184b28d6a", "baseline:visible:grid": "598807a80c71dace", "meta": "6d7522d27aad6845", "meta.scene": "e3a612c399343294", "meta.difficulty": "df5368fc172012e1", "meta.baseline_iou": "ffd04c3b81f63542"}`
- `physics/test/2`：`{"instruction": "6e9b648fa6ab6bc1", "image": "02b754022163215c", "target_mask": "d0f08a900c708887", "target_grid": "b218aa23efce774d", "object_masks": "87198f7826a99604", "baseline:const_velocity:mask": "c1f1cef22054316f", "baseline:const_velocity:grid": "90e15c402f97c6f9", "baseline:copy:mask": "1c28047293838a1c", "baseline:copy:grid": "67411e117908f1e6", "meta": "4bedc8d0269f5558", "meta.scene": "8274d7b5e69c2f94", "meta.final_poses": "4c64b07f60871e9b", "meta.difficulty": "ffb2b7c65be58f6a", "meta.baseline_iou": "e5c03a79931d5880"}`

### 3.3 d 不一致时的定位（服务器侧数据）

`physics/test/2`（默认档 bin=2、k=20）在同一 (split, index) 下，各档 k 的服务器侧结果：

| bin | k | 服务器摘要 | 场景与默认档相同 | 目标位移 px | 与默认档（k=20）目标 mask 的 IoU | max_penetration_px | final_poses (obj_id, cx, cy, angle°) |
|---|---|---|---|---|---|---|---|
| 0 | 5 | `7e0c124a32037a47` | True | 32.1314 | 0.5516 | 0.75754 | [(0, 285.96875, 439.796875, -0.89453125), (1, 72.4609375, 414.703125, 0.0), (2, 349.46875, 421.28515625, -1.62109375), (3, 228.28125, 428.83203125, 23.96484375)] |
| 1 | 10 | `7ab4226d334a3559` | True | 25.4609 | 0.7307 | 0.75754 | [(0, 282.390625, 440.171875, -0.06640625), (1, 72.4609375, 414.703125, 0.0), (2, 347.41015625, 421.6796875, -1.421875), (3, 221.62109375, 429.6484375, 27.375)] |
| 2 | 20 | `f4a84d111dd2b7d4` | True | 17.8203 | 1.0000 | 0.75754 | [(0, 282.390625, 440.171875, -0.06640625), (1, 72.4609375, 414.703125, 0.0), (2, 348.203125, 422.6015625, -0.015625), (3, 213.98046875, 429.6484375, 13.4140625)] |
| 3 | 40 | `46ea0efea3d365d1` | True | 2.53907 | 0.5296 | 0.75754 | [(0, 282.390625, 440.171875, -0.06640625), (1, 72.4609375, 414.703125, 0.0), (2, 348.203125, 422.6015625, -0.015625), (3, 198.69921875, 429.6484375, -14.50390625)] |

- 开发机侧的钉住值只有整样本摘要，服务器上**无法**算出"与开发机的位姿最大差（像素与角度）"和"目标 mask 的 IoU"，也无法判断差异从哪一步开始。
- 要得到这三个数，需要在开发机上对同一 case 跑 `p10_synth_probe.py digests`，再与上表的 final_poses 及 §3.2 的分量子摘要逐项对照（只需几秒）。

### 3.4 诊断：关闭 numpy SIMD 分派后的摘要（原样）

```
### NPY_DISABLE_CPU_FEATURES=''
  runtime AVX512F True AVX2 True FMA3 True
imagery|test|2 19303b5b3aab28b1f15962950aeb31ed5874c3ba3a9d2b92613dba29c4617f6a
occlusion|test|2 d2ef5d62b5258d4abb36a33d4d7d652777c9dc9fee01c4a72996a74693327c1f
physics|test|2 f4a84d111dd2b7d40320f2594c3555b8c0429e2552f4a604b7bf322468725f10
### NPY_DISABLE_CPU_FEATURES='X86_V4 AVX512_ICL AVX512_SPR'
  runtime AVX512F True AVX2 True FMA3 True
imagery|test|2 19303b5b3aab28b1f15962950aeb31ed5874c3ba3a9d2b92613dba29c4617f6a
occlusion|test|2 d2ef5d62b5258d4abb36a33d4d7d652777c9dc9fee01c4a72996a74693327c1f
physics|test|2 f4a84d111dd2b7d40320f2594c3555b8c0429e2552f4a604b7bf322468725f10
### NPY_DISABLE_CPU_FEATURES='X86_V3 X86_V4 AVX512_ICL AVX512_SPR'
  runtime AVX512F True AVX2 True FMA3 True
imagery|test|2 19303b5b3aab28b1f15962950aeb31ed5874c3ba3a9d2b92613dba29c4617f6a
occlusion|test|2 d2ef5d62b5258d4abb36a33d4d7d652777c9dc9fee01c4a72996a74693327c1f
physics|test|2 f4a84d111dd2b7d40320f2594c3555b8c0429e2552f4a604b7bf322468725f10
### ''
simd_extensions': {'baseline': ['X86_V2'], 'found': ['X86_V3', 'X86_V4', 'AVX512_ICL'], 'not_found': ['AVX512_SPR']}}, {'ignore_floating_point_errors_in_matmul': False}] 

### 'X86_V3 X86_V4 AVX512_ICL AVX512_SPR'
simd_extensions': {'baseline': ['X86_V2'], 'found': [], 'not_found': ['X86_V3', 'X86_V4', 'AVX512_ICL', 'AVX512_SPR']}}, {'ignore_floating_point_errors_in_matmul': False}]
```

### 3.5 生成速度（`speed.json`；单进程；train 序号 0–199，档位按默认轮转）

| 任务 | n | 档位轮转 | 墙钟 s | CPU s | 样本/s（墙钟） | 样本/s（CPU） |
|---|---|---|---|---|---|---|
| occlusion | 200 | bin = index % 3 | 2.067 | 2.067 | 96.76 | 96.78 |
| physics | 200 | bin = index % 4 | 1.026 | 1.026 | 195.0 | 195.02 |
| imagery | 200 | bin = index % 4 | 0.659 | 0.658 | 303.66 | 303.73 |

## 4. 与任务书假设不符的发现

1. **c（遮挡补全）也有一个 case 不一致**：`occlusion/test/2`。任务书与 D23-3 的预期是 c、e 逐位相同，只有 d 可能不同。e 的 3 个 case 全部相同。
2. **两个不一致的 case 共用 scene_seed 562949953421314**，即 `scene_seed(split="test", index=2)`。同一 seed 的 `imagery/test/2` 相同。其余 7 个相同的 case 用的 scene_seed 分别是 0、281474976710657、3。
3. **与 numpy 的 SIMD 分派无关**（只陈述诊断结果）：这台机器的 numpy 运行时分派到 X86_V3、X86_V4、AVX512_ICL。用 `NPY_DISABLE_CPU_FEATURES` 关掉后，`np.show_runtime()` 的 `found` 为空，已确认生效，而三个 test-2 case 的摘要逐字相同。`src/synth` 里用到的 `math.cos/sin/atan2`（Python libm）与 pymunk 不受这个开关影响。
4. §4.2 要求的"位姿最大差、目标 mask 的 IoU、差异从哪一步开始"，只靠服务器数据算不出来（§3.3）。已给出服务器侧逐档位姿与分量子摘要，供开发机对照。

## 5. 自己造成的错误或浪费

`p10_synth_probe.py` 的 speed 模式第一次运行时，因为字符串格式化写错（`"… % %d" % n`）在记录结果时抛 `ValueError`。改成 f-string 后重跑。代价：约 2 s。

## 6. 生成文件清单与磁盘收支

**仓库内**：`experiments/P0/0006_env_synth/`：
- `pip_dryrun.txt`、`pip_install.log`、`pytest.txt`
- `synth_digests.json`、`speed.json`、`diag_npy_disable_cpu_features.txt`
- `disk_ledger.tsv`、`README.md`、`probe_scripts/`

另有本 report。

**venv**：新装 cffi 2.1.1、pycparser 3.0、pymunk 7.3.0（下载约 1.3 MB）。

**磁盘收支**（`disk_ledger.tsv`；GB = 1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 11:13:34 | 0006 start / before pip | 5.647 | 1.775 | 0.490 |
| 11:13:45 | after pip install pymunk | 5.641 | 1.775 | 0.496 |
| 11:15:45 | after pytest | 5.639 | 1.823 | 0.497 |
| 11:19:40 | before 0006 commit | 5.638 | 1.823 | 0.498 |

- 安装前后：`df /` 可用 5.647 → 5.641 GB，`$PROJ` 0.490 → 0.496 GB。
- 根盘最低 5.638 GB（红线 3 GB）。内存盘没有变动：只有 Qwen3.5-0.8B，外加 pytest 下载的小文件。
- 没有删除、移动或修改任何非本项目文件。
