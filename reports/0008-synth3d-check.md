# RUN-0008 — E5 生成器第二部分（a、b，3D）的跨平台逐位复现核对

服务器 CC，b200，2026-09-30 15:22–15:27 UTC（值守）。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- 全量测试 `1102 passed, 11 skipped, 23 warnings in 265.44s (0:04:25)`，rc=0。跳过项与 RUN-0007 相同：10 个缺 punkt_tab，1 个缺 cv2。
- `synth_pin_digests.py check`：五个任务都是 60/60 与钉住的摘要相同，共 300 个，rc=0。其中 rotation3d（a）与 viewpoint（b）的任务版本为 3，c/d/e 为 2；d 在本平台（规范平台）上计入。

## 1. 前置校验

- `git pull --ff-only` 后 HEAD 为 `fbc8570`（本任务书所在的 commit）；`bd23fc0` 在 HEAD 历史中。
- `pip install --dry-run -r requirements.txt`：没有任何 "Would install"（`pip_dryrun.txt`）。`requirements.txt` 自 RUN-0007（`a566ede`）以来没有变动。
- 没有安装任何包；没有修改仓库代码，也没有改 `tests/data/`。

## 2. 执行记录

| 步骤 | 命令 | 墙钟 | 结果 |
|---|---|---|---|
| 依赖 dry-run | `pip install --dry-run -r requirements.txt` | 数秒 | 没有 "Would install" |
| 全量测试 | `python -m pytest tests/ -q -rsx` | 4m27.357s（pytest 自报 265.44 s） | 1102 passed, 11 skipped, 23 warnings in 265.44s (0:04:25) |
| 摘要核对 | `python scripts/synth_pin_digests.py check` | 3.9 s | rc=0，300/300 相同 |

## 3. 结果表（原样）

全量测试：`1102 passed, 11 skipped, 23 warnings in 265.44s (0:04:25)`

```
SKIPPED [9] tests/test_health_ifeval.py:177: 找不到 punkt_tab：设 NLTK_DATA 指向含 tokenizers/punkt_tab 的目录（python -m nltk.downloader -d <目录> punkt_tab）
SKIPPED [1] tests/test_health_ifeval.py:279: 找不到 punkt_tab：设 NLTK_DATA 指向含 tokenizers/punkt_tab 的目录（python -m nltk.downloader -d <目录> punkt_tab）
SKIPPED [1] tests/test_metrics_seg.py:129: could not import 'cv2': No module named 'cv2'
```

`synth_check.txt`：

```
platform x86_64 Linux; canonical: True; code version 3
occlusion: 60/60 equal (task version 2) 
physics: 60/60 equal (task version 2) canonical
imagery: 60/60 equal (task version 2) 
rotation3d: 60/60 equal (task version 3) 
viewpoint: 60/60 equal (task version 3) 

real	0m3.933s
user	0m8.253s
sys	0m0.381s
```

## 4. 与任务书假设不符的发现

无：结果与任务书的预期一致。

## 5. 自己造成的错误或浪费

无。

## 6. 生成文件清单与磁盘收支

**仓库内**：`experiments/P0/0008_synth3d_check/`：`pip_dryrun.txt`、`pytest.txt`、`synth_check.txt`、`disk_ledger.tsv`、`README.md`；另有本 report。没有出现不同，所以没有写 `synth_digests_server.json` 与 `component_digests.json`。

**磁盘收支**（`disk_ledger.tsv`；GB = 1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 15:22:12 | 0008 start | 42.748 | 1.778 | 0.526 |
| 15:26:52 | after pytest + check | 42.746 | 1.826 | 0.527 |
| 15:27:20 | before 0008 commit | 42.746 | 1.826 | 0.527 |

- 与预期一致，基本没有变化。根盘的小幅波动在 MB 级；`$PROJ` 的增量来自 pytest 临时目录与本目录的文件。
- 内存盘仍只有 Qwen3.5-0.8B，外加测试下载的小文件。
