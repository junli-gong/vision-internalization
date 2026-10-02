[DEV-TASK 0026] E1 的 RefCOCO 落地支持本地 refer zip（A800 连不上 web.archive.org）+ 落地代码的两个遗留问题（D49-5）

## 背景

- RUN-0022（D55）：A800 上 web.archive.org 连接超时，E1 的 `refcoco` 源没落地，训练接口核对因此跳过。E2 的指代分割评测也要用 E1 的 RefCOCO 评测总体，所以同样受影响。
- 训练源 `refcoco_train`（DEV-0022）已经支持 `--refer-zip-dir`：本地 zip 照样核对尾部 sha256 与 CRC32。E1 的 `refcoco` 源（`scripts/data/land_e1.py`、`src/data/land/refcoco.py`）还不支持。
- 3 个 refer zip 会由开发机从 web.archive.org 下载、经 Drive 交给合作者，共约 157 MB：refcoco.zip 53,993,993 B、refcoco+.zip 45,613,210 B、refcocog.zip 56,712,951 B。
- D49-5 的两个遗留：
  1. E1 的 `download_many` 在抛出 `DiskRedLine` 后，线程池会把已提交的下载照常跑完，不会取消；
  2. `Manifest` 没有 `script` 参数，各落地源只好自己改写清单字段。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D30、D49、D55
- `src/data/land/refcoco.py`、`src/data/land/refcoco_train.py`（它的 `--refer-zip-dir` 实现）、`src/data/land/common.py`、`net.py`、`manifest.py`、`scripts/data/land_e1.py`

## 任务

1. **`land_e1.py refcoco --refer-zip-dir DIR`**：
   - DIR 里放 `refcoco.zip`、`refcoco+.zip`、`refcocog.zip`，读本地文件，不访问 web.archive.org。
   - 核对与远端相同：中央目录或尾部的 sha256、成员的 CRC32，以及 E1 已钉住的其他核对项。核对不过就报错，不落地。
   - COCO 图像照旧从 images.cocodataset.org 用 Range 取。A800 上这个站可以访问。
   - 不给 `--refer-zip-dir` 时，行为完全不变。
   - 清单里记下用的是本地 zip，以及每个 zip 的完整 sha256。
   - **与训练源共用**：`refcoco_train` 与 E1 的 `refcoco` 读本地 zip 的代码只写一份，例如放进 `common.py`。这一点允许改 `refcoco_train.py`，但行为不变。
2. **`download_many` 在磁盘红线处取消**：抛出 `DiskRedLine` 时，取消还没开始的下载，已在进行的尽快停下，不再写新文件。补一条测试：用假网络与假磁盘统计，核对红线之后不再有新文件写出。
3. **`Manifest` 加 `script` 参数**：各落地源改为传参，不再改写清单字段。已有清单的格式不变，加一条测试核对。
4. **一个完整性工具**：`scripts/data/refer_zip_sha256.py <dir>`，打印 3 个 zip 的完整 sha256 与字节数，与钉住值比对。合作者拿到 zip 后先跑它。

## 允许改动的文件

- `src/data/land/refcoco.py`、`src/data/land/refcoco_train.py`、`src/data/land/common.py`、`src/data/land/net.py`、`src/data/land/manifest.py`、`scripts/data/land_e1.py`、`scripts/data/land_train.py`
- 新建 `scripts/data/refer_zip_sha256.py`
- 对应的测试与 fixture

**不许碰**：其他数据源的行为、`src/e1/` 到 `src/e5/`、`src/adapters/`、`configs/`。

## 环境与验收

- conda 环境 `visint`，不装包。允许联网只读：HEAD、Range 读中央目录，用来核对钉住值；不下载 zip 本体。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 用 fixture 的小 zip 跑一遍 `land_e1.py refcoco --refer-zip-dir … --limit 3`
- 总结里写明：
  - 3 个 zip 的完整 sha256 怎么钉住：如果现有 SPEC 只钉了尾部，写明能不能从已落地的清单或中央目录推出完整 sha256，推不出的话由管理员下载后补钉；
  - A800 上的运行命令。
