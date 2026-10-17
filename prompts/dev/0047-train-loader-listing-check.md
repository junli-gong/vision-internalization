[DEV-TASK 0047] 训练数据加载器的"文件存在"检查改成按目录列一次清单：结果逐字节相同，共享盘上的元数据查询从十几万次降到几百次

## 背景

- RUN-0052（hpcgpu06，4B 的 E3b × 3）每 500 步重启一次训练进程（驱动按段续跑）。每次启动都要重新建训练池。
- 10-17 起 ITSC 的 Lustre 元数据很慢：a100-69 用 strace 量到，每次 `stat` 中位 0.11 s，最长 0.9 s。
- 训练进程在加载完权重之后，主线程逐个 `stat` 训练文件：
  - 来源是 `src/data/train_loaders.py` 的 `_have()`，它对每条记录调 `(src / rel).exists()`；
  - refseg 是 refcoco、refcoco+、refcocog 的每条指代句，约 13 万次（同一张图会被重复查）；
  - depth 是 NYU 的每对 rgb 与深度图，约 5 万次。
- 后果：
  - 一次启动在 GPU 空闲的情况下花 2 个多小时（10-17 第 7 段：11:41 加载完权重，到 14:00 还在查 NYU）；
  - 平时 Lustre 正常，也要 10–20 分钟。
  - 06 还要重启约 9 次。
- 用到同一个 `_have()` 的还有 nyu_normals、ade20k 的加载器，一并受益。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `src/data/train_loaders.py`：全文，重点是 `_have`、`load_refcoco_train`、`load_nyu_depth_train`、`load_nyu_normals`、ADE20K 的加载
- `src/e3/data.py`：`load_pool`、`E3Paths.as_dict`
- `scripts/train_e3b.py` 与它的续跑逻辑（`--resume`）：续跑时核对哪些东西（config.yaml、运行身份、代码 commit？）
- 现有测试：`tests/` 里覆盖 train_loaders 与 E3 训练池的测试

## 任务

1. **改 `_have()` 的实现**，语义不变：
   - 判断 `rel` 在清单 `files` 里，并且磁盘上存在。磁盘上的存在改为按父目录一次性列出（`os.scandir` 或 `os.listdir`），在内存里查成员，每个目录只列一次，加载器调用之间可以不共享缓存。
   - **与 `Path.exists()` 的差别要处理掉**：
     - 符号链接：清单里是符号链接的条目，退回逐个 `exists()`，保证断链仍判不存在；
     - 名字的大小写与 Unicode 规范化：按字节比较，与 `exists()` 在 Linux 上一致。
   - strict 时报错的信息与现在逐字相同：还是列出不存在的那几个 `rel`。
   - 目录不存在（列不出来）就当作该目录下全都不存在，与 `exists()` 一致，不抛别的异常。
2. **不改别的**：记录、顺序、`stats`、去重剔除、`LoadResult` 的任何字段都不变；不改 `configs/`；不改 E3、E5 的任何训练或评测逻辑。
3. **测试**（`conda run -n visint pytest tests/ -q` 全部通过）：
   - 在一个临时目录里造小型的落地树：refcoco、NYU、nyu_normals、ADE20K 各几条。包括缺文件、断开的符号链接、清单里有但磁盘上没有、磁盘上有但清单里没有这几种情形。
   - 断言新实现与旧实现（测试里保留一份旧 `_have` 的拷贝，或用 monkeypatch 换回旧实现）在 strict 与非 strict 下的 `LoadResult` 完全相同：记录逐条相等、`stats` 相等，strict 时异常类型与信息相同。
   - 断言新实现的 `stat`、`exists` 次数随目录数增长，不随文件数增长：例如 monkeypatch `os.stat` 或 `Path.exists` 计数。
4. **续跑兼容**：
   - 查清 `scripts/train_e3b.py --resume` 在续跑时是否核对代码 commit 或源文件哈希。
   - 如果核对，**不要改它**，在完成总结里写明：会核对什么、换 commit 后续跑会不会被拒。
   - 如果只核对 config.yaml 与运行身份，确认这次改动不改变它们（`E3Paths.as_dict` 等），写一条测试钉住。
5. **估一下省多少**：在开发机上，对一个几千个文件的临时树，量新旧实现的 `stat` 次数，不必量时间。

## 不做

- 不改 `prompts/server_hpc/` 下的驱动脚本，不改 `scripts/server/`。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；新 `_have` 的做法。
- 与旧实现逐条相同的测试列表与结果。
- `stat` 次数的对比。
- 续跑核对什么、换 commit 后续跑会不会被拒（第 4 项）。
- 测试结果原文。
- 任何你认为需要管理员定的地方。
