[RUN-TASK 0009] punkt_tab 安装与 8 个基座的 IFEval 补评分；权重下载清单（不占 GPU）

## 为什么做

- 项目负责人已允许 IFEval 评分器使用 nltk 的 punkt_tab（D28-6）：哈希钉住、不入库、记录版本。
- RUN-0007 的 8 个基座运行当时只生成了 IFEval 的输出、没有评分，现在离线补上，不重跑模型。
- 权重继续放内存盘。按 D28-8 保留下载清单和哈希，重启后照单恢复。

## 0. 先读

- `CLAUDE.md`（磁盘一节已按 D28-8 更新）、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D26、D28
- `scripts/health_check.py --help`（`--rescore`、`--only-ifeval`）

本任务只运行，不改 `src/`、`scripts/`、`tests/`、`configs/`。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含本任务书所在的 commit。不包含就停下报告。

## 2. punkt_tab

1. `python -m nltk.downloader -d $PROJ/nltk_data punkt_tab`
   - 数据放在 `$PROJ/nltk_data`，**不要**写进全局的 `~/nltk_data`；
   - 记下 nltk 版本与下载时间。
2. `export NLTK_DATA=$PROJ/nltk_data`
3. 核对 `punkt_tab/english/` 下的文件哈希与 `src/health/protocol.py` 里的 `PUNKT_TAB_ENGLISH_SHA256` 逐个相同。
   - **不同就停下报告**：贴出两边的哈希，不要改钉住值。管理员会从开发机拷一份钉住的版本过来。

## 3. IFEval 补评分

对 `experiments/P0/0007_health_base/` 里 8 个 `health_<键>.json` 逐个执行：

```bash
python scripts/health_check.py --rescore experiments/P0/0007_health_base/health_<键>.json \
  --out experiments/P0/0009_ifeval_rescore/health_<键>.json --only-ifeval
```

- 不需要 GPU，也不加载模型。
- 退出码为 2 时，照常推回，在 report 里贴出报错，继续下一个。

## 4. 权重下载清单

写一个探针脚本，放在 `experiments/P0/0009_ifeval_rescore/probe_scripts/`，只读 HF API，不下载权重：
- 对 `src/adapters/registry.py` 里的 8 个模型，按固定 revision 列出每个文件的路径、字节数与 sha256（LFS 文件用 lfs.sha256；小文件用 blob id，并注明是哪一种）。
- 结果写进 `experiments/P0/0009_ifeval_rescore/weights_manifest.json`。
- 另写一个说明：重启后用 `snapshot_download(repo, revision=...)` 照单恢复，并按清单核对。说明放进 README。

## 5. 产出与回传

- `experiments/P0/0009_ifeval_rescore/`：
  - 8 份重评后的 JSON；
  - `weights_manifest.json`、`punkt_tab_check.txt`；
  - `README.md`、`probe_scripts/`
- `reports/0009-ifeval-rescore.md`：第 3 节原样贴每个模型的 IFEval 四个口径（prompt-level strict 为标题），并与 RUN-0007 的 MMLU-Pro、GSM8K 合成一张表。只贴数字与事实，不做解读。

一批推完：

```bash
git add <逐个路径> && git commit -m "[results] 0009: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0009-ifeval-rescore.md` 追加进完成清单，重新进入值守。
