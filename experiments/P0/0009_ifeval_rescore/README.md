# experiments/P0/0009_ifeval_rescore — RUN-0009 证据

服务器 CC，2026-10-01。内容：安装 punkt_tab 并核对哈希（D28-6）；对 RUN-0007 的 8 个基座运行做 IFEval 离线补评分（不加载模型）；写出 8 个注册模型的权重下载清单（D28-8）。运行报告：`reports/0009-ifeval-rescore.md`。

| 文件 | 内容 |
|---|---|
| `health_<键>.json` ×8 | `scripts/health_check.py --rescore experiments/P0/0007_health_base/health_<键>.json --out … --only-ifeval` 的输出：IFEval 已评分，MMLU-Pro、GSM8K 与 RUN-0007 原样相同；`scoring` 字段记有评分时间、代码 commit、punkt_tab 各文件哈希、包版本与源文件 sha256 |
| `rescore_<键>.log` ×8 | 各次重评的 stdout/stderr |
| `nltk_download.log` | `python -m nltk.downloader -d $PROJ/nltk_data punkt_tab` 的输出，以及 nltk 版本、下载起止时间 |
| `punkt_tab_check.txt` | `$PROJ/nltk_data/tokenizers/punkt_tab/english/` 四个文件的 sha256，与 `src/health/protocol.py` 的 `PUNKT_TAB_ENGLISH_SHA256` 逐个比对（全部相同），另附 punkt_tab.zip 的 sha256 |
| `weights_manifest.json` | 8 个模型在固定 revision 下的全部文件：路径、字节数、哈希（`hash_kind` 注明哈希的种类） |
| `manifest_local_check.txt` | 用本地已有的 Qwen3.5-0.8B 快照抽查清单：权重 sha256、config.json 的 git blob sha1，以及 `p12_verify_weights.py` 的全量核对 |
| `disk_ledger.tsv` | 磁盘收支 |
| `probe_scripts/` | `p12_weights_manifest.py`（生成清单，只读 HF API）、`p12_verify_weights.py`（按清单恢复与核对）、`diskcheck9.sh` |

## 重启后按清单恢复权重

内存盘 `/dev/shm/visint/hf-hub` 在重启后会清空。恢复步骤：

1. `source $PROJ/scripts/env.sh`，确保 `HF_HUB_CACHE=/dev/shm/visint/hf-hub`、`HF_XET_CACHE=/dev/shm/visint/xet`，不要让权重落到根盘。
2. 对需要的模型执行（在 `$REPO` 下）：
   ```bash
   python experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py <模型键> --download
   ```
   - 它先用 `snapshot_download(repo, revision=<清单里的 revision>)` 下载；
   - 然后逐个文件核对字节数与哈希：LFS 文件比文件内容的 sha256（`hash_kind=lfs_sha256`），小文件比 git blob sha1（`hash_kind=git_blob_sha1`，即 `sha1(b"blob <字节数>\0" + 内容)`）；
   - 全部一致时退出码为 0，否则为 1，并列出缺失或不一致的文件。
3. 内存盘上限 25 GB：一次只恢复当前要用的模型，用完按 D13-6 用 `scan_cache_dir(...).delete_revisions(<revision>)` 删除。
4. 清单是用 HF API（`model_info(repo, revision=…, files_metadata=True)`）读到的。HF 上同一 revision 的内容不会变；清单里 `api_sha == revision` 表示读到的确实是固定的 revision。
