# experiments/P0/0023_eval_hashes — RUN-0023：B200 上评测数据的感知哈希清单（跨机器去重用，硬约束 7）

服务器 CC，b200，2026-10-02 13:28 UTC。运行报告：`reports/0023-eval-hash-export.md`。

## 做了什么

在固定于任务开始时 HEAD `0238c69` 的 git worktree 里（`/dev/shm/visint/tmp/wt0023`，跑完已删除）运行：

```bash
python scripts/data/dedupe_report.py --export-hashes $PROJ/data --out $REPO/experiments/P0/0023_eval_hashes/eval_hashes_b200.json
```

- 不用 GPU。
- 哈希不是重新计算的：`src/data/train_dedupe.py` 直接读 `$PROJ/data/<源>/MANIFEST.json` 里每张图的 `image_key` 与 `phash`，这两个字段是落地时写进去的。

## 文件

| 路径 | 内容 |
|---|---|
| `eval_hashes_b200.json` | 导出的清单（kind `visint-image-hashes`，schema 1）。`sets` 里每个集合的 `items` 是 `[image_key, phash, 相对源目录的路径]`；`facts` 是各源 MANIFEST 的 stats、选择规则与 dedupe_facts。不含图像 |
| `logs/export.log` | 导出的 stdout/stderr，含 `time` |
| `logs/worktree.txt` | 运行用的 commit |
| `disk_ledger.tsv` | 磁盘收支 |
