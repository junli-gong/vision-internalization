# RUN-0023 — 导出 B200 上评测数据的感知哈希清单（跨机器去重用，硬约束 7；不占 GPU）

服务器 CC，b200，2026-10-02 13:28 UTC，值守中接单。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- **导出结果：** rc=0。`eval_hashes_b200.json` 共 9 个集合、15109 张图，每张都带 phash。
- **体积与哈希：** 清单 1,257,007 B，低于 20 MB 上限；sha256 `8434d94dbaa085f2b723b4f35f3f0132915a7a21049480560192871eeb0845a1`。
- **用时：** 0.154 s（`time` real）。哈希是直接读各源 MANIFEST.json 里落地时写进去的 `phash`，没有重新计算。
- **未进清单的源：** `$PROJ/data` 下 10 个源目录里，`coco_train2014_e1dev` 与 `countbenchqa_questions` 没有图像记录，所以不在 `sets` 里。
  - 前者只有标注，图像身份是 `coco:<image_id>`，与 refcoco 源的图同一身份键，见清单 `facts`；
  - 后者只有问句文本。
  - 两者的 MANIFEST 摘要都在 `facts` 里。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `0238c69`。`fba37dd`（DEV-0022 合入）与本任务书所在的 commit 都在 HEAD 的历史里。
- **运行用的代码：** 按更新后的 STANDBY.md 第 3 步（D46），在固定于 `0238c69` 的 worktree 里运行，跑完已删除。清单的 `git.commit` 是 `0238c698f394…`，`dirty=false`。
- **改动范围：** 没有修改任何仓库代码，也没有安装包；不用 GPU。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。

## 2. 执行记录

```
[导出] /home/ubuntu/workspace/wzg/gjl/vision-internalization/data → /home/ubuntu/workspace/wzg/gjl/vision-internalization/repo/experiments/P0/0023_eval_hashes/eval_hashes_b200.json：9 个集合，15109 张图

real	0m0.154s
user	0m2.473s
sys	0m0.044s
```

## 3. 结果表（原样）

### 3.1 每个数据源的图像张数（`eval_hashes_b200.json` 的 `sets`）

| 集合 | 源 | role | 图像张数 | 带 phash 的张数 |
|---|---|---|---|---|
| blink_reldepth/val | blink_reldepth | eval | 124 | 124 |
| coco_val2017/e1sub500 | coco_val2017 | eval | 500 | 500 |
| countbench/eval | countbench | eval | 491 | 491 |
| cvbench/test | cvbench | eval | 2638 | 2638 |
| embspatial/test | embspatial | eval | 3640 | 3640 |
| nyuv2/dev_train | nyuv2 | eval | 50 | 50 |
| nyuv2/test | nyuv2 | eval | 654 | 654 |
| odinw13/eval | odinw13 | eval | 463 | 463 |
| refcoco/eval | refcoco | eval | 6549 | 6549 |
| **合计** | | | **15109** | **15109** |

### 3.2 清单

| 项 | 值 |
|---|---|
| 路径 | `experiments/P0/0023_eval_hashes/eval_hashes_b200.json` |
| 字节数 | 1,257,007 |
| sha256 | `8434d94dbaa085f2b723b4f35f3f0132915a7a21049480560192871eeb0845a1` |
| kind / schema | `visint-image-hashes` / 1 |
| host | `metallic-matrix-fin-03` |
| root | `$PROJ/data` |
| 每条的格式 | `[image_key, phash, 相对源目录的路径]`，例：`["coco:154", "adc01bcf72366c61", "images/COCO_train2014_000000000154.jpg"]` |

### 3.3 用时

导出 0.154 s（real），user 2.473 s。

## 4. 与任务书假设不符的发现

- **用时：** 任务书预计几分钟到半小时，取决于图像张数。实际 0.154 s，原因是哈希取自落地清单，不重新读图。

## 5. 生成的文件

`experiments/P0/0023_eval_hashes/`：`eval_hashes_b200.json`、`README.md`、`logs/`（`export.log`、`worktree.txt`）、`disk_ledger.tsv`。一次推回。

## 6. 磁盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 13:28:53 | 0023 start | 38.018 | 3.328 | 4.697 |
| 13:28:53 | after export | 38.016 | 3.328 | 4.698 |
| 13:29:24 | 0023 end (worktree removed) | 38.017 | 3.080 | 4.697 |

- **根盘：** 剩 38.0 GB，远高于 3 GB 红线。
- **`$PROJ`：** 4.697 GB，在 15 GB 份额以内。
