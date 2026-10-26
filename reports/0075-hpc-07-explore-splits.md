# RUN-0075 报告：hpcgpu07（只用 CPU），devcheck3 与 explore_val 的清单

任务书 `prompts/server_hpc/0075-hpc-07-explore-splits.md`。只生成句子 id 的清单，**没有在 devcheck3 上跑任何系统**。没有碰 RUN-0072、0073、0074；没有用 GPU，没有写暂停文件，没有调 `claim_gpu`。只贴数字，不解读。

## 启动与工作树

- 经 hpcmgt 上脱离会话的 `$VI/tmp/run0075.sh`（10-26 03:42Z 前后）。前 squeue：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0075（稀疏）HEAD `9ac65354a002582ed48c42154b06f8f8fc9aecae`（含 e5ae47c8），status 空，`experiments/` 只有 P0。
- 9 个 sha256 与任务书逐字相同。

## 两次运行

两次都在 07 上 `srun --jobid=157297 --overlap -n1 -c 4`，env 为 `PATH=$VI/env/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp`，工作目录 wt0075。

| 运行 | 命令 | 退出码 | 墙钟 |
|---|---|---|---|
| 生成 | `python -m src.data.explore_splits --build --data-root $VI/b200/data --train-data-root $VI/data/train_data --out $VI/repo/experiments/X/0075_explore_splits_hpcgpu07/explore_splits.json` | 0 | 59 s（03:42:34Z – 03:43:33Z） |
| 核对 | `python -m src.data.explore_splits --check --splits <同上文件> --data-root $VI/b200/data --train-data-root $VI/data/train_data` | 0 | 18 s（03:43:33Z – 03:43:51Z） |

- 核对日志 `run_0075_check.log` 原文：`一致（整份重算）`。
- 输出文件 `explore_splits.json`：sha256 `df9f6c19141cee8c012dd3b6606e5b12f00db92557d43f07c7ac2214a3ab7892`，191,360 字节。
- 两份日志里没有警告（grep `warn` 都是 0），没有 traceback。

### 生成日志原文（`run_0075_build.log`）

```
写出 /home/mail/chenyuma/visint/repo/experiments/X/0075_explore_splits_hpcgpu07/explore_splits.json（sha256 df9f6c19141cee8c012dd3b6606e5b12f00db92557d43f07c7ac2214a3ab7892）
devcheck3_refcoco_unc_val：总体 10634 句 / 1500 张图 → 按图排除后 2777 / 475 → 按句排除后 2424 / 475 → 取 200 句（9e2c5aa4a3500f0e6397a9170989f8e2eaf0db3b38755c5033b6320d682076b7）
devcheck3_refcoco+_unc_val：总体 10558 句 / 1500 张图 → 按图排除后 2841 / 475 → 按句排除后 2570 / 475 → 取 200 句（acba358dac5b1024f08a4d8e89d2af9df2dee0bdca96f7671ab379422c17512c）
devcheck3_refcocog_umd_val：总体 4696 句 / 1298 张图 → 按图排除后 4281 / 1212 → 按句排除后 3379 / 1172 → 取 100 句（048523aaeffbfe1eeec3731c060253aa66139975761f41a3be447cf2fc2c8df4）
explore_val_refcoco_unc_val：总体 10634 句 / 1500 张图 → 按图排除后 1101 / 204 → 按句排除后 955 / 204 → 取 400 句（389a9e02482f2f85712d2d25ca25ea71f2bcc802f1f554e8f8f00add8291436e）
explore_val_refcoco+_unc_val：总体 10558 句 / 1500 张图 → 按图排除后 1127 / 204 → 按句排除后 1012 / 204 → 取 400 句（0d6069826192b34592cb6d8161874b893f5a8257164b9ddd6d91e97213d22f9c）
explore_val_refcocog_umd_val：总体 4696 句 / 1298 张图 → 按图排除后 3820 / 1110 → 按句排除后 2992 / 1070 → 取 200 句（3c558433947f7edb91fc4effdbd2fcee994a7e7799017ecac195b72d418ef71f）
组 devcheck3：500 句、365 张图（dc73561d228c0d9077a19398cc31437b4daa2f8bbb5d6f36c7478fdba2ef95a3）
组 explore_val：1000 句、379 张图（d65231fd854a0d30c6d41d2d36ebb7e1cb9d404cd048c1b6c46f18cceb8e2db2）
W2-c 训练池：核对过（287518 句、24397 张图；重叠 {'devcheck3': 0, 'explore_val': 0}）
```

## 每个划分与组

| 划分 / 组 | 封存 | n | sha256 | 总体 句 / 图 | 按图排除后 句 / 图 | 按句排除后 句 / 图 | all_taken |
|---|---|---:|---|---|---|---|---|
| devcheck3_refcoco_unc_val | True | 200 | `9e2c5aa4a3500f0e6397a9170989f8e2eaf0db3b38755c5033b6320d682076b7` | 10634 / 1500 | 2777 / 475 | 2424 / 475 | False |
| devcheck3_refcoco+_unc_val | True | 200 | `acba358dac5b1024f08a4d8e89d2af9df2dee0bdca96f7671ab379422c17512c` | 10558 / 1500 | 2841 / 475 | 2570 / 475 | False |
| devcheck3_refcocog_umd_val | True | 100 | `048523aaeffbfe1eeec3731c060253aa66139975761f41a3be447cf2fc2c8df4` | 4696 / 1298 | 4281 / 1212 | 3379 / 1172 | False |
| explore_val_refcoco_unc_val | False | 400 | `389a9e02482f2f85712d2d25ca25ea71f2bcc802f1f554e8f8f00add8291436e` | 10634 / 1500 | 1101 / 204 | 955 / 204 | False |
| explore_val_refcoco+_unc_val | False | 400 | `0d6069826192b34592cb6d8161874b893f5a8257164b9ddd6d91e97213d22f9c` | 10558 / 1500 | 1127 / 204 | 1012 / 204 | False |
| explore_val_refcocog_umd_val | False | 200 | `3c558433947f7edb91fc4effdbd2fcee994a7e7799017ecac195b72d418ef71f` | 4696 / 1298 | 3820 / 1110 | 2992 / 1070 | False |
| 组 devcheck3 | True | 500（目标 500），365 张图 | `dc73561d228c0d9077a19398cc31437b4daa2f8bbb5d6f36c7478fdba2ef95a3` | | | | |
| 组 explore_val | False | 1000（目标 1000），379 张图 | `d65231fd854a0d30c6d41d2d36ebb7e1cb9d404cd048c1b6c46f18cceb8e2db2` | | | | |

- 所有划分 `all_taken` 都是 false：explore_val 与 devcheck3 没有不够数的层。
- devcheck3 组 `gate` G-EX，`sealed` true；explore_val 组 `sealed` false。

## 清单的 `checks` 一节原文

```json
{
 "devcheck3_vs_explore_val": {
  "n_images_shared": 0,
  "n_samples_shared": 0
 },
 "vs_w2_splits_by_image": {
  "devcheck3": {
   "devtune800": 0,
   "devcheck2": 0,
   "test_refcoco_unc_testA": 0,
   "test_refcoco_unc_testB": 0,
   "test_refcoco+_unc_testA": 0,
   "test_refcoco+_unc_testB": 0,
   "test_refcocog_umd_test": 0
  },
  "explore_val": {
   "devtune800": 0,
   "devcheck2": 0,
   "test_refcoco_unc_testA": 0,
   "test_refcoco_unc_testB": 0,
   "test_refcoco+_unc_testA": 0,
   "test_refcoco+_unc_testB": 0,
   "test_refcocog_umd_test": 0
  }
 },
 "vs_e1_subsets_by_sentence": {
  "devcheck3": 0,
  "explore_val": 0
 },
 "train_pool": {
  "status": "核对过",
  "rule": "W2-c 与 E3 的训练池（src/w2/w2c_data.load_train_pool：三个数据集 train 的全部句子，训练加载器按图剔除 E3 的剔除清单，即 refcoco/eval 等评测集的图；与 W2-c 引用同一份 <训练数据根>/dedupe/train_exclusions.json）",
  "exclusions": {
   "file": "dedupe/train_exclusions.json",
   "sha256": "40f214fb14315c990de811ef9d23172d6c36ea661e40a241ad46fc667f3c1838",
   "schema": 2,
   "max_hamming": 8,
   "eval_sets": [
    "ade20k/val",
    "blink_reldepth/val",
    "coco_val2017/e1sub500",
    "coco_val2017_full/val2017",
    "countbench/eval",
    "cvbench/test",
    "embspatial/test",
    "nyu_normals/test",
    "nyuv2/dev_train",
    "nyuv2/test",
    "odinw13/eval",
    "probe/D20",
    "refcoco/eval"
   ],
   "required_eval_sets": [
    "blink_reldepth/val",
    "coco_val2017/e1sub500",
    "countbench/eval",
    "cvbench/test",
    "embspatial/test",
    "nyuv2/test",
    "odinw13/eval",
    "refcoco/eval"
   ],
   "n_sentences_removed": 86
  },
  "pool": {
   "n_sentences": 287518,
   "sha256": "5b9b93124ad865fc941e3c84e6d4d89112b56c9190ae861f78f5ba07266e918f",
   "n_images": 24397
  },
  "overlap": {
   "devcheck3": {
    "n_images": 0,
    "n_pool_sentences": 0
   },
   "explore_val": {
    "n_images": 0,
    "n_pool_sentences": 0
   }
  }
 }
}
```

- 各项重叠计数都是 0：devcheck3 与 explore_val 互相（图 0、句 0）；两组对 W2 清单 7 个划分按图都是 0；两组对 E1 子集按句都是 0；两组对训练池（图 0、句 0）。
- 与训练池的核对：status 核对过；训练池 287,518 句、24,397 张图（sha256 `5b9b9312…`），剔除清单 `dedupe/train_exclusions.json`（sha256 `40f214fb…`）去掉 86 句。

## 推送与收尾

- `explore_splits.json` 与两份日志的拷贝在 `experiments/X/0075_explore_splits_hpcgpu07/`；`ONCE=1` 推一次（TAG 0075）：`2026-10-26T03:43:59Z ok`。
- 后 squeue：157296、157297 RUNNING。
- 封存照任务书：在 G-EX 判定之前，不在 devcheck3 上跑任何系统。
