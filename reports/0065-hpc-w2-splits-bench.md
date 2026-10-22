# RUN-0065 报告：W2 的 devcheck2 清单与候选生成器耗时（只用 CPU）

任务书 `prompts/server_hpc/0065-hpc-w2-splits-bench.md`（origin 80c538e4）。不占 GPU，没写暂停文件，没有碰 06 的 RUN-0056、07 的 RUN-0057；没有往 `$VI/repo` 里 pull。经 hpcmgt 上脱离会话的 `$VI/tmp/run0065.sh`（输出 `$VI/logs/run0065.txt`）。只贴数字，不解读；不贴 devcheck2 的句子清单。

## 工作树

- 10-22 22:46:28Z squeue：157296、157297 都是 RUNNING。
- fetch 第 1 次 rc=0。wt0065：HEAD `80c538e420078eab09cb876c36cf55eb2666ec92`，含 49b7bbd2，status 空。
- sha256，5 个都与任务书相同：data.py `13fab313…`、candidates.py `f03bbbfc…`、w1_splits.json `a1371851…`、e1_eval_subsets.json `29415ea0…`、w2_splits.json `dae6afd2…`。

## 运行

- 07 上经 `srun --jobid=157297 --overlap -n1 -c 8`（`-c 8` 被接受），节点上的命令同任务书第 3 步（`HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=`；bench 加 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`）。22:46:44Z – 23:22:03Z，srun rc=0，墙钟 2,119 s。
- (b) 的 40 张图：数据根下 RefCOCO 用的 COCO train2014 图在 `$VI/b200/data/refcoco/images`（6,549 个 jpg）；按文件名排序取前 40 张尺寸为 640×480 的，软链接到 `$VI/tmp/bench0065/`；跑完已删（23:22Z 查，不存在）。
- 23:22:03Z `ONCE=1 sync_push.sh … 0065 experiments/W2/0065_w2_splits_bench`：rc=0，`sync_0065.status` `2026-10-22T23:22:12Z ok`。没有起常驻循环。
- `git worktree remove wt0065` rc=0。23:22:13Z squeue 两条 RUNNING。

## steps.log 原文

```
2026-10-22T22:46:45Z node hpcgpu07 start
build rc=0
check rc=0
bench images: 40 (first: COCO_train2014_000000000605.jpg, last: COCO_train2014_000000011774.jpg)
bench rc=0
8
2026-10-22T23:22:03Z node done
```

## build.log、check.log（全文，均不足 30 行）

```
写出 /home/mail/chenyuma/visint/repo/experiments/W2/0065_w2_splits_bench/w2_splits.json（sha256 9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1）
总体 10634 句（1783dd7fc012cb97e22275dea55cf3de8240433797b40e1340ade93c40f9876c；核对：一致）；排除 devtune800 的 800 句所在的 636 张图，去掉 5024 句；剩 5610 句（d83e22329dce4acec62a4954f0b7a9371c70030ec2adb79348281968b06c2d2d）；取 500 句（e0c090322dc324a27283aa78d3a5ce6ee5f910822646369cd6b2cac36307c546）
```

```
一致（整份重算）
```

## w2_splits.json 里 devcheck2 的统计（不含句子 id 与图像映射）

| 项 | 值 |
|---|---|
| status | built |
| 总体（refcoco_unc_val，load_refcoco('refcoco', 'val', subset='eval')） | 10,634 句，sha256 `1783dd7fc012cb97e22275dea55cf3de8240433797b40e1340ade93c40f9876c`；对 configs/e1_eval_subsets.json 的核对：一致 |
| 排除依据 | devtune800（800 句，sha256 `c8c1f298eedd88f402c0dce6de0816493bc0a01640ffebbe5ca42f62c6d25efd`） |
| 被排除的图 | 636 张（sha256 `7f70f205465330156be910725d8b4efdd309e00f47e8ce847244184743c6a26b`），去掉 5,024 句 |
| 排除后 | 5,610 句（sha256 `d83e22329dce4acec62a4954f0b7a9371c70030ec2adb79348281968b06c2d2d`） |
| 选取 | salt `w2_dev_v1`、purpose `w2_devcheck`，取 500 句 |
| **devcheck2 sha256** | `e0c090322dc324a27283aa78d3a5ce6ee5f910822646369cd6b2cac36307c546`（n = 500） |
| w2_splits.json 文件 sha256（build.log） | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |

## bench.log 最后 5 行

```
[36/40] /home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000010881.jpg（DEFAULT_SPEC 到目前的中位数 4.61 s）
[37/40] /home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000010936.jpg（DEFAULT_SPEC 到目前的中位数 4.61 s）
[38/40] /home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000011282.jpg（DEFAULT_SPEC 到目前的中位数 4.61 s）
[39/40] /home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000011426.jpg（DEFAULT_SPEC 到目前的中位数 4.39 s）
[40/40] /home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000011774.jpg（DEFAULT_SPEC 到目前的中位数 4.61 s）
```

## bench.json 原文

```json
{
 "machine": {
  "platform": "Linux-4.18.0-425.3.1.el8.x86_64-x86_64-with-glibc2.28",
  "cpu": "Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz",
  "cpu_count": 48,
  "python": "3.12.13",
  "opencv": "4.10.0",
  "opencv_threads": 8,
  "skimage": "0.24.0",
  "numpy": "2.5.2"
 },
 "images": 40,
 "image_paths": [
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000000605.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000000716.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000001261.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000001822.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000001958.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000001966.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000001994.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000002281.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000002342.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000002448.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000002742.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000003320.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000004587.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000004714.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000004830.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000005424.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000005508.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000005632.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000005638.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000005782.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000005862.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000006026.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000006338.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000006747.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000006842.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000007852.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000007946.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000008856.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000008936.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000009057.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000009353.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000009822.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000010176.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000010229.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000010728.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000010881.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000010936.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000011282.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000011426.jpg",
  "/home/mail/chenyuma/visint/tmp/bench0065/COCO_train2014_000000011774.jpg"
 ],
 "sizes": [
  "640x480"
 ],
 "boxes": [
  [
   0.25,
   0.25,
   0.75,
   0.75
  ],
  [
   0.4,
   0.35,
   0.65,
   0.7
  ],
  [
   0.05,
   0.1,
   0.9,
   0.95
  ]
 ],
 "repeats": 1,
 "per_generator": {
  "gc_default": {
   "n": 120,
   "median_s": 1.1103703249245882,
   "mean_s": 1.7220966291303437,
   "max_s": 11.928347144275904
  },
  "gc_box_plus10": {
   "n": 120,
   "median_s": 1.4225250445306301,
   "mean_s": 2.0820880851708354,
   "max_s": 24.29449873045087
  },
  "gc_soft_box_minus10": {
   "n": 120,
   "median_s": 1.3616180159151554,
   "mean_s": 1.8076165297689537,
   "max_s": 7.990604039281607
  },
  "gc_iter10": {
   "n": 120,
   "median_s": 2.000508576631546,
   "mean_s": 2.8968450923760733,
   "max_s": 18.804095275700092
  },
  "gc_iter1": {
   "n": 120,
   "median_s": 0.44619196094572544,
   "mean_s": 0.684039835476627,
   "max_s": 4.615697890520096
  },
  "gc_pts": {
   "n": 120,
   "median_s": 1.220075873658061,
   "mean_s": 1.8057612059017023,
   "max_s": 11.635446444153786
  },
  "gc_default+lcc+fill": {
   "n": 120,
   "median_s": 0.010403171181678772,
   "mean_s": 0.010261110402643681,
   "max_s": 0.022316858172416687
  },
  "slic_s30_t50": {
   "n": 120,
   "median_s": 0.10680528543889523,
   "mean_s": 0.11282955035567284,
   "max_s": 0.20913642272353172
  },
  "slic_s100_t50": {
   "n": 120,
   "median_s": 0.11974168382585049,
   "mean_s": 0.12747082756832243,
   "max_s": 0.23455653712153435
  },
  "ws_center": {
   "n": 120,
   "median_s": 0.11511083506047726,
   "mean_s": 0.11407186537981033,
   "max_s": 0.14559615030884743
  },
  "ws_pts": {
   "n": 120,
   "median_s": 0.09132995083928108,
   "mean_s": 0.08645455179115137,
   "max_s": 0.11023591458797455
  },
  "box": {
   "n": 120,
   "median_s": 4.824250936508179e-05,
   "mean_s": 5.1089624563852945e-05,
   "max_s": 9.481608867645264e-05
  }
 },
 "default_spec_total": {
  "n": 120,
  "median_s": 4.547266757115722,
  "mean_s": 6.088402652181685,
  "max_s": 39.162600588053465
 },
 "default_spec_names": [
  "gc_default",
  "gc_box_plus10",
  "gc_soft_box_minus10",
  "gc_default+lcc+fill",
  "slic_s30_t50",
  "slic_s100_t50",
  "ws_pts",
  "box"
 ]
}
```

- 没有 traceback。
