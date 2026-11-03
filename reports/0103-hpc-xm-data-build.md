# RUN-0103 报告：X-M 的两份清单、训练目标与长度统计（D165、D167）

任务书 `prompts/server_hpc/0103-hpc-xm-data-build.md`。只用 CPU。判定集与最终报告集只读真值，不跑任何系统。只贴数字，不解读。

## 启动

- hpcmgt 上排队脚本 `$VI/tmp/run0103.sh`（11-03 03:57:54Z 起）：等 RUN-0102 的 train2017 通过（ACCEPT、118,287 张）后开始。前 squeue（06:3xZ）：157296、157297 RUNNING。
- fetch 第 1 次 rc=0；wt0103（稀疏）HEAD `b01fe642477e41d606c147d5c44f788b59f4e144`（含 0caa0d6a），status 空；8 个 sha256 与任务书逐字相同。

## 第 1 步：两份清单（hpcmgt，nice，`--workers 12`；rc=0，1,627 s）

日志原文（`0103_xm_configs/logs/build_configs.log` 摘录）：

```
[X-M 划分] instances_val2017：5000 张图、36781 个标注
[X-M 划分] xm_tune：300 张图、923 对、2263 个实例
[X-M 划分] xm_decision：500 张图、1437 对、3597 个实例
[X-M 划分] xm_final：1000 张图、2940 对、7323 个实例
[X-M 划分] lvis_sub：1000 张图、3134 对
[X-M 划分] reasonseg_val：200 张图、196 对（长句 109、短语 87）
[X-M 去重] instances_train2017：118287 张图
[X-M 去重] RefCOCO 评测划分：6549 张图，其中 6549 张在 train2017
[X-M 去重] 计算 train2017 的 118287 张图的 sha256（12 个线程）…
[X-M 去重] ReasonSeg val：200 张图；与 train2017 同名 0、同 sha256 0
[X-M 去重] xm_tune：300 张图；与 train2017 同 id 0、同 sha256 0
[X-M 去重] xm_decision：500 张图；与 train2017 同 id 0、同 sha256 0
[X-M 去重] xm_final：1000 张图；与 train2017 同 id 0、同 sha256 0
[X-M 去重] lvis_sub：1000 张图；与 train2017 同 id 0、同 sha256 0
xm_tune：300 张图、923 对、2263 个实例（513b535b44aa72ad…）
xm_decision：500 张图、1437 对、3597 个实例（16187f531d58954d…）
xm_final：1000 张图、2940 对、7323 个实例（41d611f13fe83b06…）
lvis_sub：1000 张图、3134 对、10163 个实例（eeb3c8ac68a607dc…）
reasonseg_val：200 张图、196 对、196 个实例（2bce88892ed2a482…）
去重 refcoco_eval：train2017 里剔除 6549 张
去重 reasonseg_val：train2017 里剔除 0 张
去重 coco_eval_splits：train2017 里剔除 0 张
RefCOCO 评测划分 6549 张图（在 train2017 6549 张）；子集核对 13 个，全部在内
去重清单共 6549 张（2123a9a392caf6eb…）；用时 1587 s
```

- 输出：`xm_splits.json` 80,488 字节，sha256 `995345f470111e3ac73c9681754dd46eca66da7d9ca0b648873cc1429219db8d`；`xm_train_exclusions.json` 151,562 字节，sha256 `b50395674de3a33d23e028ca98cf18f06a62f982c7847005a40cbe2955546dc3`；`build_log.json` `de78bf01…`。
- 07:04:07Z 一次性同步推回（`6aa5a9026`）供审。管理员审过后按字节相同提交到 `configs/`，XM-2a 进预注册（`61abf1e39`）。

## 第 2 步：训练目标（07 上借 CPU：`srun --jobid=157297 --overlap -n1 -c 14`，`CUDA_VISIBLE_DEVICES` 为空，`--workers 14`）

- 试跑（`--limit 2000` → `0103_xm_targets_probe`）：rc=0，307 s。`targets_000.jsonl：2000 对，46.7 s`；`X-M 目标：2000 对、694 张图、5084 个实例（保留 5084）；超过 30 个的对 0（0.00%）；样条可用 2000 对，失败的实例 {}；fit_iou 中位数 0.926829268292683`。
- 全量（`0103_xm_targets`）：rc=0，1,039 s。17 片（16 片各 20,000 对，每片 52–88 s；最后一片 464 对）。
  - `X-M 目标：320464 对、110717 张图、791628 个实例（保留 791628）；超过 30 个的对 0（0.00%）；样条可用 320456 对，失败的实例 {'mask_empty': 8}；fit_iou 中位数 0.9275579616624257`
  - 占用 507 MB（试跑 3.3 MB）。manifest 在 `0103_xm_targets/xm_targets/manifest.json`（不在 `0103_xm_targets/` 顶层；RUN-0104 的驱动因此改过一次，D174）。大文件只在 HPC，记在 `LARGE_FILES.manifest.tsv`。

## 第 3 步：长度统计（hpcmgt，nice，`--workers 12`；rc=0，2,198 s）

`0103_xm_data_stats/data_stats.md` 全文：

# X-M 数据统计（DEV-0079）

分词器：{'model': 'qwen3.5-9b', 'repo': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'class': 'Qwen2Tokenizer'}；截断上限 30 个；样条 16 个控制点。

| 划分 | 对 | 图 | 实例 | 每对实例 中位/p90/p99/最大 | >30 的对 | 截掉的实例 | 样条不可用的对 | 框列表 token 中位/p90/p99/最大 | 样条 token 中位/p90/p99/最大 | 两者之和 p50/p99/最大 | fit_iou 中位 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 320464 | 110717 | 791628 | 1.0/6.0/13.0/27 | 0（0.00%） | 0 | 8 | 20/119/260/535 | 162/956/2082/4289 | 182/2342/4824 | 0.928 |
| xm_tune | 923 | 300 | 2263 | 1.0/6.0/13.0/13 | 0（0.00%） | 0 | 0 | 20/116/260/264 | 162/938/2082/2091 | 182/2342/2355 | 0.934 |
| xm_decision | 1437 | 500 | 3597 | 1.0/6.0/13.0/13 | 0（0.00%） | 0 | 0 | 20/119/260/263 | 162/960/2082/2088 | 182/2342/2349 | 0.924 |
| xm_final | 2940 | 1000 | 7323 | 1.0/6.0/13.0/22 | 0（0.00%） | 0 | 0 | 20/120/259/440 | 162/961/2081/3522 | 182/2340/3962 | 0.929 |
| lvis_sub | 3134 | 1000 | 10163 | 1.0/6.0/31.0/170 | 33（1.05%） | 1005 | 2 | 21/119/564/615 | 164/954/4519/4804 | 185/5083/5415 | 0.929 |

token 数只算模型的输出（框列表一次、样条一次），不含提示与图像 token；lvis_sub 只作参考（任务书要的是 train 与三个 COCO 划分）。
分位数是 numpy.percentile 的 linear 插值。

## 收尾

- 最后一次同步（configs、targets_probe、targets、data_stats 四个目录）08:14:46Z ok。后 squeue：157296 RUNNING 79-19:40:32 hpcgpu06；157297 RUNNING 81-17:12:41 hpcgpu07。
