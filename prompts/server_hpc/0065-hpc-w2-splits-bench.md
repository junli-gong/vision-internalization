[RUN-TASK 0065] HPC，只用 CPU：生成 W2 的新开发-检查集 devcheck2 的清单，并在服务器上实测候选生成器的耗时（D113、D116）。约 20 分钟，不占 GPU、不让路

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0056、07 上的 RUN-0057；不写任何暂停文件。
- 不往 `$VI/repo` 里 pull。

## 为什么

- **devcheck2**：W2 去留判定用的 500 句。
  - 规则写死在 `src/w2/data.py`：RefCOCO（UNC）val 的 eval 总体 10,634 句，按图像排除 W1 用过的 800 句所在的图，再按 `hash_rank(salt='w2_dev_v1', purpose='w2_devcheck')` 取前 500 句。
  - 开发机没有 RefCOCO 的数据，所以在 HPC 上生成。
  - 这一步只产出句子的 id 清单，不跑任何模型，也不看任何结果。
- **耗时**：W2-c 要在 RefCOCO train 上为几万句生成候选。开发机 M4 单核每句约 3.4 s，服务器上要实测，才能定训练集的规模。

## 文件（commit `49b7bbd2` 或更新；工作树 wt0065 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `src/w2/data.py` | `13fab313e2fd7e6566a70c953f8686951256a994444d6563a9d08ead57fbe3b7` |
| `src/w2/candidates.py` | `f03bbbfce130d937ace7d5555c5fcbc898e76c88880a2d9eb30d289207a635d5` |
| `configs/w1_splits.json` | `a1371851555e854fac6f9d0e267f19a23602ee9cf31d5c303ef08432d1f7f0f7` |
| `configs/e1_eval_subsets.json` | `29415ea0d131336af52c950c053f95534a526261870a38ff66de2285c2545139` |
| `configs/w2_splits.json` | `dae6afd22fc2bddf2a898d32b7083d89332e68692f894e9263662e7b24fa045d`（骨架，devcheck2 是 pending） |

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0065（稀疏，同 RUN-0054 的规则），建在 origin/main。fetch 失败就重试。核对上表 5 个 sha256。
3. 在 07 上经 srun 跑（只用 CPU，`CUDA_VISIBLE_DEVICES` 置空，`-c 8`；被拒就去掉 `-c`）。在节点上执行：

   ```bash
   VI=/home/mail/chenyuma/visint; W=$VI/wt/wt0065; O=$VI/repo/experiments/W2/0065_w2_splits_bench
   export HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= PATH=$VI/env/bin:/usr/bin:/bin
   mkdir -p $O && cd $W
   # (a) devcheck2 的清单
   python -m src.w2.data --build --data-root $VI/b200/data --out $O/w2_splits.json > $O/build.log 2>&1; echo "build rc=$?" >> $O/steps.log
   python -m src.w2.data --check --data-root $VI/b200/data --splits $O/w2_splits.json > $O/check.log 2>&1; echo "check rc=$?" >> $O/steps.log
   # (b) 耗时：取 RefCOCO 用到的 COCO 图 40 张，软链接到临时目录，单进程跑
   ```

   - (b) 的 40 张图：从 RefCOCO 所在的 COCO train2014 目录（数据根下，具体路径你在服务器上查）里，按文件名排序取前 40 张 640×480 的 jpg，软链接到 `$VI/tmp/bench0065/`。然后：

     ```bash
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m src.w2.candidates --bench $VI/tmp/bench0065 --repeats 1 > $O/bench.json 2> $O/bench.log; echo "bench rc=$?" >> $O/steps.log
     nproc >> $O/steps.log
     ```

   - 跑完删掉 `$VI/tmp/bench0065`。
4. 在 hpcmgt 上用 `sync_push.sh` 推一次（`ONCE=1`，TAG 0065，路径 `experiments/W2/0065_w2_splits_bench`）。不起常驻循环。
5. 删掉工作树 wt0065，再跑一次 `squeue`。

## 报告

推送 `reports/0065-hpc-w2-splits-bench.md`，内容：
- `steps.log` 原文；
- `build.log`、`check.log` 的最后 30 行；
- `w2_splits.json` 里 devcheck2 一节的统计：各步条数、被排除的图数、devcheck2 的 sha256。不要贴句子清单；
- `bench.json` 原文；
- 报错的 traceback。

只贴数字，不解读。
