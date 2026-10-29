[RUN-TASK 0095] CPU，只读：从 RUN-0072 的 W2-a 逐句预测里抽出每句的交与并，推回仓库（D152）。不用 GPU，几分钟，现在就发；不影响 06 上的 RUN-0094 与 07 上的 RUN-0093

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不进 srun，不碰任何节点上在跑的任务。

## 为什么

- 论文的 W2 test 主表里，"合并 5,000 句"一列只有配对差，没有各系统自己的读数。原因是 W2-a 的 predictions.jsonl 每个约 10 MB，没同步进仓库。
- 这里只取每句的 id、inter、union、iou，5 个划分合计约 0.5 MB。管理员在开发机上算合并读数。

## 文件（工作树 wt0095 建在 origin/main，只需这一个文件）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0095-hpc-w2a-scores/extract_scores.py` | `4816c65288253d5c17f605710e932b55ec7e826f09d71f9c505c120b9cbf605f` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0095（稀疏），核对 sha256。
2. 在 hpcmgt 上运行（只用标准库）：
   ```bash
   VI=/home/mail/chenyuma/visint
   nice -n 10 $VI/env/bin/python -I $VI/wt/wt0095/prompts/server_hpc/0095-hpc-w2a-scores/extract_scores.py \
     --run $VI/repo/experiments/W2/0072_w2_test_hpcgpu06 --setting w2a --out $VI/repo/experiments/W2/0095_w2a_scores
   ```
   - 脚本先核对 5 个 predictions.jsonl 的 sha256、字节数与 `LARGE_FILES.manifest.tsv` 逐字相同，metrics.json 为 complete；任何一条不符就退出，不写结果。
   - 每个划分打印一行：行数、由逐句交并重算的 cIoU、metrics.json 的 cIoU、match。全部 match 时 rc=0。
3. 把 `experiments/W2/0095_w2a_scores/`（5 个 jsonl 与 summary.json）提交推送，提交信息 `[results] 0095: W2-a per-sentence scores`。
4. 写 `reports/0095-hpc-w2a-scores.md`：贴 stdout 原文、rc、summary.json 里每个划分的 out_sha256。再跑一次 `squeue`。

管理员在开发机上用 W2-c 的叶目录拼了一个假的 RUN-0072 彩排过：5 个划分都 match，rc=0；目录名不带运行编号时拒绝。
