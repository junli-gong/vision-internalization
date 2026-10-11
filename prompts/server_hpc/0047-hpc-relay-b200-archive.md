[RUN-TASK 0047] 把 B200 的归档从开发机传到 HPC 的 Lustre，并逐文件核对 sha256（D91-4）

## 铁律（先于一切）

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**

- 本任务不用 GPU，不碰 RUN-0045 的驱动与推送循环。
- 开始前与结束后各跑一次 `squeue -h -j 157296,157297 -o "%i %T %M %N"`，两条都必须是 RUNNING。

## 为什么做

- B200 已退役（D91）。剩下的工作全部在 A100 上跑，要用到 B200 上的两样东西：
  - 2B 的最终检查点：E5 的起点（E3b+C3、C3+预热）、2B 的重评与 E3a+C3 的机制比较都要用；
  - E1 评测数据：0.8B、4B 检查点的评测，以及基座 E1 的重跑要用。
- 管理员已把这些只读拷到开发机，并逐文件核对过。本任务把它们传到 Lustre，在 HPC 上再核对一遍。
- 与 RUN-0045 的训练同时进行。中转只占网络与 Lustre 的 I/O，不占 GPU。

## 0. 先读

- `CLAUDE.md`；`~/a100-hpc-guide.md`
- `DECISIONS.md` 的 D89、D91
- `reports/0044-hpc-bootstrap.md` 的数据中转一节：本任务照它的做法，经 Mac → lab5 → hpcmgt → Lustre。经验也照搬：
  - macOS 的 tar 要设 `COPYFILE_DISABLE=1`，不然会带进 `._*` 文件；
  - 长连接会降速，分成几条流并行；
  - 断了只补传缺的部分。

本任务只搬数据，不改任何仓库代码与配置。

## 1. 源与去处

源在开发机：`~/Research/vision-internalization-archive/b200_20261011/`（只读，不要在源目录里写任何文件）。

| 源（相对源目录） | 内容 | 文件数（`SHA256SUMS.b200`） | 大小 |
|---|---|---:|---:|
| `ckpt/0036/` | RUN-0036 的 2B 最终检查点（E3b、C1′、C3、E3a full/C1/C2、预热 × 种子 0–2），叶目录旁有 `config.yaml` | 189 | 1.6 GB |
| `ckpt/0040/` | RUN-0040 的 2B 最终检查点（E3b+C3、C3+预热 × 种子 0–2） | 48 | 1.1 GB |
| `data/` | E1 评测数据：refcoco、nyuv2、cvbench、embspatial、blink_reldepth、countbench、countbenchqa_questions、coco_val2017、coco_train2014_e1dev、odinw13 | 15,848 | 2.9 GB |
| `nltk_data/` | IFEval 句子切分用的 punkt_tab（健康检查要用） | 78 | 15 MB |
| `SHA256SUMS.b200`、`README.md` | 清单与说明 | — | 2 MB |

- **`logs/` 不传。** 它拷贝时还在写，不在核对范围里，也用不到。
- **去处：** 统一放在 `$VI/b200/` 下，保持相对路径不变，例如 `$VI/b200/ckpt/0036/…`、`$VI/b200/data/refcoco/…`、`$VI/b200/nltk_data/…`。这样 `SHA256SUMS.b200` 可以直接在 `$VI/b200` 里核对。以后的任务书用：
  - `--data-root $VI/b200/data`；
  - `NLTK_DATA=$VI/b200/nltk_data`；
  - 检查点 `$VI/b200/ckpt/0036`、`$VI/b200/ckpt/0040`。
- **份额（D89-7，上限 80 GB）：** 这次加约 5.8 GB。RUN-0045 驱动的份额命令只量 `$VI/ckpt`，固定部分按 34 GB 算，所以会少算这 5.8 GB。距 80 GB 的停止线还远，这次不改。报告里写一个实测的份额（`lfs quota`，或你在 RUN-0044 用过的办法），以后的任务书把固定部分改成实测值。

## 2. 做法

1. `squeue`，两条 RUNNING。
2. 在 hpcmgt 上 `mkdir -p $VI/b200`。
3. 照 RUN-0044 的中转办法传 `ckpt/`、`data/`、`nltk_data/`、`SHA256SUMS.b200`、`README.md`：
   - 用 tar 流，`COPYFILE_DISABLE=1`；
   - 可以按顶层目录分成几条流并行；
   - 不在 lab5 上落地任何文件，不在 B200 上做任何事。
4. 删掉万一带进来的 `._*`、`.DS_Store`，并记下个数。
5. **逐文件核对**：在 `$VI/b200` 里，用 `SHA256SUMS.b200` 去掉 `logs/` 那些行，跑 `sha256sum -c`。
   - 全部 OK 才算完成。
   - 有缺的或不对的，只补传那些文件，再核一遍。
   - 在哪台机器上算由你定（RUN-0044 的做法即可）。在计算节点上算的话，经 `srun --overlap` 进任务的资源范围，不用 GPU。
6. 再核对文件数：`$VI/b200` 下的普通文件数，应等于清单去掉 `logs/` 以后的行数（16,163），加上 `SHA256SUMS.b200`、`README.md` 这 2 个。
7. `squeue`，两条 RUNNING。

## 3. 报告

推送 `reports/0047-hpc-relay-b200-archive.md`，只记事实：

- 每条流的字节数、用时、平均速度；断开与补传的情况；
- 删掉的 `._*`、`.DS_Store` 个数；
- `sha256sum -c` 的汇总：OK 数、FAILED 数、缺失数；
- 文件数核对；
- 实测份额；
- 前后两次 squeue 的原文。

做完回到 RUN-0045 的巡检。有事用 SendMessage 发给 `cvpr2027-7c`。
