[RUN-TASK 0102] hpcmgt（登录节点，只用网络与 CPU）：X-M 的数据落地——COCO 2017、LVIS v1 val、ReasonSeg val（D165）。不用 GPU；按下载速度，约 1–3 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不进 srun，不碰两台 GPU 节点上的任何东西。

## 为什么

- 负责人的 X-M 指令（D165）要用 COCO 2017 train（训练）、COCO val2017（调参、判定、最终报告三个划分）、LVIS v1 val 子集与 ReasonSeg val（描述性报告）。
- 这一步只把原始文件落地并记 sha256。划分、去重与样条目标由开发任务另写，不在这里做。
- **数据只用于非商业研究，不再分发**（D28-9）。都放在仓库以外的 `$VI/data/`，不进 git。

## 做法

1. hpcmgt 上 `squeue`。确认 Lustre 余量 ≥ 60 GB（只记录，D95）。
2. 目录：`$VI/data/coco2017/{zips,images,annotations}`、`$VI/data/lvis`、`$VI/data/reasonseg`。
3. **COCO 2017**（官方地址；直连慢就报告速度，再试镜像，用了镜像要记来源）：
   ```bash
   cd $VI/data/coco2017/zips
   wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   wget -c http://images.cocodataset.org/zips/val2017.zip
   wget -c http://images.cocodataset.org/zips/train2017.zip
   ```
   - 先下标注与 val2017（小，约 1 GB），解压后**先报告**；train2017（约 18 GB）接着下。
   - 解压到 `images/train2017`、`images/val2017`、`annotations/`。核对张数：train2017 118,287，val2017 5,000。
   - 每个 zip 与解压出的标注 json 记 sha256、字节数，写 `$VI/data/coco2017/MANIFEST.json`。
4. **LVIS v1 val 标注**：`https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip`，解压到 `$VI/data/lvis/`，记 sha256。图像用 COCO 2017 的，不另下。
5. **ReasonSeg val**：找 LISA 官方 README 给的发布地址（dvlab-research/LISA），选能连上的官方来源下载 val 部分，解压到 `$VI/data/reasonseg/val/`，记来源、sha256、张数。
   - 连不上任何官方来源就停下来报告，不用来路不明的拷贝。
6. 把三份 MANIFEST.json 的内容抄进 `reports/0102-hpc-data-xm.md`，连同下载速度与用时；只推报告，不推数据。再跑一次 `squeue`。
