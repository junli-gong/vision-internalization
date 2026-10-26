[RUN-TASK 0075] hpcgpu07，只用 CPU：生成 devcheck3 与 explore_val 的清单（D126-4、D126-5、D128、D130）。约 10 分钟，不占 GPU，现在就可以发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 RUN-0072、0073、0074。
- 本任务不用 GPU，不写任何暂停文件，不调 `claim_gpu`。

## 为什么

- 负责人要求"现在就密封 devcheck3"（D126-4）：从 RefCOCO 系列 val 取 500 句，按哈希固定，写进预注册。
- 同时生成探索用的 explore_val。
- 本任务**只生成句子 id 的清单，不跑任何系统**。

## 密封

- 这份清单生成之后，在 11-02 的判定（门名 G-EX）之前，**不在 devcheck3 上跑任何系统**：不跑模型、不跑工具、不算候选。
- 以后任何任务书都不会让你在 devcheck3 上跑，除非写明了 G-EX。
- 看清单里的 id 与计数没有关系。

## 文件（commit 见发出时的启动语；工作树 wt0075 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `src/data/explore_splits.py` | `3635a8de1e537cd75de93eb3de2ce6e3cbab7482be39df490d85ab9e9c0f78aa` |
| `src/w2/data.py` | `b741e01957f57e0a32678e96555432070675586391dbc76cc69c339883c1658e` |
| `src/w2/w2c_data.py` | `4fa7f9cd929592dd29a200907ba8ca07881c8702a2464df720e78bd6a5ff4c03` |
| `src/data/subsets.py` | `73e712d42a8d1ee101e7a476f3297a4e84d8e58ab706db21b92a674f54304239` |
| `src/data/loaders.py` | `f0bbd0d7984d574dd9677aef8703390a316dc707439cba4100c31bc192bcb7c8` |
| `configs/e1_eval_subsets.json` | `29415ea0d131336af52c950c053f95534a526261870a38ff66de2285c2545139` |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |
| `configs/w1_splits.json` | `a1371851555e854fac6f9d0e267f19a23602ee9cf31d5c303ef08432d1f7f0f7` |
| `configs/explore_splits.json`（骨架） | `0c4c2ebca781b4ebdd185fb0fbfc03b00aef2cefaf4a1e5733bbb51077043422` |

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0075（稀疏，规则同 RUN-0072），建在 origin/main。fetch 失败就重试。核对上表 9 个 sha256。
3. **生成**：在 07 上用 CPU，与 RUN-0073 共用节点，只占 4 核：

   ```bash
   ssh -n hpcgpu07 'cd /home/mail/chenyuma/visint/wt/wt0075 && srun --jobid=157297 --overlap -n1 -c 4 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONHASHSEED=0 \
       HF_HUB_OFFLINE=1 TMPDIR=/home/mail/chenyuma/visint/tmp \
     python -m src.data.explore_splits --build --data-root /home/mail/chenyuma/visint/b200/data \
       --train-data-root /home/mail/chenyuma/visint/data/train_data \
       --out /home/mail/chenyuma/visint/repo/experiments/X/0075_explore_splits_hpcgpu07/explore_splits.json \
     > /home/mail/chenyuma/visint/logs/run_0075_build.log 2>&1 < /dev/null'
   ```

   - 前台跑即可，预计 2–10 分钟，大部分时间在读约 29 万句的训练池。
   - 输出目录不存在就先建。退出码非 0 就贴日志原文，停下等我。
4. **核对**：在同一节点上用同样的方式跑一次整份重算：

   ```bash
   python -m src.data.explore_splits --check --splits <上一步的文件> \
     --data-root /home/mail/chenyuma/visint/b200/data --train-data-root /home/mail/chenyuma/visint/data/train_data
   ```

   日志写 `$VI/logs/run_0075_check.log`。不通过就贴原文，停下。
5. **推送**：
   - 用 `sync_push.sh` 的 ONCE=1 推一次 `experiments/X/0075_explore_splits_hpcgpu07`，里面只有这一个 json 和两份日志的拷贝；
   - 再推送报告。
6. 再跑一次 `squeue`。

## 报告

推送 `reports/0075-hpc-07-explore-splits.md`，内容：
- 两次运行的命令、退出码、墙钟；
- 输出文件的 sha256 与字节数；
- 清单 `checks` 一节的原文，各项重叠计数必须都是 0；
- 每个划分与组的 n、sha256，以及构造各步剩下的句数与图数；
- explore_val 有没有不够数的层（`all_taken`）；
- 与训练池核对的状态与计数；
- 任何警告。

**不在 devcheck3 上跑任何东西。** 清单交回后由管理员放进 `configs/explore_splits.json`。
