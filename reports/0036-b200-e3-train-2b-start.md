# RUN-0036 启动报告 — B200 后台训练队列：2B 的 E3 全部变体 × 3 个种子，加网格格式预热 × 3 个种子

服务器 CC，b200，2026-10-07 14:34 UTC 起。只贴数字与事实，不做解读。最终报告等队列跑完、管理员发 addendum 后再写。

## 1. 拉取与校验

- **HEAD：** `flock $PROJ/logs/git.lock -c 'git -C $REPO pull --ff-only'` 之后 HEAD 是 `18bf339`。本任务书所在的 `2dc92d8` 在 HEAD 的历史里。
- **工作树：** `git worktree add --detach /dev/shm/visint/tmp/wt0036 HEAD`，HEAD 是 `18bf339`。队列在跑时不删。
- **配置：** 两个 sha256 都与任务书相同：
  - `configs/e3b_train.json`：`35e69b56324281356aba22419391fe8d092e3c0f0413ee1af4ed8ea74b60f457`；
  - `configs/e3a_train.json`：`28be66e52fd3e9e5b45c5b04e299342bda469c9c2f9d20e3b2de022b0c790067`。
- **训练数据：** `/dev/shm/visint/train_data` 四个源的文件数与 MANIFEST 的 n_files 相同，也与 RUN-0034 的核对结果相同。

  | 源 | 盘上文件数（不含 MANIFEST.json） | MANIFEST n_files | RUN-0034 |
  |---|---|---|---|
  | refcoco_train | 24411 | 24411 | 24411 |
  | nyu_depth_train | 45698 | 45698 | 45698 |
  | nyu_normals | 2899 | 2899 | 2899 |
  | ade20k | 44424 | 44424 | 44424 |

  - `dedupe/train_exclusions.json` 的 sha256 `40f214fb…` 与 RUN-0034 入库的副本相同。
- **权重：** RUN-0034 结束时 2B 已从内存盘删掉，这次按 RUN-0009 的清单重新下载并核对：

  ```
  qwen3.5-2b @ 15852e8c16360a2fea060d615a32b45270f8a8fc: 13/13 files verified (/dev/shm/visint/hf-hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc)
  ```

- **环境：** `pip install --dry-run -r requirements.txt` 没有任何 "Would install"。

## 2. 内存、磁盘与 GPU（启动前）

```
Wed Oct  7 14:34:58 UTC 2026
               total        used        free      shared  buff/cache   available
Mem:             180           5          27          18         147         155
Swap:              0           0           0
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda3       485G  413G   27G  94% /
tmpfs            91G   19G   73G  21% /dev/shm
Wed Oct  7 14:34:58 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA B200                    On  |   00000000:05:00.0 Off |                    0 |
| N/A   34C    P0            248W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

du -sh $PROJ:
7.2G	/home/ubuntu/workspace/wzg/gjl/vision-internalization
```

- 可用内存 155 GiB（MemAvailable 163 GB），高于 120 GB 的门槛；GPU 上没有任何进程。
- 本项目根盘占用 7.2 GB，最终检查点放 `$PROJ/ckpt/0036`。

## 3. 启动

- **时刻：** 2026-10-07T14:35:06Z。命令照任务书 §3：`setsid nohup bash /dev/shm/visint/tmp/wt0036/prompts/server/0036-b200-e3-train-2b/queue.sh > $PROJ/logs/train_0036.driver.log 2>&1 < /dev/null &`。
- **驱动的进程**（`ps -o pid,sid,cmd`）：

  ```
      PID     SID CMD
2169926 2169926 bash /dev/shm/visint/tmp/wt0036/prompts/server/0036-b200-e3-train-2b/queue.sh
  ```

  - 它的会话号是 2169926，启动它的 CC shell 的会话号是 2169922，两者不同。

## 4. 第一条运行（warmup_s0）

- 第一段 rc=0，墙钟 480 s（14:35:07–14:43:07）。
- 预热在第 200 步满足停止规则（留出集格式错误率 0/200；上限 2,000 步），所以第一段内就完成了：
  - `stop_rule_met=True stopped_at=200/2000 format_error_rate=0.0`；
  - final loss 0.0340。
- final 已由驱动拷到根盘并逐文件核对 sha256：`$PROJ/ckpt/0036/warmup/qwen3.5-2b/seed0/ckpt/final/model`，131,764 KB。
  - 记进 `experiments/E3/0036_e3_train_2b/ckpt_ledger.tsv`；
  - 驱动推送了 `2b5cf4f [results] 0036: warmup_s0 complete`。
- **驱动日志**（`train_0036.driver.log`）原文：

  ```
  2026-10-07T14:35:06Z 队列开始：commit 18bf339e0914047239983b34427216bb7371c49d，种子 0 1 2，每段 1000 步
2026-10-07T14:43:08Z warmup_s0 完成：final → /home/ubuntu/workspace/wzg/gjl/vision-internalization/ckpt/0036/warmup/qwen3.5-2b/seed0/ckpt/final/model（131764 KB）
  ```

- **`train_0036.status`** 原文：

  ```
  2026-10-07T14:43:14Z running e3b_s0 (chunk 1)
  ```

## 5. 之后

- 已把 `0036-b200-e3-train-2b.md` 追加进完成清单，回到值守。
- 评测任务来时，按 STANDBY 末节先写 `$PROJ/logs/train.pause`，等队列停在段的边界。
- 主 checkout 上的 git 操作一律放进 `flock $PROJ/logs/git.lock`。
