# RUN-0067 报告：hpcgpu06，W2-c 训练数据第一步（base 9B 在 train 抽样 16,000 句上出框）

任务书 `prompts/server_hpc/0067-hpc-06-w2c-propose.md`（origin d896a27a）。没有碰 07（RUN-0057、RUN-0066）。只贴数字，不解读。

## 启动

- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0067.sh`（PID 491595，10-23 08:47:33Z 起）：
  - fetch rc=0；wt0067 HEAD `3feb46f4cc3d45590aff8ccaa1f3207ded238459`，含 d896a27a，status 空，`experiments/` 只有 P0。
  - 6 个 sha256 逐字与任务书相同：propose.sh `f0cfb0f9…`、build_w2c_data.py `15d09131…`、w2c_data.py `4fa7f9cd…`、w2_splits.json `9dd22dff…`、e3b_train.json `b265cd87…`、eval_lib.sh `2380464b…`；propose.sh 用本节点暂停文件。
- 插曲：一条早先因中继卡住、以为没发出去的启动命令在 09:02:53Z 迟到执行，起了第二个实例；它见 `wt0067 already exists` 立即退出，没做别的，只是把 `$VI/logs/start0067.txt` 截断了（日志不全）。以下读数取自 `eval_0067.status`、`steps.log` 与结果文件。
- 启动条件 10-24 03:08:19Z 满足（脚本记录原文）：`train status: 2026-10-23T23:29:53Z done 0056-hpcgpu06; run_0056.log: 2026-10-24T03:04:45Z done; no 0056 procs on 06; GPU idle; no 06 pause file`。前后 squeue 两条 RUNNING。
- 03:08:19Z 按任务书命令启动 propose.sh（`EV_CAP_H=3`），step 157296.168。

## steps.log 原文

```
2026-10-24T03:08:20Z start commit 3feb46f4cc3d45590aff8ccaa1f3207ded238459 node hpcgpu06
2026-10-24T03:51:39Z propose rc=0 wall_s=2599
2026-10-24T03:51:39Z done; failed steps: 0
```

- 心跳 `2026-10-24T03:51:39Z done 0067 (failed steps: 0)`；墙钟 2,599 s（0.72 h），上限 3 h。
- 03:53:20Z `ONCE=1 sync_push.sh … 0067 experiments/W2/0067_w2c_data_hpcgpu06`：第 1 次 `2026-10-24T03:53:34Z ok`。没有起常驻循环。之后 `train.pause.hpcgpu06` 不存在；squeue 两条 RUNNING。06 空着等下一份任务书。

## manifest.json（`w2c_data/manifest.json`）的去重与抽样统计

| 步 | 条数 / 值 |
|---|---|
| 总体（E3 / C3 的训练池，`src/e3/data.load_pool('refseg')`） | refcoco 107,859 → 剔除 27 → 107,832；refcoco+ 107,376 → 剔除 27 → 107,349；refcocog 72,369 → 剔除 32 → 72,337；空句 0；合计 **287,518**（sha256 `5b9b9312…`） |
| E3 剔除清单 | `dedupe/train_exclusions.json`（sha256 `40f214fb…`），去掉 86 句 |
| W2 评测图（devtune800 636 张、devcheck2 379 张、5 个 test 子集 541–831 张） | 合计 3,152 张（sha256 `2b5c42f9…`） |
| `w2_eval_exclusion` | n_before 287,518；**n_removed 0**；n_images_hit 0；n_after 287,518 |
| 抽样（salt `w2c_train_v1`，purpose `w2c_train`） | 要 16,000，得 **16,000** 句，覆盖 10,438 张图；sha256 `7f2dd6ea9f1b8128a685bbeb076d3f4835e14922eaf1c7f16228a44b22120992`；按序 sha256 `eb4ee124…` |
| **`checks.selected_on_w2_eval_images`** | **0** |

## 出框（`propose/metrics.json`）

- 任务 refexp_box，n 16,000，complete True，模型 qwen3.5-9b（revision `c2022362…`），关思考，max_new_tokens 256。
- 标题读数：acc@0.5 **0.8742** [0.8694, 0.8791]，n 16,000；mean_iou 0.8130。
- **失败率**：parse n_ok 15,691、n_failed 309，**failure_rate 0.0193**；failures：multiple_boxes 287、truncated 20、malformed 1、no_box 1。截断率 0.00125；拒答 1。
- 生成 token：中位 46，p95 55，最大 256。

## 其他

- 显存峰值：本任务的脚本不采样显存，结果文件与日志里都没有。
- `logs/propose.log` 里没有 traceback，也没有 out of memory。
