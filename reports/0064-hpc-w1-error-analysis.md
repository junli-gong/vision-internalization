# RUN-0064 报告：W1 错误分析（重放 RUN-0063 开发-检查集，只用 CPU）

任务书 `prompts/server_hpc/0064-hpc-w1-error-analysis.md`（origin c941a092）。不占 GPU，没写暂停文件，没有碰 06 的 RUN-0056、07 的 RUN-0057；没有往 `$VI/repo` 里 pull。经 hpcmgt 上脱离会话的 `$VI/tmp/run0064.sh`（输出 `$VI/logs/run0064.txt`）。只贴数字，不解读。

## 工作树

- 10-22 22:00:23Z squeue：157296、157297 都是 RUNNING。
- fetch 第 1 次 rc=0。wt0064：HEAD `c941a092402b494e586c555011fd11558a6e0a58`，含 7ab1e1d1，status 空。
- sha256，4 个都与任务书相同：replay.py `757e898c…`、w1_replay.py `331f76dc…`、workbench.py `775e0dca…`、sandbox.py `13ed14dc…`。

## 命令与结果

```
ssh -n hpcgpu07 "srun --jobid=157297 --overlap -n1 -c 8 env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=$VI/tmp PYTHONHASHSEED=0 \
    PATH=$VI/env/bin:/usr/bin:/bin \
    bash -c 'cd $VI/wt/wt0064 && $VI/env/bin/python scripts/w1_replay.py \
      --run $VI/repo/experiments/W1/0063_w1_devcheck_hpcgpu07 --data-root $VI/b200/data \
      --out $VI/repo/experiments/W1/0064_w1_error_analysis --workers 8 --classify' \
    > $VI/logs/replay_0064.log 2>&1 < /dev/null"
```

- `-c 8` 被 srun 接受，`--workers 8`。22:00:46Z – 22:03:24Z，**rc=0**，命令墙钟 158 s。
- 日志末尾原文：

```
[w1_replay] 写出 /home/mail/chenyuma/visint/repo/experiments/W1/0064_w1_error_analysis/replay.jsonl：500 条，重放不一致 5 条，墙钟 123 s
[w1_replay] 分类：进入分析 495 条；F1 失败 258 条 {'no_submission': 70, 'degraded': 8, 'bad_seed': 96, 'snap': 84}；F2 失败 235 条 {'no_submission': 50, 'degraded': 7, 'bad_seed': 72, 'snap': 106}
w1_replay n=500 replay_mismatch=5 wall_s=123
```

- 输出：`error_analysis.json`、`error_analysis.md`、`logs/`、`replay.jsonl`、`replay_meta.json`。没有 traceback。
- 22:03:24Z `ONCE=1 sync_push.sh … 0064 experiments/W1/0064_w1_error_analysis`：rc=0，`sync_0064.status` `2026-10-22T22:03:33Z ok`。没有起常驻循环。
- `git worktree remove wt0064` rc=0。22:03:34Z squeue 两条 RUNNING。

## replay_meta.json

- n_rows 500；n_replayed 500；**n_replay_mismatch 5**；n_image_mismatch_calls 0；workers 8；wall_s 123.26；sum_sample_wall_s 930.85；replay_sha256 `b33b944d…`。
- 输入核对：predictions `64842b4c…`（manifest_checked true）、config `24e2ed00…`、w1f `596b3ead…`、ownbox `db30e8c9…`、box `8ebd741d…`；tools_sha256 `dcc91402…`。
- 环境：python 3.12.13，numpy 2.5.2，cv2 4.10.0，skimage 0.24.0，scipy 1.18.0，host hpcgpu07。
- 不一致的 5 条（id：reasons）：
  - `refcoco_unc/val/32852`：text@4:run_python
  - `refcoco_unc/val/39123`：text@3:run_python、text@4:run_python
  - `refcoco_unc/val/93737`：text@4:run_python
  - `refcoco_unc/val/112453`：text@1:run_python、text@2:run_python、text@3:run_python、text@5:submit_mask、submission
  - `refcoco_unc/val/116368`：text@3:run_python

## error_analysis.md 原文

# W1 错误分析（DEV-0057；D113-4）

- 来源：`/home/mail/chenyuma/visint/repo/experiments/W1/0063_w1_devcheck_hpcgpu07`，模型 qwen3.5-9b，refseg/full/devcheck；回路记录 sha256 64842b4c996ad42c…
- 阈值：F1 final_iou < 0.5；F2 final_iou < w1f_iou − 0.05；改坏 best − final ≥ 0.1；种子放错 seed_box_iou_gt < 0.5；IoU 不变 |Δ| ≤ 0。恰好落在阈值上算达到阈值（比较留 1e-09 的余量）。
- 分类先后：没有提交 → 修改把好的结果改坏 → 种子点放错 → 贴边算法本身的问题（每条只进一类）。
- 重放：500 条，不一致 5 条（不进分类）；返回图的 sha256 不同的调用 0 次（只计数）。进入分析 495 条。

不一致的条目（前 20 条）：

- refcoco_unc/val/32852：text@4:run_python
- refcoco_unc/val/39123：text@3:run_python, text@4:run_python
- refcoco_unc/val/93737：text@4:run_python
- refcoco_unc/val/112453：text@1:run_python, text@2:run_python, text@3:run_python, text@5:submit_mask, submission
- refcoco_unc/val/116368：text@3:run_python

## 总体

- 有效提交 425/495；失败码 {'no_answer': 55, 'ok': 425, 'wrong_submission': 15}
- cIoU：实际 0.471；若交过程中最好的 mask 0.485；若交最后写出的 mask 0.460；W1-F 0.578
- 种子框的来源：{'grabcut_box': 397, 'mask_bbox:draw_box': 14, 'mask_bbox:draw_points': 1, 'mask_bbox:draw_polygon': 2, 'mask_bbox:select_superpixels': 43, 'none': 38}

## F1：final_iou < 0.5

失败 258 条（占进入分析的 52.1%；不计 0 条）。

| 类 | 条数 | 占比 | best_iou 中位 | seed_box_iou_gt 中位 | w1f_iou 中位 | final_iou 中位 | 没有种子 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 没有提交 | 70 | 27.1% | 0.000 | 0.535 | 0.511 | 0.000 | 38 |
| 修改把好的结果改坏 | 8 | 3.1% | 0.467 | 0.694 | 0.358 | 0.250 | 0 |
| 种子点放错 | 96 | 37.2% | 0.277 | 0.380 | 0.509 | 0.277 | 0 |
| 贴边算法本身的问题 | 84 | 32.6% | 0.368 | 0.700 | 0.425 | 0.368 | 0 |

"没有提交"这一类（70 条）：若交过程中最好的 mask，cIoU 0.142；若交最后写出的 mask 0.109；同一批句子上 W1-F 0.548；其中 32 条过程中有过 mask。

## F2：final_iou < w1f_iou − 0.05（没有 W1-F 记录的不计）

失败 235 条（占进入分析的 47.5%；不计 0 条）。

| 类 | 条数 | 占比 | best_iou 中位 | seed_box_iou_gt 中位 | w1f_iou 中位 | final_iou 中位 | 没有种子 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 没有提交 | 50 | 21.3% | 0.000 | 0.525 | 0.676 | 0.000 | 25 |
| 修改把好的结果改坏 | 7 | 3.0% | 0.600 | 0.748 | 0.626 | 0.438 | 0 |
| 种子点放错 | 72 | 30.6% | 0.340 | 0.401 | 0.678 | 0.326 | 0 |
| 贴边算法本身的问题 | 106 | 45.1% | 0.592 | 0.737 | 0.768 | 0.590 | 0 |

"没有提交"这一类（50 条）：若交过程中最好的 mask，cIoU 0.163；若交最后写出的 mask 0.121；同一批句子上 W1-F 0.683；其中 25 条过程中有过 mask。

全部没有有效提交的 70 条：若交最好的 mask，cIoU 0.142；若交最后写出的 mask 0.109；W1-F 0.548。

## 写进工具的框（seed_box）

| 量 | n | 缺失 | 均值 | p10 | p25 | 中位 | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 种子框 vs 自己的框 | 451 | 44 | 0.664 | 0.346 | 0.509 | 0.712 | 0.848 | 0.923 |
| 种子框 vs 真值框 | 457 | 38 | 0.645 | 0.321 | 0.483 | 0.698 | 0.842 | 0.904 |
| 自己的框 vs 真值框 | 495 | 0 | 0.841 | 0.593 | 0.863 | 0.935 | 0.963 | 0.977 |

种子框 vs 自己的框，IoU 直方图（左闭右开，最后一格含 1.0）：

| 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 15 | 16 | 27 | 48 | 48 | 58 | 71 | 96 | 67 |

（自己的框解析失败的，与真值框的 IoU 记 0，同 E1；没有自己的框时"种子框 vs 自己的框"记缺失。）

同一批 451 条（种子框与自己的框都有）：与真值框的 IoU 中位，种子框 0.698，自己的框 0.936；逐条差（种子框 − 自己的框）中位 -0.171，种子框更差的 397 条；Acc@0.5，种子框 72.5%，自己的框 93.1%。

## 每种工具调用之后 IoU 的变化

Δ = 写出的 mask 的 IoU − 参照 mask 在调用前的 IoU；参照：工具的输入 mask（refine / morph / components / 用 mask 初始化的 grabcut） → 同名的旧 mask → 调用前最后写出的 mask；都没有的记"无参照"。

| 工具 | 执行 | 成功 | 写出 mask | 升 | 降 | 不变 | 无参照 | Δ 中位 | Δ 均值 | 写出后 IoU 中位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zoom | 696 | 696 | 0 | 0 | 0 | 0 | 0 | — | — | — |
| show | 468 | 468 | 0 | 0 | 0 | 0 | 0 | — | — | — |
| submit_mask | 448 | 425 | 0 | 0 | 0 | 0 | 0 | — | — | — |
| grabcut | 422 | 422 | 422 | 25 | 18 | 3 | 376 | 0.013 | 0.087 | 0.529 |
| tile_view | 219 | 219 | 0 | 0 | 0 | 0 | 0 | — | — | — |
| superpixels | 174 | 174 | 0 | 0 | 0 | 0 | 0 | — | — | — |
| refine | 133 | 133 | 133 | 78 | 52 | 3 | 0 | 0.000 | 0.001 | 0.638 |
| select_superpixels | 108 | 108 | 108 | 36 | 18 | 2 | 52 | 0.183 | 0.167 | 0.380 |
| components | 107 | 107 | 25 | 14 | 10 | 1 | 0 | 0.000 | -0.000 | 0.680 |
| edges | 81 | 81 | 9 | 4 | 3 | 0 | 2 | 0.007 | -0.261 | 0.020 |
| draw_box | 28 | 28 | 28 | 1 | 3 | 1 | 23 | -0.039 | -0.153 | 0.402 |
| submit_box | 14 | 14 | 0 | 0 | 0 | 0 | 0 | — | — | — |
| read_pixels | 9 | 9 | 0 | 0 | 0 | 0 | 0 | — | — | — |
| morph | 6 | 6 | 6 | 3 | 2 | 1 | 0 | 0.000 | -0.072 | 0.694 |
| draw_polygon | 4 | 4 | 4 | 0 | 1 | 0 | 3 | -0.035 | -0.035 | 0.408 |
| watershed | 4 | 2 | 2 | 2 | 0 | 0 | 0 | 0.079 | 0.079 | 0.316 |
| draw_points | 2 | 2 | 2 | 0 | 1 | 0 | 1 | -0.064 | -0.064 | 0.323 |
| combine_masks | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0.025 | 0.025 | 0.657 |
| delete_mask | 1 | 1 | 0 | 0 | 0 | 0 | 0 | — | — | — |
| submit_count | 1 | 1 | 0 | 0 | 0 | 0 | 0 | — | — | — |

各类的样本 id 见 error_analysis.json（failure_defs.<F1|F2>.classes.<类>.ids）。
