# experiments/E1/0033_e1_posthoc2 — RUN-0033：E1 事后读数的后续（D72-2/3）：NYU 点对的对调标记去偏（8 个模型）、结构强制的多边形（Qwen 四个尺寸 × 两个变体）、"自己的框"基线

服务器 CC，b200，2026-10-07 03:01–03:38 UTC。运行报告：`reports/0033-e1-posthoc2.md`。都是**事后**读数（D72-2/3；E1 预注册 §10.3；实现 DEV-0036，D74）。

## 运行用的代码

- 驱动 `probe_scripts/run_e1_33.sh` 在一个固定于任务开始时 HEAD `a716a92` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0033`），跑完已删除。
- 驱动用 `setsid nohup` 启动，与 CC 的进程脱钩（STANDBY.md "长时间运行的驱动"）。
  - 每个模型跑完，由驱动自己推送（`probe_scripts/push_model.sh`）。
  - 驱动每 30 分钟以及每一步开始时更新心跳。
- `configs/e1_prompts.json` 的 sha256 是 `f3353de0…`，`configs/e1_posthoc.json` 的 sha256 是 `9a560782…`，见 `logs/worktree.txt`。
- 两份汇总也在这个 worktree 里运行，读 worktree 里的 RUN-0027、0028、0030 结果，三者在 `a716a92` 里各有 8 个模型。

## 每个模型执行的命令

模型顺序同 RUN-0027，批配置用冻结值 64，环境同 RUN-0030。

```bash
python scripts/run_e1.py --model <键> --task rel_depth_nyu --subset letter_swap --data-root $PROJ/data --out <本目录>
# 只有 Qwen 四个尺寸：
python scripts/run_e1.py --model <键> --task refexp_poly --subset forced_poly --data-root $PROJ/data --out <本目录>
```

全部跑完后，按任务书 §3 原样运行汇总（CPU）：

- `summarize_e1_posthoc.py validity --debiased experiments/E1/0030_e1_posthoc --letter-swap <本目录> --free … --main …` 生成 `validity_d72.*`；
- `summarize_e1_posthoc.py ownbox --main … --think … --forced experiments/E1/0030_e1_posthoc --forced-poly <本目录> --data-root $PROJ/data` 生成 `ownbox.*`。

## 文件

| 路径 | 内容 |
|---|---|
| `<模型>/rel_depth_nyu__letter_swap/` | 对调标记的去偏读数：`predictions.jsonl`、`readings.jsonl`（每张图每条的读数）、`metrics.json`（`letter_swap`：去偏读数、同次运行原图那一遍的字母读数与配对差、两张图各自的读数、答案字母分布）、`compute.json`、`config.yaml`、`meta.json`、`README.md`、`logs/` |
| `qwen3.5-*/refexp_poly__poly_structured/`、`refexp_poly__poly_structured_min8/` | 结构强制的多边形：至少 3 点、至少 8 点 |
| `validity_d72.md/json` | D72-2 的标题表与有效性检查 |
| `ownbox.md/json` | "自己的框"基线与预填、结构强制、主评测、思考的并列表；近矩形占比；id 核对 |
| `letter_swap_table.md/json` | 由 `probe_scripts/p32_letter_swap_table.py` 从各 `__letter_swap/metrics.json` 原样取出的表，另附用量 |
| `logs/` | 每条命令的日志、`driver.log`、`summary_*.log`、`worktree.txt`；`posthoc2_gpu.csv`、`posthoc2_apps.csv` 是 nvidia-smi 采样 |
| `disk_ledger.tsv`、`pip_dryrun.txt` | 磁盘收支；dry-run 输出（没有 "Would install"） |
| `probe_scripts/` | `run_e1_33.sh`、`push_model.sh`、`prep_push.py`、`p32_letter_swap_table.py`、`gpu_sampler.sh`、`diskcheck33.sh` |

本目录没有超过 20 MB 的文件。
