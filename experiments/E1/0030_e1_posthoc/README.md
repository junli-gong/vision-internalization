# experiments/E1/0030_e1_posthoc — RUN-0030：E1 事后读数（去偏字母读数 8 个模型；格式强制诊断 Qwen 四个尺寸）

服务器 CC，b200，2026-10-04 06:35–07:46 UTC。运行报告：`reports/0030-e1-posthoc.md`。两项都是**事后**读数（D66；E1 预注册 §10，写于运行之前；实现 DEV-0033，D68）。

## 运行用的代码

- 驱动 `probe_scripts/run_e1_posthoc.sh` 在一个固定于任务开始时 HEAD `9099f23` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0030`），跑完已删除。
- `configs/e1_prompts.json` 的 sha256 是 `f3353de0…`，`configs/e1_posthoc.json` 的 sha256 是 `e02971cd…`，见 `logs/worktree.txt`。44 个 metrics.json 记的也是这两个值。
- 两份汇总与 `estimate` 也在这个 worktree 里运行。RUN-0027、RUN-0028 的结果从 worktree 里读，它们在 `9099f23` 里都是完整的 8 个模型。
- 输出用绝对路径写进本目录。

## 每个模型执行的命令

模型顺序同 RUN-0027。

```bash
python scripts/run_e1.py --model <键> --task all --subset debiased --data-root $PROJ/data --out <本目录>
# 只有 Qwen 四个尺寸，紧跟在上一条之后：
python scripts/run_e1.py --model <键> --task all --subset forced --data-root $PROJ/data --out <本目录>
```

- **批配置：** 冻结值 64，没有加 `--batch-size`。
- **环境：** 同 RUN-0027。
- **`--task all`：** 对 debiased 展开为 4 个字母读数任务，对 forced 展开为 refexp_poly、grid16。

**全部模型跑完后：**

```bash
python scripts/summarize_e1_posthoc.py validity --debiased <本目录> --free experiments/E1/0027_e1_main \
  --main experiments/E1/0027_e1_main --out-json validity.json --out-md validity.md
python scripts/summarize_e1_posthoc.py forced --forced <本目录> --main experiments/E1/0027_e1_main \
  --think experiments/E1/0028_e1_think --out-json forced.json --out-md forced.md
# 任务书没有要求；只读，用来给"与估算的对照"提供逐模型的估算：
python scripts/summarize_e1_posthoc.py estimate --main experiments/E1/0027_e1_main --out-json estimate.json --out-md estimate.md
```

## 文件

| 路径 | 内容 |
|---|---|
| `<模型>/<任务>__debiased/` | 去偏字母读数：`predictions.jsonl`（每条的答案）、`readings.jsonl`（每种顺序每条的读数）、`metrics.json`（`debiased`：去偏读数、同次运行的原字母读数与配对差、每种顺序、随机水平、答案字母分布）、`compute.json`、`config.yaml`、`meta.json`、`README.md`、`logs/` |
| `qwen3.5-*/refexp_poly__forced_prefill/`、`grid16__forced_restricted/`、`grid16__forced_structured/` | 格式强制，同样的文件，没有 `readings.jsonl` |
| `validity.md/json` | §10.1 的有效性检查（`summarize_e1_posthoc.py validity`） |
| `forced.md/json` | §10.2 的配对比较（`summarize_e1_posthoc.py forced`） |
| `estimate.md/json` | 按 RUN-0027 实测吞吐的用时估算（`summarize_e1_posthoc.py estimate`，只读） |
| `debiased_table.md/json` | 由 `probe_scripts/p30_debiased_table.py` 生成：从各 `__debiased/metrics.json` 原样取出去偏读数表与每种顺序的读数；用量表取自 `logs/driver.log` 与 `estimate.json` |
| `logs/` | 每条命令的日志 `<模型>__{debiased,forced}.log`、`driver.log`、`summary_{validity,forced}.log`、`worktree.txt`；`posthoc_gpu.csv`、`posthoc_apps.csv` 是每 5 s 一次的 nvidia-smi 采样 |
| `disk_ledger.tsv`、`pip_dryrun.txt` | 磁盘收支；dry-run 的输出（没有 "Would install"） |
| `probe_scripts/` | `run_e1_posthoc.sh`、`push_model.sh`、`prep_push.py`、`p30_debiased_table.py`、`gpu_sampler.sh`、`diskcheck30.sh` |

- 本目录没有超过 20 MB 的文件，没有 gzip。最大的是 gemma 两个模型 embspatial 的 `readings.jsonl`，约 10.6 MB。
- `validity.md/json` 末尾有 4 行"probe_scripts … 缺 ['debiased', 'free']"。原因是汇总脚本的 `models_in` 把本目录下除 `logs` 以外的子目录都当成模型，见报告 §4。
