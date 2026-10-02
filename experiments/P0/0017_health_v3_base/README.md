# experiments/P0/0017_health_v3_base — RUN-0017：健康检查协议 v3，8 个基座重跑，加 GPU 上的字母读数核对

服务器 CC，b200，2026-10-02 05:47–07:16 UTC。运行报告：`reports/0017-health-v3-base.md`。

## 运行用的代码

8 个模型都用 commit `c3f6983` 的代码运行，各 health JSON 里 `run.meta.git.commit` 记的都是它。

- **前 4 个模型**（qwen3.5-0.8b、qwen3.5-2b、gemma-4-e2b-it、internvl3.5-2b-hf）：在主 checkout 里跑，驱动脚本是 `probe_scripts/run_health_v3.sh`。
- **后 4 个模型**（qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b）：在固定于 `c3f6983` 的 git worktree 里跑，worktree 在 `/dev/shm/visint/tmp/wt0017`，驱动脚本是 `probe_scripts/run_health_v3_wt.sh`。跑完 worktree 已删除。
  - **原因：** 推第一批时 `git pull --ff-only` 把主 checkout 快进到了 `97d12d1`，其中有 DEV-0020 对 `src/adapters` 的改动（7 个文件，+2025/−161 行）。
  - **代价：** 当时正在跑的 qwen3.5-4b 健康检查已经运行约 1.5 分钟，被停掉后从 worktree 重跑。

## 每个模型执行的命令

```bash
python scripts/health_check.py --model <键> --device cuda --dtype bf16 --out experiments/P0/0017_health_v3_base/health_<键>.json
python scripts/health_check.py --compare experiments/P0/0011_health_v2_base/health_<键>.json \
  experiments/P0/0017_health_v3_base/health_<键>.json --out experiments/P0/0017_health_v3_base/compare_v2_v3_<键>.json
python experiments/P0/0017_health_v3_base/probe_scripts/audit_letter.py <键> experiments/P0/0017_health_v3_base/health_<键>.json \
  experiments/P0/0017_health_v3_base/audit_letter_<键>.json 100
```

## 文件

| 路径 | 内容 |
|---|---|
| `health_<键>.json` ×8 | 健康检查 v3 的输出：`mmlu_pro_letter`、`mmlu_pro_free256`、`mmlu_pro_fmt32`、`gsm8k`、`ifeval` 五个读数；`option_spellings` 是各字母用到的单 token 写法 |
| `compare_v2_v3_<键>.json` ×8 | 与 RUN-0011（v2）的比较 |
| `audit_letter_<键>.json` ×8 | 字母读数核对，每题一行，另给三项的相同条数（见下） |
| `health_summary_v3.md`、`health_summary_v3.json` | 由 `probe_scripts/p21_v3_summary.py` 从上面的文件生成：分数、与 v2 的逐题核对、字母读数核对、字母写法、截断率、计算量 |
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt` 的输出，没有 "Would install" |
| `disk_ledger.tsv` | 磁盘收支，单位字节 |
| `logs/` | 各模型的 `health_`、`compare_`、`audit_` 日志，以及 `driver.log`（含手动停止的记录）；`h3_*.csv` 和 `h3wt_*.csv` 是两个驱动期间每 5 s 一次的 nvidia-smi 采样 |
| `probe_scripts/` | `audit_letter.py`、`p21_v3_summary.py`、`run_health_v3.sh`、`run_health_v3_wt.sh`、`gpu_sampler.sh`、`diskcheck17.sh` |

## `audit_letter.py`（任务书 §3.1）

只调用仓库里现有的函数，不改仓库代码：

- **适配器**：`scripts/health_check.py` 的 `make_adapter`，参数是 `--device cuda --dtype bf16`。
- **每题两次调用**：
  1. `src.health.run.letter_text_only` 做一次 `option_logits`；
  2. `adapter.generate(max_new_tokens=1, enable_thinking=False, allowed_token_ids=本题全部字母的全部单 token 写法, **P.GREEDY_KWARGS)`。
- **记录三项**：
  1. 生成的 token 是否等于 option_logits 选中的 token；
  2. 两者对应的字母是否相同；
  3. option_logits 选中的字母是否等于 health JSON 里该题 `mmlu_pro_letter` 的 choice。

后 4 个模型跑的是 worktree 里的同一份探针副本，内容与本目录的版本逐字节相同。探针按自身位置定位仓库根目录，所以加载的是 worktree 里的代码。
