[RUN-TASK 0007] 环境更新 + 全量测试 + E5 v2 摘要（填入 d 的规范值）+ 8 个基座模型的健康检查（IFEval 先只生成、不评分）

## 为什么做

- **健康检查**（简报 §8.11）：每个门都要把合并后模型与基座在固定 200 题上比较。基座分数现在就测，作为以后各门的参照。协议已在 D26 固定，这是第一次出分。
- **E5 生成器 v2**（D24、D25）：
  - c、e 必须在任何平台上逐位相同；
  - d 以本服务器为规范平台，这次填入 d 的规范摘要。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D24、D25、D26
- `DECISIONS_NEEDED.md` 的 N12：**不要下载 nltk 的 punkt_tab 数据**
- `scripts/health_check.py --help`、`scripts/synth_pin_digests.py --help`：参数以 argparse 为准

本任务只运行，不改 `src/`、`scripts/`、`tests/`、`configs/`。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含本任务书所在的 commit，以及 `647b007`。不包含就停下报告。

## 2. 环境更新

1. 在 venv 里跑 `pip install --dry-run -r requirements.txt`，结果存进 `pip_dryrun.txt`。
   - 预期：pymunk 已是 7.3.0，不变；新装 langdetect、nltk、immutabledict 及其依赖。
   - 会升降任何已装的包就**停下报告**，不要装。
2. 通过就执行 `PIP_NO_CACHE_DIR=1 pip install -r requirements.txt`。
3. **不要运行 `nltk.downloader`，不要设 NLTK_DATA**（N12）。
4. 安装前后各记一次 `df /` 与 `$PROJ` 的占用。

## 3. 全量测试

用 venv 跑 `python -m pytest tests/ -q -rs -rx`，完整输出存进 `pytest.txt`。

- 预期 11 个 skip：10 个是缺 punkt_tab 数据，1 个是缺 cv2。
- d 的规范平台比对：文件状态还是 `pending_canonical`，所以摘要不同时显示为 xfail，相同时显示为 pass，两者都正常。
- 其余全部通过。
- **任何其他失败都停下报告**，不改代码。c、e 的钉住摘要在这里失败也算"其他失败"。

## 4. E5 摘要

1. `python scripts/synth_pin_digests.py check`，原样存进 `synth_check.txt`。
   - c、e 的 120 个必须全部相同；
   - d 的比对结果照实记录。
2. `python scripts/synth_pin_digests.py write --only physics --out experiments/P0/0007_health_base/synth_digests_v2.json`
   - 这是本平台上 d 的规范值；
   - 不要改 `tests/data/`，由管理员复制回去。

## 5. 基座健康检查

### 5.1 模型与内存盘

8 个模型，依次跑：

1. `qwen3.5-0.8b`
2. `qwen3.5-2b`
3. `gemma-4-e2b-it`
4. `internvl3.5-2b-hf`
5. `qwen3.5-4b`
6. `gemma-4-e4b-it`
7. `internvl3.5-4b-hf`
8. `qwen3.5-9b`

内存盘照 RUN-0005 的做法：
- 每个模型跑完就删，只保留 `qwen3.5-0.8b`；
- 任何时刻不超过 25 GB；
- 下载与删除前后都记进 `disk_ledger.tsv`。

### 5.2 命令

每个模型一条：

```bash
python scripts/health_check.py --model <键> --device cuda --dtype bf16 --skip-ifeval-scoring \
  --out experiments/P0/0007_health_base/health_<键>.json
```

### 5.3 失败与时间

- 退出码为 2 时，输出文件里已有报错：照常推回，report 里贴出报错，然后继续下一个模型。
- 预计每个模型 20–45 分钟。
- 累计超过 8 小时：跑完当前模型就停，推回已完成的部分。

## 6. 产出

`experiments/P0/0007_health_base/`：
- `pip_dryrun.txt`、`pytest.txt`、`synth_check.txt`、`synth_digests_v2.json`
- 8 份 `health_<键>.json`
- `disk_ledger.tsv`、`README.md`

`reports/0007-health-base.md`，按 report 模板：
- 第 3 节原样贴：
  - 测试结果与 E5 摘要比对；
  - 每个模型一行的分数表：MMLU-Pro 准确率、GSM8K flexible 与 strict、IFEval 标"未评分"；
  - 解析失败计数、达到 max_new_tokens 的题数、墙钟、生成 token 数。
- 第 6 节写磁盘与内存盘收支。
- 只贴数字与事实，不做解读。

## 7. 回传

分三批：

1. §2–§4 完成后，推环境、测试与 E5 摘要；
2. 前 4 个模型跑完后，推它们的输出；
3. 全部跑完后，推其余输出和 report。

```bash
git add <逐个路径> && git commit -m "[results] 0007: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0007-health-base.md` 追加进完成清单，重新进入值守。
