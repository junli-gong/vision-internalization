[RUN-TASK 0016] E1 算力估算 v3（协议 v2 代码）：大批量下的思考吞吐 + 四个大模型的首次估算 + 检测开发图标注落地

## 为什么做

- RUN-0012、RUN-0013 两次都只估算完 4 个小模型。qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b 一次都没测过。
- 4 个小模型在旧协议下合计 29.2 GPU 小时：
  - 主评测 12.1；
  - 思考子集 11.2；
  - 8192 敏感性 5.9。
- 思考子集只按批大小 8、每任务 8 条测过。当时 GPU 利用率均值约 55%，估计偏保守。
- DEV-0017（E1 协议 v2，D38）改了三处：
  - 选择题与点对任务改成受限字母读数，只生成 1 个 token；
  - 新增 `free` 子集，是这四个任务的自由作答诊断，关思考、上限 1024；
  - 批配置冻结为批大小 16、左填充、样本按选择顺序。主评测按 16 外推。
- 本任务要回答三个问题：
  1. 思考子集在批大小 32 或 64 下快多少；
  2. 关思考的主评测在 32、64 下比 16 快多少：要不要把冻结的批大小改大，由管理员在验收 v3 之前定；
  3. 4 个大模型的各子集要多少。
- 结果用于 E1 预注册 §9 的算力表。
- 本任务只用开发子集，不碰任何评测子集。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D33、D36（第 1、6 条）、D38
- `reports/0013-e1-accept-estimate-v2.md` 的 §2.2、§3.2、§4
- `python scripts/run_e1.py --help`、`python scripts/data/land_e1.py --help`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `319c2fe`（DEV-0017 的合入）与本任务书所在的 commit。不包含就停下报告。

## 2. 环境与落地

- `pip install --dry-run -r requirements.txt` 应当没有任何 "Would install"。有的话停下报告。
- 设下列环境变量：
  - `VISINT_DATA_ROOT=$PROJ/data`
  - `NLTK_DATA=$PROJ/nltk_data`
  - `TMPDIR=/dev/shm/visint/tmp`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 设 `PYTORCH_CUDA_ALLOC_CONF` 的原因：RUN-0013 第 13 行，nvidia-smi 看到 162 GB，而 `max_memory_allocated` 只有 14 GB；GPU 是多人共用的。这个变量只改缓存分配，不改数值。
  - report 里贴出每个模型的 nvidia-smi 峰值，与 RUN-0013 对照。
- **落地检测开发图的标注**（验收 v3 要用，本任务先落好）：

  ```bash
  python scripts/data/land_e1.py coco_train2014_e1dev --root $PROJ/data
  ```

  - 会往 TMPDIR 写约 107 MB 压缩数据，解出来约 332 MB，解析时要几 GB 内存；`--root` 里只多出几 MB。
  - 清单副本放进运行目录。
  - 失败照实记下，继续做估算。
- 模型与内存盘规则同 RUN-0013：
  - 每个模型跑完即删，只保留 qwen3.5-0.8b；
  - 内存盘 ≤ 25 GB；
  - 权重按 `experiments/P0/0009_ifeval_rescore/weights_manifest.json` 核对。

## 3. 估算

顺序：**先跑 4 个大模型，再跑 4 个小模型**。

1. qwen3.5-4b
2. gemma-4-e4b-it
3. internvl3.5-4b-hf
4. qwen3.5-9b
5. qwen3.5-0.8b
6. qwen3.5-2b
7. gemma-4-e2b-it
8. internvl3.5-2b-hf

每个模型一条命令：

```bash
python scripts/run_e1.py --estimate --model <键> --task all --batch-sizes 16 32 64 \
  --estimate-n 64 --estimate-think --estimate-think-n 64 \
  --out experiments/E1/0016_estimate
```

- 主评测按冻结的 16 外推；32 与 64 只测，供比较。
- 思考的批大小按现有代码取 min(推荐批大小, 64)，以 estimate.json 里记的为准。
- 附加口径只跑 `--task refexp_box ovd --batch-sizes 16 32 64`，不加 `--estimate-think`：
  - qwen3.5-9b 与 qwen3.5-0.8b 加 `--policy qwen:1024`；
  - internvl3.5-4b-hf 与 internvl3.5-2b-hf 加 `--policy internvl:native`；
  - 各紧跟在对应模型的主口径之后跑。
- 某个批大小报 OOM 或 `mem_ok=False`，照实记下，不重试，不改参数。

## 4. 时间与停止

- 预计合计 4–6 GPU 小时：大模型每个约 40–70 分钟，小模型每个约 25–40 分钟。
- 每个模型开始前检查累计耗时：超过 6 小时（21,600 s）就不再开始新模型，推回已完成的部分。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E1/0016_estimate/`：
  - run_e1 的全部输出；
  - `estimate_summary.md/json`：可以在 RUN-0013 的 `p16_estimate_summary.py` 基础上改，放进 `probe_scripts/`；
  - 落地清单副本、`disk_ledger.tsv`、`README.md`、`logs/`、`probe_scripts/`。
- `reports/0016-e1-estimate-v3.md`，按 report 模板，第 3 节原样贴：
  - **估算汇总**：主口径，逐模型 × 逐任务。列与 RUN-0013 §3.2 相同，另加四列：
    - 外推用的批大小（extrapolated_with）；
    - 思考实测批大小；
    - free 子集的 n 与 GPU 小时；
    - 16、32、64 各自的样本/秒。
  - **每个模型的合计**：eval、free、think、think8192。
  - **附加口径**。
  - **开思考实测**：批大小、样本数、样本/秒、平均生成 token、截断率、峰值显存。
  - **同模型的思考吞吐对照**：
    - 4 个小模型逐任务列 RUN-0013 的批大小 8 与本次的思考样本/秒，以及两者之比；
    - 大模型没有可比数字，写"—"。
  - **nvidia-smi 峰值**：每个模型一行，与 RUN-0013 对照。
  - **落地结果**：rc、文件数、核对、体积、根盘前后。
- 只贴数字与事实，不做解读。
- 分两批：
  1. 4 个大模型跑完推一次；
  2. 其余跑完后，连同 report 一起推。

```bash
git add <逐个路径> && git commit -m "[results] 0016: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0016-e1-estimate-v3.md` 追加进完成清单，重新进入值守。
