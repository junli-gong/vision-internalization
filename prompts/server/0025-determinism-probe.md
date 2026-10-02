[RUN-TASK 0025] E1 验收 (i) 的诊断：qwen3.5-2b 定位同配置重跑有 1 条不同，找原因（GPU 上的非确定性）

## 为什么做

- 项目负责人的验收 v3 第 (i) 条（D36-1）：同一批配置重跑，输出逐位相同。
- RUN-0020 与 RUN-0024 合计 72 个任务文件，71 个逐字节相同。只有 qwen3.5-2b 的 refexp_box 有 1 条不同：
  - 样本 `refcoco+_unc/val/7909`，第 4 批，批大小 64；
  - A1 的框是 `[506, 496, 631, 988]`，A2 是 `[508, …]`；
  - 同一批的其余 63 条逐字节相同；
  - A1、A2 在同一个进程里先后跑，GPU 上只有这一个进程。
- 同一进程、同一批组成却出现不同，说明有 GPU 上的非确定性。本任务要回答三个问题：
  1. 能不能复现，频率多高；
  2. 换成确定性设置能不能消掉；
  3. 代价多大。
- 这不是实验，只是仪器诊断。用开发子集，不碰评测子集。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`（第 3 步：从固定 commit 的工作树运行）、`.claude/skills/exp-triad-flow/references/server.md`
- `reports/0024-e1-acceptance-v3-ext.md` 的 §0、§4
- `DECISIONS.md` 的 D36（第 1 条）、D38、D43

本任务只运行，不改任何仓库代码。探针脚本放运行目录的 `probe_scripts/`，只调用现有接口。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含本任务书所在的 commit。不包含就停下报告。

## 2. 环境

同 RUN-0024：
- `VISINT_DATA_ROOT=$PROJ/data`
- `TMPDIR=/dev/shm/visint/tmp`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

模型只用 qwen3.5-2b。

## 3. 运行

探针 `probe_scripts/rerun_batch.py`：
- 用 `src/e1` 现有的函数取 refexp_box 开发子集，批划分与 RUN-0024 的 A1 相同（冻结配置 64）；
- 只取出第 4 批这 64 条；
- 用 `adapter.generate_batch` 按 refexp_box 的提示与生成上限重复生成；
- 记录每遍每条的 raw_text 与生成的 token id，以及与第 1 遍不同的条数和位置。

四种设置，每种都在**新开的进程**里跑：

| 设置 | 内容 | 遍数 |
|---|---|---|
| S1 | 与 RUN-0024 相同的环境 | 同一进程里重复 20 遍 |
| S2 | 与 S1 相同 | 再开 3 个新进程，各跑 3 遍。看跨进程是否不同，例如 Triton autotune 每个进程选的配置不同 |
| S3 | S1 加 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并在探针里 `torch.use_deterministic_algorithms(True, warn_only=True)` | 同一进程里重复 20 遍，记下 warn 提到的算子 |
| S4 | S3 再加 fla 的确定性开关。先查 fla 0.5.2 的源码：GDN 前向用到的 Triton 内核有没有 autotune 或 `atomic_add`，有没有关掉 autotune、固定配置的环境变量，照实记下 | 同一进程里重复 20 遍 |

- 另外在 S1、S3 下各跑一遍完整的 refexp_box 开发子集（600 条，`run_e1.py --acceptance --task refexp_box` 只看 A1 与 A2），比较不同的条数，并记下每种设置的 generate 墙钟。
- 有不同时，对第 1 个不同的位置，记下两遍在这一步的前两名 token 与 logit 差。用 `option_logits` 不行的话，就用 `generate(output_logits=True)`。

## 4. 时间与停止

- 预计 1–1.5 GPU 小时。累计超过 3 小时就停，推回已完成的部分。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E1/0025_determinism_probe/`：探针输出、`probe_scripts/`、`logs/`、`README.md`。
- `reports/0025-determinism-probe.md`，按 report 模板，第 3 节原样贴：
  - **每种设置**：遍数、与第 1 遍不同的条数（每遍一列）、不同的位置。
  - **跨进程（S2）**：各进程之间的比较。
  - **完整开发子集**：S1、S3 下 A1 与 A2 不同的条数，以及墙钟。
  - **第 1 个不同位置**：前两名 token 与 logit 差。
  - **fla 源码的事实**：autotune、atomic、环境变量，引出文件与行号。
  - **warn_only 列出的非确定性算子**。
- 只贴数字与事实，不做解读。
- 一次推回：

```bash
git add <逐个路径> && git commit -m "[results] 0025: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0025-determinism-probe.md` 追加进完成清单，重新进入值守。
