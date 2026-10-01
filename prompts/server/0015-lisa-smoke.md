[RUN-TASK 0015] LISA-7B 外部对照：在 B200 上用独立 venv 做冒烟（D36-5）

## 为什么做

- E5 要一个 LISA 式的外部对照（简报 §8.8）。项目负责人选定原版 LISA-7B（`xinlai/LISA-7B-v1`），见 D36-5。
- 原版环境是旧版 torch 与 CUDA，在 B200（sm_100）上大概率跑不起来。所以先试一个独立 venv：
  - 继承基础环境的新版 torch；
  - 只把 transformers 等依赖钉在 LISA 要求的旧版本上；
  - 做一次冒烟：加载权重，跑通一张图的分割。
- 不顺就停下报告，管理员会把 LISA 整个打包给 A100，用原版环境跑。
- **隔离规则**（简报 §4 第 1 条）：
  - LISA 只作外部对照，单独运行；
  - 它的输出绝不能进入我们的训练数据、标签或奖励；
  - 所有产物只放在 `$PROJ/ext/lisa/` 与本任务的运行目录，不放进 `$PROJ/data`。

## 0. 先读

- `CLAUDE.md`（磁盘与内存盘规则）、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D34、D36（第 5、7 条）
- LISA 官方仓库 `dvlab-research/LISA` 的 README 与 `requirements.txt`

本任务不改我们仓库的任何代码。允许在 `$PROJ/ext/lisa/` 下 clone LISA 仓库、建独立 venv、写探针脚本。探针脚本的副本放进运行目录。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含本任务书所在的 commit。

## 2. 准备

1. **记账**：开始前记 `df /`、内存盘占用与 `$PROJ` 占用。本项目根盘合计不得超过 15 GB（D28-8）。
2. **代码**：clone `https://github.com/dvlab-research/LISA` 到 `$PROJ/ext/lisa/LISA`，记下 commit。
3. **环境**：
   - 建独立 venv：`python -m venv --system-site-packages $PROJ/ext/lisa/venv`，继承基础环境的 torch 2.13，不重装 torch。
   - 按 LISA 的 requirements 安装其余依赖，但跳过 torch、torchvision、deepspeed、flash-attn、bitsandbytes 这类与 CUDA 绑定或推理用不到的包，并把跳过的写进 report。
   - transformers 等钉在 LISA 要求的版本。
   - 一律 `PIP_NO_CACHE_DIR=1`。安装前后记 venv 的体积。
4. **权重**：先清空内存盘上本项目的其他权重，再下载到内存盘（`HF_HUB_CACHE` 照旧）：
   - `xinlai/LISA-7B-v1` @ `43c754eef75871fcc11c84d3930a402f7b0a754f`（约 16.1 GB）；
   - CLIP 视觉塔 `openai/clip-vit-large-patch14` @ `32bd64288804d66eefd0ccbe215aa642df71cc41`，只下一种格式，约 1.7 GB，用 `allow_patterns`。
   - 内存盘合计不得超过 25 GB。

## 3. 冒烟

- 写一个非交互的探针脚本，按 LISA 官方 `chat.py` 的推理流程：
  - 加载模型（bf16）；
  - 对 `$PROJ/tmp/000000039769.jpg` 跑一次，提示为 `Can you segment the cat on the left in this image?`；
  - 输出 mask。
- 记录：
  - 是否成功；
  - 报错的完整 traceback（如果有）；
  - 加载时间与推理时间；
  - 显存峰值；
  - mask 的像素数与外接框。
- mask 与叠加图存进运行目录（只这一张，用于目检）。
- 失败时：
  - 先试一次官方 README 里对应的已知修法，例如某个依赖的版本，并记录；
  - 仍失败就停下，照实报告。不要改 LISA 的源码。

## 4. 收尾

- 删掉内存盘上的 LISA 与 CLIP 权重，记账。
- venv 与 LISA 代码保留在 `$PROJ/ext/lisa/`，并记下体积。

## 5. 产出与回传

- `experiments/P0/0015_lisa_smoke/`：
  - `pip_freeze_lisa.txt`、`install.log`
  - `probe_scripts/`、`smoke.json`、mask 与叠加图
  - `disk_ledger.tsv`、`README.md`
- `reports/0015-lisa-smoke.md`，按 report 模板，只贴数字与事实。

```bash
git add <逐个路径> && git commit -m "[results] 0015: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0015-lisa-smoke.md` 追加进完成清单，重新进入值守。
