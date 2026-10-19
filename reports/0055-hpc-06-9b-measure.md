# RUN-0055 报告：hpcgpu06 上 9B 的实测（只贴耗时、显存与事实，不贴指标读数）

任务书 `prompts/server_hpc/0055-hpc-06-9b-measure.md`（origin 23f5ad20；脚本 commit e496ea0b）。

## 第一部分：下载并核对 9B 权重

### 工作树

- 10-19 08:39:46Z squeue：157296、157297 都是 RUNNING。
- `wt0055`：fetch rc=0；`worktree add --no-checkout --detach … e496ea0b9bd368b29240cdb153b1312cfbf984ac` rc=0；`sparse-checkout set` rc=0；`read-tree -mu HEAD` rc=0。HEAD `e496ea0b9bd368b29240cdb153b1312cfbf984ac`，status 空，`experiments/` 只有 P0。
- sha256，7 个都与任务书相同：

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0055-hpc-06-9b-measure/fetch_9b.sh` | `8ef9a46511010e4989f1b6c61200c5606f963a9c802bad5834d23df472ed303b` |
| `prompts/server_hpc/0055-hpc-06-9b-measure/measure_9b.sh` | `d680172ee26d606b472fe88b810da430c753652b5e9c7bf7997cde58868a0fc1` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |
| `configs/e3b_train.json` | `b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375` |
| `src/data/train_loaders.py` | `76535736def11a13a19a94a930ead215684ee4a697db3c05f6fd8c35d54d10cb` |
| `experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py` | `0fcd224eb3b0025e7d34045a5243af8acd8eafc14daef98aa7007f3452504550` |
| `experiments/P0/0009_ifeval_rescore/weights_manifest.json` | `8535b8cb89e6def1f8a79feff669b315ca0578c75d4b7ce310d4ff8aa86efa46` |

- 启动前：`$VI/hf/hub/` 下没有 `models--Qwen--Qwen3.5-9B`，没有 `fetch_0055*` 日志。

### 下载

- 08:40:34Z 按任务书命令在 06 上启动 fetch_9b.sh，step 157296.163；之后 squeue 两条 RUNNING，big_s0 的 step 157296.162 照常。
- 心跳：`2026-10-19T08:40:42Z running download qwen3.5-9b (attempt 1)` → `2026-10-19T09:05:09Z done fetch_0055 (attempt 1)`。没有重启。
- 日志里核对那一行原文：`qwen3.5-9b @ c202236235762e1c871ad0ccb60c8ee5ba337b9a: 16/16 files verified (/home/mail/chenyuma/visint/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a)`
- `blobs/` 下 16 个文件，`ls -l` 字节数合计 19,329,393,661 B（19.33 GB）；最大四个：5,368,717,440、5,335,161,512、5,276,436,216、3,325,995,712 B。没有剩下的 `.incomplete`。
- 用时 08:40:42Z – 09:05:09Z，1,467 s；平均 13.2 MB/s（合计字节 / 用时）。
- 我这边每 10 分钟的速度采样脚本有引号错误，取到的字节数一直是 0，没有起到监测作用；下载在第一个采样周期内正常完成，没有因此误停或漏停。
- 09:13:50Z squeue：两条 RUNNING；step 157296.163 已不在；hpcmgt 上没有 fetch_9b 进程。

## 第二部分：实测

（等 06 跑完 E5 的 big_s0。）
