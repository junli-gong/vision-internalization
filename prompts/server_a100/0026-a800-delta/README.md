# RUN-0026：A800 增量包（数据落地、去重、训练接口核对、E2 算力估算、E3b 训练吞吐）

**给协助运行的合作者。** 谢谢你跑完 RUN-0022：A800 就绪了，8 个模型的权重、E5 生成器、冒烟都与另一台机器一致。

这个包**沿用 RUN-0022 的工作目录**（例如 `/root/autodl-tmp/visint-0022`）：venv 与权重都不重装、不重下，只换代码快照，然后做下面几件事：
- 落地 E1 的 RefCOCO 与 NYUv2，以及 E2/E3 的四个训练数据源；
- 训练数据与评测数据去重，生成剔除清单；
- 训练接口核对：上次因为 RefCOCO 没落地，跳过了；
- E2 探针的算力估算；
- E3b 训练吞吐的短实测：每种 20 步，只测速度与显存，不是正式训练。

**不需要改任何代码。** 遇到报错不要自己修代码，原样记下来；环境与网络的问题可以照上次的办法处理，在 notes.txt 里写明。

## 0. 包里有什么

| 文件 | 作用 |
|---|---|
| `vision-internalization-<commit>.tar.gz` | 新的代码快照，完整 commit 见 `VERSION.txt` |
| `step1_update_and_land.sh` | 第 1 步（需联网）：换代码、核对 refer zip、落地数据 |
| `step2_checks.sh` | 第 2 步（GPU，离线）：去重、训练接口核对、E2 估算、E3b 吞吐 |
| `step3_collect.sh` | 第 3 步：打包结果，不含检查点与数据 |

## 1. 准备

- **refer zip**：web.archive.org 在 A800 上连不上。Drive 的同一个文件夹里有 `refer_zips/`，内含 `refcoco.zip`、`refcoco+.zip`、`refcocog.zip`，共约 157 MB。请放到 `<工作目录>/refer_zips/`。
  - 第 1 步会先逐个核对完整 sha256、字节数与成员 CRC，不符就不用。
  - 这三个文件只供本项目内部使用，不再转发。
- **磁盘**：比 RUN-0022 再多约 15 GB，数据合计约 13 GB 加临时文件。上次剩 169 GiB，够用。
- **网络**：
  - 第 1 步默认 `HF_ENDPOINT=https://hf-mirror.com`；
  - images.cocodataset.org、ADE20K 官方站、horatio.cs.nyu.edu 要能访问；
  - Google Drive（法向数据）不通的话那一项会失败，照实回传即可。

## 2. 步骤

```bash
W=/root/autodl-tmp/visint-0022          # RUN-0022 的工作目录
bash step1_update_and_land.sh $W        # 约 1–2 小时，下载约 12 GB；中断了重跑即可，已落好的会跳过
bash step2_checks.sh $W                 # 约 2–3 小时，建议放进 tmux 或 nohup
bash step3_collect.sh $W
```

每一步的退出码都记在 `<工作目录>/results_0026/steps.log`，某一项失败不影响后面。

## 3. 预期结果（供核对，不需要你判断对错）

- **`refer_zip_check.txt`**：三行都是"= 钉住的值"。
- **`land_logs/`**：每个源一个日志。refcoco、nyuv2、ade20k、nyu_depth_train、refcoco_train 应该成功；nyu_normals 视 Google Drive 能否访问而定。
- **`dedupe/dedupe_report.md`**：训练集与评测集之间的重合统计。剔除清单留在 `<工作目录>/data/dedupe/`，以后训练要用，**请不要删 `<工作目录>/data`**。
- **`train_iface_<模型>.json`**：`pass` 为 true 或 false 都照实传回。
- **`e2_estimate/`**：8 个模型的 E2 算力估算。
- **`e3b_probe/`**：4 种组合（0.8B、9B × 网格、文本）各 20 步的每步耗时与显存。

## 4. 回传

`step3_collect.sh` 生成的 `a800_results_0026_<时间>.tar.gz`，加上一份 notes.txt（写明报错与处理、是否用了镜像），传到 Google Drive 的 `vision-internalization-a100/`。

## 5. 规则

- 不修改代码、配置与测试。
- 模型与数据只用于本项目的非商业研究，不再转发给第三方。
- 只在工作目录里读写，不改机器上的全局配置。
- HF 凭据：权重都已在本地缓存，这次用不到 token。
