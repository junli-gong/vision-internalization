# RUN-0022（第二版）：A800 环境搭建、就绪检查与训练数据落地

**给协助运行的合作者。** 上一个包（RUN-0014）在 A100 上跑通了，谢谢你补装了漏掉的依赖。这个包换到 A800 上：
- 修掉了上次的三处疏漏：torchvision、accelerate 和 README 文件；
- 多做几件事，为以后在这台机器上训练做准备：
  - 下齐 Qwen 与 InternVL 的全部权重，并逐文件核对；
  - 跑一遍训练接口的核对；
  - 用健康检查和另一台机器逐题对比；
  - 落地训练数据（约 10.5 GB），并与评测数据做去重。

上次的 A800 结果（重跑 RUN-0014 原包）已经收到：三个家族都通过了。这个包请在扩好的大盘上跑。

**不需要改任何代码。** 照下面的步骤跑，把结果压缩包和一份 notes.txt 传回同一个 Google Drive 文件夹。遇到报错不要自己修代码，原样记下来；环境类的问题（缺包、网络）可以像上次那样处理，在 notes.txt 里写明。

## 0. 包里有什么

| 文件 | 作用 |
|---|---|
| `vision-internalization-<commit>.tar.gz` | 代码快照（固定 commit，完整号见 `VERSION.txt`） |
| `setup_env.sh` | 第 1 步：解开代码、建 Python 3.12 环境、装依赖（需联网） |
| `prefetch.sh` | 第 2 步：下载权重并核对哈希、落地 RefCOCO 开发数据、下 1 张测试图（需联网） |
| `verify_weights.py` | `prefetch.sh` 调用：按另一台机器的清单逐文件核对权重 |
| `run_all.sh` | 第 3 步：全部检查（GPU，不需联网） |
| `collect_results.sh` | 第 4 步：把结果打成一个压缩包 |
| `reference_b200_pip_freeze.txt` | 另一台机器的包版本，`setup_env.sh` 用它做对照 |

## 1. 需要准备

- **一块 A800**，80 GB 最好；40 GB 也能跑完本包。
- **大盘上约 100 GB 空间**（50 GB 的盘放不下）：
  - 权重约 50 GB；带 Gemma 约 75 GB；
  - 环境约 8 GB；
  - RefCOCO 开发数据约 1.1 GB；
  - NYUv2 官方集约 1.3 GB，落地时临时多占约 3 GB；
  - E2/E3 训练数据约 10.5 GB；
  - 其余是代码与结果。
- **Python 3.12**：没有的话用 conda 建一个，再用 `PYTHON=<路径>` 指给脚本。
- **Gemma 4**：是受限模型。
  - 上次的 Gemma 权重如果还在，可以 `export HF_HOME=<上次的 hf 目录>` 复用，不用再下。
  - 要重新下，请用一个只读权限的 token。不要把有写权限的 token 发给第三方镜像。
  - 拿不到的话，第 2、3 步都加 `SKIP_GEMMA=1`，回传时说明。
- **网络**：第 1、2 步要能访问：
  - PyPI、download.pytorch.org；
  - huggingface.co：慢的话 `export HF_ENDPOINT=https://hf-mirror.com`，每个文件都会核对哈希；
  - images.cocodataset.org：COCO 图像，用 Range 只取需要的约 4 GB；
  - web.archive.org：RefCOCO 标注的存档；
  - horatio.cs.nyu.edu：NYUv2 官方集；
  - ADE20K 官方站，以及 Google Drive（法向数据）。

  第 3 步离线跑。

**连不上时怎么办**：每个数据源都是独立的，失败只记日志，不影响别的步骤。
- **web.archive.org 不通**：在能访问的电脑上下载下面 3 个文件，放进 `<工作目录>/refer_zips/`，再重跑第 2 步。已落好的会自动跳过。
  - `https://web.archive.org/web/20220413011718if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip`
  - `https://web.archive.org/web/20220413011656if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip`
  - `https://web.archive.org/web/20220413012904if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip`
- **Google Drive 不通**：法向数据（nyu_normals）会失败，照实回传即可，暂时用不到。
- **huggingface.co 不通**：`export HF_ENDPOINT=https://hf-mirror.com`。

## 2. 步骤

`<工作目录>` 是大盘上的一个空目录，例如 `/root/autodl-tmp/visint`。四步都用同一个。

```bash
# 第 1 步：约 15–30 分钟
bash setup_env.sh <工作目录>

# 第 2 步：约 1–3 小时，下载约 65 GB（取决于网速；中断了重跑即可，已下好的会跳过）
bash prefetch.sh <工作目录>

# 第 3 步：约 2.5–3.5 小时（大头是 CPU 上的全量测试，约 1 小时）
bash run_all.sh <工作目录>

# 第 4 步：打包结果
bash collect_results.sh <工作目录>
```

- 第 3 步的每一步都把退出码写进 `results_0022/steps.log`，某一步失败不影响后面的步骤。
- 第 3 步较长，建议放进 `tmux` 或 `nohup` 里跑。

## 3. 预期结果（供核对，不需要你判断对错）

- **`pip_vs_b200.txt`**：torch、torchvision、transformers、peft、accelerate、fla 等关键包与另一台机器相同。
- **`prefetch_logs/verify_weights.txt`**：最后一行是 `ALL_OK`。
- **`pytest.txt`**：约 1,700 条测试，绝大多数 passed，约 11 条 skipped。生成器的单核速度测试（`test_speed_*`）在较慢的 CPU 上可能不过，属正常，照实传回。
- **`synth_digests.txt`**：每一行都是 `60/60 equal`。
- **三个 `smoke_tiny_*.json`、三个 `smoke_<模型>.json`**：`all_ok` 为 true。
- **`health_vs_b200.txt`**：逐题和另一台机器比。相同或不同都正常，这正是要量的跨机器差异。
- **两个 `train_iface_<模型>.json`**：`pass` 为 true 或 false 都照实传回。如果 RefCOCO 没落地成功，这一步会跳过，`steps.log` 里写着 `train_iface_skipped_no_refcoco`。
- **`prefetch_logs/land_*.log`**：每个数据源的落地日志，失败的照实传回。
- **`dedupe/dedupe_report.md`**：训练数据与评测数据之间的重合统计。剔除清单 `train_exclusions.json` 留在 `<工作目录>/data/dedupe/`，以后训练要用，**不要删 `<工作目录>/data`**。

## 4. 回传

- 第 4 步生成的 `a800_results_0022_<时间>.tar.gz`，上传到 Google Drive 的 `vision-internalization-a100/`（放进你上次用的子文件夹也可以）。
- 另写一个 `notes.txt`，一起传：
  - 机器、GPU 型号与显存、磁盘；
  - 哪一步报错、你怎么处理的；
  - 是否加了 `SKIP_GEMMA=1`，是否用了 `HF_ENDPOINT` 镜像。

## 5. 规则

- 不修改代码、配置与测试。
- 模型与数据只用于本项目的非商业研究，不再转发给第三方。
- 只在 `<工作目录>` 里读写，不改机器上的全局配置。
