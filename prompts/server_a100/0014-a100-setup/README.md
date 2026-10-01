# RUN-0014：A100 环境搭建与就绪检查

**给协助运行的合作者。** 本包是一个研究项目（让大语言模型内化视觉能力）在 A100 服务器上的第一个任务，目的是：
- 把环境搭好；
- 确认代码在这台机器上行为正确；
- 测几个基本数字。

以后需要大盘的实验（训练数据和检查点较大），会以同样的方式打成新包交给你。

**不需要改任何代码。** 照下面的步骤跑，把结果压缩包传回同一个 Google Drive 文件夹即可。遇到报错不要自己修，原样记下来一起传回。

## 0. 包里有什么

| 文件 | 作用 |
|---|---|
| `vision-internalization-<commit>.tar.gz` | 代码快照（某个固定 commit），`VERSION.txt` 里有完整 commit 号 |
| `setup_env.sh` | 第 1 步：解开代码、建 Python 3.12 环境、装依赖（登录节点，需联网） |
| `prefetch.sh` | 第 2 步：按固定版本下载模型权重与 1 张测试图（登录节点，需联网） |
| `run_all.sh` | 第 3 步：全部检查（GPU 节点，不需联网） |
| `slurm_example.sbatch` | 用 SLURM 提交第 3 步的示例 |
| `collect_results.sh` | 第 4 步：把结果打成一个压缩包 |

## 1. 需要准备

- **一块 A100**（40 GB 或 80 GB 都可以）。
- **大盘上约 40 GB 空间**：模型权重约 17 GB，环境约 8 GB，其余是代码与结果。
- **Python 3.12**：没有的话用 conda 建一个，再用 `PYTHON=<路径>` 指给脚本。
- **Hugging Face 账号**：Gemma 4 是受限模型，要先在 <https://huggingface.co/google/gemma-4-E2B-it> 接受条款，再 `hf auth login`。拿不到的话，第 2、3 步都加 `SKIP_GEMMA=1`，其余照跑，回传时说明一句。
- **网络**：第 1、2 步要能访问 PyPI、download.pytorch.org、huggingface.co、images.cocodataset.org。GPU 节点不能联网也没关系，第 3 步离线跑。

## 2. 步骤

下面的 `<工作目录>` 是大盘上的一个空目录，例如 `/scratch/$USER/visint`。四步都用同一个。

```bash
# 第 1 步（登录节点）：约 15–30 分钟
bash setup_env.sh <工作目录>

# 第 2 步（登录节点）：约 10–20 分钟，下载约 17 GB
bash prefetch.sh <工作目录>

# 第 3 步（GPU 节点）：约 30–60 分钟
#   集群：改好 slurm_example.sbatch 里的两个路径后 sbatch slurm_example.sbatch
#   直接有 GPU 的机器：
bash run_all.sh <工作目录>

# 第 4 步：打包结果
bash collect_results.sh <工作目录>
```

- torch 的 CUDA 版本按驱动自动选。驱动太旧时 setup 会失败，把报错传回来。也可以用 `TORCH_CUDA=cu126` 之类的变量强制指定。
- 第 3 步的每一步都会把退出码写进 `results_0014/steps.log`，某一步失败不影响后面的步骤。

## 3. 预期结果（供核对，不需要你判断对错）

- **`pytest.txt`**：约 1,400 条测试，绝大多数 passed，另有约 10 条 skipped（缺 cv2 或 nltk 数据，属正常）。
  - 如果 `tests/test_synth_repro.py` 里 physics 的"规范平台"比对失败，照实传回：那正是要核对的跨机器差异，不是环境问题。
- **三个 `smoke_tiny_*.json`**：`all_ok` 为 true。
- **三个真实权重冒烟 `smoke_<模型>.json`**：`all_ok` 为 true。其中检查 `2_task_modulation` 的 `image_first.equal_layers`，在另一台机器上是：
  - qwen3.5-0.8b：25/25；
  - gemma-4-e2b-it：36/36；
  - internvl3.5-2b-hf：29/29。
- **`vb_*.json`**：批量生成与逐条生成的一致率和吞吐。只是测量，数值高低都正常。

## 4. 回传

- 第 4 步生成的 `a100_results_0014_<时间>.tar.gz`，上传到 Google Drive 里放本包的同一个文件夹（`vision-internalization-a100/`）。
- 另写一个 `notes.txt`，一起传：
  - 机器名称与 GPU 型号；
  - 哪一步报错、你怎么处理的；
  - 是否加了 `SKIP_GEMMA=1`。

## 5. 规则

- 不修改代码、配置与测试。脚本跑不通就停下，把报错原样传回。
- 模型与数据只用于本项目的非商业研究，不再转发给第三方。
- 只在 `<工作目录>` 里读写，不改机器上的全局配置。
