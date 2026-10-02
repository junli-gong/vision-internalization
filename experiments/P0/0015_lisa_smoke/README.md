# experiments/P0/0015_lisa_smoke — RUN-0015：LISA-7B-v1 外部对照，B200 独立 venv 冒烟

服务器 CC，b200，2026-10-02 00:12–00:25 UTC。运行报告：`reports/0015-lisa-smoke.md`。

## 隔离（简报 §4 第 1 条、D36-5）

- **LISA 只作外部对照。** 代码、venv 只在 `$PROJ/ext/lisa/` 下：LISA 仓库在 `LISA/`（commit `3cb2d43`），独立 venv 在 `venv/`。产物只在本目录。
- **不碰项目数据和代码：** 没有写 `$PROJ/data`，没有改我们仓库的代码，也没有改 LISA 源码（clone 下来的仓库 `git status` 为空）。
- **权重：**
  - LISA-7B-v1 @ `43c754e` 与 CLIP ViT-L/14 @ `32bd642`（只取 safetensors）下载到内存盘 `/dev/shm/visint/hf-hub`；
  - 冒烟后已用 `delete_revisions` 删除；
  - 下载前先清掉了内存盘上的 Qwen3.5-0.8B。

## 文件

| 路径 | 内容 |
|---|---|
| `smoke.json` | 第 3 次运行（唯一一次修法，成功）的结果：版本、分词器的特殊 token、加载与推理时间、显存峰值、输出文本、mask 的像素数与外接框 |
| `000000039769_mask_0.png`、`000000039769_masked_img_0.jpg` | 第 3 次运行的 mask（0/255）和叠加图。写法同 LISA `chat.py`：mask 区域按 0.5 混合红色 |
| `smoke_run1.json`、`logs/smoke_run1.log` | 第 1 次运行：按 chat.py 的流程，未加修法，失败，含 traceback |
| `smoke_run2_cuda_launch_blocking.json`、`logs/smoke_run2_cuda_launch_blocking.log` | 第 2 次运行：与第 1 次相同，只加了 `CUDA_LAUNCH_BLOCKING=1`，用来取得准确的报错位置，属于诊断，不是修法 |
| `logs/smoke_run3_fix.log` | 第 3 次运行的日志 |
| `diag_tokens.txt` | CPU 上的分词器诊断，未加修法与加了修法各一份：分词器长度、特殊 token 的 id、提示的 id 范围 |
| `install.log`、`pip_freeze_lisa.txt`、`pip_freeze_lisa_before.txt` | 依赖安装的全部输出，以及安装后、安装前的 `pip freeze`（前者继承自基础环境） |
| `logs/binary_availability.txt` | 安装前逐个核对 LISA 钉住的版本在 Python 3.12 下有没有二进制包（`pip install --dry-run --only-binary :all: --no-deps`） |
| `hf_repo_sizes.txt` | 下载前用 HF API 读取的两个仓库的文件清单与体积（D36-5：先核对体积） |
| `weights_download.json` | 下载并逐文件核对字节数与哈希：LFS 文件比 sha256，小文件比 git blob sha1 |
| `lisa_commit.txt` | LISA 仓库的 commit，以及 clone 里被改动的文件数（0） |
| `disk_ledger.tsv`、`logs/driver.log`、`logs/download.log`、`logs/ext_sizes.txt` | 磁盘收支、删除与释放量、`$PROJ/ext/lisa` 的体积 |
| `probe_scripts/` | `p18_download.py`（下载与核对，用项目 venv）、`p19_lisa_smoke.py`（冒烟，用 LISA venv）、`run_smoke.sh`（第 1、2 次）、`run_smoke_fix.sh`（第 3 次，加 `--tokenizer-special-str`）、`diskcheck15.sh` |

## 复现

1. 按 `install.log` 建 venv：`$ROOT/env/bin/python -m venv --system-site-packages $PROJ/ext/lisa/venv`，然后按日志里的三步安装。
2. 用项目 venv 运行 `p18_download.py`，把两份权重下载到内存盘。
3. 运行 `bash probe_scripts/run_smoke_fix.sh`。

`p19_lisa_smoke.py` 与 LISA `chat.py` 的差别，见 `reports/0015-lisa-smoke.md` §4。
