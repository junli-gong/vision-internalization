#!/usr/bin/env bash
# RUN-0014 第 1 步：解开代码快照、建 Python 环境、装依赖。在登录节点上跑（需要联网）。
# 用法：bash setup_env.sh <工作目录>        工作目录放在大盘上，例如 /scratch/$USER/visint
# 可选环境变量：
#   PYTHON=python3.12      用哪个 Python 建 venv（必须是 3.12）
#   TORCH_CUDA=cu126       强制 torch 的 CUDA 轮子（cu126 / cu128 / cu130）；不给就按驱动版本自动选
set -euo pipefail
WORK=${1:?用法: bash setup_env.sh <工作目录（放在大盘上）>}
PKG=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK"
cd "$WORK"
echo "[setup] 解开代码快照"
tar -xzf "$PKG"/vision-internalization-*.tar.gz
cp "$PKG/VERSION.txt" "$WORK/vision-internalization/PACKAGE_VERSION.txt"

PY=${PYTHON:-python3.12}
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || { echo "需要 Python 3.12：用 PYTHON=<路径> 指定，或先 conda create -n visint python=3.12 再把 PYTHON 指过去"; exit 1; }
"$PY" -m venv "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
export PIP_NO_CACHE_DIR=1
pip install --upgrade pip

if [ -z "${TORCH_CUDA:-}" ]; then
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "$drv" -ge 580 ]; then TORCH_CUDA=cu130; elif [ "$drv" -ge 570 ]; then TORCH_CUDA=cu128; else TORCH_CUDA=cu126; fi
  echo "[setup] 驱动主版本 $drv → torch 轮子 $TORCH_CUDA"
fi
pip install "torch==2.13.0" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
pip install -r "$WORK/vision-internalization/requirements.txt"
pip install "flash-linear-attention==0.5.2" "fla-core==0.5.2"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY
pip freeze > "$WORK/pip_freeze.txt"
echo "[setup] 完成。下一步：bash prefetch.sh $WORK"
