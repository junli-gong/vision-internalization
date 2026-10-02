#!/usr/bin/env bash
# RUN-0022 第 1 步：解开代码快照、建 Python 3.12 环境、装依赖。需要联网。
# 用法：bash setup_env.sh <工作目录>        工作目录放在大盘上，例如 /root/autodl-tmp/visint
# 可选环境变量：
#   PYTHON=python3.12      用哪个 Python 建 venv（必须是 3.12）
#   TORCH_CUDA=cu130       强制 torch 的 CUDA 轮子（cu126 / cu128 / cu130）；不给就按驱动版本自动选
#   PIP_INDEX_URL=...      PyPI 慢时可以指向国内镜像（torch 与 torchvision 仍从 download.pytorch.org 装）
set -euo pipefail
WORK=${1:?用法: bash setup_env.sh <工作目录（放在大盘上）>}
PKG=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK"
cd "$WORK"
echo "[setup] 解开代码快照"
rm -rf "$WORK/vision-internalization"
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
# torch 与 torchvision 必须配套，从同一个 CUDA 索引装（RUN-0014 漏了 torchvision）
pip install "torch==2.13.0" "torchvision==0.28.0" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
# requirements.txt 已含 peft、accelerate（RUN-0014 漏了 accelerate）
pip install -r "$WORK/vision-internalization/requirements.txt"
pip install "flash-linear-attention==0.5.2" "fla-core==0.5.2"

python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print("device", i, p.name, "capability", (p.major, p.minor), "memory_GB", round(p.total_memory / 1e9, 1))
PY
pip freeze > "$WORK/pip_freeze.txt"
pip check > "$WORK/pip_check.txt" 2>&1 || true
# 与 B200 的版本对照（只记录，不判；关键包应当相同）
python - "$PKG/reference_b200_pip_freeze.txt" "$WORK/pip_freeze.txt" > "$WORK/pip_vs_b200.txt" <<'PY'
import sys
def load(p):
    out = {}
    for line in open(p):
        line = line.strip()
        if "==" in line:
            k, v = line.split("==", 1); out[k.lower().replace("_", "-")] = v
    return out
ref, mine = load(sys.argv[1]), load(sys.argv[2])
key = ["torch", "torchvision", "transformers", "peft", "accelerate", "flash-linear-attention", "fla-core",
       "numpy", "pymunk", "safetensors", "tokenizers", "huggingface-hub", "pillow", "scipy", "shapely"]
for k in key:
    print(f"{k:24s} b200={ref.get(k)}  here={mine.get(k)}  {'same' if ref.get(k) == mine.get(k) else 'DIFFERENT'}")
PY
cat "$WORK/pip_vs_b200.txt"
echo "[setup] 完成。下一步：bash prefetch.sh $WORK"
