#!/bin/bash
# RUN-0044 §2 (v2: python=3.12.13 like B200; pytorch index + tsinghua extra index so constraints resolve): build $VI/env (py3.12) with B200 versions pinned. Runs inside srun --overlap on hpcgpu06.
set -uo pipefail
export VI=/home/mail/chenyuma/visint
export CONDA_PKGS_DIRS=$VI/conda-pkgs
export TMPDIR=$VI/tmp
export PIP_NO_CACHE_DIR=1
export PIP_CACHE_DIR=$VI/tmp/pip
export XDG_CACHE_HOME=$VI/cache/hpcgpu06/xdg
P=$VI/probe/0044
C=/home/mail/chenyuma/soft/miniconda3/bin/conda
echo "== $(date -u +%FT%TZ) host=$(hostname -s) nproc=$(nproc) CVD=${CUDA_VISIBLE_DEVICES:-}"
t0=$(date +%s)
rm -rf $VI/env; $C create -y -p $VI/env --solver classic python=3.12.13 pip 2>&1 | grep -v libmamba | tail -25; echo "conda create rc=${PIPESTATUS[0]} $(( $(date +%s)-t0 ))s"
PY=$VI/env/bin/python; $PY -V
t0=$(date +%s)
$PY -m pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple torch==2.13.0+cu130 torchvision==0.28.0+cu130 -c $P/b200_constraints.txt 2>&1 | tail -15; echo "torch rc=${PIPESTATUS[0]} $(( $(date +%s)-t0 ))s"
t0=$(date +%s)
$PY -m pip install -c $P/b200_constraints.txt -r $P/requirements.txt flash-linear-attention==0.5.2 fla-core==0.5.2 einops 2>&1 | tail -25; echo "reqs rc=${PIPESTATUS[0]} $(( $(date +%s)-t0 ))s"
$PY -m pip freeze > $P/hpc_pip_freeze.txt
$PY -m pip check 2>&1 | tail -5
$PY - <<'PY'
import json, torch
b = torch.backends
conv = getattr(b.cudnn, "conv", None)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "arch_list": torch.cuda.get_arch_list(),
 "cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 "cap": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
 "cudnn.allow_tf32": b.cudnn.allow_tf32, "cudnn.deterministic": b.cudnn.deterministic, "cudnn.benchmark": b.cudnn.benchmark,
 "cudnn.conv.fp32_precision": getattr(conv, "fp32_precision", None), "cuda.matmul.fp32_precision": getattr(b.cuda.matmul, "fp32_precision", None)}, indent=1))
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16); print("bf16 matmul ok", float((x @ x).float().abs().mean()))
PY
echo "== done $(date -u +%FT%TZ)"
