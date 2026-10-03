#!/usr/bin/env bash
# RUN-0029 各步共用：环境变量、记步函数。由各步 source，不单独运行。
# TINY=1 时是开发机彩排：小型随机模型、CPU、仓库里的 fixture，不需要 GPU、数据与权重（管理员打包前用它把每条命令空跑一遍）。
set -uo pipefail
WORK=${1:?用法: bash <步骤脚本> <工作目录，例如 /root/autodl-tmp/visint-0022>}
TINY=${TINY:-0}
OUT="$WORK/results_0029"; mkdir -p "$OUT" "$WORK/tmp"
step() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1 rc=$2" | tee -a "$OUT/steps.log"; }
progress() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" > "$OUT/progress.txt"; }
if [ "$TINY" = 1 ]; then
  CODE=${CODE:?TINY=1 时要给 CODE=<仓库根>}
  TFLAG=(--tiny); DATA_ARGS=()
else
  CODE="$WORK/vision-internalization"
  # shellcheck disable=SC1091
  source "$WORK/venv/bin/activate"
  export HF_HOME=${HF_HOME:-$WORK/hf} HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  NCPU=$(nproc 2>/dev/null || echo 8); [ "$NCPU" -gt 16 ] && NCPU=16; export OMP_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
  TFLAG=(); DATA_ARGS=(--data-root "$WORK/data")
fi
export TMPDIR=$WORK/tmp PIP_NO_CACHE_DIR=1
QWEN=(qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b)
REFS=(--reference post_proj pre_proj)
CAP=${CAP:-32}   # 批准的 GPU 小时上限（N23）；项目负责人批的数不同时由管理员改这里
