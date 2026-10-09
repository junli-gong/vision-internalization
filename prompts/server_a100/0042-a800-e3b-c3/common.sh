#!/usr/bin/env bash
# RUN-0042 各步共用（由 RUN-0035 的 common.sh 改来，只换了运行编号与目录）：环境变量、记步函数、训练命令的包装。由各步 source，不单独运行。
# TINY=1：开发机彩排（小型随机模型、CPU、仓库里的 fixture）；管理员打包前用它把每条命令空跑一遍。
set -uo pipefail
WORK=${1:?用法: bash <步骤脚本> <工作目录，例如 /root/autodl-tmp/visint-0022>}
TINY=${TINY:-0}
OUT="$WORK/results_0042"; mkdir -p "$OUT" "$WORK/tmp"
CKPT="$WORK/e3_ckpt_0042"; mkdir -p "$CKPT"
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
SEEDS=(0 1 2)

# 一个训练运行：$1 = 运行名（日志用），其余 = 传给训练脚本的参数（不含 --data-root/--tiny/--resume/--ckpt-root）
train() {
  local name=$1; shift
  progress "running $name"
  python "$@" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} ${TFLAG[@]+"${TFLAG[@]}"} --resume >> "$OUT/logs/$name.log" 2>&1
  local rc=$?; step "$name" $rc; return $rc
}
mkdir -p "$OUT/logs"
