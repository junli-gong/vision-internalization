#!/usr/bin/env bash
# RUN-0035 各步共用：环境变量、记步函数、训练命令的包装。由各步 source，不单独运行。
# TINY=1：开发机彩排（小型随机模型、CPU、仓库里的 fixture）；管理员打包前用它把每条命令空跑一遍。
set -uo pipefail
WORK=${1:?用法: bash <步骤脚本> <工作目录，例如 /root/autodl-tmp/visint-0022>}
TINY=${TINY:-0}
OUT="$WORK/results_0035"; mkdir -p "$OUT" "$WORK/tmp"
CKPT="$WORK/e3_ckpt_0035"; mkdir -p "$CKPT"
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

# 一个尺寸 × 一个种子的全部变体（E3 预注册 v1 §2、§9；D77）。$1 = 模型键，$2 = 种子，$3 = 1 时另跑同预算对照（只 0.8B）
size_seed() {
  local m=$1 s=$2 twotask=${3:-0}
  local R="$OUT/0035_e3_train" C="$CKPT/train"
  train "e3a_full_${m}_s${s}"  scripts/train_e3a.py --model "$m" --variant full --seed "$s" --out "$R" --ckpt-root "$C"
  train "e3a_c1_${m}_s${s}"    scripts/train_e3a.py --model "$m" --variant c1   --seed "$s" --out "$R" --ckpt-root "$C"
  train "e3a_c2_${m}_s${s}"    scripts/train_e3a.py --model "$m" --variant c2   --seed "$s" --out "$R" --ckpt-root "$C"
  train "e3b_${m}_s${s}"       scripts/train_e3b.py --model "$m" --supervision grid --seed "$s" --out "$R" --ckpt-root "$C"
  train "c1prime_${m}_s${s}"   scripts/train_e3b.py --model "$m" --supervision grid --variant c1prime --seed "$s" --out "$R" --ckpt-root "$C"
  train "c3_${m}_s${s}"        scripts/train_e3b.py --model "$m" --supervision text --seed "$s" --out "$R" --ckpt-root "$C"
  if [ "$twotask" = 1 ]; then  # 同预算对照：E3a 只用指代分割与深度（与 E3b 同一组数据、同一日程）
    train "e3a2t_full_${m}_s${s}" scripts/train_e3a.py --model "$m" --variant full --tasks refseg depth --seed "$s" \
      --out "$OUT/0035_e3_train_e3a2task" --ckpt-root "$CKPT/e3a2task"
  fi
}
