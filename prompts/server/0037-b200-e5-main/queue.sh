#!/usr/bin/env bash
# RUN-0037 的训练队列（B200）：E5 主比较（2B；E5 预注册 §5、§8；D77-3、D78；N30 待批准，顺序与范围按批准结果改）。
# 起点都是 RUN-0036 的最终检查点（$PROJ/ckpt/0036）：base_warmup = 预热，e3b = E3b，e3a = E3a full。配种子：E5 的种子 i 用起点的种子 i。
# 顺序：1K、4K 的三个种子（H5 的主信号）→ 16K 的三个种子 → 64K 的种子 0 → a–d 迁移（4K，E3b 与预热）→ 诊断起点 base（1K、4K，种子 0）。
# 驱动的规则见 scripts/server/train_queue.sh；RUN-0036 的队列跑完才开始（wait_for_queue 0036）。
#
# 用法（B200）：setsid nohup bash /dev/shm/visint/tmp/wt0037/prompts/server/0037-b200-e5-main/queue.sh \
#   > $PROJ/logs/train_0037.driver.log 2>&1 < /dev/null &
# 彩排（开发机）：TINY=1 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… START_ROOT=<0036 彩排的 CK_FINAL> bash queue.sh
RUN_ID=0037
M=qwen3.5-2b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}" "${START_ROOT:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  M=qwen3.5-0.8b
  SMALL=(10); MID=(20); BIG=(); SEEDS=(${SEEDS:-0}); TIER_AD=10
else
  export ROOT=/home/ubuntu/workspace/wzg/gjl
  export PROJ=$ROOT/vision-internalization
  WT=/dev/shm/visint/tmp/wt0037
  REPO=$PROJ/repo
  LOGS=$PROJ/logs
  OUT=$REPO/experiments/E5/0037_e5_main
  CK_TMP=/dev/shm/visint/ckpt_0037
  CK_FINAL=$PROJ/ckpt/0037
  START_ROOT=$PROJ/ckpt/0036
  PY=$PROJ/venv/bin/python
  CHUNK=1000
  export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  EXTRA=(); DATA=(--data-root "$PROJ/data")  # 起点检查点的合并核对用 E1 的 grid16 开发子集（根盘上的 E1 落地）
  SMALL=(1000 4000); MID=(16000); BIG=(64000); SEEDS=(0 1 2); TIER_AD=4000
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"

start_ckpt() {  # $1 = 起点，$2 = 种子 → RUN-0036 的最终检查点（model/；驱动按训练叶目录的结构存放，旁边有叶目录的 config.yaml）
  case "$1" in
    base_warmup) echo "$START_ROOT/warmup/$M/seed$2/ckpt/final/model" ;;
    e3b)         echo "$START_ROOT/e3b/$M/seed$2/ckpt/final/model" ;;
    e3a)         echo "$START_ROOT/e3a/full/$M/seed$2/ckpt/final/model" ;;
  esac
}
e5() {  # $1 = 起点，$2 = 数据量档，$3 = 种子，其余 = 额外参数（--tasks …）
  local init=$1 n=$2 s=$3; shift 3
  local tag="${init}_n${n}_s${s}" ck
  [ $# -gt 0 ] && tag="${tag}_$(echo "$*" | sed 's/--tasks //; s/ //g')"
  if [ "$init" = base ]; then
    run "$tag" scripts/train_e5.py --init base --model $M --n-train "$n" --seed "$s" "$@"
  else
    ck=$(start_ckpt "$init" "$s")
    [ -d "$ck" ] || { note "$tag：起点检查点不在（$ck），跳过"; echo "$(now) $tag missing-start $ck" >> "$DRV/failed.txt"; return 1; }
    run "$tag" scripts/train_e5.py --init "$init" --start-checkpoint "$ck" --model $M --n-train "$n" --seed "$s" \
      ${DATA[@]+"${DATA[@]}"} "$@"
  fi
}

[ "$TINY" = 1 ] || wait_for_queue 0036
note "队列开始：commit $(git -C "$WT" rev-parse HEAD)，起点根 $START_ROOT，每段 $CHUNK 步"
for n in "${SMALL[@]}"; do for s in "${SEEDS[@]}"; do for init in base_warmup e3b e3a; do e5 "$init" "$n" "$s"; done; done; done
for n in ${MID[@]+"${MID[@]}"}; do for s in "${SEEDS[@]}"; do for init in base_warmup e3b e3a; do e5 "$init" "$n" "$s"; done; done; done
for n in ${BIG[@]+"${BIG[@]}"}; do for init in base_warmup e3b e3a; do e5 "$init" "$n" 0; done; done
for s in "${SEEDS[@]}"; do for init in e3b base_warmup; do e5 "$init" "$TIER_AD" "$s" --tasks a b c d; done; done
for n in "${SMALL[@]}"; do e5 base "$n" 0; done
finish
